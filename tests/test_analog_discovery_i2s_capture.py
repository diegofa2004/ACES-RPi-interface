import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from rpi3b_i2s_fft.analog_discovery_i2s_capture import (
    DecodeConfig,
    decode_i2s_words,
    write_frames_csv,
    write_frames_raw,
    write_samples_csv,
    write_words_csv,
)


def _ws_for_channel(channel: str, ws_low_channel: str) -> int:
    low_channel = ws_low_channel.lower()
    if channel == low_channel:
        return 0
    return 1


def _bus_value(sck: int, ws: int, sd: int, *, dio_sck: int, dio_ws: int, dio_sd: int) -> int:
    value = 0
    value |= (sck & 0x1) << dio_sck
    value |= (ws & 0x1) << dio_ws
    value |= (sd & 0x1) << dio_sd
    return value


def _synthesize_i2s_samples(
    slots: list[tuple[str, int]],
    *,
    bits_per_word: int,
    dio_sck: int,
    dio_ws: int,
    dio_sd: int,
    ws_low_channel: str,
) -> list[int]:
    samples: list[int] = []
    for index, (channel, word) in enumerate(slots):
        current_ws = _ws_for_channel(channel, ws_low_channel)
        next_ws = current_ws
        if index + 1 < len(slots):
            next_ws = _ws_for_channel(slots[index + 1][0], ws_low_channel)

        for bit_index in range(bits_per_word - 1, -1, -1):
            ws = next_ws if bit_index == 0 else current_ws
            sd = (word >> bit_index) & 0x1
            samples.append(_bus_value(0, ws, sd, dio_sck=dio_sck, dio_ws=dio_ws, dio_sd=dio_sd))
            samples.append(_bus_value(1, ws, sd, dio_sck=dio_sck, dio_ws=dio_ws, dio_sd=dio_sd))

    return samples


class AnalogDiscoveryI2SCaptureTests(unittest.TestCase):
    def test_decodes_raw_words_and_pairs_stereo_frames(self):
        config = DecodeConfig(
            sample_rate_hz=100_000_000.0,
            dio_sck=13,
            dio_ws=14,
            dio_sd=15,
            bits_per_word=32,
            ws_low_channel="left",
            sample_shift_bits=8,
        )

        preamble_word = 0x00000000
        left_word = 0x00123400
        right_word = 0xFFF00000
        tail_word = 0x00000000

        slots = [
            ("left", preamble_word),
            ("right", right_word),
            ("left", left_word),
            ("right", tail_word),
        ]
        samples = _synthesize_i2s_samples(
            slots,
            bits_per_word=config.bits_per_word,
            dio_sck=config.dio_sck,
            dio_ws=config.dio_ws,
            dio_sd=config.dio_sd,
            ws_low_channel=config.ws_low_channel,
        )

        summary = decode_i2s_words(samples, config)

        self.assertEqual(len(summary.words), 2)
        self.assertEqual([word.channel for word in summary.words], ["right", "left"])
        self.assertEqual(summary.words[0].value_signed, -1048576)
        self.assertEqual(summary.words[0].sample_signed, -4096)
        self.assertEqual(summary.words[1].value_signed, 1192960)
        self.assertEqual(summary.words[1].sample_signed, 4660)
        self.assertEqual(len(summary.frames), 1)
        self.assertEqual(summary.frames[0].left.sample_signed, 4660)
        self.assertEqual(summary.frames[0].right.sample_signed, -4096)
        self.assertEqual(summary.error_counts, {"ws_missing_on_last_bit": 1})

    def test_writers_emit_csv_and_raw_stereo_outputs(self):
        config = DecodeConfig(
            sample_rate_hz=100_000_000.0,
            dio_sck=13,
            dio_ws=14,
            dio_sd=15,
            bits_per_word=32,
            ws_low_channel="left",
            sample_shift_bits=8,
        )

        slots = [
            ("left", 0),
            ("right", 0xFFF00000),
            ("left", 0x00123400),
            ("right", 0),
        ]
        samples = _synthesize_i2s_samples(
            slots,
            bits_per_word=config.bits_per_word,
            dio_sck=config.dio_sck,
            dio_ws=config.dio_ws,
            dio_sd=config.dio_sd,
            ws_low_channel=config.ws_low_channel,
        )
        summary = decode_i2s_words(samples, config)

        with tempfile.TemporaryDirectory() as tmpdir:
            samples_csv = Path(tmpdir) / "samples.csv"
            words_csv = Path(tmpdir) / "words.csv"
            frames_csv = Path(tmpdir) / "frames.csv"
            frames_raw = Path(tmpdir) / "frames.raw"

            write_samples_csv(samples_csv, samples, summary, config)
            write_words_csv(words_csv, summary.words)
            write_frames_csv(frames_csv, summary.frames)
            write_frames_raw(frames_raw, summary.frames)

            with samples_csv.open(newline="", encoding="utf-8") as handle:
                sample_rows = list(csv.DictReader(handle))
            with words_csv.open(newline="", encoding="utf-8") as handle:
                word_rows = list(csv.DictReader(handle))
            with frames_csv.open(newline="", encoding="utf-8") as handle:
                frame_rows = list(csv.DictReader(handle))
            raw_bytes = frames_raw.read_bytes()

        self.assertEqual(len(sample_rows), len(samples))
        self.assertEqual(len(word_rows), len(summary.words))
        self.assertEqual(len(frame_rows), len(summary.frames))
        self.assertEqual(word_rows[0]["word_hex"], "0xFFF00000")
        self.assertEqual(word_rows[0]["value_signed"], "-1048576")
        self.assertEqual(word_rows[0]["sample_signed"], "-4096")
        decoded_rows = [row for row in sample_rows if row["decoded_word_hex"]]
        self.assertEqual(len(decoded_rows), 2)
        self.assertEqual(decoded_rows[0]["decoded_channel"], "right")
        self.assertEqual(decoded_rows[0]["decoded_sample_signed"], "-4096")
        self.assertEqual(frame_rows[0]["left_sample_signed"], "4660")
        self.assertEqual(frame_rows[0]["right_sample_signed"], "-4096")
        raw_stereo = np.frombuffer(raw_bytes, dtype=np.int32).reshape(-1, 2)
        np.testing.assert_array_equal(raw_stereo, np.asarray([[4660, -4096]], dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
