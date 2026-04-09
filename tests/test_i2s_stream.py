import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from rpi3b_i2s_fft import i2s_stream
from tests.test_support import pack_tagged_word


class I2SStreamTests(unittest.TestCase):
    @staticmethod
    def _misframe_pairs(pairs: np.ndarray, bit_offset: int) -> np.ndarray:
        words = np.asarray(pairs, dtype=np.int32).reshape(-1).astype(np.uint32)
        shifted = i2s_stream._reframe_tagged_words(words, bit_offset)
        shifted = shifted[: (shifted.size // 2) * 2]
        return shifted.view(np.int32).reshape(-1, 2)

    @staticmethod
    def _is_valid_tagged_output(pairs: np.ndarray) -> bool:
        if pairs.size == 0:
            return True
        metrics = i2s_stream._collect_tagged_alignment_metrics(
            np.asarray(pairs, dtype=np.int32).reshape(-1).astype(np.uint32),
            packet_index_shift=i2s_stream.DEFAULT_PACKET_INDEX_SHIFT,
            packet_index_bits=i2s_stream.DEFAULT_PACKET_INDEX_BITS,
            tag_shift=i2s_stream.DEFAULT_TAG_SHIFT,
            tag_mask=i2s_stream.DEFAULT_TAG_MASK,
            payload_bits=i2s_stream.DEFAULT_PAYLOAD_BITS,
            tag_idle=i2s_stream.DEFAULT_TAG_IDLE,
            tag_bfpexp=i2s_stream.DEFAULT_TAG_BFPEXP,
            tag_fft=i2s_stream.DEFAULT_TAG_FFT,
            fft_packet_index_base=i2s_stream.DEFAULT_FFT_PACKET_INDEX_BASE,
            search_pair_limit=max(4, len(pairs)),
        )
        return metrics.pair_count > 0 and metrics.invalid_pairs == 0 and metrics.good_pairs == metrics.pair_count

    @staticmethod
    def _make_reference_pairs() -> np.ndarray:
        bfpexp_word = pack_tagged_word(1, 0x12, packet_index=0)
        fft_word = pack_tagged_word(2, 0x15555, packet_index=i2s_stream.DEFAULT_FFT_PACKET_INDEX_BASE)
        fft_word_1 = pack_tagged_word(2, 0x15555, packet_index=i2s_stream.DEFAULT_FFT_PACKET_INDEX_BASE + 1)
        return np.asarray(
            [
                [bfpexp_word, bfpexp_word],
                [fft_word, fft_word],
                [fft_word_1, fft_word_1],
                [bfpexp_word, bfpexp_word],
                [fft_word, fft_word],
                [fft_word_1, fft_word_1],
            ],
            dtype=np.int32,
        )

    def test_build_arecord_cmd_uses_expected_format(self):
        cmd = i2s_stream.build_arecord_cmd("hw:1,0", i2s_stream.DEFAULT_CAPTURE_RATE_HZ)
        self.assertEqual(
            cmd,
            [
                "arecord",
                "-q",
                "-D",
                "hw:1,0",
                "-B",
                str(i2s_stream.DEFAULT_ARECORD_BUFFER_TIME_US),
                "-F",
                str(i2s_stream.DEFAULT_ARECORD_PERIOD_TIME_US),
                "-f",
                "S32_LE",
                "-c",
                "2",
                "-r",
                str(i2s_stream.DEFAULT_CAPTURE_RATE_HZ),
                "-t",
                "raw",
            ],
        )

    def test_resolve_capture_command_defaults_to_arecord_when_native_helper_is_missing(self):
        with mock.patch.object(i2s_stream, "find_native_capture_binary", return_value=None):
            backend, cmd = i2s_stream.resolve_capture_command("hw:1,0", i2s_stream.DEFAULT_CAPTURE_RATE_HZ)

        self.assertEqual(backend, i2s_stream.CAPTURE_BACKEND_ARECORD)
        self.assertEqual(cmd, i2s_stream.build_arecord_cmd("hw:1,0", i2s_stream.DEFAULT_CAPTURE_RATE_HZ))

    def test_resolve_capture_command_prefers_native_helper_when_present(self):
        helper_path = "/tmp/alsa_logger"
        with mock.patch.object(i2s_stream, "find_native_capture_binary", return_value=helper_path):
            backend, cmd = i2s_stream.resolve_capture_command("hw:1,0", i2s_stream.DEFAULT_CAPTURE_RATE_HZ)

        self.assertEqual(backend, i2s_stream.CAPTURE_BACKEND_NATIVE)
        self.assertEqual(cmd[0], helper_path)
        self.assertIn("--mode", cmd)
        self.assertIn("raw", cmd)

    def test_build_native_capture_cmd_enables_jsonl_telemetry_when_requested(self):
        cmd = i2s_stream.build_native_capture_cmd(
            "/tmp/alsa_logger",
            "hw:1,0",
            i2s_stream.DEFAULT_CAPTURE_RATE_HZ,
            capture_telemetry=True,
        )

        self.assertIn("--telemetry-format", cmd)
        telemetry_idx = cmd.index("--telemetry-format")
        self.assertEqual(cmd[telemetry_idx + 1], "jsonl")

    def test_build_native_capture_cmd_enables_helper_realign_when_requested(self):
        cmd = i2s_stream.build_native_capture_cmd(
            "/tmp/alsa_logger",
            "hw:1,0",
            i2s_stream.DEFAULT_CAPTURE_RATE_HZ,
            capture_realign_initial_word_skip=1,
            capture_realign_swap_channels=True,
        )

        self.assertIn("--realign-initial-word-skip", cmd)
        realign_idx = cmd.index("--realign-initial-word-skip")
        self.assertEqual(cmd[realign_idx + 1], "1")
        self.assertIn("--realign-swap-channels", cmd)

    def test_resolve_capture_command_requires_binary_for_explicit_native_backend(self):
        with mock.patch.object(i2s_stream, "find_native_capture_binary", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                i2s_stream.resolve_capture_command(
                    "hw:1,0",
                    i2s_stream.DEFAULT_CAPTURE_RATE_HZ,
                    backend=i2s_stream.CAPTURE_BACKEND_NATIVE,
                )

        self.assertIn("no compiled helper was found", str(ctx.exception))

    def test_resolve_capture_command_rejects_helper_realign_with_arecord_backend(self):
        with self.assertRaises(RuntimeError) as ctx:
            i2s_stream.resolve_capture_command(
                "hw:1,0",
                i2s_stream.DEFAULT_CAPTURE_RATE_HZ,
                backend=i2s_stream.CAPTURE_BACKEND_ARECORD,
                capture_realign_initial_word_skip=1,
            )

        self.assertIn("requires the native alsa_logger backend", str(ctx.exception))

    def test_resolve_capture_command_rejects_helper_realign_when_native_helper_is_missing(self):
        with mock.patch.object(i2s_stream, "find_native_capture_binary", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                i2s_stream.resolve_capture_command(
                    "hw:1,0",
                    i2s_stream.DEFAULT_CAPTURE_RATE_HZ,
                    capture_realign_initial_word_skip=1,
                )

        self.assertIn("helper realignment was requested", str(ctx.exception))

    def test_parse_capture_telemetry_line_accepts_jsonl_and_plain_text(self):
        json_payload = i2s_stream.parse_capture_telemetry_line(b'{"type":"capture_chunk","frames":512}\n')
        text_payload = i2s_stream.parse_capture_telemetry_line("plain stderr line\n")

        self.assertEqual(json_payload, {"type": "capture_chunk", "frames": 512})
        self.assertEqual(
            text_payload,
            {
                "type": "capture_stderr_text",
                "message": "plain stderr line",
            },
        )

    def test_decode_tagged_i2s_word_sign_extends_18bit_payload(self):
        decoded = i2s_stream.decode_tagged_i2s_word(0x0003FFDF)

        self.assertEqual(decoded["payload"], -33)
        self.assertEqual(decoded["tag"], 0)
        self.assertEqual(decoded["packet_index"], 0)
        self.assertEqual(decoded["reserved"], 0)

    def test_trim_incomplete_frames_discards_partial_tail(self):
        raw = b"\x00" * 19
        trimmed = i2s_stream.trim_incomplete_frames(raw, bytes_per_frame=8)
        self.assertEqual(len(trimmed), 16)

    def test_resolve_audio_device_prefers_keyword_match(self):
        proc = SimpleNamespace(
            stdout=(
                "card 0: Loopback [Loopback], device 0: Loopback PCM [Loopback PCM]\n"
                "card 2: acesfpgafft [aces-fpgafft], device 0: FPGA FFT Capture [FPGA FFT Capture]\n"
            )
        )
        with mock.patch.object(i2s_stream.subprocess, "run", return_value=proc):
            self.assertEqual(i2s_stream.resolve_audio_device("auto"), "hw:2,0")

    def test_resolve_audio_device_requires_explicit_choice_when_multiple_devices_match_nothing(self):
        proc = SimpleNamespace(
            stdout=(
                "card 0: Loopback [Loopback], device 0: Loopback PCM [Loopback PCM]\n"
                "card 1: USB [USB Audio], device 0: USB Audio [USB Audio]\n"
            )
        )
        with mock.patch.object(i2s_stream.subprocess, "run", return_value=proc):
            with self.assertRaises(RuntimeError) as ctx:
                i2s_stream.resolve_audio_device("auto")
        self.assertIn("Multiple ALSA capture devices were found", str(ctx.exception))

    def test_detect_tagged_alignment_recovers_expected_words_from_bit_shifted_stream(self):
        true_pairs = self._make_reference_pairs()
        observed = self._misframe_pairs(true_pairs, 14)

        alignment = i2s_stream.detect_tagged_i2s_alignment(observed)
        self.assertIsNotNone(alignment)

        realigner = i2s_stream.TaggedI2SRealigner(confirm_pairs=4, validate_pairs=4)
        recovered = realigner.push_pairs(observed)
        self.assertGreaterEqual(recovered.shape[0], 3)

        recovered_words = recovered.reshape(-1).astype(np.uint32)
        true_words = true_pairs.reshape(-1).astype(np.uint32)

        found_match = False
        for start in range(len(true_words) - len(recovered_words[:6]) + 1):
            if np.array_equal(recovered_words[:6], true_words[start : start + 6]):
                found_match = True
                break
        self.assertTrue(found_match)

    def test_tagged_realigner_survives_chunk_boundaries(self):
        true_pairs = self._make_reference_pairs()
        observed = self._misframe_pairs(true_pairs, 18)

        realigner = i2s_stream.TaggedI2SRealigner(confirm_pairs=4, validate_pairs=4)
        chunks = [observed[:2], observed[2:4], observed[4:]]
        recovered_parts = [realigner.push_pairs(chunk) for chunk in chunks]
        recovered = np.concatenate([part for part in recovered_parts if part.size], axis=0)

        self.assertGreaterEqual(recovered.shape[0], 3)
        self.assertTrue(
            np.array_equal(recovered[:3], true_pairs[1:4]) or np.array_equal(recovered[:3], true_pairs[2:5])
        )

    def test_tagged_realigner_relocks_after_midstream_phase_jump(self):
        bfpexp_word = pack_tagged_word(1, 0x12, packet_index=0)
        fft_word = pack_tagged_word(2, 0x15555, packet_index=i2s_stream.DEFAULT_FFT_PACKET_INDEX_BASE)
        segment_a = np.asarray(
            [[bfpexp_word, bfpexp_word]] + [[fft_word, fft_word]] * 8,
            dtype=np.int32,
        )
        segment_b = np.asarray(
            [[bfpexp_word, bfpexp_word]] + [[fft_word, fft_word]] * 8,
            dtype=np.int32,
        )
        observed_a = self._misframe_pairs(segment_a, 9)
        observed_b = self._misframe_pairs(segment_b, 23)
        observed = np.concatenate((observed_a, observed_b), axis=0)

        realigner = i2s_stream.TaggedI2SRealigner(confirm_pairs=4, validate_pairs=8)
        chunks = [observed[idx : idx + 3] for idx in range(0, observed.shape[0], 3)]
        recovered_parts = [realigner.push_pairs(chunk) for chunk in chunks]
        recovered = np.concatenate([part for part in recovered_parts if part.size], axis=0)

        self.assertGreaterEqual(recovered.shape[0], 4)
        self.assertTrue(self._is_valid_tagged_output(recovered))
        self.assertTrue(np.any(np.all(recovered == segment_a[1], axis=1)))


if __name__ == "__main__":
    unittest.main()
