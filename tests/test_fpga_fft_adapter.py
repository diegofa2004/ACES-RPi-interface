import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from rpi3b_i2s_fft import fpga_fft_adapter
from rpi3b_i2s_fft import i2s_stream
from rpi3b_i2s_fft.fpga_fft_adapter import FFTAdapterConfig, FPGAFFTReceiver
from tests.test_support import FakeProcess, pack_raw_pairs, pack_tagged_pairs, pack_tagged_word


class FFTAdapterConfigTests(unittest.TestCase):
    def test_rejects_overlapping_tag_and_payload(self):
        with self.assertRaises(ValueError):
            FFTAdapterConfig(use_i2s_tags=True, tag_shift=10, payload_bits=18)


class FPGAFFTReceiverTests(unittest.TestCase):
    def test_receiver_polls_at_least_one_frame_per_read(self):
        cfg = FFTAdapterConfig(frame_bins=512, useful_bins=256)
        rx = FPGAFFTReceiver(cfg)

        self.assertEqual(rx._poll_pairs, 2048)
        self.assertEqual(rx._poll_bytes, 16384)

    def test_read_available_pairs_can_return_partial_chunk(self):
        cfg = FFTAdapterConfig(frame_bins=4, useful_bins=4)
        rx = FPGAFFTReceiver(cfg)
        rx._proc = FakeProcess(pack_raw_pairs([(1, 2), (3, 4), (5, 6)]), max_chunk_bytes=16)

        pairs = rx.read_available_pairs(3)
        self.assertIsNotNone(pairs)
        assert pairs is not None
        self.assertEqual(pairs.shape, (2, 2))
        np.testing.assert_array_equal(pairs, np.asarray([[1, 2], [3, 4]], dtype=np.int32))

    def test_start_and_stop_use_capture_process_helpers(self):
        fake_proc = FakeProcess(b"")
        cfg = FFTAdapterConfig(device="auto")
        rx = FPGAFFTReceiver(cfg)

        with mock.patch.object(fpga_fft_adapter, "resolve_audio_device", return_value="hw:2,0"), \
             mock.patch.object(fpga_fft_adapter, "build_capture_cmd", return_value=["fake-capture", "--raw"]) as build_mock, \
             mock.patch.object(fpga_fft_adapter, "start_capture_process", return_value=fake_proc) as start_mock, \
             mock.patch.object(fpga_fft_adapter, "stop_process") as stop_mock:
            rx.start()
            build_mock.assert_called_once_with(
                "hw:2,0",
                48828,
                backend="auto",
                capture_binary=None,
            )
            start_mock.assert_called_once_with(
                "hw:2,0",
                48828,
                backend="auto",
                capture_binary=None,
            )
            rx.stop()
            stop_mock.assert_called_once_with(fake_proc)

    def test_read_frame_in_tagged_mode_decodes_bfpexp_then_fft_pairs(self):
        cfg = FFTAdapterConfig(
            frame_bins=4,
            useful_bins=4,
            use_i2s_tags=True,
            bfpexp_pairs_required=1,
            handshake_timeout_seconds=0.01,
        )
        rx = FPGAFFTReceiver(cfg)
        stream = pack_tagged_pairs(
            [
                (0, 0, 0),
                (1, 7, 7),
                (1, 7, 7),
                (2, 3, 4),
                (2, 5, 12),
                (2, -8, 15),
                (2, 7, -24),
            ]
        )
        rx._proc = FakeProcess(stream)

        frame = rx.read_frame()
        self.assertIsNotNone(frame)
        fft_bins, mfcc = frame
        np.testing.assert_allclose(fft_bins, np.asarray([5.0, 13.0, 17.0, 25.0], dtype=np.float32))
        self.assertEqual(mfcc.shape, (13,))

    def test_tagged_mode_requires_bfpexp_by_default_and_can_be_relaxed(self):
        stream = pack_tagged_pairs(
            [
                (2, 1, 2),
                (2, 3, 4),
                (2, 5, 12),
                (2, 8, 15),
            ]
        )

        strict_rx = FPGAFFTReceiver(
            FFTAdapterConfig(frame_bins=4, useful_bins=4, use_i2s_tags=True, handshake_timeout_seconds=0.01)
        )
        strict_rx._proc = FakeProcess(stream)
        self.assertIsNone(strict_rx.read_frame())

        relaxed_rx = FPGAFFTReceiver(
            FFTAdapterConfig(
                frame_bins=4,
                useful_bins=4,
                use_i2s_tags=True,
                require_bfpexp_before_fft=False,
                handshake_timeout_seconds=0.01,
            )
        )
        relaxed_rx._proc = FakeProcess(stream)
        frame = relaxed_rx.read_frame()
        self.assertIsNotNone(frame)
        fft_bins, _ = frame
        np.testing.assert_allclose(fft_bins, np.asarray([np.sqrt(5.0), 5.0, 13.0, 17.0], dtype=np.float32))

    def test_tagged_mode_with_done_line_can_bootstrap_from_fft_without_bfpexp(self):
        cfg = FFTAdapterConfig(
            frame_bins=4,
            useful_bins=4,
            use_i2s_tags=True,
            done_line=24,
            handshake_timeout_seconds=0.01,
        )
        rx = FPGAFFTReceiver(cfg)
        rx._proc = FakeProcess(
            pack_tagged_pairs(
                [
                    (2, 1, 2),
                    (2, 3, 4),
                    (2, 5, 12),
                    (2, 8, 15),
                ]
            )
        )

        frame = rx.read_frame()
        self.assertIsNotNone(frame)
        fft_bins, _ = frame
        np.testing.assert_allclose(fft_bins, np.asarray([np.sqrt(5.0), 5.0, 13.0, 17.0], dtype=np.float32))

    def test_tagged_mode_discards_idle_pairs_while_searching_for_frame(self):
        cfg = FFTAdapterConfig(
            frame_bins=4,
            useful_bins=4,
            use_i2s_tags=True,
            bfpexp_pairs_required=1,
            handshake_timeout_seconds=0.01,
        )
        rx = FPGAFFTReceiver(cfg)
        rx._proc = FakeProcess(
            pack_tagged_pairs(
                [
                    (0, 0, 0),
                    (0, 0, 0),
                    (1, 7, 7),
                    (2, 3, 4),
                    (2, 5, 12),
                    (2, -8, 15),
                    (2, 7, -24),
                ]
            )
        )

        frame = rx.read_frame()
        self.assertIsNotNone(frame)
        fft_bins, _ = frame
        np.testing.assert_allclose(fft_bins, np.asarray([5.0, 13.0, 17.0, 25.0], dtype=np.float32))

    def test_tagged_mode_can_require_full_bfpexp_preamble(self):
        strict_cfg = FFTAdapterConfig(
            frame_bins=4,
            useful_bins=4,
            use_i2s_tags=True,
            bfpexp_pairs_required=3,
            handshake_timeout_seconds=0.01,
        )
        strict_rx = FPGAFFTReceiver(strict_cfg)
        strict_rx._proc = FakeProcess(
            pack_tagged_pairs(
                [
                    (0, 0, 0),
                    (1, 9, 9),
                    (1, 9, 9),
                    (2, 3, 4),
                    (2, 5, 12),
                    (2, -8, 15),
                    (2, 7, -24),
                ]
            )
        )

        self.assertIsNone(strict_rx.read_frame())

        valid_rx = FPGAFFTReceiver(strict_cfg)
        valid_rx._proc = FakeProcess(
            pack_tagged_pairs(
                [
                    (0, 0, 0),
                    (1, 9, 9),
                    (1, 9, 9),
                    (1, 9, 9),
                    (2, 3, 4),
                    (2, 5, 12),
                    (2, -8, 15),
                    (2, 7, -24),
                ]
            )
        )

        frame = valid_rx.read_frame()
        self.assertIsNotNone(frame)
        fft_bins, _ = frame
        np.testing.assert_allclose(fft_bins, np.asarray([5.0, 13.0, 17.0, 25.0], dtype=np.float32))

    def test_tagged_mode_resets_partial_bfpexp_run_when_idle_breaks_preamble(self):
        cfg = FFTAdapterConfig(
            frame_bins=4,
            useful_bins=4,
            use_i2s_tags=True,
            bfpexp_pairs_required=3,
            handshake_timeout_seconds=0.01,
        )
        rx = FPGAFFTReceiver(cfg)
        rx._proc = FakeProcess(
            pack_tagged_pairs(
                [
                    (0, 0, 0),
                    (1, 9, 9),
                    (1, 9, 9),
                    (0, 0, 0),
                    (1, 5, 5),
                    (1, 5, 5),
                    (1, 5, 5),
                    (2, 3, 4),
                    (2, 5, 12),
                    (2, -8, 15),
                    (2, 7, -24),
                ]
            )
        )

        frame = rx.read_frame()
        self.assertIsNotNone(frame)
        fft_bins, _ = frame
        np.testing.assert_allclose(fft_bins, np.asarray([5.0, 13.0, 17.0, 25.0], dtype=np.float32))

    def test_tagged_mode_tolerates_loss_inside_preamble_and_fft_frame(self):
        cfg = FFTAdapterConfig(
            frame_bins=4,
            useful_bins=4,
            use_i2s_tags=True,
            bfpexp_pairs_required=2,
            loss_tolerance_pairs=1,
            handshake_timeout_seconds=0.01,
        )
        rx = FPGAFFTReceiver(cfg)
        mismatch_stream = np.asarray(
            [
                [pack_tagged_word(1, 9), pack_tagged_word(1, 9)],
                [pack_tagged_word(0, 0), pack_tagged_word(0, 0)],  # tolerated loss inside BFPEXP preamble
                [pack_tagged_word(2, 3), pack_tagged_word(2, 4)],
                [pack_tagged_word(2, 0), pack_tagged_word(1, 1)],  # tolerated tag mismatch inside FFT frame
                [pack_tagged_word(2, -8), pack_tagged_word(2, 15)],
                [pack_tagged_word(2, 7), pack_tagged_word(2, -24)],
                [pack_tagged_word(2, 11), pack_tagged_word(2, 5)],
                [pack_tagged_word(2, -3), pack_tagged_word(2, 6)],
            ],
            dtype=np.int32,
        ).tobytes()
        rx._proc = FakeProcess(mismatch_stream)

        frame = rx.read_frame()
        self.assertIsNotNone(frame)
        fft_bins, _ = frame
        np.testing.assert_allclose(fft_bins, np.asarray([5.0, 0.0, 17.0, 25.0], dtype=np.float32))

    def test_tagged_mode_ignores_idle_padding_inside_fft_frame(self):
        cfg = FFTAdapterConfig(
            frame_bins=4,
            useful_bins=4,
            use_i2s_tags=True,
            bfpexp_pairs_required=1,
            loss_tolerance_pairs=0,
            handshake_timeout_seconds=0.01,
        )
        rx = FPGAFFTReceiver(cfg)
        rx._proc = FakeProcess(
            pack_tagged_pairs(
                [
                    (1, 9, 9),
                    (2, 3, 4),
                    (0, 0, 0),
                    (2, 5, 12),
                    (0, 0, 0),
                    (2, -8, 15),
                    (2, 7, -24),
                ]
            )
        )

        frame = rx.read_frame()
        self.assertIsNotNone(frame)
        fft_bins, _ = frame
        np.testing.assert_allclose(fft_bins, np.asarray([5.0, 13.0, 17.0, 25.0], dtype=np.float32))

    def test_tagged_mode_resynchronizes_after_broken_frame(self):
        cfg = FFTAdapterConfig(
            frame_bins=4,
            useful_bins=4,
            use_i2s_tags=True,
            bfpexp_pairs_required=1,
            handshake_timeout_seconds=0.01,
        )
        rx = FPGAFFTReceiver(cfg)
        stream = pack_tagged_pairs(
            [
                (1, 9, 9),
                (2, 1, 2),
                (2, 3, 4),
                (0, 0, 0),
                (1, 5, 5),
                (2, 3, 4),
                (2, 5, 12),
                (2, 8, 15),
                (2, 7, 24),
            ]
        )
        rx._proc = FakeProcess(stream)

        frame = rx.read_frame()
        self.assertIsNotNone(frame)
        fft_bins, _ = frame
        np.testing.assert_allclose(fft_bins, np.asarray([5.0, 13.0, 17.0, 25.0], dtype=np.float32))

    def test_tagged_mode_realigns_bit_shifted_stream(self):
        cfg = FFTAdapterConfig(
            frame_bins=4,
            useful_bins=4,
            use_i2s_tags=True,
            require_bfpexp_before_fft=False,
            handshake_timeout_seconds=0.01,
        )
        rx = FPGAFFTReceiver(cfg)

        stream = pack_tagged_pairs(
            [
                (1, 7, 7),
                (2, 3, 4),
                (2, 5, 12),
                (2, -8, 15),
                (2, 7, -24),
                (1, 9, 9),
            ]
        )
        words = np.frombuffer(stream, dtype=np.int32).astype(np.uint32)
        misframed = i2s_stream._reframe_tagged_words(words, 14)
        misframed = misframed[: (misframed.size // 2) * 2].astype(np.uint32).view(np.int32).tobytes()
        rx._proc = FakeProcess(misframed)

        frame = rx.read_frame()
        self.assertIsNotNone(frame)
        fft_bins, _ = frame
        np.testing.assert_allclose(fft_bins, np.asarray([5.0, 13.0, 17.0, 25.0], dtype=np.float32))

    def test_tagged_mode_uses_configured_tag_layout_in_realigner(self):
        cfg = FFTAdapterConfig(
            frame_bins=4,
            useful_bins=4,
            use_i2s_tags=True,
            tag_shift=28,
            payload_bits=16,
            require_bfpexp_before_fft=False,
            handshake_timeout_seconds=0.01,
        )
        rx = FPGAFFTReceiver(cfg)
        rx._proc = FakeProcess(
            pack_tagged_pairs(
                [
                    (1, 7, 7),
                    (2, 3, 4),
                    (2, 5, 12),
                    (2, -8, 15),
                    (2, 7, -24),
                ],
                payload_bits=16,
                tag_shift=28,
            )
        )

        frame = rx.read_frame()
        self.assertIsNotNone(frame)
        fft_bins, _ = frame
        np.testing.assert_allclose(fft_bins, np.asarray([5.0, 13.0, 17.0, 25.0], dtype=np.float32))

    def test_pair_kind_reports_tag_mismatch_explicitly(self):
        cfg = FFTAdapterConfig(frame_bins=4, useful_bins=4, use_i2s_tags=True)
        rx = FPGAFFTReceiver(cfg)
        pair = np.asarray(
            [
                pack_tagged_word(2, 5),
                pack_tagged_word(1, 5),
            ],
            dtype=np.int32,
        )

        kind, payload = rx._pair_kind_and_payload(pair)

        self.assertEqual(kind, "tag_mismatch")
        self.assertEqual(payload, (5, 5))

    def test_raw_mode_reads_exact_frame_without_tags(self):
        cfg = FFTAdapterConfig(frame_bins=3, useful_bins=3, use_i2s_tags=False)
        rx = FPGAFFTReceiver(cfg)
        rx._proc = FakeProcess(pack_raw_pairs([(1, 2), (3, 4), (5, 12)]))

        frame = rx.read_frame()
        self.assertIsNotNone(frame)
        fft_bins, _ = frame
        np.testing.assert_allclose(fft_bins, np.asarray([np.sqrt(5.0), 5.0, 13.0], dtype=np.float32))

    def test_raw_mode_can_wait_for_gpio_falling_edge_before_consuming_frame(self):
        cfg = FFTAdapterConfig(
            frame_bins=3,
            useful_bins=3,
            use_i2s_tags=False,
            bfpexp_flag_line=23,
            handshake_timeout_seconds=0.01,
        )
        rx = FPGAFFTReceiver(cfg)
        rx._proc = FakeProcess(
            pack_raw_pairs([(100, 100), (200, 200), (1, 2), (3, 4), (5, 12)]),
            max_chunk_bytes=16,
        )
        rx._bfpexp_line = object()
        rx._gpio_api = "v1"
        rx._read_flag_active = mock.Mock(side_effect=[True, False])

        frame = rx.read_frame()
        self.assertIsNotNone(frame)
        fft_bins, _ = frame
        np.testing.assert_allclose(fft_bins, np.asarray([np.sqrt(5.0), 5.0, 13.0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
