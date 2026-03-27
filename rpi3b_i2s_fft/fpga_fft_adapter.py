import subprocess
from dataclasses import dataclass
from typing import Optional, Tuple

import librosa
import numpy as np
from scipy.fftpack import dct


@dataclass
class FFTAdapterConfig:
    device: str = "hw:0,0"
    sample_rate: int = 48000
    frame_bins: int = 512
    useful_bins: int = 256


class FPGAFFTReceiver:
    def __init__(self, cfg: FFTAdapterConfig):
        self.cfg = cfg
        self._proc: Optional[subprocess.Popen] = None
        self._bytes_per_pair = 8  # real(int32) + imag(int32)
        self._frame_bytes = self.cfg.frame_bins * self._bytes_per_pair

        n_fft = max(2, 2 * (self.cfg.useful_bins - 1))
        self.mel_filter = librosa.filters.mel(
            sr=self.cfg.sample_rate,
            n_fft=n_fft,
            n_mels=32,
        ).astype(np.float32)

    @staticmethod
    def _build_arecord_cmd(device: str, sample_rate: int) -> list:
        return [
            "arecord",
            "-q",
            "-D",
            device,
            "-f",
            "S32_LE",
            "-c",
            "2",
            "-r",
            str(sample_rate),
            "-t",
            "raw",
        ]

    def start(self) -> None:
        cmd = self._build_arecord_cmd(self.cfg.device, self.cfg.sample_rate)
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def stop(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def read_frame(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self._proc is None or self._proc.stdout is None:
            raise RuntimeError("Receiver not started")

        raw = self._proc.stdout.read(self._frame_bytes)
        if len(raw) != self._frame_bytes:
            return None

        pairs = np.frombuffer(raw, dtype=np.int32).reshape(-1, 2)
        real = pairs[:, 0].astype(np.float32)
        imag = pairs[:, 1].astype(np.float32)

        # Magnitude spectrum from complex bins streamed by FPGA.
        fft_mag = np.sqrt(real * real + imag * imag)
        fft_useful = fft_mag[: self.cfg.useful_bins]

        mel = self.mel_filter @ fft_useful
        mel = np.log(mel + 1e-9)
        mfcc = dct(mel, type=2, norm="ortho")[:13].astype(np.float32)

        return fft_useful.astype(np.float32), mfcc
