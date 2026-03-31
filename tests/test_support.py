import io
from typing import Optional

import numpy as np


class ChunkedBytesIO(io.BytesIO):
    def __init__(self, data: bytes, max_chunk_bytes: Optional[int] = None):
        super().__init__(data)
        self._max_chunk_bytes = max_chunk_bytes

    def read(self, size: int = -1) -> bytes:
        if self._max_chunk_bytes is not None and size >= 0:
            size = min(size, self._max_chunk_bytes)
        return super().read(size)


class FakeProcess:
    def __init__(self, data: bytes, *, max_chunk_bytes: Optional[int] = None):
        self.stdout = ChunkedBytesIO(data, max_chunk_bytes=max_chunk_bytes)
        self.returncode = None

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        if self.stdout.tell() >= len(self.stdout.getbuffer()):
            self.returncode = 0
            return self.returncode
        return None

    def terminate(self):
        self.returncode = 0

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self):
        self.returncode = -9


def pack_raw_pairs(pairs):
    return np.asarray(list(pairs), dtype=np.int32).reshape(-1, 2).tobytes()


def pack_tagged_word(tag: int, payload: int, *, payload_bits: int = 18, tag_shift: int = 30) -> int:
    mask = (1 << payload_bits) - 1
    payload_u32 = payload & mask
    word = (int(tag) << tag_shift) | payload_u32
    return int(np.asarray([np.uint32(word)], dtype=np.uint32).view(np.int32)[0])


def pack_tagged_pairs(entries, *, payload_bits: int = 18, tag_shift: int = 30):
    pairs = [
        (
            pack_tagged_word(tag, left, payload_bits=payload_bits, tag_shift=tag_shift),
            pack_tagged_word(tag, right, payload_bits=payload_bits, tag_shift=tag_shift),
        )
        for tag, left, right in entries
    ]
    return pack_raw_pairs(pairs)
