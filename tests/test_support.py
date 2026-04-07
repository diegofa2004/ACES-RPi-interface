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


def pack_tagged_word(
    tag: int,
    payload: int,
    *,
    packet_index: int = 0,
    packet_index_bits: int = 10,
    packet_index_shift: int = 22,
    payload_bits: int = 18,
    tag_shift: int = 20,
) -> int:
    mask = (1 << payload_bits) - 1
    payload_u32 = payload & mask
    packet_index_mask = (1 << packet_index_bits) - 1
    packet_index_u32 = int(packet_index) & packet_index_mask
    word = (packet_index_u32 << packet_index_shift) | (int(tag) << tag_shift) | payload_u32
    return int(np.asarray([np.uint32(word)], dtype=np.uint32).view(np.int32)[0])


def pack_tagged_pairs(
    entries,
    *,
    packet_index_bits: int = 10,
    packet_index_shift: int = 22,
    fft_packet_index_base: int = 1 << 9,
    payload_bits: int = 18,
    tag_shift: int = 20,
):
    next_bfpexp_index = 0
    next_fft_index = 0
    previous_tag = None
    last_active_tag = None
    pairs = []
    for entry in entries:
        if len(entry) == 4:
            packet_index, tag, left, right = entry
        else:
            tag, left, right = entry
            if tag == 1 and tag != previous_tag:
                if tag == 1:
                    next_bfpexp_index = 0
            elif tag == 2 and last_active_tag != 2:
                next_fft_index = 0
            if tag == 1:
                packet_index = next_bfpexp_index
            elif tag == 2:
                packet_index = fft_packet_index_base + next_fft_index
            else:
                packet_index = 0
        pairs.append(
            (
                pack_tagged_word(
                    tag,
                    left,
                    packet_index=packet_index,
                    packet_index_bits=packet_index_bits,
                    packet_index_shift=packet_index_shift,
                    payload_bits=payload_bits,
                    tag_shift=tag_shift,
                ),
                pack_tagged_word(
                    tag,
                    right,
                    packet_index=packet_index,
                    packet_index_bits=packet_index_bits,
                    packet_index_shift=packet_index_shift,
                    payload_bits=payload_bits,
                    tag_shift=tag_shift,
                ),
            )
        )
        if tag == 1:
            next_bfpexp_index += 1
            last_active_tag = 1
        elif tag == 2:
            next_fft_index += 1
            last_active_tag = 2
        previous_tag = tag
    return pack_raw_pairs(pairs)
