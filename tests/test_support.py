import io
from typing import Optional


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
