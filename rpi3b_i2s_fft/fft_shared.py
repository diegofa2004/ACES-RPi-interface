import struct
import time
from multiprocessing import shared_memory
from typing import Dict

# Layout: real(int32), imag(int32), seq(uint32), status(uint32)
_STRUCT = struct.Struct("<iiII")
DEFAULT_SHM_NAME = "fft_i2s_latest"
STATUS_OK = 0
STATUS_NO_DATA = 1


class FFTSharedState:
    def __init__(self, name: str = DEFAULT_SHM_NAME, create: bool = False):
        self.name = name
        self._owns = create
        if create:
            try:
                self.shm = shared_memory.SharedMemory(name=name, create=True, size=_STRUCT.size)
            except FileExistsError:
                # Recover from stale segments left behind by unclean shutdowns.
                stale = shared_memory.SharedMemory(name=name, create=False)
                stale.close()
                stale.unlink()
                self.shm = shared_memory.SharedMemory(name=name, create=True, size=_STRUCT.size)
            self.write(0, 0, 0, STATUS_NO_DATA)
        else:
            self.shm = shared_memory.SharedMemory(name=name, create=False)

    def close(self) -> None:
        self.shm.close()

    def unlink(self) -> None:
        if self._owns:
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass

    def write(self, real: int, imag: int, seq: int, status: int = STATUS_OK) -> None:
        self.shm.buf[:_STRUCT.size] = _STRUCT.pack(int(real), int(imag), int(seq), int(status))

    def read(self) -> Dict[str, int]:
        real, imag, seq, status = _STRUCT.unpack(bytes(self.shm.buf[:_STRUCT.size]))
        return {
            "real": real,
            "imag": imag,
            "seq": seq,
            "status": status,
            "timestamp_ns": time.time_ns(),
        }
