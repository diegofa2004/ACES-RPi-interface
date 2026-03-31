import sys
import unittest
import importlib
from pathlib import Path

import numpy as np

TEST_ROOT = Path(__file__).resolve().parents[1]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

comparar_evento_module = importlib.import_module("rpi3b_i2s_fft.compararEvento")


class CompararEventoHelperTests(unittest.TestCase):
    def test_window_mean_helpers_match_naive_reference(self):
        data_1d = np.asarray([1, 2, 3, 4, 5], dtype=np.float32)
        data_2d = np.asarray([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=np.float32)

        mean_1d = comparar_evento_module._window_mean_1d(data_1d, 3, 1)
        mean_2d = comparar_evento_module._window_mean_2d(data_2d, 2, 1)

        np.testing.assert_allclose(mean_1d, np.asarray([2.0, 3.0, 4.0], dtype=np.float32))
        np.testing.assert_allclose(
            mean_2d,
            np.asarray([[1.5, 15.0], [2.5, 25.0], [3.5, 35.0]], dtype=np.float32),
        )

    def test_melhor_bloco_continuo_tracks_high_energy_region(self):
        energia = np.zeros(40, dtype=np.float32)
        energia[12:20] = np.asarray([2, 4, 6, 8, 8, 6, 4, 2], dtype=np.float32)
        i0, i1 = comparar_evento_module._melhor_bloco_continuo(energia, min_frac=0.2, max_frac=0.4)
        self.assertGreaterEqual(i0, 12)
        self.assertLessEqual(i1, 20)


if __name__ == "__main__":
    unittest.main()
