import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


TEST_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = TEST_ROOT / "rpi3b_i2s_fft" / "run_channel_debug_matrix.sh"


class RunChannelDebugMatrixScriptTests(unittest.TestCase):
    def test_dry_run_generates_shared_raw_capture_and_replay_commands(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "artifacts"
            proc = subprocess.run(
                [
                    "bash",
                    str(SCRIPT_PATH),
                    "--dry-run",
                    "--python",
                    sys.executable,
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=TEST_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(proc.returncode, 0, msg=proc.stderr)

            commands = (output_dir / "replay_commands.sh").read_text(encoding="utf-8")
            manifest = (output_dir / "session_info.txt").read_text(encoding="utf-8")
            readme = (output_dir / "README.txt").read_text(encoding="utf-8")
            summary = (output_dir / "scenario_summary.tsv").read_text(encoding="utf-8")

        self.assertIn("mirror_capture.raw", commands)
        self.assertIn("mirror_capture_auto.csv", commands)
        self.assertEqual(commands.count("fft_i2s_logger.py"), 1)
        self.assertGreaterEqual(commands.count("verify_transport_stream.py"), 3)
        self.assertIn("capture_backend=auto", manifest)
        self.assertIn("channel_modes=auto left right", manifest)
        self.assertIn("Mic mirror debug session", readme)
        self.assertIn("mirror_capture.raw", readme)
        self.assertIn("scenario\tstatus\tsample_count", summary)


if __name__ == "__main__":
    unittest.main()
