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
            summary_header = (output_dir / "scenario_summary.tsv").read_text(encoding="utf-8").splitlines()[0]

        self.assertIn("--debug-raw-capture", commands)
        self.assertEqual(commands.count("--debug-raw-capture"), 1)
        self.assertGreaterEqual(commands.count("--debug-replay-raw"), 3)
        self.assertIn("raw_capture=", manifest)
        self.assertIn("raw_index=", manifest)
        self.assertIn("channel_capture.raw", readme)
        self.assertIn("channel_capture.index.jsonl", readme)
        self.assertIn("capture_xruns", summary_header)
        self.assertIn("capture_queue_high_water", summary_header)


if __name__ == "__main__":
    unittest.main()
