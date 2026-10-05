import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

from config import DEMO_EXECUTION_ENABLED, TRADING_ENABLED


RUNNER_COMMANDS = ["PAPER", "PAPER EURUSD", "MT5 DEMO AUTO RUN", "LIVE PORTFOLIO"]
# El resumen diario de Telegram solo se agrega desde las 17:00 (hora local).
EXPECTED_RUNNER_COMMANDS = [RUNNER_COMMANDS, RUNNER_COMMANDS + ["TELEGRAM DAILY"]]


class RunnerBatTests(unittest.TestCase):
    def make_runner_project(self, main_source=None):
        tmp = tempfile.TemporaryDirectory(dir=Path.cwd())
        root = Path(tmp.name)
        shutil.copy2("run_paper.bat", root / "run_paper.bat")
        (root / "scripts").mkdir()
        shutil.copy2(Path("scripts") / "paper_lock.ps1", root / "scripts" / "paper_lock.ps1")

        scripts_dir = root / ".venv" / "Scripts"
        scripts_dir.mkdir(parents=True)
        source_python = Path(".venv") / "Scripts" / "python.exe"
        source_cfg = Path(".venv") / "pyvenv.cfg"
        if not source_python.exists() or not source_cfg.exists():
            tmp.cleanup()
            self.skipTest("Local Windows venv is required to execute run_paper.bat")
        shutil.copy2(source_python, scripts_dir / "python.exe")
        shutil.copy2(source_cfg, root / ".venv" / "pyvenv.cfg")

        if main_source is None:
            main_source = """
                import sys
                from pathlib import Path

                Path("logs").mkdir(exist_ok=True)
                with Path("logs/args.txt").open("a", encoding="utf-8") as handle:
                    handle.write(" ".join(sys.argv[1:]) + "\\n")
            """
        (root / "main.py").write_text(textwrap.dedent(main_source), encoding="utf-8")
        return tmp, root

    def run_temp_bat(self, root, timeout=30):
        return subprocess.run(
            [str(root / "run_paper.bat")],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    def read_summary_log(self, root):
        return (root / "logs" / "paper_runner.log").read_text(encoding="utf-8")

    def wait_for_lock(self, root, timeout=10):
        lock = root / "paper" / "paper.lock"
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if lock.exists() and lock.read_text(encoding="ascii").strip():
                    return lock
            except PermissionError:
                pass
            time.sleep(0.05)
        self.fail("Timed out waiting for paper lock")

    def test_run_paper_order_gold_eurusd_auto(self):
        text = Path("run_paper.bat").read_text(encoding="utf-8")

        gold = text.index("@('PAPER') $goldLog")
        eurusd = text.index("@('PAPER', 'EURUSD') $eurusdLog")
        auto = text.index("@('MT5', 'DEMO', 'AUTO', 'RUN') $autoLog")

        self.assertLess(gold, eurusd)
        self.assertLess(eurusd, auto)

    def test_run_paper_runs_portfolio_after_forwards(self):
        text = Path("run_paper.bat").read_text(encoding="utf-8")

        self.assertIn("mt5_demo_auto_runner.log", text)
        self.assertIn("live_portfolio_runner.log", text)
        # Los forwards EURUSD/GOLD se ejecutan solo via MT5 DEMO AUTO, nunca via LIVE.
        self.assertNotIn("@('LIVE', 'GOLD')", text)
        self.assertNotIn("@('LIVE', 'EURUSD')", text)
        auto = text.index("@('MT5', 'DEMO', 'AUTO', 'RUN') $autoLog")
        portfolio = text.index("@('LIVE', 'PORTFOLIO') $portfolioLog")
        self.assertLess(auto, portfolio)

    def test_flags_remain_safe_by_default(self):
        self.assertIs(TRADING_ENABLED, False)
        self.assertIs(DEMO_EXECUTION_ENABLED, False)

    @unittest.skipIf(os.name != "nt", "run_paper.bat lock behavior is Windows-only")
    def test_run_paper_continues_without_existing_lock(self):
        tmp, root = self.make_runner_project()
        with tmp:
            result = self.run_temp_bat(root)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((root / "paper" / "paper.lock").exists())
            args = (root / "logs" / "args.txt").read_text(encoding="utf-8").splitlines()
            self.assertIn(args, EXPECTED_RUNNER_COMMANDS)
            self.assertNotIn("STALE LOCK DETECTED", self.read_summary_log(root))

    @unittest.skipIf(os.name != "nt", "run_paper.bat lock behavior is Windows-only")
    def test_run_paper_exits_when_lock_pid_is_active_runner(self):
        tmp, root = self.make_runner_project()
        process = None
        with tmp:
            (root / "paper").mkdir()
            process = subprocess.Popen(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    (
                        "$path = Join-Path $PWD 'paper\\paper.lock'; "
                        "$stream = [System.IO.File]::Open($path, [System.IO.FileMode]::Create, "
                        "[System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::Read); "
                        "$bytes = [System.Text.Encoding]::ASCII.GetBytes([string]$PID); "
                        "$stream.Write($bytes, 0, $bytes.Length); $stream.Flush(); "
                        "Start-Sleep -Seconds 30; $stream.Close()"
                    ),
                ],
                cwd=root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                self.wait_for_lock(root)

                result = self.run_temp_bat(root)

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual((root / "paper" / "paper.lock").read_text(encoding="ascii").strip(), str(process.pid))
                summary = self.read_summary_log(root)
                self.assertIn(f"PAPER ya esta en ejecucion. PID activo: {process.pid}", summary)
                self.assertFalse((root / "logs" / "args.txt").exists())
            finally:
                if process is not None and process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)

    @unittest.skipIf(os.name != "nt", "run_paper.bat lock behavior is Windows-only")
    def test_run_paper_removes_stale_lock_with_missing_pid(self):
        tmp, root = self.make_runner_project()
        with tmp:
            (root / "paper").mkdir()
            (root / "paper" / "paper.lock").write_text("999999", encoding="ascii")

            result = self.run_temp_bat(root)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((root / "paper" / "paper.lock").exists())
            summary = self.read_summary_log(root)
            self.assertIn("STALE LOCK DETECTED", summary)
            self.assertIn("LOCK PID: 999999", summary)

    @unittest.skipIf(os.name != "nt", "run_paper.bat lock behavior is Windows-only")
    def test_run_paper_removes_malformed_lock(self):
        tmp, root = self.make_runner_project()
        with tmp:
            (root / "paper").mkdir()
            (root / "paper" / "paper.lock").write_text("not-a-pid", encoding="ascii")

            result = self.run_temp_bat(root)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((root / "paper" / "paper.lock").exists())
            summary = self.read_summary_log(root)
            self.assertIn("STALE LOCK DETECTED", summary)
            self.assertIn("LOCK PID: not-a-pid", summary)

    @unittest.skipIf(os.name != "nt", "run_paper.bat lock behavior is Windows-only")
    def test_run_paper_cleans_own_lock_when_finished(self):
        main_source = """
            from pathlib import Path

            lock = Path("paper/paper.lock")
            Path("logs").mkdir(exist_ok=True)
            Path("logs/lock_seen.txt").write_text(
                str(lock.exists()) + ":" + lock.read_text(encoding="ascii").strip(),
                encoding="utf-8",
            )
        """
        tmp, root = self.make_runner_project(main_source)
        with tmp:
            result = self.run_temp_bat(root)

            self.assertEqual(result.returncode, 0, result.stderr)
            seen = (root / "logs" / "lock_seen.txt").read_text(encoding="utf-8")
            self.assertRegex(seen, r"^True:\d+$")
            self.assertFalse((root / "paper" / "paper.lock").exists())

    @unittest.skipIf(os.name != "nt", "run_paper.bat lock behavior is Windows-only")
    def test_two_simultaneous_runners_do_not_both_execute(self):
        main_source = """
            import sys
            import time
            from pathlib import Path

            Path("logs").mkdir(exist_ok=True)
            with Path("logs/args.txt").open("a", encoding="utf-8") as handle:
                handle.write(" ".join(sys.argv[1:]) + "\\n")
            time.sleep(1)
        """
        tmp, root = self.make_runner_project(main_source)
        first = None
        with tmp:
            first = subprocess.Popen(
                [str(root / "run_paper.bat")],
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.wait_for_lock(root)

            second = self.run_temp_bat(root, timeout=15)
            first_stdout, first_stderr = first.communicate(timeout=20)

            self.assertEqual(first.returncode, 0, first_stderr or first_stdout)
            self.assertEqual(second.returncode, 0, second.stderr)
            args = (root / "logs" / "args.txt").read_text(encoding="utf-8").splitlines()
            self.assertIn(args, EXPECTED_RUNNER_COMMANDS)
            self.assertIn("PAPER ya esta en ejecucion. PID activo:", self.read_summary_log(root))

    @unittest.skipIf(os.name != "nt", "run_paper.bat lock behavior is Windows-only")
    def test_stale_lock_after_interrupted_runner_is_removed(self):
        tmp, root = self.make_runner_project()
        process = None
        with tmp:
            (root / "paper").mkdir()
            process = subprocess.Popen(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    (
                        "$path = Join-Path $PWD 'paper\\paper.lock'; "
                        "$stream = [System.IO.File]::Open($path, [System.IO.FileMode]::Create, "
                        "[System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::Read); "
                        "$bytes = [System.Text.Encoding]::ASCII.GetBytes([string]$PID); "
                        "$stream.Write($bytes, 0, $bytes.Length); $stream.Flush(); "
                        "Start-Sleep -Seconds 30"
                    ),
                ],
                cwd=root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            lock = self.wait_for_lock(root)
            locked_pid = lock.read_text(encoding="ascii").strip()

            process.kill()
            process.wait(timeout=10)

            result = self.run_temp_bat(root)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(lock.exists())
            summary = self.read_summary_log(root)
            self.assertIn("STALE LOCK DETECTED", summary)
            self.assertIn(f"LOCK PID: {locked_pid}", summary)


if __name__ == "__main__":
    unittest.main()
