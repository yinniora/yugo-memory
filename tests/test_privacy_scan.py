from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCANNER = ROOT / "scripts" / "privacy-scan.py"


class PrivacyScanTests(unittest.TestCase):
    def _repo(self) -> tuple[tempfile.TemporaryDirectory, Path]:
        temporary = tempfile.TemporaryDirectory(prefix="long-memory-privacy-")
        root = Path(temporary.name) / "repo"
        root.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "Synthetic Test"], cwd=root, check=True)
        synthetic_email = "synthetic" + "@" + "example.invalid"
        subprocess.run(["git", "config", "user.email", synthetic_email], cwd=root, check=True)
        return temporary, root

    def _commit(self, root: Path, message: str) -> None:
        subprocess.run(["git", "add", "-A"], cwd=root, check=True)
        subprocess.run(["git", "commit", "-q", "-m", message], cwd=root, check=True)

    def test_clean_publishable_fixture_passes(self) -> None:
        temporary, root = self._repo()
        try:
            (root / "README.md").write_text("Synthetic memory plugin fixture.\n", encoding="utf-8")
            run = subprocess.run(
                [sys.executable, str(SCANNER), "--root", str(root), "--json"],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertTrue(json.loads(run.stdout)["passed"])
        finally:
            temporary.cleanup()

    def test_private_conversation_and_secret_are_rejected_without_echoing_content(self) -> None:
        temporary, root = self._repo()
        try:
            private_path = "/" + "Us" + "ers/example.person/private/conversation.jsonl"
            delegation = "<codex_" + "delegation>real transcript</codex_" + "delegation>"
            credential = "private_" + "token=EXAMPLE_NOT_A_REAL_TOKEN_12345"
            content = "\n".join((private_path, delegation, credential))
            (root / "accidental-history.txt").write_text(content, encoding="utf-8")
            run = subprocess.run(
                [sys.executable, str(SCANNER), "--root", str(root), "--json"],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(run.returncode, 2)
            payload = json.loads(run.stdout)
            self.assertFalse(payload["passed"])
            rules = {finding["rule"] for finding in payload["findings"]}
            self.assertTrue({"personal-home-path", "raw-delegation", "credential-assignment"} <= rules)
            self.assertNotIn("EXAMPLE_NOT_A_REAL_TOKEN_12345", run.stdout)
        finally:
            temporary.cleanup()

    def test_deleted_private_file_is_still_rejected_from_history(self) -> None:
        temporary, root = self._repo()
        try:
            private_path = "/" + "Us" + "ers/example.person/private/notes.txt"
            leaked = root / "removed.txt"
            leaked.write_text(private_path, encoding="utf-8")
            self._commit(root, "add synthetic leak")
            leaked.unlink()
            self._commit(root, "remove synthetic leak")
            run = subprocess.run(
                [sys.executable, str(SCANNER), "--root", str(root), "--json"],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(run.returncode, 2)
            payload = json.loads(run.stdout)
            historical = [item for item in payload["findings"] if item["revision"]]
            self.assertTrue(any(item["rule"] == "personal-home-path" for item in historical))
            self.assertNotIn("example.person", run.stdout)
        finally:
            temporary.cleanup()

    def test_external_private_denylist_is_not_echoed(self) -> None:
        temporary, root = self._repo()
        try:
            private_phrase = "Synthetic Seagrass allocation 4711"
            (root / "fixture.txt").write_text(private_phrase, encoding="utf-8")
            self._commit(root, "add fixture")
            denylist = Path(temporary.name) / "private-denylist.txt"
            denylist.write_text(private_phrase + "\n", encoding="utf-8")
            run = subprocess.run(
                [
                    sys.executable, str(SCANNER), "--root", str(root),
                    "--denylist", str(denylist), "--json",
                ],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(run.returncode, 2)
            payload = json.loads(run.stdout)
            self.assertTrue(any(item["rule"] == "private-denylist" for item in payload["findings"]))
            self.assertNotIn(private_phrase, run.stdout)
        finally:
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
