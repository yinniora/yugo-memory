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


    def test_scanner_file_and_new_identifier_formats_are_checked(self) -> None:
        temporary, root = self._repo()
        try:
            target = root / "scripts" / "privacy-scan.py"
            target.parent.mkdir()
            fabricated_id = "01af" + "0000-1111-7222-8333-444444444444"
            fabricated_token = "gh" + "p_" + ("Z" * 36)
            private_host = "fictional-vault." + "internal"
            target.write_text("\n".join((fabricated_id, fabricated_token, private_host)), encoding="utf-8")
            run = subprocess.run(
                [sys.executable, str(SCANNER), "--root", str(root), "--json"],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(run.returncode, 2, run.stdout + run.stderr)
            rules = {item["rule"] for item in json.loads(run.stdout)["findings"]}
            self.assertTrue({"real-thread-id", "provider-token", "internal-domain"} <= rules)
            for private_value in (fabricated_id, fabricated_token, private_host):
                self.assertNotIn(private_value, run.stdout)
        finally:
            temporary.cleanup()

    def test_commit_and_tag_messages_are_scanned(self) -> None:
        temporary, root = self._repo()
        try:
            (root / "README.md").write_text("Invented fixture.\n", encoding="utf-8")
            fabricated_token = "gh" + "p_" + ("Y" * 36)
            self._commit(root, "Synthetic marker " + fabricated_token)
            fabricated_secret = "api" + "_key=EXAMPLE_NOT_A_REAL_SECRET_12345"
            subprocess.run(["git", "tag", "-a", "fictional-v1", "-m", fabricated_secret], cwd=root, check=True)
            run = subprocess.run(
                [sys.executable, str(SCANNER), "--root", str(root), "--all-refs", "--json"],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(run.returncode, 2, run.stdout + run.stderr)
            findings = json.loads(run.stdout)["findings"]
            self.assertTrue(any(x["path"].startswith("<commit:") and x["rule"] == "provider-token" for x in findings))
            self.assertTrue(any(x["path"].startswith("<tag:") and x["rule"] == "credential-assignment" for x in findings))
            self.assertFalse(any(x["rule"] == "email-address" for x in findings))
            self.assertNotIn(fabricated_token, run.stdout)
            self.assertNotIn(fabricated_secret, run.stdout)
        finally:
            temporary.cleanup()

    def test_email_in_commit_body_is_not_exempted(self) -> None:
        temporary, root = self._repo()
        try:
            (root / "README.md").write_text("Invented fixture.\n", encoding="utf-8")
            address = "fictional-person" + "@" + "example.invalid"
            self._commit(root, "Synthetic contact " + address)
            run = subprocess.run(
                [sys.executable, str(SCANNER), "--root", str(root), "--json"],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(run.returncode, 2)
            self.assertTrue(any(x["rule"] == "email-address" for x in json.loads(run.stdout)["findings"]))
            self.assertNotIn(address, run.stdout)
        finally:
            temporary.cleanup()

    def test_nonexistent_history_ref_fails_closed(self) -> None:
        temporary, root = self._repo()
        try:
            (root / "README.md").write_text("Invented fixture.\n", encoding="utf-8")
            self._commit(root, "synthetic fixture")
            run = subprocess.run(
                [sys.executable, str(SCANNER), "--root", str(root), "--history-ref", "missing-fictional-ref", "--json"],
                text=True, capture_output=True, check=False,
            )
            self.assertNotEqual(run.returncode, 0)
        finally:
            temporary.cleanup()

if __name__ == "__main__":
    unittest.main()
