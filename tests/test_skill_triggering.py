import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / "plugins/yugo-memory/skills/yugo-memory-auto-recall/SKILL.md"
OPENAI_YAML = SKILL.parent / "agents/openai.yaml"
HOOKS = ROOT / "plugins/yugo-memory/hooks/hooks.json"


class SkillTriggeringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.skill_text = SKILL.read_text(encoding="utf-8")
        frontmatter = cls.skill_text.split("---", 2)[1]
        match = re.search(r"^description:\s*(.+)$", frontmatter, re.MULTILINE)
        assert match is not None
        cls.description = match.group(1)

    def test_description_frontloads_cross_session_history_signals(self) -> None:
        for signal in (
            "previous",
            "earlier",
            "latest",
            "Nth",
            "branch",
            "thread",
            "window",
            "decisions",
            "commands",
            "paths",
            "IDs",
            "之前那个",
            "上次",
            "当时",
        ):
            with self.subTest(signal=signal):
                self.assertIn(signal, self.description)

    def test_description_preserves_false_positive_boundary(self) -> None:
        self.assertIn("Do not trigger", self.description)
        for boundary in ("visible text", "local files", "current external events"):
            with self.subTest(boundary=boundary):
                self.assertIn(boundary, self.description)

    def test_implicit_invocation_remains_enabled(self) -> None:
        metadata = OPENAI_YAML.read_text(encoding="utf-8")
        self.assertIn("allow_implicit_invocation: true", metadata)

    def test_no_prompt_submit_or_forced_recall_hook(self) -> None:
        hooks = HOOKS.read_text(encoding="utf-8")
        self.assertNotIn("UserPromptSubmit", hooks)
        self.assertNotIn("PreToolUse", hooks)

    def test_skill_uses_only_standalone_memory_tools(self) -> None:
        forbidden = "episodic" + "-memory"
        self.assertNotIn(forbidden, self.skill_text.lower())
        self.assertIn("do not call another memory plugin", self.skill_text.lower())
        self.assertIn("read_evidence", self.skill_text)


if __name__ == "__main__":
    unittest.main()
