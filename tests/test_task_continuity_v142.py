import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "plugins/yugo-memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from memory_control import sync_task, task_status  # noqa: E402
from recall_mcp import handle, resolved_session_id, tool_definitions  # noqa: E402


class TaskContinuityV142Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.control = self.root / "control.sqlite"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_natural_followups_preserve_objective_and_amend(self) -> None:
        started = sync_task(
            "fictional-session-a",
            "实现一个本地运行的虚构星图索引，并验证测试通过。",
            control_path=self.control,
        )
        original_objective = started["active_task"]["objective"]
        for request in (
            "测试用例还要覆盖最早记录和最近记录。",
            "把输出格式改成 JSON。",
            "修复刚才发现的边界条件。",
        ):
            result = sync_task(
                "fictional-session-a", request, control_path=self.control, profile="minimal",
            )
            self.assertEqual(result["transition"], "amended", request)
            self.assertEqual(result["active_task"]["objective"], original_objective)
            self.assertLessEqual(len(result["active_task"]["items"]), 4)

    def test_non_mutating_turns_do_not_create_or_change_a_task(self) -> None:
        empty = sync_task("fictional-session-b", "继续", control_path=self.control)
        self.assertEqual(empty["transition"], "unchanged")
        self.assertIsNone(empty["active_task"])
        sync_task("fictional-session-b", "制作一份虚构月球清单。", control_path=self.control)
        before = task_status("fictional-session-b", self.control)["active_task"]
        for request in ("好的", "目前进度怎么样？", "thanks"):
            result = sync_task("fictional-session-b", request, control_path=self.control)
            self.assertEqual(result["transition"], "unchanged")
            self.assertEqual(result["active_task"]["task_id"], before["task_id"])
            self.assertEqual(result["active_task"]["items"], before["items"])

    def test_explicit_and_clearly_independent_tasks_replace(self) -> None:
        first = sync_task(
            "fictional-session-c", "实现一个虚构星图索引。", control_path=self.control,
        )
        independent = sync_task(
            "fictional-session-c", "设计一份虚构温室灌溉表。", control_path=self.control,
        )
        self.assertEqual(independent["transition"], "replaced")
        self.assertNotEqual(independent["active_task"]["task_id"], first["active_task"]["task_id"])
        explicit = sync_task(
            "fictional-session-c", "新任务：编写虚构彗星菜单。", control_path=self.control,
        )
        self.assertEqual(explicit["transition"], "replaced")

    def test_ambiguous_turn_preserves_without_storing_uncertain_text(self) -> None:
        sync_task(
            "fictional-session-d", "实现一个虚构星图索引。", control_path=self.control,
        )
        before = task_status("fictional-session-d", self.control)["active_task"]
        result = sync_task(
            "fictional-session-d", "考虑后续安排。", control_path=self.control,
        )
        self.assertEqual(result["transition"], "ambiguous")
        self.assertTrue(result["needs_disambiguation"])
        self.assertLess(result["objective_similarity"], 0.14)
        self.assertEqual(result["active_task"]["task_id"], before["task_id"])
        self.assertEqual(result["active_task"]["items"], before["items"])

    def test_sessions_remain_isolated(self) -> None:
        left = sync_task("fictional-left", "整理虚构蓝色星图。", control_path=self.control)
        right = sync_task("fictional-right", "整理虚构红色星图。", control_path=self.control)
        sync_task("fictional-left", "另外补充校验步骤。", control_path=self.control)
        self.assertEqual(
            task_status("fictional-right", self.control)["active_task"]["task_id"],
            right["active_task"]["task_id"],
        )
        self.assertNotEqual(left["active_task"]["task_id"], right["active_task"]["task_id"])

    def test_session_resolution_has_safe_priority_and_no_guess(self) -> None:
        clean_env = {
            "CODEX_THREAD_ID": "fictional-env-codex",
            "QODER_SESSION_ID": "fictional-env-qoder",
            "YUGO_MEMORY_SESSION_ID": "fictional-env-yugo",
        }
        with patch.dict(os.environ, clean_env, clear=False):
            self.assertEqual(
                resolved_session_id(
                    "fictional-explicit", "fictional-current",
                    {"context": {"threadId": "fictional-meta"}},
                ),
                "fictional-explicit",
            )
            self.assertEqual(
                resolved_session_id(None, "fictional-current", {"threadId": "fictional-meta"}),
                "fictional-current",
            )
            self.assertEqual(
                resolved_session_id(None, None, {"context": {"threadId": "fictional-meta"}}),
                "fictional-meta",
            )
        with patch.dict(
            os.environ,
            {"CODEX_THREAD_ID": "", "QODER_SESSION_ID": "", "YUGO_MEMORY_SESSION_ID": ""},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "will not guess another session"):
                resolved_session_id()
            with self.assertRaisesRegex(ValueError, "invalid session id"):
                resolved_session_id("fictional session with spaces")

    def test_mcp_accepts_current_id_or_metadata_and_fails_closed_without_one(self) -> None:
        env = {
            "YUGO_MEMORY_HOME": str(self.root / "memory-home"),
            "YUGO_MEMORY_SOURCE_DIR": str(self.root / "missing-sessions"),
            "CODEX_THREAD_ID": "",
            "QODER_SESSION_ID": "",
            "YUGO_MEMORY_SESSION_ID": "",
        }
        with patch.dict(os.environ, env, clear=False):
            current = handle({
                "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": {"name": "prepare_context", "arguments": {
                    "current_session_id": "fictional-current-only",
                    "user_request": "实现虚构星图索引。",
                    "include_recall": "no",
                }},
            })
            self.assertFalse(current["result"]["isError"])
            metadata = handle({
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {
                    "name": "task_status", "arguments": {},
                    "_meta": {"context": {"conversationId": "fictional-meta-only"}},
                },
            })
            self.assertFalse(metadata["result"]["isError"])
            missing = handle({
                "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                "params": {"name": "task_status", "arguments": {}},
            })
            self.assertTrue(missing["result"]["isError"])
            error = json.loads(missing["result"]["content"][0]["text"])["error"]
            self.assertIn("will not guess another session", error)

    def test_task_tools_publish_current_session_alias_and_minimal_profile(self) -> None:
        definitions = {tool["name"]: tool for tool in tool_definitions()}
        for name in ("prepare_context", "task_update", "task_status"):
            self.assertIn("current_session_id", definitions[name]["inputSchema"]["properties"])
        profile = definitions["task_update"]["inputSchema"]["properties"]["profile"]
        self.assertEqual(profile["default"], "minimal")


if __name__ == "__main__":
    unittest.main()
