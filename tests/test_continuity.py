import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "plugins/yugo-memory/scripts"
sys.path.insert(0, str(SCRIPTS))

from archive_parser import ParserState, iter_tool_evidence, parse_increment, probe_source  # noqa: E402
from memory_control import (  # noqa: E402
    manage_experience,
    prepare_context,
    recall_experiences,
    sync_task,
    task_status,
)
from recall_index import read_evidence, search_index, sync_index  # noqa: E402


def codex_message(role: str, text: str, timestamp: str) -> dict:
    return {
        "timestamp": timestamp,
        "type": "response_item",
        "payload": {"type": "message", "role": role, "content": [{"type": "input_text", "text": text}]},
    }


class ContinuityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.archives = self.root / "archives"
        self.index = self.root / "index.sqlite"
        self.control = self.root / "control.sqlite"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_archive(self, session_id: str, exchanges: int = 2) -> Path:
        path = self.archives / "2042" / f"fictional-{session_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [{
            "type": "session_meta",
            "payload": {"id": session_id, "cwd": "/demo/fictional", "context_window": 10_000},
        }]
        for number in range(exchanges):
            rows.extend([
                codex_message("user", f"处理虚构星图批次 {number}。", f"2042-01-01T00:{number:02d}:00Z"),
                codex_message("assistant", f"虚构星图批次 {number} 完成，标记 nebula-{number:04d}。", f"2042-01-01T00:{number:02d}:01Z"),
            ])
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
        return path

    def test_large_session_reindex_does_not_exceed_sqlite_variable_limit(self) -> None:
        path = self.write_archive("large-session", exchanges=430)
        sync_index(self.archives, self.index)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(codex_message(
                "user", "追加虚构星图校验。", "2042-01-02T00:00:00Z"
            ), ensure_ascii=False) + "\n")
        report = sync_index(self.archives, self.index)
        self.assertEqual(report["incrementally_indexed_files"], 1)
        result = search_index(self.index, "nebula-0429", mode="auto")
        self.assertTrue(result["safe_to_answer"])

    def test_auto_profile_uses_context_budget(self) -> None:
        self.write_archive("budget-session")
        sync_index(self.archives, self.index)
        result = search_index(
            self.index,
            "nebula-0001",
            current_session_id="budget-session",
            context_window=10_000,
            context_tokens_used=9_500,
            response_profile="auto",
        )
        self.assertEqual(result["response_profile"], "minimal")
        self.assertEqual(result["context_budget"]["recommended_evidence_ranges"], 1)
        self.assertNotIn("timings_ms", result)
        self.assertIn("top_locator", result)
        compact = search_index(
            self.index, "nebula-0001", response_profile="compact", mode="auto",
        )
        standard = search_index(
            self.index, "nebula-0001", response_profile="standard", mode="auto",
        )
        minimal_size = len(json.dumps(result, ensure_ascii=False))
        compact_size = len(json.dumps(compact, ensure_ascii=False))
        standard_size = len(json.dumps(standard, ensure_ascii=False))
        self.assertLess(minimal_size, compact_size)
        self.assertLess(compact_size, standard_size)
        self.assertLess(minimal_size, 2_500)

    def test_qoder_transcript_parses_and_probes_context_window(self) -> None:
        path = self.archives / "qoder" / "fictional-qoder.jsonl"
        path.parent.mkdir(parents=True)
        rows = [
            {"type": "runtime-config", "sessionId": "qoder-fictional", "contextWindow": 4_000, "model": "fictional"},
            {"type": "session_meta", "sessionId": "qoder-fictional", "cwd": "/demo/fictional", "data": {}},
            {"type": "user", "sessionId": "qoder-fictional", "timestamp": "2042-02-01T00:00:00Z", "message": {"role": "user", "content": "紫晶航线 " * 700}},
            {"type": "user", "sessionId": "qoder-fictional", "timestamp": "2042-02-01T00:00:00Z", "toolUseResult": {"stdout": "FICTIONAL_QODER_TOOL_OK"}, "message": {"role": "user", "content": "tool result"}},
            {"type": "assistant", "sessionId": "qoder-fictional", "timestamp": "2042-02-01T00:00:01Z", "message": {"role": "assistant", "content": "已完成。"}},
        ]
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
        exchanges, state, _seeks, _compactions, _scanned = parse_increment(path, ParserState())
        self.assertEqual(state.source_agent, "qoder")
        self.assertEqual(state.context_window, 4_000)
        self.assertEqual(exchanges[0].session_id, "qoder-fictional")
        self.assertEqual(len(exchanges), 1)
        tool_evidence = list(iter_tool_evidence(path, "qoder-fictional"))
        self.assertEqual(len(tool_evidence), 1)
        self.assertIn("FICTIONAL_QODER_TOOL_OK", tool_evidence[0].text)
        probe = probe_source(path, long_ratio=0.05, minimum_tokens=1_000)
        self.assertTrue(probe["long"])
        sync_index(self.archives, self.index)
        recalled = search_index(self.index, "FICTIONAL_QODER_TOOL_OK", mode="auto")
        self.assertTrue(recalled["safe_to_answer"])
        self.assertEqual(recalled["results"][0]["session_id"], "qoder-fictional")

    def test_task_auto_replaces_unrelated_objective_and_clears_on_completion(self) -> None:
        first = sync_task(
            "session-a", "实现虚构彗星索引，必须保持本地运行，并验证测试通过。",
            control_path=self.control,
        )
        self.assertEqual(first["transition"], "started")
        amended = sync_task(
            "session-a", "另外补充失败时不能编造结果。", control_path=self.control,
        )
        self.assertEqual(amended["transition"], "amended")
        replaced = sync_task(
            "session-a", "新任务：设计虚构温室灌溉表。", control_path=self.control,
        )
        self.assertEqual(replaced["transition"], "replaced")
        cleared = sync_task("session-a", action="complete", control_path=self.control)
        self.assertTrue(cleared["cleared"])
        self.assertIsNone(task_status("session-a", self.control)["active_task"])

    def test_experience_is_versioned_recalled_and_hard_deleted(self) -> None:
        archive = self.write_archive("experience-session")
        sync_index(self.archives, self.index)
        ref = {"archive_path": str(archive.resolve()), "line_start": 2, "line_end": 3}
        first = manage_experience(
            "upsert", "fictional.star-map", "虚构星图工具", "需要检查星图批次",
            "先运行只读探针，再检查完成标记。", "批次成功完成。", ["tool", "star"], [ref],
            control_path=self.control, index_path=self.index,
        )
        self.assertEqual(first["version"], 1)
        second = manage_experience(
            "upsert", "fictional.star-map", "虚构星图工具", "需要检查星图批次",
            "先运行只读探针，再核对两个完成标记。", "新版流程成功完成。", ["tool", "star"], [ref],
            control_path=self.control, index_path=self.index,
        )
        self.assertEqual(second["version"], 2)
        recalled = recall_experiences("星图完成标记", control_path=self.control)
        self.assertEqual(recalled["results"][0]["version"], 2)
        self.assertEqual(
            recall_experiences("不存在的虚构火山协议", control_path=self.control)["results"], []
        )
        deleted = manage_experience(
            "delete", "fictional.star-map", control_path=self.control, index_path=self.index,
        )
        self.assertTrue(deleted["hard_deleted"])
        self.assertEqual(recall_experiences("星图", control_path=self.control)["results"], [])

    def test_prepare_context_skips_conversation_recall_without_history_signal(self) -> None:
        payload = prepare_context(
            "session-b", "实现一个虚构索引并运行测试。", include_recall="auto",
            control_path=self.control, index_path=self.index,
        )
        self.assertIsNone(payload["conversation_recall"])
        self.assertIsNotNone(payload["active_task"])

    def test_prepare_context_reports_unready_index_without_building_it(self) -> None:
        payload = prepare_context(
            "session-c", "查找以前虚构星图任务中的精确决定。", include_recall="yes",
            control_path=self.control, index_path=self.index,
        )
        self.assertEqual(payload["conversation_recall"]["answerability"], "index_not_ready")
        self.assertFalse(payload["conversation_recall"]["safe_to_answer"])
        self.assertFalse(self.index.exists())

    def test_qoder_adapter_round_trip_preserves_unrelated_configuration(self) -> None:
        qoder_home = self.root / "qoder-home"
        qoder_home.mkdir()
        (qoder_home / "settings.json").write_text(json.dumps({
            "theme": "fictional-dark",
            "hooks": {"SessionStart": [{"matcher": "startup", "hooks": [{
                "type": "command", "command": "echo fictional-existing", "timeout": 1,
            }]}]},
        }), encoding="utf-8")
        (qoder_home / "mcp.json").write_text(json.dumps({
            "mcpServers": {"fictional-existing": {"command": "echo", "args": []}},
        }), encoding="utf-8")
        adapter = ROOT / "scripts/qoder-adapter.py"
        installed = subprocess.run([
            sys.executable, str(adapter), "install", "--repo", str(ROOT),
            "--qoder-home", str(qoder_home),
        ], text=True, capture_output=True, check=True)
        self.assertTrue(json.loads(installed.stdout)["installed"])
        status = subprocess.run([
            sys.executable, str(adapter), "status", "--repo", str(ROOT),
            "--qoder-home", str(qoder_home),
        ], text=True, capture_output=True, check=True)
        self.assertTrue(json.loads(status.stdout)["configured"])
        subprocess.run([
            sys.executable, str(adapter), "uninstall", "--repo", str(ROOT),
            "--qoder-home", str(qoder_home),
        ], text=True, capture_output=True, check=True)
        settings = json.loads((qoder_home / "settings.json").read_text(encoding="utf-8"))
        servers = json.loads((qoder_home / "mcp.json").read_text(encoding="utf-8"))["mcpServers"]
        self.assertEqual(settings["theme"], "fictional-dark")
        self.assertIn("fictional-existing", servers)
        self.assertNotIn("yugo-memory", servers)

    def test_session_end_hook_clears_only_that_task(self) -> None:
        memory_home = self.root / "memory-home"
        control = memory_home / "control.sqlite"
        sync_task("session-end-a", "完成虚构索引。", control_path=control)
        sync_task("session-end-b", "完成虚构报告。", control_path=control)
        hook = SCRIPTS / "task-lifecycle.mjs"
        run = subprocess.run(
            ["node", str(hook)],
            input=json.dumps({"session_id": "session-end-a"}),
            text=True,
            capture_output=True,
            env={**os.environ, "YUGO_MEMORY_HOME": str(memory_home)},
            check=False,
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIsNone(task_status("session-end-a", control)["active_task"])
        self.assertIsNotNone(task_status("session-end-b", control)["active_task"])


if __name__ == "__main__":
    unittest.main()
