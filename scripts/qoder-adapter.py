#!/usr/bin/env python3
"""Install or remove the dependency-free Yugo Memory adapter for Qoder."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return default
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(raw_temp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.chmod(temp, 0o600)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def hook_group(command: str, matcher: str | None = None, timeout: int = 10) -> dict[str, Any]:
    value: dict[str, Any] = {
        "hooks": [{"type": "command", "command": command, "timeout": timeout}],
    }
    if matcher is not None:
        value["matcher"] = matcher
    return value


def remove_yugo_hooks(settings: dict[str, Any]) -> None:
    hooks = settings.get("hooks")
    if not isinstance(hooks, dict):
        return
    for event in list(hooks):
        groups = hooks.get(event)
        if not isinstance(groups, list):
            continue
        kept = []
        for group in groups:
            commands = [
                str(item.get("command") or "")
                for item in (group.get("hooks") if isinstance(group, dict) else []) or []
                if isinstance(item, dict)
            ]
            if not any("yugo-memory" in command for command in commands):
                kept.append(group)
        hooks[event] = kept


def install(repo: Path, qoder_home: Path) -> dict[str, Any]:
    runtime = repo / "plugins/yugo-memory/scripts/yugo-memory.mjs"
    context = repo / "plugins/yugo-memory/scripts/compact-recall-context.mjs"
    mcp = repo / "plugins/yugo-memory/scripts/recall_mcp.py"
    skill_source = repo / "plugins/yugo-memory/skills/yugo-memory-auto-recall"
    for target in (runtime, context, mcp, skill_source / "SKILL.md"):
        if not target.exists():
            raise FileNotFoundError(target)

    settings_path = qoder_home / "settings.json"
    mcp_path = qoder_home / "mcp.json"
    settings = load_json(settings_path, {})
    mcp_json = load_json(mcp_path, {"mcpServers": {}})
    settings.setdefault("hooks", {})
    mcp_json.setdefault("mcpServers", {})
    remove_yugo_hooks(settings)

    maintenance_command = (
        f'YUGO_MEMORY_INCLUDE_QODER=1 node "{runtime}" --background --quiet'
    )
    context_command = f'node "{context}"'
    settings["hooks"].setdefault("SessionStart", []).extend([
        hook_group(context_command, "startup"),
        hook_group(maintenance_command, "startup"),
    ])
    settings["hooks"].setdefault("Stop", []).append(
        hook_group(maintenance_command, "*", timeout=3)
    )
    mcp_json["mcpServers"]["yugo-memory"] = {
        "command": "python3",
        "args": [str(mcp)],
        "enabled": True,
        "env": {"YUGO_MEMORY_INCLUDE_QODER": "1"},
    }

    skill_target = qoder_home / "skills/yugo-memory-auto-recall"
    skill_target.parent.mkdir(parents=True, exist_ok=True)
    if skill_target.is_symlink():
        skill_target.unlink()
    elif skill_target.exists():
        raise RuntimeError(f"refusing to replace non-symlink skill path: {skill_target}")
    skill_target.symlink_to(skill_source, target_is_directory=True)
    atomic_json(settings_path, settings)
    atomic_json(mcp_path, mcp_json)
    return {
        "installed": True,
        "qoder_home": str(qoder_home),
        "shared_memory_home": os.environ.get(
            "YUGO_MEMORY_HOME", str(Path.home() / ".config/yugo-memory")
        ),
        "settings": str(settings_path),
        "mcp": str(mcp_path),
        "skill": str(skill_target),
        "packages_installed": False,
    }


def uninstall(qoder_home: Path) -> dict[str, Any]:
    settings_path = qoder_home / "settings.json"
    mcp_path = qoder_home / "mcp.json"
    settings = load_json(settings_path, {})
    mcp_json = load_json(mcp_path, {"mcpServers": {}})
    remove_yugo_hooks(settings)
    if isinstance(mcp_json.get("mcpServers"), dict):
        mcp_json["mcpServers"].pop("yugo-memory", None)
    skill_target = qoder_home / "skills/yugo-memory-auto-recall"
    if skill_target.is_symlink():
        skill_target.unlink()
    atomic_json(settings_path, settings)
    atomic_json(mcp_path, mcp_json)
    return {
        "installed": False,
        "qoder_home": str(qoder_home),
        "memory_preserved": True,
    }


def status(qoder_home: Path) -> dict[str, Any]:
    settings = load_json(qoder_home / "settings.json", {})
    mcp_json = load_json(qoder_home / "mcp.json", {"mcpServers": {}})
    commands = []
    hooks = settings.get("hooks") if isinstance(settings.get("hooks"), dict) else {}
    for groups in hooks.values():
        for group in groups if isinstance(groups, list) else []:
            for item in group.get("hooks", []) if isinstance(group, dict) else []:
                if isinstance(item, dict) and "yugo-memory" in str(item.get("command") or ""):
                    commands.append(str(item.get("command")))
    skill = qoder_home / "skills/yugo-memory-auto-recall"
    servers = mcp_json.get("mcpServers") if isinstance(mcp_json.get("mcpServers"), dict) else {}
    return {
        "qoder_home": str(qoder_home),
        "mcp_configured": "yugo-memory" in servers,
        "hook_commands": len(commands),
        "skill_linked": skill.is_symlink(),
        "configured": "yugo-memory" in servers and bool(commands) and skill.is_symlink(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "uninstall", "status"))
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--qoder-home", type=Path, default=Path.home() / ".qoder")
    args = parser.parse_args()
    if args.action == "install":
        result = install(args.repo.resolve(), args.qoder_home.expanduser())
    elif args.action == "uninstall":
        result = uninstall(args.qoder_home.expanduser())
    else:
        result = status(args.qoder_home.expanduser())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
