"""Regression tests for supervised TR-Hash-i64 inference services."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from tr_hash_i64 import cli
from tr_hash_i64.parallel import launcher
from tr_hash_i64.service import ServiceManager, ServiceSpec, load_protected_secret


def _spec(tmp_path: Path, **overrides) -> ServiceSpec:
    values = {
        "name": "public_demo",
        "command": (sys.executable, "-m", "tr_hash_i64.cli", "serve", "model"),
        "directory": tmp_path,
        "log_path": tmp_path / "logs" / "server.log",
        "host": "127.0.0.1",
        "port": 7860,
        "devices": "0,1",
    }
    values.update(overrides)
    return ServiceSpec(**values)


def test_service_config_is_private_and_controls_the_whole_group(tmp_path, monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    manager = ServiceManager(tmp_path / "supervisor")
    path = manager.install(_spec(tmp_path))

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    rendered = path.read_text(encoding="utf-8")
    assert "autorestart=unexpected" in rendered
    assert "stopasgroup=true" in rendered
    assert "killasgroup=true" in rendered
    assert "stopsignal=TERM" in rendered
    assert 'environment=CUDA_VISIBLE_DEVICES="0,1"' in rendered
    assert [entry[0] for entry in calls] == [
        ["supervisorctl", "reread"],
        ["supervisorctl", "update"],
    ]
    assert all(entry[1].get("check") is True for entry in calls)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "bad name"),
        ("command", (sys.executable, "serve\nmalicious")),
        ("directory", Path("relative")),
        ("log_path", Path("relative.log")),
        ("devices", "0,$(touch pwned)"),
    ],
)
def test_service_rejects_unsafe_configuration(tmp_path, field, value):
    with pytest.raises(ValueError):
        _spec(tmp_path, **{field: value})


def test_service_rejects_newline_in_absolute_paths(tmp_path):
    with pytest.raises(ValueError):
        _spec(tmp_path, log_path=Path("/tmp/server.log\nuser=root"))


def test_service_detects_port_and_gpu_conflicts(tmp_path):
    config_dir = tmp_path / "supervisor"
    config_dir.mkdir()
    existing = _spec(tmp_path, name="first", port=7860, devices="0,1")
    (config_dir / "tr_hash_i64_first.conf").write_text(existing.render(), encoding="utf-8")
    manager = ServiceManager(config_dir)

    with pytest.raises(ValueError, match="port 7860"):
        manager._check_conflicts(_spec(tmp_path, name="second", devices="2,3"))
    with pytest.raises(ValueError, match="CUDA device"):
        manager._check_conflicts(
            _spec(tmp_path, name="second", port=7861, devices="1,2")
        )


def test_api_key_file_must_be_absolute_private_and_nonempty(tmp_path):
    secret = tmp_path / "api.key"
    secret.write_text("secret-value\n", encoding="utf-8")
    secret.chmod(0o600)
    assert load_protected_secret(secret) == "secret-value"

    secret.chmod(0o644)
    with pytest.raises(ValueError, match="mode 0600"):
        load_protected_secret(secret)

    with pytest.raises(ValueError, match="absolute"):
        load_protected_secret(Path("api.key"))


def test_service_lifecycle_never_uses_a_shell(tmp_path, monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    manager = ServiceManager(tmp_path)
    assert manager.restart("public_demo") == "ok"
    command, kwargs = calls[0]
    assert command == ["supervisorctl", "restart", "tr_hash_i64_public_demo"]
    assert "shell" not in kwargs


def test_torchrun_disables_rank_local_restarts(monkeypatch):
    captured = {}

    def fake_run(command, env):
        captured["command"] = command
        captured["env"] = env
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(launcher, "_find_free_port", lambda: "29501")
    assert launcher.launch_distributed(2, ["serve", "model"], pp_size=2) == 0
    assert "--max_restarts=0" in captured["command"]
    assert captured["env"]["TR_HASH_I64_TP_SIZE"] == "2"
    assert captured["env"]["TR_HASH_I64_PP_SIZE"] == "2"


def test_service_install_persists_performance_settings_without_secret(monkeypatch, tmp_path):
    secret = tmp_path / "api.key"
    secret.write_text("do-not-put-this-on-the-command-line", encoding="utf-8")
    secret.chmod(0o600)
    captured = {}

    def fake_install(self, spec):
        captured["spec"] = spec
        return tmp_path / "tr_hash_i64_demo.conf"

    monkeypatch.setattr(ServiceManager, "install", fake_install)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tr-hash-i64",
            "service",
            "install",
            "demo",
            "tr-hash-moe-200m",
            "--config-dir",
            str(tmp_path / "supervisor"),
            "--api-key-file",
            str(secret),
            "--max-batch-size",
            "48",
            "--chunk-size",
            "768",
            "--max-kv-blocks",
            "512",
            "--rate-limit",
            "120",
            "--max-pending",
            "64",
            "--compile",
        ],
    )

    cli.main()
    command = list(captured["spec"].command)
    assert command[command.index("--max-batch-size") + 1] == "48"
    assert command[command.index("--chunk-size") + 1] == "768"
    assert command[command.index("--max-kv-blocks") + 1] == "512"
    assert command[command.index("--rate-limit") + 1] == "120"
    assert command[command.index("--max-pending") + 1] == "64"
    assert "--compile" in command
    assert "do-not-put-this-on-the-command-line" not in command
    assert command[command.index("--api-key-file") + 1] == str(secret)
