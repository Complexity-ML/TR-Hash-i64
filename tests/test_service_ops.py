"""Regression tests for production service operations."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tr_hash_i64 import service_ops
from tr_hash_i64.service import ServiceManager, ServiceSpec
from tr_hash_i64.service_ops import (
    ReadinessWatchdog,
    ServiceAutotuner,
    ServiceUpgrader,
    TuneCandidate,
    resolve_profile,
    run_doctor,
    status_snapshot,
)


def _spec(tmp_path: Path, *, watchdog: bool = True) -> ServiceSpec:
    watchdog_command = None
    if watchdog:
        watchdog_command = (
            sys.executable,
            "-m",
            "tr_hash_i64.cli",
            "service",
            "watchdog",
            "demo",
        )
    return ServiceSpec(
        name="demo",
        command=(
            sys.executable,
            "-m",
            "tr_hash_i64.cli",
            "serve",
            "tr-hash-moe-200m",
            "--max-batch-size",
            "32",
            "--chunk-size",
            "512",
            "--max-kv-blocks",
            "512",
        ),
        directory=tmp_path,
        log_path=tmp_path / "server.log",
        host="127.0.0.1",
        port=7860,
        devices="0",
        model="tr-hash-moe-200m",
        checkpoint=str(tmp_path / "model"),
        watchdog_command=watchdog_command,
    )


def _write_config(tmp_path: Path, *, watchdog: bool = True) -> ServiceManager:
    manager = ServiceManager(tmp_path / "supervisor")
    manager.config_dir.mkdir()
    manager.config_path("demo").write_text(
        _spec(tmp_path, watchdog=watchdog).render(), encoding="utf-8"
    )
    manager.config_path("demo").chmod(0o600)
    return manager


def test_profiles_apply_safe_defaults_and_explicit_overrides():
    balanced = resolve_profile("balanced")
    assert (balanced.max_batch_size, balanced.chunk_size, balanced.max_kv_blocks) == (
        32,
        512,
        512,
    )
    custom = resolve_profile(
        "latency", max_batch_size=12, chunk_size=384, max_kv_blocks=0
    )
    assert (custom.max_batch_size, custom.chunk_size, custom.max_kv_blocks) == (
        12,
        384,
        0,
    )
    with pytest.raises(ValueError):
        resolve_profile("balanced", max_batch_size=0)


def test_supervisor_config_contains_profile_metadata_and_watchdog(tmp_path):
    rendered = _spec(tmp_path).render()
    assert "; tr-hash-i64-profile=balanced" in rendered
    assert "; tr-hash-i64-watchdog=enabled" in rendered
    assert "[program:tr_hash_i64_demo_watchdog]" in rendered
    assert "service watchdog demo" in rendered


def test_manual_restart_stops_watchdog_around_whole_group(tmp_path, monkeypatch):
    manager = _write_config(tmp_path)
    calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    manager.restart("demo")
    assert calls == [
        ["supervisorctl", "stop", "tr_hash_i64_demo_watchdog"],
        ["supervisorctl", "restart", "tr_hash_i64_demo"],
        ["supervisorctl", "start", "tr_hash_i64_demo_watchdog"],
    ]


def test_replace_executable_updates_server_and_watchdog_together(tmp_path):
    manager = _write_config(tmp_path)
    replacement = tmp_path / "release" / "bin" / "python"
    replacement.parent.mkdir(parents=True)
    replacement.write_text("", encoding="utf-8")
    manager.replace_executable(
        "demo", replacement, metadata={"release-sha": "a" * 40}, apply=False
    )
    assert manager.command("demo")[0] == str(replacement)
    assert manager.watchdog_command("demo")[0] == str(replacement)
    assert manager.metadata("demo")["release-sha"] == "a" * 40


def test_watchdog_restarts_only_after_consecutive_failures():
    class Manager:
        restarts = 0

        def restart_server(self, _name):
            self.restarts += 1

    outcomes = iter([False, False, True, False, False, False])
    manager = Manager()
    watchdog = ReadinessWatchdog(
        manager, "demo", failure_threshold=3, probe=lambda *_: next(outcomes)
    )
    assert [watchdog.tick() for _ in range(6)] == [
        "not-ready (1/3)",
        "not-ready (2/3)",
        "ready",
        "not-ready (1/3)",
        "not-ready (2/3)",
        "restarted",
    ]
    assert manager.restarts == 1


def test_doctor_checks_supervisor_checkpoint_secret_cuda_port_and_ready(
    tmp_path, monkeypatch
):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}", encoding="utf-8")
    secret = tmp_path / "api.key"
    secret.write_text("secret", encoding="utf-8")
    secret.chmod(0o600)
    spec = _spec(tmp_path, watchdog=False)
    command = (*spec.command, "--api-key-file", str(secret))
    spec = ServiceSpec(
        **{
            field: getattr(spec, field)
            for field in spec.__dataclass_fields__
            if field != "command"
        },
        command=command,
    )
    manager = ServiceManager(tmp_path / "supervisor")
    manager.config_dir.mkdir()
    manager.config_path("demo").write_text(spec.render(), encoding="utf-8")
    manager.config_path("demo").chmod(0o600)
    monkeypatch.setattr(
        manager,
        "_run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 0, stdout="tr_hash_i64_demo RUNNING pid 1, uptime 0:01\n", stderr=""
        ),
    )
    monkeypatch.setattr(
        service_ops.shutil, "which", lambda _name: "/usr/bin/nvidia-smi"
    )
    monkeypatch.setattr(
        service_ops.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, stdout="0, RTX 5090, 32768, 16384\n", stderr=""
        ),
    )
    monkeypatch.setattr(service_ops, "readiness", lambda *_args, **_kwargs: True)

    class Socket:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

    monkeypatch.setattr(
        service_ops.socket, "create_connection", lambda *_args, **_kwargs: Socket()
    )
    checks = {check.name: check for check in run_doctor(manager, "demo")}
    for name in (
        "supervisor",
        "config",
        "permissions",
        "checkpoint",
        "api-key",
        "gpu:0",
        "port",
        "process",
        "readiness",
    ):
        assert checks[name].status == "PASS"


def test_status_snapshot_uses_live_monitor_data(tmp_path, monkeypatch):
    manager = _write_config(tmp_path, watchdog=False)
    monkeypatch.setattr(
        manager,
        "status",
        lambda _name: "tr_hash_i64_demo RUNNING pid 42, uptime 0:05",
    )
    monkeypatch.setattr(service_ops, "readiness", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        service_ops,
        "fetch_json",
        lambda *_args, **_kwargs: (
            200,
            {
                "gpu": {"total_mb": 32768, "free_mb": 8192},
                "perf": {"tok_per_s": 42.5},
                "scheduler": {"max_batch_size": 48},
            },
        ),
    )
    snapshot = status_snapshot(manager, "demo")
    assert snapshot.ready is True
    assert snapshot.vram == "24.0/32.0 GB"
    assert snapshot.throughput == "42.5 tok/s"
    assert snapshot.batch_size == "48"


class _TuneManager:
    def __init__(self):
        self.configuration = "original"
        self.command_value = (
            "/python",
            "serve",
            "--max-batch-size",
            "1",
            "--chunk-size",
            "1",
            "--max-kv-blocks",
            "1",
        )
        self.watchdog_events: list[str] = []

    def config_text(self, _name):
        return self.configuration

    def command(self, _name):
        return self.command_value

    def replace_command(self, _name, command, *, metadata):
        self.command_value = command
        self.configuration = f"candidate-{metadata['max-batch-size']}"

    def restore_config(self, _name, configuration):
        self.configuration = configuration

    def restart_server(self, _name):
        return ""

    def logs(self, _name, *, lines):
        return "CUDA out of memory" if self.configuration == "candidate-64" else ""

    def has_watchdog(self, _name):
        return True

    def stop_watchdog(self, _name):
        self.watchdog_events.append("stop")

    def start_watchdog(self, _name):
        self.watchdog_events.append("start")


def test_autotune_retains_fastest_ready_candidate_and_pauses_watchdog():
    manager = _TuneManager()
    throughput = {"candidate-16": 10.0, "candidate-32": 20.0}
    tuner = ServiceAutotuner(
        manager,
        "demo",
        ready_waiter=lambda current, _name: current.configuration != "candidate-64",
        benchmark=lambda current, _name: throughput[current.configuration],
    )
    best, results = tuner.run(
        [
            TuneCandidate(16, 256, 256),
            TuneCandidate(32, 512, 512),
            TuneCandidate(64, 1024, 1024),
        ]
    )
    assert best.candidate.max_batch_size == 32
    assert manager.configuration == "candidate-32"
    assert [result.status for result in results] == ["ok", "ok", "rejected"]
    assert manager.watchdog_events == ["stop", "start"]


def test_autotune_restores_original_when_every_candidate_fails():
    manager = _TuneManager()
    tuner = ServiceAutotuner(
        manager,
        "demo",
        ready_waiter=lambda *_args: False,
        benchmark=lambda *_args: 0.0,
    )
    with pytest.raises(RuntimeError, match="all autotune candidates failed"):
        tuner.run([TuneCandidate(64, 1024, 1024)])
    assert manager.configuration == "original"
    assert manager.watchdog_events == ["stop", "start"]


class _UpgradeManager:
    def __init__(self, configuration="old-config"):
        self.configuration = configuration
        self.executable = Path("/old/python")
        self.events: list[str] = []

    def config_text(self, _name):
        return self.configuration

    def has_watchdog(self, _name):
        return True

    def stop_watchdog(self, _name):
        self.events.append("watchdog-stop")

    def start_watchdog(self, _name):
        self.events.append("watchdog-start")

    def replace_executable(self, _name, executable, *, metadata):
        self.executable = executable
        self.configuration = f"new-{metadata['release-sha']}"

    def restore_config(self, _name, configuration):
        self.configuration = configuration
        self.executable = Path("/old/python")

    def restart_server(self, _name):
        self.events.append("restart")


def test_transactional_upgrade_switches_versioned_python(tmp_path):
    sha = "a" * 40
    venv_python = tmp_path / sha[:12] / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")
    (tmp_path / sha[:12] / ".complete").write_text(f"{sha}\n", encoding="utf-8")
    manager = _UpgradeManager()

    def runner(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            [], 0, stdout=f"{sha}\trefs/heads/main\n", stderr=""
        )

    upgrader = ServiceUpgrader(
        manager, "demo", runner=runner, ready_waiter=lambda *_args: True
    )
    assert (
        upgrader.run(
            source="https://example.test/repo.git", ref="main", release_root=tmp_path
        )
        == sha
    )
    assert manager.executable == venv_python
    assert manager.events == ["watchdog-stop", "restart", "watchdog-start"]


def test_transactional_upgrade_rolls_back_when_readiness_fails(tmp_path):
    sha = "b" * 40
    venv_python = tmp_path / sha[:12] / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")
    (tmp_path / sha[:12] / ".complete").write_text(f"{sha}\n", encoding="utf-8")
    manager = _UpgradeManager()
    outcomes = iter([False, True])

    def runner(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            [], 0, stdout=f"{sha}\trefs/heads/main\n", stderr=""
        )

    upgrader = ServiceUpgrader(
        manager, "demo", runner=runner, ready_waiter=lambda *_args: next(outcomes)
    )
    with pytest.raises(RuntimeError, match="did not become ready"):
        upgrader.run(
            source="https://example.test/repo.git", ref="main", release_root=tmp_path
        )
    assert manager.configuration == "old-config"
    assert manager.executable == Path("/old/python")
    assert manager.events == ["watchdog-stop", "restart", "restart", "watchdog-start"]


def test_upgrade_rebuilds_incomplete_versioned_release(tmp_path):
    sha = "c" * 40
    release = tmp_path / sha[:12]
    release.mkdir()
    (release / ".complete").write_text("wrong-sha\n", encoding="utf-8")
    manager = _UpgradeManager()
    commands: list[list[str]] = []

    def runner(command, **_kwargs):
        commands.append(command)
        if command[:2] == ["git", "ls-remote"]:
            return subprocess.CompletedProcess(
                command, 0, stdout=f"{sha}\trefs/heads/main\n", stderr=""
            )
        if command[1:3] == ["-m", "venv"]:
            python = Path(command[-1]) / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    upgrader = ServiceUpgrader(
        manager, "demo", runner=runner, ready_waiter=lambda *_args: True
    )
    assert (
        upgrader.run(
            source="https://example.test/repo.git", ref="main", release_root=tmp_path
        )
        == sha
    )
    assert (release / ".complete").read_text(encoding="utf-8").strip() == sha
    assert any(command[1:3] == ["-m", "venv"] for command in commands)
