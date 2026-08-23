"""Operational helpers for supervised TR-Hash-i64 inference services.

This module deliberately uses only the standard library.  It keeps lifecycle
policy outside the inference hot path while exposing deterministic, testable
building blocks for profiles, diagnostics, readiness monitoring, autotuning
and transactional upgrades.
"""

from __future__ import annotations

import json
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from tr_hash_i64.service import ServiceManager, load_protected_secret, status_fields


@dataclass(frozen=True, slots=True)
class ServiceProfile:
    name: str
    max_batch_size: int
    chunk_size: int
    max_kv_blocks: int
    description: str


PROFILES: dict[str, ServiceProfile] = {
    "latency": ServiceProfile(
        "latency",
        8,
        256,
        256,
        "small batches and short prefill chunks",
    ),
    "balanced": ServiceProfile(
        "balanced",
        32,
        512,
        512,
        "safe default for mixed interactive traffic",
    ),
    "throughput": ServiceProfile(
        "throughput",
        64,
        1024,
        1024,
        "large batches for sustained throughput",
    ),
}


def resolve_profile(
    name: str,
    *,
    max_batch_size: int | None = None,
    chunk_size: int | None = None,
    max_kv_blocks: int | None = None,
) -> ServiceProfile:
    if name not in PROFILES:
        raise ValueError(f"unknown service profile: {name}")
    base = PROFILES[name]
    values = (
        base.max_batch_size if max_batch_size is None else max_batch_size,
        base.chunk_size if chunk_size is None else chunk_size,
        base.max_kv_blocks if max_kv_blocks is None else max_kv_blocks,
    )
    if values[0] < 1 or values[1] < 1 or values[2] < 0:
        raise ValueError("profile settings must be positive (KV blocks may be zero)")
    return ServiceProfile(name, *values, base.description)


def service_url(metadata: dict[str, str], path: str) -> str:
    endpoint = metadata.get("endpoint", "")
    host, separator, port = endpoint.rpartition(":")
    if not separator or not port.isdigit():
        raise ValueError("service configuration has no valid endpoint metadata")
    if host in {"0.0.0.0", "::", "[::]"}:
        host = "127.0.0.1"
    return f"http://{host}:{port}/{path.lstrip('/')}"


def fetch_json(url: str, *, timeout: float = 1.0) -> tuple[int, dict]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return response.status, payload
    except urllib.error.HTTPError as error:
        try:
            payload = json.loads(error.read().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {"error": str(error)}
        return error.code, payload


def readiness(manager: ServiceManager, name: str, *, timeout: float = 1.0) -> bool:
    try:
        status, payload = fetch_json(
            service_url(manager.metadata(name), "/ready"), timeout=timeout
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return status == 200 and payload.get("status") == "ready"


def wait_until_ready(
    manager: ServiceManager,
    name: str,
    *,
    timeout: float = 180.0,
    interval: float = 2.0,
    probe: Callable[[ServiceManager, str], bool] | None = None,
) -> bool:
    probe = probe or readiness
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if probe(manager, name):
            return True
        time.sleep(interval)
    return False


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    name: str
    status: str
    detail: str


def _check(name: str, condition: bool, success: str, failure: str) -> DoctorCheck:
    return DoctorCheck(
        name, "PASS" if condition else "FAIL", success if condition else failure
    )


def _supervisor_check(manager: ServiceManager) -> DoctorCheck:
    try:
        result = manager._run("version", check=False)
    except FileNotFoundError:
        return DoctorCheck("supervisor", "FAIL", "supervisorctl is not installed")
    if result.returncode == 0:
        return DoctorCheck("supervisor", "PASS", result.stdout.strip() or "reachable")
    return DoctorCheck("supervisor", "FAIL", result.stderr.strip() or "unreachable")


def _cuda_checks(devices: str) -> list[DoctorCheck]:
    if not devices:
        return [DoctorCheck("cuda", "WARN", "CPU service; no CUDA devices pinned")]
    executable = shutil.which("nvidia-smi")
    if not executable:
        return [DoctorCheck("cuda", "FAIL", "nvidia-smi is not available")]
    result = subprocess.run(
        [
            executable,
            "--query-gpu=index,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return [DoctorCheck("cuda", "FAIL", result.stderr.strip() or "query failed")]
    rows: dict[str, tuple[str, int, int]] = {}
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 4 and parts[0].isdigit():
            rows[parts[0]] = (parts[1], int(parts[2]), int(parts[3]))
    checks = []
    for device in devices.split(","):
        device = device.strip()
        if device not in rows:
            checks.append(DoctorCheck(f"gpu:{device}", "FAIL", "device not found"))
            continue
        model, total, free = rows[device]
        checks.append(
            DoctorCheck(
                f"gpu:{device}",
                "PASS" if free >= 1024 else "WARN",
                f"{model}; VRAM {free}/{total} MiB free",
            )
        )
    return checks


def run_doctor(manager: ServiceManager, name: str) -> list[DoctorCheck]:
    checks = [_supervisor_check(manager)]
    path = manager.config_path(name)
    checks.append(
        _check("config", path.is_file(), str(path), "configuration is missing")
    )
    if not path.is_file():
        return checks
    private = path.stat().st_mode & 0o077 == 0
    checks.append(
        _check("permissions", private, "configuration is private", "expected mode 0600")
    )
    metadata = manager.metadata(name)
    try:
        command = manager.command(name)
        executable = Path(command[0])
        checks.append(
            _check(
                "executable",
                executable.is_file(),
                str(executable),
                f"missing: {executable}",
            )
        )
    except ValueError as error:
        command = ()
        checks.append(DoctorCheck("executable", "FAIL", str(error)))

    checkpoint = metadata.get("checkpoint", "")
    if checkpoint.startswith("/"):
        checkpoint_path = Path(checkpoint)
        valid_checkpoint = checkpoint_path.exists() and (
            checkpoint_path.is_file() or (checkpoint_path / "config.json").is_file()
        )
        checks.append(
            _check(
                "checkpoint",
                valid_checkpoint,
                str(checkpoint_path),
                f"missing checkpoint or config.json: {checkpoint_path}",
            )
        )
    else:
        checks.append(
            DoctorCheck("checkpoint", "PASS", checkpoint or "registry/Hugging Face")
        )

    if "--api-key-file" in command:
        index = command.index("--api-key-file") + 1
        try:
            load_protected_secret(Path(command[index]))
            checks.append(DoctorCheck("api-key", "PASS", "private secret file"))
        except (IndexError, ValueError) as error:
            checks.append(DoctorCheck("api-key", "FAIL", str(error)))
    else:
        checks.append(
            DoctorCheck("api-key", "WARN", "public endpoint; no API key file")
        )

    checks.extend(_cuda_checks(metadata.get("devices", "")))
    try:
        host, _, raw_port = metadata.get("endpoint", "").rpartition(":")
        connect_host = "127.0.0.1" if host in {"0.0.0.0", "::", "[::]"} else host
        with socket.create_connection((connect_host, int(raw_port)), timeout=0.75):
            pass
        checks.append(
            DoctorCheck(
                "port", "PASS", f"{connect_host}:{raw_port} accepts connections"
            )
        )
    except (OSError, TypeError, ValueError):
        checks.append(
            DoctorCheck(
                "port", "FAIL", "configured endpoint does not accept connections"
            )
        )
    raw_status = manager.status(name)
    _, state, _ = status_fields(raw_status)
    checks.append(
        DoctorCheck("process", "PASS" if state == "RUNNING" else "FAIL", state)
    )
    is_ready = readiness(manager, name)
    checks.append(
        DoctorCheck(
            "readiness",
            "PASS" if is_ready else "FAIL",
            "/ready is 200" if is_ready else "/ready is unavailable",
        )
    )
    return checks


@dataclass(slots=True)
class ReadinessWatchdog:
    manager: ServiceManager
    name: str
    failure_threshold: int = 3
    timeout: float = 2.0
    failures: int = 0
    probe: Callable[[ServiceManager, str], bool] = readiness

    def tick(self) -> str:
        if self.probe(self.manager, self.name):
            self.failures = 0
            return "ready"
        self.failures += 1
        if self.failures < self.failure_threshold:
            return f"not-ready ({self.failures}/{self.failure_threshold})"
        self.manager.restart_server(self.name)
        self.failures = 0
        return "restarted"


def run_watchdog(
    manager: ServiceManager,
    name: str,
    *,
    interval: float,
    failures: int,
    grace: float,
) -> None:
    if interval <= 0 or failures < 1 or grace < 0:
        raise ValueError("invalid watchdog interval, threshold or grace period")
    watchdog = ReadinessWatchdog(manager, name, failure_threshold=failures)
    if grace:
        time.sleep(grace)
    while True:
        print(f"watchdog {name}: {watchdog.tick()}", flush=True)
        time.sleep(interval)


@dataclass(frozen=True, slots=True)
class StatusSnapshot:
    name: str
    state: str
    ready: bool
    profile: str
    devices: str
    vram: str
    throughput: str
    batch_size: str
    details: str


def status_snapshot(manager: ServiceManager, name: str) -> StatusSnapshot:
    raw = manager.status(name)
    _, state, details = status_fields(raw)
    metadata = manager.metadata(name)
    ready = readiness(manager, name, timeout=0.75) if state == "RUNNING" else False
    monitor: dict = {}
    if state == "RUNNING":
        try:
            status, monitor = fetch_json(
                service_url(metadata, "/v1/monitor"), timeout=0.75
            )
            if status != 200:
                monitor = {}
        except (OSError, ValueError, json.JSONDecodeError):
            monitor = {}
    gpu = monitor.get("gpu", {})
    if gpu:
        total_mb = int(gpu.get("total_mb", 0))
        used_mb = total_mb - int(gpu.get("free_mb", 0))
        vram = f"{used_mb / 1024:.1f}/{total_mb / 1024:.1f} GB"
    else:
        vram = "n/a"
    perf = monitor.get("perf", {})
    throughput = f"{perf['tok_per_s']} tok/s" if "tok_per_s" in perf else "n/a"
    scheduler = monitor.get("scheduler", {})
    batch = str(scheduler.get("max_batch_size", metadata.get("max-batch-size", "?")))
    return StatusSnapshot(
        name=name,
        state=state,
        ready=ready,
        profile=metadata.get("profile", "custom"),
        devices=metadata.get("devices", "cpu") or "cpu",
        vram=vram,
        throughput=throughput,
        batch_size=batch,
        details=details,
    )


def replace_command_options(
    command: Iterable[str], updates: dict[str, int]
) -> tuple[str, ...]:
    result = list(command)
    for option, value in updates.items():
        flag = f"--{option}"
        if flag in result:
            index = result.index(flag) + 1
            if index >= len(result):
                raise ValueError(f"missing value for {flag}")
            result[index] = str(value)
        else:
            result.extend([flag, str(value)])
    return tuple(result)


@dataclass(frozen=True, slots=True)
class TuneCandidate:
    max_batch_size: int
    chunk_size: int
    max_kv_blocks: int


@dataclass(frozen=True, slots=True)
class TuneResult:
    candidate: TuneCandidate
    status: str
    tokens_per_second: float = 0.0
    detail: str = ""


AUTOTUNE_CANDIDATES: dict[str, tuple[TuneCandidate, ...]] = {
    "latency": (
        TuneCandidate(4, 128, 256),
        TuneCandidate(8, 256, 256),
        TuneCandidate(16, 512, 384),
    ),
    "balanced": (
        TuneCandidate(16, 256, 384),
        TuneCandidate(32, 512, 512),
        TuneCandidate(48, 768, 768),
    ),
    "throughput": (
        TuneCandidate(32, 512, 512),
        TuneCandidate(48, 768, 768),
        TuneCandidate(64, 1024, 1024),
    ),
}


def benchmark_endpoint(
    manager: ServiceManager,
    name: str,
    *,
    requests: int = 8,
    concurrency: int = 4,
    max_tokens: int = 32,
) -> float:
    if requests < 1 or concurrency < 1 or max_tokens < 1:
        raise ValueError("benchmark requests, concurrency and tokens must be positive")
    metadata = manager.metadata(name)
    url = service_url(metadata, "/v1/completions")
    body = json.dumps(
        {
            "model": metadata.get("model", "tr-hash"),
            "prompt": "Explain deterministic token routing in one short paragraph.",
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
    ).encode("utf-8")

    def request_once() -> int:
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return int(payload.get("usage", {}).get("completion_tokens", 0))

    started = time.monotonic()
    total_tokens = 0
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = [executor.submit(request_once) for _ in range(max(1, requests))]
        for future in as_completed(futures):
            total_tokens += future.result()
    elapsed = max(time.monotonic() - started, 1e-6)
    return total_tokens / elapsed


class ServiceAutotuner:
    def __init__(
        self,
        manager: ServiceManager,
        name: str,
        *,
        ready_waiter: Callable[[ServiceManager, str], bool] | None = None,
        benchmark: Callable[[ServiceManager, str], float] | None = None,
    ) -> None:
        self.manager = manager
        self.name = name
        self.ready_waiter = ready_waiter or (
            lambda manager, service: wait_until_ready(manager, service)
        )
        self.benchmark = benchmark or benchmark_endpoint

    def run(
        self, candidates: Iterable[TuneCandidate]
    ) -> tuple[TuneResult, list[TuneResult]]:
        original = self.manager.config_text(self.name)
        watchdog_enabled = self.manager.has_watchdog(self.name)
        results: list[TuneResult] = []
        best: TuneResult | None = None
        best_config = ""
        try:
            if watchdog_enabled:
                self.manager.stop_watchdog(self.name)
            for candidate in candidates:
                updates = {
                    "max-batch-size": candidate.max_batch_size,
                    "chunk-size": candidate.chunk_size,
                    "max-kv-blocks": candidate.max_kv_blocks,
                }
                command = replace_command_options(
                    self.manager.command(self.name), updates
                )
                self.manager.replace_command(
                    self.name,
                    command,
                    metadata={key: str(value) for key, value in updates.items()},
                )
                self.manager.restart_server(self.name)
                if not self.ready_waiter(self.manager, self.name):
                    detail = "readiness failed"
                    try:
                        if (
                            "out of memory"
                            in self.manager.logs(self.name, lines=100).lower()
                        ):
                            detail = "OOM detected"
                    except (OSError, ValueError):
                        pass
                    result = TuneResult(candidate, "rejected", detail=detail)
                else:
                    try:
                        throughput = self.benchmark(self.manager, self.name)
                        result = TuneResult(candidate, "ok", throughput)
                    except Exception as error:
                        result = TuneResult(candidate, "rejected", detail=str(error))
                results.append(result)
                if result.status == "ok" and (
                    best is None or result.tokens_per_second > best.tokens_per_second
                ):
                    best = result
                    best_config = self.manager.config_text(self.name)
            if best is None:
                raise RuntimeError("all autotune candidates failed")
            self.manager.restore_config(self.name, best_config)
            self.manager.restart_server(self.name)
            if not self.ready_waiter(self.manager, self.name):
                raise RuntimeError("best candidate failed final readiness check")
            if watchdog_enabled:
                self.manager.start_watchdog(self.name)
            return best, results
        except Exception:
            self.manager.restore_config(self.name, original)
            self.manager.restart_server(self.name)
            self.ready_waiter(self.manager, self.name)
            if watchdog_enabled:
                self.manager.start_watchdog(self.name)
            raise


_SHA = re.compile(r"^[0-9a-f]{40}$")
_REF = re.compile(r"^[A-Za-z0-9._/-]{1,128}$")


class ServiceUpgrader:
    def __init__(
        self,
        manager: ServiceManager,
        name: str,
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        ready_waiter: Callable[[ServiceManager, str], bool] | None = None,
    ) -> None:
        self.manager = manager
        self.name = name
        self.runner = runner
        self.ready_waiter = ready_waiter or (
            lambda manager, service: wait_until_ready(manager, service)
        )

    def _run(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        return self.runner(command, check=True, capture_output=True, text=True)

    def resolve_revision(self, source: str, ref: str) -> str:
        if not _REF.fullmatch(ref):
            raise ValueError("invalid upgrade ref")
        target_ref = ref if ref.startswith("refs/") else f"refs/heads/{ref}"
        result = self._run(["git", "ls-remote", source, target_ref])
        sha = result.stdout.split(maxsplit=1)[0] if result.stdout.strip() else ""
        if not _SHA.fullmatch(sha):
            raise RuntimeError(f"could not resolve {ref!r} from {source!r}")
        return sha

    def run(self, *, source: str, ref: str, release_root: Path) -> str:
        if "\n" in source or "\r" in source or not source:
            raise ValueError("invalid upgrade source")
        release_root = Path(release_root)
        if not release_root.is_absolute():
            raise ValueError("release root must be absolute")
        sha = self.resolve_revision(source, ref)
        release = release_root / sha[:12]
        venv_python = release / "venv" / "bin" / "python"
        complete = release / ".complete"
        release_root.mkdir(parents=True, exist_ok=True)
        release_root.chmod(0o700)
        complete_release = (
            complete.is_file()
            and venv_python.is_file()
            and complete.read_text(encoding="utf-8").strip() == sha
        )
        if not complete_release:
            if release.exists():
                shutil.rmtree(release)
            release.mkdir(mode=0o700, exist_ok=False)
            self._run(
                [
                    sys.executable,
                    "-m",
                    "venv",
                    "--system-site-packages",
                    str(release / "venv"),
                ]
            )
            self._run(
                [
                    str(venv_python),
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    f"git+{source}@{sha}",
                ]
            )
            self._run(
                [
                    str(venv_python),
                    "-c",
                    "import tr_hash_i64; print(tr_hash_i64.__version__)",
                ]
            )
            complete.write_text(f"{sha}\n", encoding="utf-8")
            complete.chmod(0o600)

        original = self.manager.config_text(self.name)
        try:
            if self.manager.has_watchdog(self.name):
                self.manager.stop_watchdog(self.name)
            self.manager.replace_executable(
                self.name,
                venv_python,
                metadata={"release-sha": sha, "release-source": source},
            )
            self.manager.restart_server(self.name)
            if not self.ready_waiter(self.manager, self.name):
                raise RuntimeError("upgraded service did not become ready")
            if self.manager.has_watchdog(self.name):
                self.manager.start_watchdog(self.name)
        except Exception:
            self.manager.restore_config(self.name, original)
            self.manager.restart_server(self.name)
            if not self.ready_waiter(self.manager, self.name):
                raise RuntimeError("upgrade failed and rollback did not become ready")
            if self.manager.has_watchdog(self.name):
                self.manager.start_watchdog(self.name)
            raise
        return sha


__all__ = [
    "AUTOTUNE_CANDIDATES",
    "PROFILES",
    "DoctorCheck",
    "ReadinessWatchdog",
    "ServiceAutotuner",
    "ServiceProfile",
    "ServiceUpgrader",
    "StatusSnapshot",
    "TuneCandidate",
    "TuneResult",
    "benchmark_endpoint",
    "fetch_json",
    "readiness",
    "replace_command_options",
    "resolve_profile",
    "run_doctor",
    "run_watchdog",
    "service_url",
    "status_snapshot",
    "wait_until_ready",
]
