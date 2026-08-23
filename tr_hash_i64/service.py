"""Secure lifecycle management for long-lived TR-Hash-i64 servers.

The inference process remains a normal ``tr-hash-i64 serve`` command.  This
module only renders and controls a private Supervisor definition around the
whole process group, so a TP/PP failure always restarts every rank together.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$")
_CONTROL_CHARS = re.compile(r"[\x00\r\n]")


def _validate_text(value: str, label: str) -> str:
    value = str(value)
    if not value or _CONTROL_CHARS.search(value):
        raise ValueError(f"invalid {label}")
    return value


def _absolute(path: Path, label: str) -> Path:
    path = Path(_validate_text(str(path), label))
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    return path


@dataclass(frozen=True, slots=True)
class ServiceSpec:
    """Validated server process and its Supervisor policy."""

    name: str
    command: tuple[str, ...]
    directory: Path
    log_path: Path
    host: str
    port: int
    devices: str | None = None
    startsecs: int = 10
    startretries: int = 3
    stopwaitsecs: int = 60

    def __post_init__(self) -> None:
        if not _SAFE_NAME.fullmatch(self.name):
            raise ValueError("service name must contain only letters, digits, '_' or '-'")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if self.startsecs < 0 or self.startretries < 0 or self.stopwaitsecs < 1:
            raise ValueError("invalid Supervisor retry or timeout value")
        object.__setattr__(self, "directory", _absolute(self.directory, "directory"))
        object.__setattr__(self, "log_path", _absolute(self.log_path, "log path"))
        command = tuple(_validate_text(part, "command argument") for part in self.command)
        if not command or not Path(command[0]).is_absolute():
            raise ValueError("command executable must be an absolute path")
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "host", _validate_text(self.host, "host"))
        if self.devices is not None:
            devices = _validate_text(self.devices, "CUDA device list")
            if not all(part.strip().isdigit() for part in devices.split(",")):
                raise ValueError("CUDA devices must be a comma-separated list of integers")
            object.__setattr__(self, "devices", devices)

    @property
    def program_name(self) -> str:
        return f"tr_hash_i64_{self.name}"

    def render(self) -> str:
        lines = [
            "; managed by tr-hash-i64; edit through `tr-hash-i64 service`",
            f"; tr-hash-i64-endpoint={self.host}:{self.port}",
            f"; tr-hash-i64-devices={self.devices or ''}",
            f"[program:{self.program_name}]",
            f"directory={self.directory}",
            f"command={shlex.join(self.command)}",
            "autostart=true",
            "autorestart=unexpected",
            f"startsecs={self.startsecs}",
            f"startretries={self.startretries}",
            "stopasgroup=true",
            "killasgroup=true",
            "stopsignal=TERM",
            f"stopwaitsecs={self.stopwaitsecs}",
            "redirect_stderr=true",
            f"stdout_logfile={self.log_path}",
            "stdout_logfile_maxbytes=50MB",
            "stdout_logfile_backups=3",
            "umask=077",
        ]
        if self.devices is not None:
            lines.append(f'environment=CUDA_VISIBLE_DEVICES="{self.devices}"')
        return "\n".join(lines) + "\n"


class ServiceManager:
    """Install and control TR-Hash-i64 services through Supervisor."""

    def __init__(self, config_dir: Path | None = None) -> None:
        configured = os.environ.get("TR_HASH_I64_SUPERVISOR_DIR")
        self.config_dir = Path(config_dir or configured or "/etc/supervisor/conf.d")

    def config_path(self, name: str) -> Path:
        if not _SAFE_NAME.fullmatch(name):
            raise ValueError("invalid service name")
        return self.config_dir / f"tr_hash_i64_{name}.conf"

    def install(self, spec: ServiceSpec) -> Path:
        self._check_conflicts(spec)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        spec.log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            spec.log_path.parent.chmod(0o700)
        except PermissionError:
            pass
        destination = self.config_path(spec.name)
        temporary = destination.with_suffix(".conf.tmp")
        temporary.write_text(spec.render(), encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(destination)
        self.apply()
        return destination

    def remove(self, name: str, *, missing_ok: bool = False) -> None:
        path = self.config_path(name)
        if not path.exists():
            if missing_ok:
                return
            raise FileNotFoundError(path)
        path.unlink()
        self.apply()

    def apply(self) -> None:
        self._run("reread")
        self._run("update")

    def list(self) -> str:
        result = self._run("status", check=False)
        lines = [line for line in result.stdout.splitlines() if line.startswith("tr_hash_i64_")]
        return "\n".join(lines)

    def status(self, name: str) -> str:
        return self._run("status", self._program(name), check=False).stdout.rstrip()

    def start(self, name: str) -> str:
        return self._run("start", self._program(name)).stdout.rstrip()

    def stop(self, name: str) -> str:
        return self._run("stop", self._program(name)).stdout.rstrip()

    def restart(self, name: str) -> str:
        return self._run("restart", self._program(name)).stdout.rstrip()

    def log_path(self, name: str) -> Path:
        for line in self.config_path(name).read_text(encoding="utf-8").splitlines():
            if line.startswith("stdout_logfile="):
                return _absolute(Path(line.split("=", 1)[1]), "configured log path")
        raise ValueError(f"service {name!r} has no configured log")

    def logs(self, name: str, *, lines: int = 100) -> str:
        if lines < 1:
            raise ValueError("lines must be greater than zero")
        with self.log_path(name).open(encoding="utf-8", errors="replace") as handle:
            return "".join(deque(handle, maxlen=lines))

    def follow_logs(self, name: str, *, lines: int = 100) -> Iterable[str]:
        if lines < 1:
            raise ValueError("lines must be greater than zero")
        process = subprocess.Popen(
            ["tail", "-n", str(lines), "-F", str(self.log_path(name))],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            if process.stdout is None:
                raise RuntimeError("tail did not expose stdout")
            yield from process.stdout
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()

    def _program(self, name: str) -> str:
        if not _SAFE_NAME.fullmatch(name):
            raise ValueError("invalid service name")
        return f"tr_hash_i64_{name}"

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["supervisorctl", *args],
            check=check,
            capture_output=True,
            text=True,
        )

    def _check_conflicts(self, spec: ServiceSpec) -> None:
        if not self.config_dir.exists():
            return
        requested_devices = set(spec.devices.split(",")) if spec.devices else set()
        for path in self.config_dir.glob("tr_hash_i64_*.conf"):
            if path == self.config_path(spec.name):
                continue
            metadata = _metadata(path.read_text(encoding="utf-8"))
            endpoint = metadata.get("endpoint", "")
            if endpoint:
                host, _, raw_port = endpoint.rpartition(":")
                if raw_port.isdigit() and int(raw_port) == spec.port:
                    if host == spec.host or host == "0.0.0.0" or spec.host == "0.0.0.0":
                        raise ValueError(f"port {spec.port} conflicts with {path.stem}")
            current_devices = set(filter(None, metadata.get("devices", "").split(",")))
            overlap = requested_devices & current_devices
            if overlap:
                raise ValueError(
                    f"CUDA device(s) {','.join(sorted(overlap))} conflict with {path.stem}"
                )


def _metadata(configuration: str) -> dict[str, str]:
    result: dict[str, str] = {}
    prefix = "; tr-hash-i64-"
    for line in configuration.splitlines():
        if line.startswith(prefix) and "=" in line:
            key, value = line[len(prefix) :].split("=", 1)
            result[key] = value
    return result


def load_protected_secret(path: Path) -> str:
    """Read a non-empty secret file only when its permissions are private."""
    path = _absolute(Path(path), "secret file")
    if not path.is_file():
        raise ValueError(f"secret file does not exist: {path}")
    if path.stat().st_mode & 0o077:
        raise ValueError(f"secret file must be mode 0600: {path}")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ValueError(f"secret file is empty: {path}")
    return value


def status_fields(line: str) -> tuple[str, str, str]:
    parts = line.split(maxsplit=2)
    return (
        parts[0] if parts else "",
        parts[1].upper() if len(parts) > 1 else "UNKNOWN",
        parts[2] if len(parts) > 2 else "",
    )


__all__ = [
    "ServiceManager",
    "ServiceSpec",
    "load_protected_secret",
    "status_fields",
]
