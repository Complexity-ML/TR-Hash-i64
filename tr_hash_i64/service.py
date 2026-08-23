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
    model: str = ""
    profile: str = "balanced"
    checkpoint: str = ""
    max_batch_size: int = 32
    chunk_size: int = 512
    max_kv_blocks: int = 512
    watchdog_command: tuple[str, ...] | None = None
    watchdog_interval: float = 10.0
    watchdog_failures: int = 3
    watchdog_grace: float = 120.0
    startsecs: int = 10
    startretries: int = 3
    stopwaitsecs: int = 60

    def __post_init__(self) -> None:
        if not _SAFE_NAME.fullmatch(self.name):
            raise ValueError(
                "service name must contain only letters, digits, '_' or '-'"
            )
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if self.startsecs < 0 or self.startretries < 0 or self.stopwaitsecs < 1:
            raise ValueError("invalid Supervisor retry or timeout value")
        if self.max_batch_size < 1 or self.chunk_size < 1 or self.max_kv_blocks < 0:
            raise ValueError("invalid performance setting")
        if (
            self.watchdog_interval <= 0
            or self.watchdog_failures < 1
            or self.watchdog_grace < 0
        ):
            raise ValueError("invalid watchdog setting")
        object.__setattr__(self, "directory", _absolute(self.directory, "directory"))
        object.__setattr__(self, "log_path", _absolute(self.log_path, "log path"))
        command = tuple(
            _validate_text(part, "command argument") for part in self.command
        )
        if not command or not Path(command[0]).is_absolute():
            raise ValueError("command executable must be an absolute path")
        object.__setattr__(self, "command", command)
        object.__setattr__(self, "host", _validate_text(self.host, "host"))
        object.__setattr__(
            self, "model", _validate_text(self.model or "unknown", "model")
        )
        object.__setattr__(self, "profile", _validate_text(self.profile, "profile"))
        if self.checkpoint:
            object.__setattr__(
                self, "checkpoint", _validate_text(self.checkpoint, "checkpoint")
            )
        if self.watchdog_command is not None:
            watchdog = tuple(
                _validate_text(part, "watchdog command argument")
                for part in self.watchdog_command
            )
            if not watchdog or not Path(watchdog[0]).is_absolute():
                raise ValueError("watchdog executable must be an absolute path")
            object.__setattr__(self, "watchdog_command", watchdog)
        if self.devices is not None:
            devices = _validate_text(self.devices, "CUDA device list")
            if not all(part.strip().isdigit() for part in devices.split(",")):
                raise ValueError(
                    "CUDA devices must be a comma-separated list of integers"
                )
            object.__setattr__(self, "devices", devices)

    @property
    def program_name(self) -> str:
        return f"tr_hash_i64_{self.name}"

    def render(self) -> str:
        lines = [
            "; managed by tr-hash-i64; edit through `tr-hash-i64 service`",
            f"; tr-hash-i64-endpoint={self.host}:{self.port}",
            f"; tr-hash-i64-devices={self.devices or ''}",
            f"; tr-hash-i64-model={self.model}",
            f"; tr-hash-i64-profile={self.profile}",
            f"; tr-hash-i64-checkpoint={self.checkpoint}",
            f"; tr-hash-i64-max-batch-size={self.max_batch_size}",
            f"; tr-hash-i64-chunk-size={self.chunk_size}",
            f"; tr-hash-i64-max-kv-blocks={self.max_kv_blocks}",
            f"; tr-hash-i64-watchdog={'enabled' if self.watchdog_command else 'disabled'}",
            f"; tr-hash-i64-watchdog-interval={self.watchdog_interval}",
            f"; tr-hash-i64-watchdog-failures={self.watchdog_failures}",
            f"; tr-hash-i64-watchdog-grace={self.watchdog_grace}",
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
        if self.watchdog_command is not None:
            lines.extend(
                [
                    "",
                    f"[program:{self.program_name}_watchdog]",
                    f"directory={self.directory}",
                    f"command={shlex.join(self.watchdog_command)}",
                    "autostart=true",
                    "autorestart=unexpected",
                    "startsecs=2",
                    "startretries=3",
                    "stopasgroup=true",
                    "killasgroup=true",
                    "stopsignal=TERM",
                    "stopwaitsecs=10",
                    "redirect_stderr=true",
                    f"stdout_logfile={self.log_path}.watchdog",
                    "stdout_logfile_maxbytes=10MB",
                    "stdout_logfile_backups=2",
                    "umask=077",
                ]
            )
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
        self._write_config(destination, spec.render())
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
        lines = [
            line
            for line in result.stdout.splitlines()
            if line.startswith("tr_hash_i64_")
            and not line.split(maxsplit=1)[0].endswith("_watchdog")
        ]
        return "\n".join(lines)

    def status(self, name: str) -> str:
        return self._run("status", self._program(name), check=False).stdout.rstrip()

    def start(self, name: str) -> str:
        outputs = [self._run("start", self._program(name)).stdout.rstrip()]
        if self.has_watchdog(name):
            outputs.append(
                self._run(
                    "start", self._watchdog_program(name), check=False
                ).stdout.rstrip()
            )
        return "\n".join(filter(None, outputs))

    def stop(self, name: str) -> str:
        outputs = []
        if self.has_watchdog(name):
            outputs.append(
                self._run(
                    "stop", self._watchdog_program(name), check=False
                ).stdout.rstrip()
            )
        outputs.append(self._run("stop", self._program(name)).stdout.rstrip())
        return "\n".join(filter(None, outputs))

    def restart(self, name: str) -> str:
        outputs = []
        if self.has_watchdog(name):
            outputs.append(
                self._run(
                    "stop", self._watchdog_program(name), check=False
                ).stdout.rstrip()
            )
        outputs.append(self.restart_server(name))
        if self.has_watchdog(name):
            outputs.append(
                self._run(
                    "start", self._watchdog_program(name), check=False
                ).stdout.rstrip()
            )
        return "\n".join(filter(None, outputs))

    def restart_server(self, name: str) -> str:
        """Restart only the full inference process group, never its watchdog."""
        return self._run("restart", self._program(name)).stdout.rstrip()

    def start_watchdog(self, name: str) -> str:
        if not self.has_watchdog(name):
            return ""
        return self._run(
            "start", self._watchdog_program(name), check=False
        ).stdout.rstrip()

    def stop_watchdog(self, name: str) -> str:
        if not self.has_watchdog(name):
            return ""
        return self._run(
            "stop", self._watchdog_program(name), check=False
        ).stdout.rstrip()

    def has_watchdog(self, name: str) -> bool:
        path = self.config_path(name)
        return path.is_file() and self.metadata(name).get("watchdog") == "enabled"

    def metadata(self, name: str) -> dict[str, str]:
        return _metadata(self.config_text(name))

    def config_text(self, name: str) -> str:
        return self.config_path(name).read_text(encoding="utf-8")

    def command(self, name: str) -> tuple[str, ...]:
        return self._section_command(name, self._program(name))

    def watchdog_command(self, name: str) -> tuple[str, ...] | None:
        if not self.has_watchdog(name):
            return None
        return self._section_command(name, self._watchdog_program(name))

    def _section_command(self, name: str, program: str) -> tuple[str, ...]:
        section = f"[program:{program}]"
        active = False
        for line in self.config_text(name).splitlines():
            if line.startswith("["):
                active = line == section
            elif active and line.startswith("command="):
                return tuple(shlex.split(line.split("=", 1)[1]))
        raise ValueError(f"service {name!r} has no command")

    def replace_executable(
        self,
        name: str,
        executable: Path,
        *,
        metadata: dict[str, str] | None = None,
        apply: bool = True,
    ) -> None:
        executable = _absolute(Path(executable), "command executable")
        text = self.config_text(name)
        programs = [self._program(name)]
        if self.has_watchdog(name):
            programs.append(self._watchdog_program(name))
        lines = text.splitlines()
        active = False
        replaced: set[str] = set()
        current_program = ""
        for index, line in enumerate(lines):
            if line.startswith("[program:") and line.endswith("]"):
                current_program = line[len("[program:") : -1]
                active = current_program in programs
            elif active and line.startswith("command="):
                command = shlex.split(line.split("=", 1)[1])
                if not command:
                    raise ValueError(f"program {current_program!r} has no command")
                command[0] = str(executable)
                lines[index] = f"command={shlex.join(command)}"
                replaced.add(current_program)
        if replaced != set(programs):
            missing = ", ".join(sorted(set(programs) - replaced))
            raise ValueError(f"service {name!r} has no command for: {missing}")
        updated = "\n".join(lines) + "\n"
        if metadata:
            updated = _replace_metadata(updated, metadata)
        self._write_config(self.config_path(name), updated)
        if apply:
            self.apply()

    def replace_command(
        self,
        name: str,
        command: Iterable[str],
        *,
        metadata: dict[str, str] | None = None,
        apply: bool = True,
    ) -> None:
        command = tuple(_validate_text(part, "command argument") for part in command)
        if not command or not Path(command[0]).is_absolute():
            raise ValueError("command executable must be an absolute path")
        text = self.config_text(name)
        section = f"[program:{self._program(name)}]"
        lines = text.splitlines()
        active = False
        replaced = False
        for index, line in enumerate(lines):
            if line.startswith("["):
                active = line == section
            elif active and line.startswith("command="):
                lines[index] = f"command={shlex.join(command)}"
                replaced = True
                break
        if not replaced:
            raise ValueError(f"service {name!r} has no command")
        updated = "\n".join(lines) + "\n"
        if metadata:
            updated = _replace_metadata(updated, metadata)
        self._write_config(self.config_path(name), updated)
        if apply:
            self.apply()

    def restore_config(
        self, name: str, configuration: str, *, apply: bool = True
    ) -> None:
        if f"[program:{self._program(name)}]" not in configuration:
            raise ValueError("backup does not match the requested service")
        self._write_config(self.config_path(name), configuration)
        if apply:
            self.apply()

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

    def _watchdog_program(self, name: str) -> str:
        return f"{self._program(name)}_watchdog"

    @staticmethod
    def _write_config(destination: Path, content: str) -> None:
        temporary = destination.with_suffix(".conf.tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.chmod(0o600)
        temporary.replace(destination)

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


def _replace_metadata(configuration: str, updates: dict[str, str]) -> str:
    safe_updates = {
        _validate_text(key, "metadata key"): _validate_text(value, "metadata value")
        for key, value in updates.items()
    }
    lines = configuration.splitlines()
    seen: set[str] = set()
    prefix = "; tr-hash-i64-"
    for index, line in enumerate(lines):
        if line.startswith(prefix) and "=" in line:
            key = line[len(prefix) :].split("=", 1)[0]
            if key in safe_updates:
                lines[index] = f"{prefix}{key}={safe_updates[key]}"
                seen.add(key)
    insert_at = next(
        (i for i, line in enumerate(lines) if line.startswith("[program:")), 1
    )
    for key in sorted(set(safe_updates) - seen):
        lines.insert(insert_at, f"{prefix}{key}={safe_updates[key]}")
        insert_at += 1
    return "\n".join(lines) + "\n"


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
