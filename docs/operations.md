# Production operations

TR-Hash-i64 includes a Supervisor-based service manager for hosts where the engine owns a persistent model process. It keeps the configured runtime warm; it does not itself accelerate kernels.

For eGPU recovery, stable UUID selection, and host-level systemd policy, use the separate [TR-Hash-Server](https://github.com/Complexity-ML/tr-hash-server) project.

## Install a supervised service

Install Supervisor through the host package manager, then prepare a key file:

```bash
sudo install -d -m 700 /etc/tr-hash-i64
sudo install -m 600 /dev/null /etc/tr-hash-i64/api.key
sudoedit /etc/tr-hash-i64/api.key
```

Install the current SFT release:

```bash
sudo tr-hash-i64 service install public-demo tr-hash-moe-200m \
  --checkpoint /models/TR-HASH-MoE-200M-160B-SFT \
  --directory /opt/TR-Hash-i64 \
  --host 0.0.0.0 \
  --port 7860 \
  --devices 0 \
  --api-key-file /etc/tr-hash-i64/api.key \
  --profile balanced \
  --max-pending 128
```

The manager writes private metadata and Supervisor configuration, starts the service, and can install a readiness watchdog. GPU identity and power policy should be stabilized at the host layer before the process launches.

## Performance profiles

| Profile | Max batch | Prefill chunk | KV blocks | Intended use |
| --- | ---: | ---: | ---: | --- |
| `latency` | 8 | 256 | 256 | Interactive, low concurrency |
| `balanced` | 32 | 512 | 512 | Mixed traffic |
| `throughput` | 64 | 1024 | 1024 | Sustained concurrency |

```bash
tr-hash-i64 service profiles
```

`--max-batch-size`, `--chunk-size`, and `--max-kv-blocks` override individual values. Treat profiles as starting points: available VRAM, context length, quantization, and request shape determine the safe capacity.

## Lifecycle and diagnostics

```bash
tr-hash-i64 service list
tr-hash-i64 service status public-demo
tr-hash-i64 service doctor public-demo
tr-hash-i64 service restart public-demo
tr-hash-i64 service logs public-demo -f
sudo tr-hash-i64 service remove public-demo
```

`status` combines Supervisor state with readiness and, for an unauthenticated local service, live monitor data. The current service client does not attach the configured bearer token when requesting `/v1/monitor`, so monitor-derived fields are unavailable when API authentication is enabled. `doctor` still checks configuration permissions, executable and checkpoint paths, CUDA visibility, VRAM, process state, listening port, and readiness. It exits non-zero when a required check fails.

The generated Supervisor process uses grouped stop/kill behavior so distributed workers are not left behind. A tensor- or pipeline-parallel service is restarted as one process group.

## Readiness watchdog

The default watchdog starts checking after a warmup grace period. Repeated `/ready` failures trigger a process-group restart. Manual maintenance suspends the watchdog before changing process state.

Relevant install options:

```text
--no-watchdog
--watchdog-interval SECONDS
--watchdog-failures COUNT
--watchdog-grace SECONDS
```

Use `--no-watchdog` when another orchestrator already owns restart policy.

## Autotuning

```bash
sudo tr-hash-i64 service autotune public-demo --profile balanced
```

Each candidate receives a full restart and readiness check. Failed or out-of-memory candidates are rejected. The fastest successful candidate is retained; if all candidates fail, the original configuration is restored.

The current autotune client does not send a bearer token to `/v1/completions`. Do not use `service autotune` on a service configured with `--api-key-file` until that client path supports authenticated requests; benchmark it directly or tune before enabling API authentication.

Autotune measures the installed model on the actual host. Results from another GPU, checkpoint, quantization mode, batch shape, or power limit are not interchangeable.

## Atomic upgrades and rollback

```bash
sudo tr-hash-i64 service upgrade public-demo --ref main
```

The requested Git ref is resolved to an immutable commit and installed in a versioned virtual environment under `/var/lib/tr-hash-i64/releases`. The service switches only after installation. If readiness fails, the manager restores the previous private configuration, restarts the old release, and verifies readiness again.

For production, prefer a reviewed full commit SHA over a moving branch name:

```bash
sudo tr-hash-i64 service upgrade public-demo \
  --ref 833ec49660e9fd4b11b9d29b0aa60eef79edf71e
```

## Security checklist

- Bind to `127.0.0.1` unless external access is intentional.
- Put TLS and network access control in a reverse proxy or trusted ingress.
- Use a mode-`0600` API-key file, not a command-line secret.
- Set `--rate-limit` and `--max-pending` for public endpoints.
- Leave `/v1/execute` disabled unless sandbox execution is required and reviewed.
- Monitor `/ready`, `/v1/monitor`, logs, GPU errors, and KV-cache pressure.
- Pin releases to immutable commits and retain a known-good rollback release.

See [Getting started](getting-started.md) for direct foreground serving and [API guide](api.md) for health and monitoring endpoints.
