# GPU coordination

AutoDub assumes the GPU may be shared and that overheating or overcommitting it is worse than
waiting. Every GPU action passes three gates (`src/autodub/gpu_session.py`):

1. **Arm.** The reviewer arms one action for one job (`POST /api/jobs/<id>/arm-gpu`). The arm
   expires after five minutes and is consumed by exactly one request, so a stale page or a replayed
   request cannot start GPU work.
2. **Preflight.** A fresh check that fails closed: any configured conflict port with a listener
   (`AUTODUB_ADAPT_CONFLICT_PORTS`), or an unreachable broker when one is configured, blocks the
   action. `GET /api/gpu/status` shows the result.
3. **Lease.** Exclusive GPU access for the duration of the work, always released in `finally`.

## Lease back-ends

**Local (default).** A process-wide lock: one AutoDub process never runs two GPU jobs at once.
A second request waits (up to its wait limit) and then fails with a clear error.

**Broker (`AUTODUB_GPU_BROKER_URL`).** For machines where several applications share one card, AutoDub
speaks a small HTTP lease protocol:

| Request | Body | Response |
|---|---|---|
| `POST /queue/request` | `{"owner": "autodub", "reason", "resources": ["gpu"], "ttl_seconds", "queue_ttl_seconds"}` | `{"ok", "state": "granted"\|"queued", "request_id", "lease_id"}` |
| `GET /queue/request?id=<request_id>` | — | same shape; polled while `queued` |
| `POST /queue/heartbeat` | `{"lease_id", "ttl_seconds"}` | `{"ok", "expires_at"}` |
| `POST /queue/release` | `{"lease_id"}` | `{"ok"}` |
| `POST /queue/cancel` | `{"request_id"}` | `{"ok"}` |
| `GET /lease/status` | — | `{"lease_count", "queue_count"}` |

Leases are renewed by a heartbeat thread every 45 s with a 180 s TTL. After three consecutive failed
heartbeats the lease is treated as lost and the next GPU submission raises instead of running
without a lease. A queued request that times out is cancelled with the broker.

AutoDub never stops or restarts other processes to free the GPU.

## Thermal guard

The episode queue reads temperatures before each GPU episode and after it:

- default source: `nvidia-smi --query-gpu=temperature.gpu` (hottest card wins);
- `AUTODUB_GPU_TEMP_COMMAND`: any command printing `core NN C`, `hot spot NN C`, `memory junction NN C`
  lines, for tools that expose sensors nvidia-smi does not (memory junction is usually the hottest
  sensor on GDDR6X cards).

The hottest readable sensor (`peak`) is compared against `THERMAL_WARN_C` (93 °C, logged) and
`THERMAL_ABORT_C` (95 °C, queue stops before the next episode; the item stays queued). Reads retry
three times; if no sensor can be read the queue stops — the guard fails closed. Episodes are
separated by an interruptible cooldown (`THERMAL_COOLDOWN_S`).
