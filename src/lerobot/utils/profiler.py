# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Opt-in, low-overhead stage profiler for real-time control loops.

Why this exists
---------------
``cProfile`` cannot attribute GPU time on MPS/CUDA (work is dispatched
asynchronously, so the cost lands on the next synchronization point, e.g.
``.cpu()``), and ``python -m cProfile`` only profiles the main thread (so
background camera/encoder threads are invisible). This profiler solves both:
it can force a device synchronization around stages that wrap device compute,
so wall-clock time is attributed to the correct stage.

Usage
-----
Enable by setting the environment variable ``LEROBOT_PROFILE=1``. Results are
written as JSON to ``LEROBOT_PROFILE_PATH`` (default:
``benchmarks/profiling/results/rollout_<timestamp>.json``) when the process
exits.

In code::

    from lerobot.utils.profiler import init_profiler, get_profiler

    init_profiler(device="mps", meta={"fps": 30})  # once, at startup
    prof = get_profiler()  # anywhere
    with prof.stage("loop.model", sync=True):  # sync=True for GPU stages
        action = policy.select_action(obs)

When profiling is disabled, ``get_profiler`` returns a no-op profiler whose
``stage`` is a cheap null context manager and which never synchronizes, so the
instrumentation has negligible overhead in production.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TRUE = {"1", "true", "yes", "on"}


def _make_sync(device: str | None):
    """Return a callable that blocks until pending device work completes."""
    if not device:
        return lambda: None
    try:
        import torch
    except Exception:  # torch not importable for some reason
        return lambda: None
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.synchronize
    if device == "mps" and torch.backends.mps.is_available():
        return torch.mps.synchronize
    return lambda: None


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation percentile on a pre-sorted list (q in [0, 1])."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * q
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


class StageProfiler:
    """Accumulates per-stage wall-clock durations and dumps summary statistics."""

    enabled = True

    def __init__(
        self,
        device: str | None = None,
        output_path: str | Path | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        self.device = device
        self._sync = _make_sync(device)
        self._durations: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()
        self.output_path = Path(output_path) if output_path else None
        self.meta = dict(meta or {})
        self._created = time.time()

    @contextmanager
    def stage(self, name: str, sync: bool = False):
        """Time the wrapped block under ``name``.

        Args:
            name: Stage label, e.g. ``"loop.model"``. Use dotted prefixes to
                group related stages in the report.
            sync: If True, synchronize the device before *and* after the block
                so asynchronous GPU compute is attributed to this stage rather
                than to a later synchronization point. Only pass True for
                stages that actually launch device kernels — synchronizing is
                not free.
        """
        if sync:
            self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            if sync:
                self._sync()
            dur = time.perf_counter() - start
            with self._lock:
                self._durations[name].append(dur)

    def record(self, name: str, seconds: float) -> None:
        """Record a single pre-measured duration (in seconds) under ``name``."""
        with self._lock:
            self._durations[name].append(seconds)

    def add_meta(self, **kwargs: Any) -> None:
        self.meta.update(kwargs)

    def summary(self) -> dict[str, dict[str, float]]:
        """Return per-stage statistics in milliseconds."""
        out: dict[str, dict[str, float]] = {}
        with self._lock:
            items = {k: list(v) for k, v in self._durations.items()}
        for name, vals in items.items():
            ms = sorted(v * 1e3 for v in vals)
            total = sum(ms)
            out[name] = {
                "count": len(ms),
                "total_ms": total,
                "mean_ms": total / len(ms) if ms else 0.0,
                "p50_ms": _percentile(ms, 0.50),
                "p95_ms": _percentile(ms, 0.95),
                "p99_ms": _percentile(ms, 0.99),
                "min_ms": ms[0] if ms else 0.0,
                "max_ms": ms[-1] if ms else 0.0,
            }
        return out

    def dump(self, path: str | Path | None = None) -> Path | None:
        """Write a JSON report. Returns the written path (or None if no path)."""
        path = Path(path) if path else self.output_path
        if path is None:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "lerobot.stage_profile.v1",
            "device": self.device,
            "created": datetime.fromtimestamp(self._created).isoformat(timespec="seconds"),
            "wall_duration_s": time.time() - self._created,
            "meta": self.meta,
            "stages": self.summary(),
        }
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path


class _NullProfiler:
    """No-op profiler used when profiling is disabled (negligible overhead)."""

    enabled = False

    @contextmanager
    def stage(self, name: str, sync: bool = False):
        yield

    def record(self, name: str, seconds: float) -> None:
        pass

    def add_meta(self, **kwargs: Any) -> None:
        pass

    def summary(self) -> dict[str, dict[str, float]]:
        return {}

    def dump(self, path: str | Path | None = None) -> Path | None:
        return None


_NULL = _NullProfiler()
_PROFILER: StageProfiler | _NullProfiler = _NULL
_INIT_LOCK = threading.Lock()


def profiling_enabled() -> bool:
    return os.environ.get("LEROBOT_PROFILE", "0").strip().lower() in _TRUE


def init_profiler(device: str | None = None, meta: dict[str, Any] | None = None):
    """Initialise the global profiler if ``LEROBOT_PROFILE`` is set.

    Idempotent: repeated calls return the existing instance but merge ``meta``.
    Registers an ``atexit`` hook that writes the JSON report on process exit.
    Returns the active profiler (real or no-op).
    """
    global _PROFILER
    if not profiling_enabled():
        return _PROFILER
    with _INIT_LOCK:
        if isinstance(_PROFILER, StageProfiler):
            if meta:
                _PROFILER.add_meta(**meta)
            return _PROFILER
        default = Path("benchmarks/profiling/results") / f"rollout_{datetime.now():%Y%m%d_%H%M%S}.json"
        output_path = os.environ.get("LEROBOT_PROFILE_PATH", str(default))
        _PROFILER = StageProfiler(device=device, output_path=output_path, meta=meta)
        atexit.register(_dump_on_exit)
        logger.info("StageProfiler enabled (device=%s) -> %s", device, output_path)
        return _PROFILER


def get_profiler():
    """Return the active profiler, or a no-op profiler when disabled."""
    return _PROFILER


def _dump_on_exit() -> None:
    try:
        path = _PROFILER.dump()
        if path is not None:
            logger.info("StageProfiler wrote %s", path)
            print(f"[lerobot.profiler] wrote {path}")
    except Exception as e:  # never let profiling crash shutdown
        logger.warning("StageProfiler dump failed: %s", e)
