#!/usr/bin/env python
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

"""Standalone policy inference benchmark (no robot, no camera required).

This isolates the model — the one stage ``cProfile`` cannot measure on MPS
because GPU work is asynchronous. It:

  1. Loads a policy exactly like the rollout does (config + weights + device).
  2. Builds synthetic, correctly-shaped observations on-device.
  3. Times ``predict_action_chunk`` (the heavy forward) and ``select_action``
     (the per-tick call that pops a cached chunk most of the time) with proper
     ``synchronize()`` so the numbers are real.
  4. Sweeps input resolution to quantify the cost of feeding large frames.
  5. For ACT, splits the forward into backbone / encoder / decoder via hooks.

Results are written as JSON to ``benchmarks/profiling/results/``.

Example
-------
    .venv/bin/python benchmarks/profiling/bench_policy.py \
        --policy.path su-dm/act-baseline-seed1000 \
        --revision chunk100 --device mps \
        --resolutions 1080x1920,720x1280,480x640,256x256
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.policies.factory import get_policy_class

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def make_sync(device: str):
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.synchronize
    if device == "mps" and torch.backends.mps.is_available():
        return torch.mps.synchronize
    return lambda: None


def percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * q
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def timeit(fn, sync, n: int, warmup: int) -> dict[str, float]:
    """Time ``fn`` ``n`` times after ``warmup`` calls, synchronizing each time."""
    for _ in range(warmup):
        fn()
    sync()
    xs: list[float] = []
    for _ in range(n):
        s = time.perf_counter()
        fn()
        sync()
        xs.append((time.perf_counter() - s) * 1e3)
    xs.sort()
    total = sum(xs)
    return {
        "count": len(xs),
        "mean_ms": total / len(xs),
        "p50_ms": percentile(xs, 0.50),
        "p95_ms": percentile(xs, 0.95),
        "min_ms": xs[0],
        "max_ms": xs[-1],
    }


def build_observation(policy, config, device: str, hw: tuple[int, int]) -> dict[str, torch.Tensor]:
    """Synthesize a normalized-ish batch matching the policy's input features.

    Values are random; ACT has no data-dependent control flow, so timing is
    independent of the actual pixel values. Images are float32 in [0, 1].
    """
    batch: dict[str, torch.Tensor] = {}
    h, w = hw
    for key, feat in config.input_features.items():
        if feat.type is FeatureType.VISUAL:
            c = feat.shape[0] if len(feat.shape) == 3 else 3
            batch[key] = torch.rand(1, c, h, w, device=device)
        else:
            dim = feat.shape[0]
            batch[key] = torch.randn(1, dim, device=device)
    batch["task"] = ""
    batch["robot_type"] = ""
    return batch


def attach_act_submodule_hooks(policy, sync) -> dict[str, list[float]]:
    """Register hooks that time ACT backbone/encoder/decoder. Returns timing dict."""
    timings: dict[str, list[float]] = {"backbone": [], "encoder": [], "decoder": []}
    state: dict[str, float] = {}
    model = policy.model

    def pre(name):
        def _pre(_m, _inp):
            sync()
            state[name] = time.perf_counter()

        return _pre

    def post(name):
        def _post(_m, _inp, _out):
            sync()
            timings[name].append((time.perf_counter() - state[name]) * 1e3)

        return _post

    handles = []
    for name, module in (
        ("backbone", getattr(model, "backbone", None)),
        ("encoder", getattr(model, "encoder", None)),
        ("decoder", getattr(model, "decoder", None)),
    ):
        if module is not None:
            handles.append(module.register_forward_pre_hook(pre(name)))
            handles.append(module.register_forward_hook(post(name)))
    return timings, handles


def summarize(xs: list[float]) -> dict[str, float]:
    xs = sorted(xs)
    if not xs:
        return {}
    total = sum(xs)
    return {
        "count": len(xs),
        "mean_ms": total / len(xs),
        "p50_ms": percentile(xs, 0.50),
        "p95_ms": percentile(xs, 0.95),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Standalone policy inference benchmark")
    p.add_argument("--policy.path", dest="policy_path", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--device", default="mps")
    p.add_argument("--n", type=int, default=30, help="timed iterations per measurement")
    p.add_argument("--warmup", type=int, default=10, help="warmup iterations (MPS kernel compile)")
    p.add_argument(
        "--resolutions",
        default="native",
        help="comma list HxW, e.g. 1080x1920,720x1280,480x640,256x256. 'native' uses the config shape.",
    )
    p.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    args = p.parse_args()

    device = args.device
    sync = make_sync(device)

    print(f"[bench_policy] loading {args.policy_path} (rev={args.revision}) on {device} ...")
    config = PreTrainedConfig.from_pretrained(args.policy_path, revision=args.revision)
    config.device = device
    policy = get_policy_class(config.type).from_pretrained(
        args.policy_path, config=config, revision=args.revision
    )
    policy = policy.to(device).eval()
    # fp16 is measured via autocast (not .half()): policies keep some fp32
    # buffers (e.g. ACT's sinusoidal position embeddings) that break a global
    # weight cast on MPS. autocast is also the production-correct approach.
    autocast_ctx = (
        torch.autocast(device_type=device, dtype=torch.float16) if args.dtype == "float16" else nullcontext()
    )

    # Discover native resolution from the visual feature shape.
    image_features = {k: v for k, v in config.input_features.items() if v.type is FeatureType.VISUAL}
    native_hw = None
    if image_features:
        shape = next(iter(image_features.values())).shape
        if len(shape) == 3:
            native_hw = (shape[1], shape[2])
    print(f"[bench_policy] type={config.type} image_features={list(image_features)} native_hw={native_hw}")
    print(
        f"[bench_policy] n_action_steps={getattr(config, 'n_action_steps', '?')} "
        f"chunk_size={getattr(config, 'chunk_size', '?')}"
    )

    if args.resolutions == "native":
        resolutions = [native_hw] if native_hw else [(480, 640)]
    else:
        resolutions = []
        for tok in args.resolutions.split(","):
            h, w = tok.lower().split("x")
            resolutions.append((int(h), int(w)))

    results = {
        "schema": "lerobot.policy_bench.v1",
        "created": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "dtype": args.dtype,
        "platform": platform.platform(),
        "policy_path": args.policy_path,
        "revision": args.revision,
        "policy_type": config.type,
        "native_hw": list(native_hw) if native_hw else None,
        "n_action_steps": getattr(config, "n_action_steps", None),
        "chunk_size": getattr(config, "chunk_size", None),
        "image_features": list(image_features),
        "n": args.n,
        "warmup": args.warmup,
        "by_resolution": {},
    }

    for hw in resolutions:
        tag = f"{hw[0]}x{hw[1]}"
        print(f"\n[bench_policy] === resolution {tag} ===")
        batch = build_observation(policy, config, device, hw)

        # Heavy forward: full ResNet + transformer, fills the action queue.
        def heavy():
            with torch.inference_mode(), autocast_ctx:
                policy.predict_action_chunk(dict(batch))

        chunk_stats = timeit(heavy, sync, args.n, args.warmup)
        print(
            f"  predict_action_chunk: mean={chunk_stats['mean_ms']:.1f}ms p95={chunk_stats['p95_ms']:.1f}ms"
        )

        # Per-submodule split for ACT.
        submodule = {}
        if config.type == "act":
            timings, handles = attach_act_submodule_hooks(policy, sync)
            with torch.inference_mode(), autocast_ctx:
                for _ in range(max(5, args.n // 3)):
                    policy.predict_action_chunk(dict(batch))
                    sync()
            for h in handles:
                h.remove()
            submodule = {k: summarize(v) for k, v in timings.items() if v}
            if submodule:
                parts = "  ".join(f"{k}={v['mean_ms']:.1f}ms" for k, v in submodule.items())
                print(f"  submodule split: {parts}")

        results["by_resolution"][tag] = {
            "predict_action_chunk": chunk_stats,
            "submodule": submodule,
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"policy_{config.type}_{device}_{args.dtype}_{stamp}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[bench_policy] wrote {out}")


if __name__ == "__main__":
    main()
