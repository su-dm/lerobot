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

"""Standalone webcam benchmark (no robot, no policy).

Measures the true camera path that ``cProfile`` cannot see (capture runs in a
background thread). For each requested FOURCC it reports:

  * the resolution / FPS / FOURCC the driver actually negotiated,
  * blocking ``VideoCapture.read()`` latency distribution -> sustained FPS,
  * the BGR->RGB ``cvtColor`` cost (what lerobot's OpenCVCamera does per frame).

On USB webcams the FOURCC is usually decisive: MJPG sustains 1080p30 while raw
YUYV often collapses to ~5 fps at 1080p. Run this to find out for *your* camera.

Example
-------
    .venv/bin/python benchmarks/profiling/bench_camera.py \
        --index 0 --width 1920 --height 1080 --fps 30 --frames 120 \
        --fourccs MJPG,none
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import cv2

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * q
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def fourcc_to_str(code: float) -> str:
    n = int(code)
    return "".join(chr((n >> (8 * i)) & 0xFF) for i in range(4))


def bench_one(index: int, width: int, height: int, fps: int, frames: int, fourcc: str) -> dict:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return {"error": f"could not open camera index {index}"}

    if fourcc and fourcc.lower() != "none":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    negotiated = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "fourcc": fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC)),
        "backend": cap.getBackendName(),
    }

    # Discard the first few frames (driver warmup / exposure settling).
    for _ in range(5):
        cap.read()

    read_ms: list[float] = []
    last_frame = None
    for _ in range(frames):
        s = time.perf_counter()
        ok, frame = cap.read()
        dt = (time.perf_counter() - s) * 1e3
        if not ok:
            break
        read_ms.append(dt)
        last_frame = frame

    cvt_ms: list[float] = []
    if last_frame is not None:
        for _ in range(30):
            s = time.perf_counter()
            cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
            cvt_ms.append((time.perf_counter() - s) * 1e3)

    cap.release()

    read_sorted = sorted(read_ms)
    cvt_sorted = sorted(cvt_ms)
    mean_read = sum(read_ms) / len(read_ms) if read_ms else 0.0
    return {
        "requested": {"width": width, "height": height, "fps": fps, "fourcc": fourcc},
        "negotiated": negotiated,
        "frames_read": len(read_ms),
        "read_ms": {
            "mean": mean_read,
            "p50": percentile(read_sorted, 0.50),
            "p95": percentile(read_sorted, 0.95),
            "min": read_sorted[0] if read_sorted else 0.0,
            "max": read_sorted[-1] if read_sorted else 0.0,
        },
        "sustained_fps": (1000.0 / mean_read) if mean_read else 0.0,
        "cvtColor_bgr2rgb_ms": {
            "mean": sum(cvt_ms) / len(cvt_ms) if cvt_ms else 0.0,
            "p95": percentile(cvt_sorted, 0.95),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Standalone webcam benchmark")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--fourccs", default="MJPG,none", help="comma list, e.g. MJPG,YUYV,none")
    args = p.parse_args()

    results = {
        "schema": "lerobot.camera_bench.v1",
        "created": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "opencv_version": cv2.__version__,
        "index": args.index,
        "runs": [],
    }

    for fourcc in args.fourccs.split(","):
        fourcc = fourcc.strip()
        print(f"\n[bench_camera] testing fourcc={fourcc or 'none'} {args.width}x{args.height}@{args.fps} ...")
        res = bench_one(args.index, args.width, args.height, args.fps, args.frames, fourcc)
        if "error" in res:
            print(f"  ERROR: {res['error']}")
        else:
            n = res["negotiated"]
            print(f"  negotiated: {n['width']}x{n['height']} fourcc={n['fourcc']} backend={n['backend']}")
            print(
                f"  read mean={res['read_ms']['mean']:.1f}ms p95={res['read_ms']['p95']:.1f}ms "
                f"-> sustained {res['sustained_fps']:.1f} fps"
            )
            print(f"  cvtColor BGR2RGB mean={res['cvtColor_bgr2rgb_ms']['mean']:.1f}ms")
        results["runs"].append(res)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"camera_idx{args.index}_{args.width}x{args.height}_{stamp}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[bench_camera] wrote {out}")


if __name__ == "__main__":
    main()
