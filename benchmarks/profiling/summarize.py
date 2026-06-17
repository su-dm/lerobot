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

"""Aggregate profiling result JSONs into a single Markdown report.

Reads every ``*.json`` in ``benchmarks/profiling/results/`` and renders a
report grouped by schema (rollout stage profile / policy bench / camera bench).
The report is designed to be self-contained so it can be read in a fresh
session without the raw data.

    .venv/bin/python benchmarks/profiling/summarize.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
REPORT = Path(__file__).resolve().parent / "REPORT.md"


def load_results() -> list[dict]:
    out = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            data["_file"] = f.name
            out.append(data)
        except Exception as e:
            print(f"skip {f.name}: {e}")
    return out


def fmt_stage_profile(d: dict) -> str:
    lines = [f"### Rollout stage profile — `{d['_file']}`", ""]
    meta = d.get("meta", {})
    lines.append(f"- device: `{d.get('device')}` | wall: {d.get('wall_duration_s', 0):.1f}s")
    if meta:
        kv = ", ".join(f"{k}={v}" for k, v in meta.items())
        lines.append(f"- meta: {kv}")
    lines.append("")
    stages = d.get("stages", {})
    lines.append("| stage | count | mean ms | p50 | p95 | max | total ms |")
    lines.append("|---|--:|--:|--:|--:|--:|--:|")
    for name in sorted(stages, key=lambda n: -stages[n]["total_ms"]):
        s = stages[name]
        lines.append(
            f"| {name} | {s['count']} | {s['mean_ms']:.2f} | {s['p50_ms']:.2f} | "
            f"{s['p95_ms']:.2f} | {s['max_ms']:.2f} | {s['total_ms']:.0f} |"
        )
    lines.append("")
    return "\n".join(lines)


def fmt_policy_bench(d: dict) -> str:
    lines = [f"### Policy benchmark — `{d['_file']}`", ""]
    lines.append(f"- policy: `{d.get('policy_type')}` @ `{d.get('policy_path')}` (rev {d.get('revision')})")
    lines.append(
        f"- device: `{d.get('device')}` dtype: `{d.get('dtype')}` | "
        f"native_hw={d.get('native_hw')} n_action_steps={d.get('n_action_steps')} "
        f"chunk_size={d.get('chunk_size')}"
    )
    lines.append("")
    lines.append("| resolution | predict_chunk mean ms | p95 | backbone | encoder | decoder |")
    lines.append("|---|--:|--:|--:|--:|--:|")
    for tag, r in d.get("by_resolution", {}).items():
        c = r["predict_action_chunk"]
        sm = r.get("submodule", {})

        def g(k):
            return f"{sm[k]['mean_ms']:.1f}" if k in sm else "-"

        lines.append(
            f"| {tag} | {c['mean_ms']:.1f} | {c['p95_ms']:.1f} | "
            f"{g('backbone')} | {g('encoder')} | {g('decoder')} |"
        )
    lines.append("")
    return "\n".join(lines)


def fmt_camera_bench(d: dict) -> str:
    lines = [f"### Camera benchmark — `{d['_file']}`", ""]
    lines.append(f"- opencv: `{d.get('opencv_version')}` | index: {d.get('index')}")
    lines.append("")
    lines.append("| fourcc req | negotiated | read mean ms | p95 | sustained fps | cvtColor ms |")
    lines.append("|---|---|--:|--:|--:|--:|")
    for run in d.get("runs", []):
        if "error" in run:
            lines.append(f"| - | ERROR: {run['error']} | - | - | - | - |")
            continue
        n = run["negotiated"]
        req = run["requested"]["fourcc"] or "none"
        neg = f"{n['width']}x{n['height']} {n['fourcc']}"
        lines.append(
            f"| {req} | {neg} | {run['read_ms']['mean']:.1f} | {run['read_ms']['p95']:.1f} | "
            f"{run['sustained_fps']:.1f} | {run['cvtColor_bgr2rgb_ms']['mean']:.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    results = load_results()
    if not results:
        print(f"No result JSONs found in {RESULTS_DIR}")
        return

    sections = {"stage": [], "policy": [], "camera": [], "other": []}
    for d in results:
        schema = d.get("schema", "")
        if schema == "lerobot.stage_profile.v1":
            sections["stage"].append(fmt_stage_profile(d))
        elif schema == "lerobot.policy_bench.v1":
            sections["policy"].append(fmt_policy_bench(d))
        elif schema == "lerobot.camera_bench.v1":
            sections["camera"].append(fmt_camera_bench(d))
        else:
            sections["other"].append(f"- `{d['_file']}` (unknown schema `{schema}`)")

    parts = ["# LeRobot profiling report", ""]
    parts.append(f"Generated from {len(results)} result file(s) in `benchmarks/profiling/results/`.")
    parts.append("")
    if sections["policy"]:
        parts += ["## Policy inference (synchronized, isolated)", ""] + sections["policy"]
    if sections["camera"]:
        parts += ["## Camera capture", ""] + sections["camera"]
    if sections["stage"]:
        parts += ["## Live rollout control loop", ""] + sections["stage"]
    if sections["other"]:
        parts += ["## Other", ""] + sections["other"]

    REPORT.write_text("\n".join(parts))
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
