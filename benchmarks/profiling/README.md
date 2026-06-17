# Rollout profiling suite

Tools to measure where time goes when deploying a policy with `lerobot-rollout`,
designed for Apple Silicon (MPS) where naive timing and `cProfile` mislead you.

## Why three tools

| Tool                   | Measures                                   | Sees GPU time?     | Sees bg threads? |
| ---------------------- | ------------------------------------------ | ------------------ | ---------------- |
| `bench_policy.py`      | model forward, isolated + resolution sweep | yes (synchronized) | n/a              |
| `bench_camera.py`      | webcam read latency / FPS / FOURCC         | n/a                | yes (direct)     |
| `StageProfiler` (live) | per-stage split during a real rollout      | yes (synchronized) | yes              |

`cProfile` is intentionally **not** the primary tool: on MPS the GPU runs
asynchronously, so model time is misattributed to the next `.cpu()` sync, and
`python -m cProfile` only profiles the main thread (the camera capture thread is
invisible). The tools here synchronize explicitly and/or run on the relevant
thread.

## 1. Policy benchmark (no hardware)

```bash
.venv/bin/python benchmarks/profiling/bench_policy.py \
  --policy.path su-dm/act-baseline-seed1000 --revision chunk100 --device mps \
  --resolutions 1080x1920,720x1280,480x640,256x256
```

Isolates the model and reports `predict_action_chunk` latency plus a
backbone/encoder/decoder split (ACT) at each resolution. Add `--dtype float16`
to measure half precision.

## 2. Camera benchmark (camera only, no robot)

```bash
.venv/bin/python benchmarks/profiling/bench_camera.py \
  --index 0 --width 1920 --height 1080 --fps 30 --fourccs MJPG,none
```

Reports the negotiated format and sustained FPS per FOURCC.

## 3. Live rollout profiler (real robot)

Set `LEROBOT_PROFILE=1` and run any rollout. A JSON report is written on exit
(Ctrl-C is fine — it exits cleanly through teardown).

```bash
LEROBOT_PROFILE=1 .venv/bin/lerobot-rollout \
  --strategy.type=base --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5AE60806291 --robot.id=my_follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
  --policy.path=su-dm/act-baseline-seed1000 --policy.revision=chunk100 \
  --task="..." --device=mps --fps=30 --play_sounds=false
```

Optional: `LEROBOT_PROFILE_PATH=/abs/path.json` to choose the output file.

Stage names: `loop.*` (control loop), `dispatch.*` (action dispatch),
`infer.*` (inside the sync inference engine). `infer.select_action` is the
model; it is called every tick but only runs the heavy forward once per
`n_action_steps`, so look at its **max/p95**, not its mean.

## 4. Build the report

```bash
.venv/bin/python benchmarks/profiling/summarize.py
```

Aggregates every `results/*.json` into `REPORT.md`.
