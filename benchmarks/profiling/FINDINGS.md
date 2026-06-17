# ACT-on-MPS rollout performance — findings

**Goal:** run `su-dm/act-baseline-seed1000` (ACT, trained at **1080×1920**) on a
MacBook M1 at 30 FPS with a 1920×1080 webcam via `lerobot-rollout`.

**Status:** model + camera data collected; live-loop (real-robot) data still
pending. Reusable suite under `benchmarks/profiling/`. This file is the
human-readable analysis; `REPORT.md` holds the auto-generated tables; raw JSON
is in `results/`.

---

## Environment

- Apple Silicon M1, macOS, Python 3.13, torch 2.11.0, MPS available.
- Policy: ACT, ResNet18 backbone, `dim_model=512`, 4 encoder layers, 1 decoder
  layer, `chunk_size=100`, `n_action_steps=100`. One camera
  (`observation.images.front`), native visual feature shape `3×1080×1920`.

## Methodology (and why not just cProfile)

MPS executes asynchronously, so wall-clock timing only attributes GPU work to
the next synchronization point. The user's `cProfile` run confirmed this: the
"model" ops were near-zero (`torch.conv2d` 0.107s, `linear` 0.056s) while the
loop crawled, with time piling onto `tensor.to` (3.48s) — the `.cpu()` sync.
`python -m cProfile` also only profiles the main thread, hiding the camera
capture thread. We therefore measure with explicit `torch.mps.synchronize()`:

- `bench_policy.py` — isolates the model, sweeps resolution, splits
  backbone/encoder/decoder via forward hooks. (No hardware.)
- `bench_camera.py` — measures true webcam read latency / sustained FPS /
  FOURCC. (Camera only; needs macOS camera permission.)
- `StageProfiler` (`LEROBOT_PROFILE=1`) — per-stage split during a real
  rollout, synchronized. (Real robot.)

---

## Result 1 — Model forward is the periodic stall (MEASURED)

`predict_action_chunk` on MPS, float32, mean over 30 iters (10 warmup):

| resolution             | forward mean | backbone | encoder | decoder |
| ---------------------- | -----------: | -------: | ------: | ------: |
| **1080×1920 (native)** | **145.8 ms** |     77.2 |    54.4 |     3.8 |
| 720×1280               |      64.0 ms |     36.9 |    20.4 |     3.1 |
| 480×640                |      25.0 ms |     14.7 |     7.0 |     2.2 |
| 256×256                |       9.8 ms |      4.8 |     3.4 |     1.9 |

Interpretation:

- At native 1080p the forward is **~146 ms**. With `n_action_steps=100`, this
  fires once per ~100 control ticks, i.e. a ~146 ms hiccup roughly every 3 s at
  30 FPS — this matches the user's logs (warnings clustered ~3 s apart at the
  instantaneous "5–6 Hz" of that single slow tick). The loop is mostly fine on
  the 99 cheap ticks; the problem is the periodic stall (bad for control
  smoothness) plus per-tick preprocessing tax (below).
- Cost is **super-linear in resolution**: backbone scales ~with pixels
  (77→15 ms, 480p is ~5× fewer pixels), the **encoder (self-attention) scales
  quadratically** with token count (54→7 ms). 1080p→480×640 ≈ **5.8× faster
  forward**; →256×256 ≈ **15× faster**.
- Decoder is negligible (≤4 ms) — it's fixed at `chunk_size` queries.

## Result 2 — fp16 autocast is SLOWER on MPS (MEASURED, overturns a hypothesis)

`predict_action_chunk`, float16 via `torch.autocast(device_type="mps")`:

| resolution |     fp32 | fp16 autocast |
| ---------- | -------: | ------------: |
| 1080×1920  | 145.8 ms |  **184.4 ms** |
| 480×640    |  25.0 ms |      106.1 ms |
| 256×256    |   9.8 ms |      103.8 ms |

fp16 autocast is _slower everywhere_, with a suspicious ~100 ms floor
independent of resolution. autocast inserts many fp32↔fp16 cast ops around
ACT's numerous small ops; on MPS that overhead dominates. **Do not use autocast
fp16 for ACT on MPS.** (A static `.half()` weight cast is not usable as-is: ACT
keeps fp32 positional-embedding buffers and crashes with a broadcast error — it
would need dtype-correctness work, and may still not beat fp32 given these
numbers.) This is the opposite of my earlier recommendation; the measurement
corrected it.

## Result 3 — From the user's cProfile (CPU-side, reliable parts)

- `tensor.to` 3.48 s / 3114 calls — largest torch cost; the 25 MB float32 1080p
  image uploaded to MPS **every tick** plus sync absorption.
- `prepare_observation_for_inference` 2.29 s cumulative — CPU `/255` + permute +
  `contiguous` on the 1080p frame, **every tick**.
- `cvtColor` ~0.24–0.40 s; `contiguous` 0.514 s; `type`(float) 0.344 s.
- From DEBUG logs: motor read `~1.0–1.4 ms`, camera consume `read front 0.0 ms`
  (cached buffer peek). **Control plane / robot I/O is not the bottleneck.**

Key structural inefficiency observed in code: `SyncInferenceEngine.get_action`
runs `prepare_observation` + `preprocess` (the 25 MB upload + normalize) on
**every** tick, even though `select_action` only consumes the image once per
`n_action_steps` (the other 99 ticks just pop a cached action). So the per-tick
image tax is largely wasted work.

## Result 4 — Camera is NOT a bottleneck (MEASURED)

`bench_camera.py` at 1920×1080@30, 120 frames, AVFoundation backend:

| mode    | negotiated | read mean |     p95 | sustained fps | cvtColor BGR2RGB |
| ------- | ---------- | --------: | ------: | ------------: | ---------------: |
| MJPG    | 1920×1080  |   33.1 ms | 36.8 ms |      **30.2** |           0.7 ms |
| default | 1920×1080  |   34.9 ms | 47.2 ms |      **28.6** |           1.1 ms |

The ~33 ms read time _is_ the 30 FPS frame interval — `cap.read()` blocks until
the next frame, it is not processing cost. The camera comfortably delivers
30 fps at 1080p, and `cvtColor` is ~1 ms. This confirms the loop-side
`read front: 0.0ms` (cached buffer peek): **the camera keeps up; it is not the
bottleneck.** (AVFoundation ignores the OpenCV FOURCC hint, so the FOURCC field
reads blank in both modes.)

## Result 5 — Live control loop (PENDING — needs user + robot)

```bash
LEROBOT_PROFILE=1 .venv/bin/lerobot-rollout \
  --strategy.type=base --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5AE60806291 --robot.id=my_follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
  --policy.path=su-dm/act-baseline-seed1000 --policy.revision=chunk100 \
  --task="Grab the white cube and place it in the grey container" \
  --device=mps --use_torch_compile=false --fps=30 --play_sounds=false
```

Run ~30–60 s, Ctrl-C (exits cleanly, writes JSON). Then
`.venv/bin/python benchmarks/profiling/summarize.py`. Look at
`infer.select_action` **max/p95** (the heavy forward) vs `infer.preprocess`
(per-tick image tax) vs `dispatch.robot_send` and `loop.sleep`.

---

## Recommendations, ranked by MEASURED return

1. **Lower the policy input resolution (biggest win).** 1080p→480×640 turns the
   ~146 ms forward into ~25 ms (5.8×); →256×256 into ~9.8 ms (15×). Because the
   model was trained at 1080p, the _proper_ path is to **retrain/fine-tune at
   the lower resolution** (downscaling at inference only is a train/test shift;
   test it, but retrain to be safe). This single change most likely makes 30 FPS
   feasible.
2. **Add a resize step to the sync inference path** (parity with the async path's
   `resize_robot_observation_image`). Lets you capture 1080p for recording while
   the policy sees a small frame — also shrinks the 25 MB→~few-MB device upload.
3. **Stop re-preprocessing on cached-action ticks.** Only upload+normalize the
   image when the policy will actually run the backbone (queue empty), saving the
   per-tick 25 MB transfer + normalize on ~99% of ticks.
4. **Upload uint8 then convert on-device** (instead of CPU `/255`+permute+
   `contiguous` on 1080p before `.to`): smaller host→device copy, work on GPU.
5. **Fix the `torch.compile` bug** (passes `mode` _and_ `options` → always
   throws; `triton.cudagraphs` is CUDA-only). Low priority on MPS (inductor is
   immature there) but it's a clear bug.
6. **Do NOT pursue fp16 autocast on MPS** — measured slower. Revisit only with a
   dtype-correct static-half model and re-measure.

```

```
