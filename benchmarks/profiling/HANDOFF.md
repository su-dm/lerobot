# ACT-on-M1 performance work — handoff

> Read this first if you're picking up the "make ACT run at 30 FPS on a MacBook M1
> at 1080p" task. It's the narrative + index. The hard data lives in the JSON files
> under `results/`; the ranked technical audit lives in `FINDINGS.md` (same folder).
> This file tells you what we tried, what we measured, what shipped, what's still
> broken, and what to do next.

## 1. The goal

Run an **ACT policy** in real time on an **Apple M1 (MPS)** driving an **SO-101
follower** arm, with a **1920×1080** camera, at **30 FPS**. The policy
(`su-dm/act-baseline-seed1000`, revision `chunk100`) was **trained on 1080×1920**
data, so naive downscaling is a train/test shift (accuracy risk), not a free win.

Repro command (the one the user actually runs):

```bash
.venv/bin/lerobot-rollout --strategy.type=base --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem5AE60806291 --robot.id=my_follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
  --policy.path=su-dm/act-baseline-seed1000 --policy.revision=chunk100 \
  --task="..." --device=mps --fps=30 --play_sounds=false --inference.type=sync
```

Symptom: `Record loop is running slower (~1–5 Hz) than the target FPS (30 Hz)`.

## 2. How the control loop actually works (read this before optimizing)

Per tick, `BaseStrategy.run` (`src/lerobot/rollout/strategies/base.py`) does:

1. `robot.get_observation()` — camera frame (cached, background thread) + motor read.
2. `_process_observation_and_notify()` — runs the **robot** observation processor,
   **gated** by `interpolator.needs_new_action()`.
3. `send_next_action()` (`src/lerobot/rollout/strategies/core.py`) — also gated by
   `needs_new_action()`: `build_dataset_frame` → `engine.get_action()` →
   interpolate → `robot_action_processor` → `robot.send_action()`.

Key subtlety that drives everything: with `interpolation_multiplier == 1` (default),
`needs_new_action()` is **True every tick**, so `get_action` is called every tick.

Inside `SyncInferenceEngine.get_action` (`src/lerobot/rollout/inference/sync.py`) the
per-tick pipeline is: `prepare_observation_for_inference` → policy `preprocessor`
(normalize) → `policy.select_action` → `postprocessor` → `.cpu()` → reorder.

**ACT has its own internal `_action_queue`** (`policies/act/modeling_act.py`):
`select_action` only runs the heavy forward (`predict_action_chunk`) once every
`n_action_steps` calls; the other calls just `popleft()` and **ignore the
observation**. So pre-D, the image was uploaded + normalized every tick even though
the forward (and thus the image) was only consumed every `n_action_steps` ticks.

## 3. Profiling methodology (and how to reproduce)

`cProfile` is misleading here: MPS work is async (cost lands at the next
`.cpu()`/sync), and it only sees the main thread (camera capture is a background
thread). So we built an **MPS-sync-aware `StageProfiler`**
(`src/lerobot/utils/profiler.py`) that optionally `torch.mps.synchronize()`s around
device stages.

- **Enable in a live rollout:** prefix the command with `LEROBOT_PROFILE=1`. On exit
  (Ctrl-C) it writes `benchmarks/profiling/results/rollout_<ts>.json`. Override the
  path with `LEROBOT_PROFILE_PATH=...`.
- **Standalone policy benchmark:** `benchmarks/profiling/bench_policy.py` — synced
  `predict_action_chunk` latency across a resolution sweep, with backbone/encoder/
  decoder split. Use `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` if cached.
- **Standalone camera benchmark:** `benchmarks/profiling/bench_camera.py` — read
  latency, sustained FPS, MJPG vs raw, `cvtColor` cost. **Must run in a terminal with
  macOS camera (TCC) permission** — a sandboxed process can't inherit it.
- **Aggregate to a report:** `benchmarks/profiling/summarize.py` → `REPORT.md`.
- Schema / details: `benchmarks/profiling/README.md`.

## 4. Where the results are + the headline numbers

All under `benchmarks/profiling/results/`:

| File                                          | What                                                              | Headline                                                                                                          |
| --------------------------------------------- | ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `policy_act_mps_float32_20260617_125422.json` | Policy sweep, fp32                                                | 1080p `predict_action_chunk` **~146 ms** (backbone 77 ms, encoder 54 ms); 480×640 **~25 ms**; 256×256 **~9.8 ms** |
| `policy_act_mps_float16_20260617_133436.json` | Policy sweep, fp16 autocast                                       | 1080p **~184 ms** — **SLOWER than fp32** on MPS                                                                   |
| `camera_idx0_1920x1080_20260617_142820.json`  | Webcam @1080p                                                     | MJPG 33.1 ms → **30.2 fps**; default 34.9 ms → 28.6 fps; `cvtColor` ~1 ms                                         |
| `rollout_20260617_143940.json`                | Live instrumented rollout @1080p (**baseline, pre-optimization**) | see below                                                                                                         |

Baseline live rollout (1080p, every-tick preprocessing, `n_action_steps` internal):

- `infer.prepare_observation`: mean **7.08 ms**, every tick (count 1037)
- `infer.preprocess` (normalize): mean **6.74 ms**, every tick
- `infer.select_action`: p50 **0.49 ms** (queue pop), p99 **96.7 ms**, max **607 ms** (forward + MPS warmup)
- `loop.get_observation`: mean **1.18 ms** (cheap — camera is cached, not the bottleneck)
- `loop.compute`: p50 **16 ms**, p99 **131 ms**
- `dispatch.robot_send`: 0.12 ms (serial I/O is not the bottleneck)

Interpretation: the steady-state tax was the **per-tick 1080p image work**
(`prepare` + `preprocess` ≈ 13.8 ms) plus the periodic **forward spike** (~150 ms,
up to ~600 ms cold).

## 5. What shipped (code changes, all tested + ruff-clean)

1. **On-device uint8 preprocessing** (findings A/C/G) —
   `src/lerobot/policies/utils.py::prepare_observation_for_inference`. Upload raw
   **uint8 (H,W,C)** to device first, then float/`/255`/permute/`contiguous` on the
   accelerator; `non_blocking` on CUDA. 4× smaller host→device copy, no CPU memcpy.
   Numerically identical (test-verified). Added optional `image_resize_shapes` for
   on-device bilinear resize.
2. **Opt-in sync-path resize** (rec #2) — `SyncInferenceConfig.resize_observation_images`
   in `src/lerobot/rollout/inference/factory.py`; derives target shapes from the
   policy's VISUAL input features and passes them to the engine.
3. **Chunked action cache (finding D, ACT-only)** —
   `SyncInferenceConfig.chunked_action_cache` + `_resolve_chunk_action_steps()`
   (factory) + `SyncInferenceEngine._get_action_chunked / _run_policy_chunk`
   (`sync.py`). Drives the policy via `predict_action_chunk`, postprocesses the whole
   chunk once, serves CPU actions from a local FIFO, so image upload+normalize only
   run on refill ticks. Behaviour-preserving for ACT **without** temporal ensembling
   (factory refuses non-ACT, ensembling, and validates `n_action_steps`).

Tests: `tests/policies/test_utils.py` (preprocessing parity/shape/range/resize),
`tests/test_rollout.py` (resize wiring + chunked cache serve/refill/validation).

Instrumentation added to `base.py`, `core.py`, `sync.py` (the `prof.stage(...)` calls).

## 6. Experiment results — what worked, what didn't

- ❌ **fp16 autocast on MPS**: _slower_ than fp32 for ACT (184 vs 146 ms). Do not
  pursue without a dtype-correct static-half model and re-measurement.
- ➖ **uint8 on-device preprocessing**: correct and cleaner, but the user reported
  "not much difference" — because the dominant remaining cost is the **forward spike**,
  not the per-tick transfer, once you're at 1080p. Still worth keeping (it removes the
  CPU memcpy and cuts the transfer 4×; matters more at higher tick rates / smaller res).
- ✅ **Chunked action cache (D)**: **works.** After enabling it, the slow-loop
  warnings went from _continuous_ to _one every ~3.3 s_ (= `chunk_steps/fps` =
  100/30). That proves ~99/100 ticks are now fast and only the **refill tick** stalls.
  Confirmed by log-timestamp cadence (the warning has no rate-limiting, so one per
  3.3 s ⇒ one slow tick per 3.3 s).

## 7. THE current open problem

The refill tick runs the **full ACT forward (~150 ms, up to ~770 ms during MPS
kernel warmup) synchronously on the control loop**. So the arm runs smoothly for
~3 s, then freezes ~0.2–0.8 s while the next chunk computes. A 150 ms forward
cannot fit in a 33 ms tick, so caching alone can't fix this — the forward must move
**off** the control loop.

## 8. Recommended next steps (in priority order)

1. **Background chunk prefetch (the real fix — proposed, not yet built).** Add a
   background thread to the chunked sync engine that prefetches the next chunk when
   the cache drops below a watermark (e.g. 20 actions left ≈ 0.66 s of runway at
   30 fps — plenty to hide a 150 ms forward), while the main loop keeps serving cached
   actions. The threading pattern is already proven by `RTCInferenceEngine`
   (`src/lerobot/rollout/inference/rtc.py`) — copy its lifecycle (start/stop/pause/
   resume, obs holder + lock, error backoff). **Why not just use RTC?** RTC calls
   `predict_action_chunk(inference_delay=..., prev_chunk_left_over=...)`, a signature
   plain ACT doesn't implement. So this is a lighter "async chunk prefetch" without
   RTC's reanchoring/latency logic. On cheap ticks the main thread does **zero MPS
   work**, so there's no MPS contention with the background forward.
   - Watch out for: MPS warmup on the very first forward(s) (pre-warm before the loop
     starts); resetting the thread + cache on `engine.reset()`; what to return if the
     cache momentarily empties (return `None` → loop reuses last action, or block
     briefly). Add tests mirroring the existing chunked-cache tests.
2. **Resolution reduction + retrain.** The forward is ~146 ms @1080p vs ~25 ms
   @480×640 vs ~9.8 ms @256×256 — near-quadratic in the transformer. Combine
   `--inference.resize_observation_images=true` (already shipped) to shrink the input,
   but **retrain/fine-tune at the lower resolution** to avoid the train/test shift.
   This is the highest-accuracy-preserving way to make the forward itself cheap enough
   that even synchronous refills fit.
3. **Fix the `torch.compile` bug** (early finding, **not yet fixed**). In
   `src/lerobot/rollout/context.py` the compile kwargs pass both `mode` **and**
   `options` (torch raises "Either mode or options can be specified…"), and
   `triton.cudagraphs` is CUDA-only. Fix so `--use_torch_compile` works on MPS at all;
   then measure whether inductor helps on MPS (historically immature — measure, don't
   assume).
4. **`channels_last` end-to-end** to remove the physical CHW transpose (the
   `contiguous`), letting ResNet consume NHWC directly. Low priority; measure.

## 9. Code map (where to look)

- Control loop: `src/lerobot/rollout/strategies/base.py`, `.../core.py`
- Sync engine (chunked cache lives here): `src/lerobot/rollout/inference/sync.py`
- Engine config + factory + validation: `src/lerobot/rollout/inference/factory.py`
- Async/RTC reference for background threading: `src/lerobot/rollout/inference/rtc.py`
- Preprocessing (uint8/on-device/resize): `src/lerobot/policies/utils.py`
- ACT internals (`select_action`, `predict_action_chunk`, `_action_queue`,
  `ACTTemporalEnsembler`): `src/lerobot/policies/act/modeling_act.py`
- Normalizer (image MEAN_STD): `src/lerobot/processor/normalize_processor.py`
- Device move step: `src/lerobot/processor/device_processor.py`
- Camera (bg thread, resolution enforcement, `read_latest`, `cvtColor`):
  `src/lerobot/cameras/opencv/camera_opencv.py`
- Profiler: `src/lerobot/utils/profiler.py`
- Ranked technical audit: `benchmarks/profiling/FINDINGS.md`

## 10. Gotchas / advice (things that cost us time)

- **The OpenCV camera enforces resolution.** `_validate_width_and_height` _raises_ if
  the webcam won't honor the requested `width/height`. So if a run starts, the camera
  really is capturing at that size and the model truly sees that size. Setting
  `width: 640, height: 480` in the camera config _is_ real source downscaling (and is
  different from the software `resize_observation_images`, which resizes _after_
  capture — only useful when you must capture high-res).
- **Camera is NOT the bottleneck.** `get_observation` is ~1.18 ms (frames are served
  from a cache filled by a background thread). Don't chase the camera.
- **MPS is async.** Any timing without `torch.mps.synchronize()` is attributed to the
  wrong stage. Use the `StageProfiler` with `sync=True` for device stages.
- **`LEROBOT_LOG_LEVEL=DEBUG` does nothing** — `init_logging` (`src/lerobot/utils/utils.py`)
  hardcodes console level to INFO and ignores the env var (minor, unfixed).
- **Sandbox/network:** model download needs network (`HF_HUB_OFFLINE=1` +
  `TRANSFORMERS_OFFLINE=1` when cached); MPS + camera don't work inside the sandbox
  (need real GPU/TCC). Some tests `mkdir` in `~/.cache/huggingface` — run them outside
  the sandbox.
- **Chunked cache is behaviour-preserving only for ACT w/o temporal ensembling.** With
  ensembling, `n_action_steps==1` and every tick needs a fresh forward + the ensembler
  (which lives in `select_action`) — the factory correctly refuses it. SAC's
  `predict_action_chunk` raises; Diffusion-family populate obs-history queues as a
  side effect of `select_action`. Don't blindly generalize D to other policies.
