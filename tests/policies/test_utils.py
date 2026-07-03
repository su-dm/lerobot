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

"""Tests for ``lerobot.policies.utils.prepare_observation_for_inference``.

These validate the on-device image preprocessing rewrite: the output must be
numerically identical to the previous host-side implementation, while images
are uploaded as ``uint8`` and converted on the target device. The optional
on-device resize path is also covered.
"""

from __future__ import annotations

import numpy as np
import torch

from lerobot.policies.utils import prepare_observation_for_inference

IMAGE_KEY = "observation.images.cam"
STATE_KEY = "observation.state"


def _reference_image(img_hwc: np.ndarray) -> torch.Tensor:
    """The original host-side implementation, used as a numerical oracle."""
    t = torch.from_numpy(img_hwc).type(torch.float32) / 255
    t = t.permute(2, 0, 1).contiguous()
    return t.unsqueeze(0)


def _make_observation(h: int = 12, w: int = 16) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        IMAGE_KEY: rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8),
        STATE_KEY: rng.standard_normal(6).astype(np.float32),
    }


def test_image_dtype_shape_and_range():
    obs = _make_observation()
    out = prepare_observation_for_inference(obs, torch.device("cpu"))

    img = out[IMAGE_KEY]
    assert img.dtype == torch.float32
    assert img.shape == (1, 3, 12, 16)  # (B, C, H, W)
    assert img.is_contiguous()
    assert float(img.min()) >= 0.0
    assert float(img.max()) <= 1.0


def test_image_values_match_reference_implementation():
    obs = _make_observation()
    expected = _reference_image(obs[IMAGE_KEY].copy())
    out = prepare_observation_for_inference(obs, torch.device("cpu"))
    torch.testing.assert_close(out[IMAGE_KEY], expected)


def test_state_keeps_dtype_and_gets_batch_dim():
    obs = _make_observation()
    expected = torch.from_numpy(obs[STATE_KEY].copy())
    out = prepare_observation_for_inference(obs, torch.device("cpu"))

    state = out[STATE_KEY]
    assert state.dtype == torch.float32
    assert state.shape == (1, 6)
    torch.testing.assert_close(state.squeeze(0), expected)


def test_task_and_robot_type_defaults_and_overrides():
    out = prepare_observation_for_inference(_make_observation(), torch.device("cpu"))
    assert out["task"] == ""
    assert out["robot_type"] == ""

    out = prepare_observation_for_inference(
        _make_observation(), torch.device("cpu"), task="pick", robot_type="so101"
    )
    assert out["task"] == "pick"
    assert out["robot_type"] == "so101"


def test_resize_downscales_image_to_target_shape():
    obs = _make_observation(h=48, w=64)
    out = prepare_observation_for_inference(
        obs,
        torch.device("cpu"),
        image_resize_shapes={IMAGE_KEY: (3, 24, 32)},
    )
    img = out[IMAGE_KEY]
    assert img.shape == (1, 3, 24, 32)
    assert img.dtype == torch.float32
    assert img.is_contiguous()
    assert float(img.min()) >= 0.0
    assert float(img.max()) <= 1.0


def test_resize_is_noop_when_shape_already_matches():
    """When the captured frame already matches the target, output is unchanged."""
    obs = _make_observation(h=24, w=32)
    expected = _reference_image(obs[IMAGE_KEY].copy())
    out = prepare_observation_for_inference(
        obs,
        torch.device("cpu"),
        image_resize_shapes={IMAGE_KEY: (3, 24, 32)},
    )
    torch.testing.assert_close(out[IMAGE_KEY], expected)


def test_resize_only_applies_to_listed_keys():
    obs = _make_observation(h=20, w=20)
    # Target map references a different key, so the image must not be resized.
    out = prepare_observation_for_inference(
        obs,
        torch.device("cpu"),
        image_resize_shapes={"observation.images.other": (3, 8, 8)},
    )
    assert out[IMAGE_KEY].shape == (1, 3, 20, 20)
