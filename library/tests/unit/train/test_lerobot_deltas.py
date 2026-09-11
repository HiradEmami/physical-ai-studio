# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Training windows must follow policy config rather than execution length."""

from types import SimpleNamespace

import pytest

pytest.importorskip("lerobot")

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

from physicalai.train.utils import _get_delta_indices


def test_diffusion_training_uses_shifted_horizon():
    config = DiffusionConfig(horizon=16, n_action_steps=3, n_obs_steps=2)
    model = SimpleNamespace(config=config)
    assert _get_delta_indices(model, "action_delta_indices") == list(range(-1, 15))
    assert _get_delta_indices(model, "observation_delta_indices") == [-1, 0]


def test_act_training_uses_full_chunk():
    model = SimpleNamespace(config=ACTConfig(chunk_size=8, n_action_steps=3))
    assert _get_delta_indices(model, "action_delta_indices") == list(range(8))
    assert _get_delta_indices(model, "observation_delta_indices") is None


def test_legacy_config_fallback_and_direct_model_indices():
    model = SimpleNamespace(config=SimpleNamespace(n_action_steps=3, n_obs_steps=2))
    assert _get_delta_indices(model, "action_delta_indices") == [0, 1, 2]
    assert _get_delta_indices(model, "observation_delta_indices") == [-1, 0]
    model.action_delta_indices = [1, 3]
    assert _get_delta_indices(model, "action_delta_indices") == [1, 3]
