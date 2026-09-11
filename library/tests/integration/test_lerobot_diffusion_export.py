# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Offline round trips with real LeRobot models and processors."""

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("lerobot")

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from physicalai.inference.model import InferenceModel

from physicalai.policies.lerobot import LeRobotPolicy


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(params=["environment", "image"])
def diffusion_policy(request):
    inputs = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,))}
    if request.param == "image":
        inputs["observation.images.camera"] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64))
    else:
        inputs["observation.environment_state"] = PolicyFeature(type=FeatureType.ENV, shape=(2,))
    config = DiffusionConfig(
        input_features=inputs,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        normalization_mapping={
            "STATE": NormalizationMode.MEAN_STD,
            "ENV": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
            "VISUAL": NormalizationMode.IDENTITY,
        },
        horizon=8,
        n_action_steps=3,
        n_obs_steps=2,
        down_dims=(32, 64),
        n_groups=8,
        diffusion_step_embed_dim=32,
        num_inference_steps=2,
        pretrained_backbone_weights=None,
        crop_shape=None,
        spatial_softmax_num_keypoints=4,
        device="cpu",
    )
    stats = {
        key: {"mean": torch.full(feature.shape, 2.0), "std": torch.full(feature.shape, 3.0)}
        for key, feature in {**inputs, **config.output_features}.items()
        if feature.type != FeatureType.VISUAL
    }
    return LeRobotPolicy(policy_name="diffusion", config=config, dataset_stats=stats).eval()


def observation(policy, step, *, batched=False, history=False):
    result = {}
    for name, feature in policy.config.input_features.items():
        runtime_name = name.replace("observation.state", "state").replace("observation.images.", "images.")
        prefix = (1, 2) if history else ((1,) if batched else ())
        result[runtime_name] = np.full((*prefix, *feature.shape), step, dtype=np.float32)
    return result


def native_batch(inputs):
    return {
        key.replace("state", "observation.state", 1)
        if key == "state"
        else key.replace("images.", "observation.images.", 1): torch.from_numpy(value.copy())
        for key, value in inputs.items()
    }


@pytest.mark.parametrize("batched", [False, True])
def test_real_diffusion_export_single_observation(diffusion_policy, tmp_path, batched):
    policy = diffusion_policy
    policy.to_torch(tmp_path)
    model = InferenceModel(tmp_path, backend="torch", device="cpu")
    assert type(model.adapter._policy.lerobot_policy) is type(policy.lerobot_policy)
    assert (tmp_path / "diffusion.pt").is_file()
    assert [(feature.name, tuple(feature.shape)) for feature in model.input_features] == [
        (feature.name, feature.shape) for feature in policy.inputs_schema
    ]
    assert tuple(model.output_features[0].shape) == (3, 2)
    # Saved action statistics must denormalize one to mean + std, not identity.
    torch.testing.assert_close(model.adapter._policy._postprocessor(torch.ones(1, 3, 2)), torch.full((1, 3, 2), 5.0))
    for step in range(3):
        inputs = observation(policy, step, batched=batched)
        torch.manual_seed(123 + step)
        expected = policy.predict_action_chunk(native_batch(inputs)).detach().numpy()[0]
        torch.manual_seed(123 + step)
        actual = model.predict_action_chunk(inputs)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        assert np.isfinite(actual).all()


def test_real_diffusion_runtime_buffer_and_reset(diffusion_policy, tmp_path):
    policy = diffusion_policy
    policy.to_torch(tmp_path)
    model = InferenceModel(tmp_path, device="cpu")
    # Native select_action sees every observation, including between chunk predictions.
    torch.manual_seed(456)
    expected = [policy.select_action(native_batch(observation(policy, step))).numpy()[0] for step in range(7)]
    torch.manual_seed(456)
    actual = [model.select_action(observation(policy, step)) for step in range(7)]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    for name in ("observation.state", "observation.environment_state", "observation.images"):
        if name in policy.lerobot_policy._queues:
            expected_history = torch.stack(list(policy.lerobot_policy._queues[name]))
            actual_history = torch.stack(list(model.adapter._policy.lerobot_policy._queues[name]))
            torch.testing.assert_close(actual_history, expected_history)
    # Known statistics must be restored by checkpoint loading, not merely
    # produce matching results through two identically misconfigured paths.
    state_history = torch.stack(list(model.adapter._policy.lerobot_policy._queues["observation.state"]))
    torch.testing.assert_close(state_history[:, 0, 0], torch.tensor([1.0, 4.0 / 3.0]))
    model.reset()
    assert all(len(queue) == 0 for queue in model.adapter._policy.lerobot_policy._queues.values())
    policy.reset()
    inputs = observation(policy, 10)
    torch.manual_seed(789)
    expected = policy.select_action(native_batch(inputs)).numpy()[0]
    torch.manual_seed(789)
    np.testing.assert_allclose(model.select_action(inputs), expected, rtol=1e-5, atol=1e-5)


def test_real_diffusion_explicit_history_after_online(diffusion_policy, tmp_path):
    policy = diffusion_policy
    policy.to_torch(tmp_path)
    model = InferenceModel(tmp_path, device="cpu")
    model.predict_action_chunk(observation(policy, 99))
    saved_history = list(model.adapter._policy.lerobot_policy._queues["observation.state"])
    history = observation(policy, 1, history=True)
    torch.manual_seed(123)
    expected = policy.predict_action_chunk(native_batch(history)).numpy()[0]
    torch.manual_seed(123)
    np.testing.assert_allclose(model.predict_action_chunk(history), expected, rtol=1e-5, atol=1e-5)
    for before, after in zip(
        saved_history, model.adapter._policy.lerobot_policy._queues["observation.state"], strict=True
    ):
        torch.testing.assert_close(before, after)


def test_act_output_schema_matches_full_chunk():
    config = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(3,)),
            "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(2,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        normalization_mapping={name: NormalizationMode.IDENTITY for name in ("STATE", "ENV", "ACTION")},
        chunk_size=8,
        n_action_steps=3,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        use_vae=False,
        pretrained_backbone_weights=None,
        device="cpu",
    )
    policy = LeRobotPolicy(policy_name="act", config=config).eval()
    with torch.no_grad():
        actual = policy({"observation.state": torch.zeros(1, 3), "observation.environment_state": torch.zeros(1, 2)})
    assert tuple(actual.shape[1:]) == policy.outputs_schema[0].shape == (8, 2)
