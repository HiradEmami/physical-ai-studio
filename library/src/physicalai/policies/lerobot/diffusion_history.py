# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Observation history for the pinned LeRobot diffusion implementation.

LeRobot 0.6.0 exposes queue updates only through select_action. Chunk-based
Runtime inference must also update them when no new action is generated.
Keep the private queue access confined here rather than copying the model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

if TYPE_CHECKING:
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy


def observe(policy: DiffusionPolicy, batch: dict[str, torch.Tensor]) -> None:
    """Append a normalized single observation without sampling an action.

    Args:
        policy: Initialized native diffusion policy.
        batch: Preprocessed, batched observation tensors.

    Raises:
        ValueError: State contains a temporal window or the policy is not initialized.
    """
    if batch[OBS_STATE].ndim != 2:  # noqa: PLR2004
        msg = "Online diffusion observations require state shape (batch, state_dim)."
        raise ValueError(msg)
    features = policy.config.input_features
    queues = policy._queues  # noqa: SLF001
    if features is None or queues is None:
        msg = "Diffusion features and observation queues must be initialized."
        raise ValueError(msg)
    # Clone retained values: robot clients may reuse their input array buffers.
    observation = {key: value.detach().clone() for key, value in batch.items() if key in features}
    if policy.config.image_features:
        observation[OBS_IMAGES] = torch.stack([observation[key] for key in policy.config.image_features], dim=-4)
    policy._queues = populate_queues(queues, observation, exclude_keys=[ACTION])  # noqa: SLF001


def predict_chunk(policy: DiffusionPolicy, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Predict from an online observation or an independent temporal window.

    Args:
        policy: Initialized native diffusion policy.
        batch: Preprocessed inputs, optionally with explicit batched history.

    Returns:
        Normalized action chunk of shape (batch, n_action_steps, action_dim).
    """
    if batch[OBS_STATE].ndim == 3:  # noqa: PLR2004
        # Native chunk prediction otherwise ignores supplied windows once the
        # online queues are populated. Restore both history and pending actions.
        queues = policy._queues  # noqa: SLF001
        policy.reset()
        try:
            return policy.predict_action_chunk(batch)
        finally:
            policy._queues = queues  # noqa: SLF001

    observe(policy, batch)
    # Native online prediction uses the combined image queue key.
    online = {key: value for key, value in batch.items() if key != ACTION}
    if policy.config.image_features:
        online[OBS_IMAGES] = torch.stack([batch[key] for key in policy.config.image_features], dim=-4)
    return policy.predict_action_chunk(online)
