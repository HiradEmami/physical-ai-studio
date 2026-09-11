# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Record a reproducible diffusion loss trend using public PushT samples."""

# ruff: noqa: INP001

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
from collections.abc import Mapping
from pathlib import Path

import lightning
import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger

from physicalai.data.lerobot import LeRobotDataModule
from physicalai.policies.lerobot import Diffusion
from physicalai.train import Trainer

logger = logging.getLogger(__name__)


class LossRecorder(Callback):
    """Record actual losses from Trainer optimizer steps."""

    def __init__(self) -> None:
        """Initialize the per-step loss record."""
        self.losses: list[float] = []

    def on_train_batch_end(
        self,
        _trainer: lightning.Trainer,
        _pl_module: lightning.LightningModule,
        outputs: object,
        _batch: object,
        _batch_idx: int,
    ) -> None:
        """Retain the loss produced by the completed training batch.

        Raises:
            TypeError: The batch did not produce a tensor loss.
        """
        loss = outputs.get("loss") if isinstance(outputs, Mapping) else outputs
        if not isinstance(loss, torch.Tensor):
            message = "Expected a tensor training loss."
            raise TypeError(message)
        self.losses.append(float(loss.detach().cpu()))
        if len(self.losses) % 25 == 0:
            logger.info("Step %d, last-25 mean loss %.6f", len(self.losses), np.mean(self.losses[-25:]))


def main() -> None:  # noqa: PLR0914 - retain the full verification recipe in one CLI entry point
    """Train or smoke-test the recorded configuration and write measured results.

    Raises:
        SystemExit: Training or validation does not meet the recorded criteria.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/lerobot/diffusion_verification.yaml"))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = yaml.safe_load(args.config.read_text())
    torch.set_num_threads(2)
    lightning.seed_everything(config["seed"], workers=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    datamodule = LeRobotDataModule(root=args.dataset_root.resolve(), **config["data"])
    policy = Diffusion(**config["model"])
    recorder = LossRecorder()
    trainer_args = dict(config["trainer"])
    if args.smoke:
        trainer_args["fast_dev_run"] = True
    trainer = Trainer(
        **trainer_args,
        callbacks=[recorder],
        logger=CSVLogger(output, name="logs"),
        default_root_dir=output,
    )
    trainer.fit(policy, datamodule=datamodule)
    validation = trainer.validate(policy, datamodule=datamodule)
    window = config["loss_window"]
    first = float(np.mean(recorder.losses[:window]))
    last = float(np.mean(recorder.losses[-window:]))
    finite = bool(np.isfinite(recorder.losses).all())
    val_finite = bool(validation and np.isfinite(validation[0].get("val/loss", np.nan)))
    success = finite and val_finite and (args.smoke or (len(recorder.losses) >= 2 * window and last < first))
    result = {
        "config": config,
        "smoke": args.smoke,
        "losses": recorder.losses,
        "initial_window_mean": first,
        "final_window_mean": last,
        "validation": validation,
        "passed": success,
        "versions": {name: importlib.metadata.version(name) for name in ["torch", "lerobot", "lightning", "av"]},
    }
    (output / "results.json").write_text(json.dumps(result, indent=2))
    if not args.smoke:
        trainer.save_checkpoint(output / "diffusion.ckpt")
        policy.to_torch(output / "export")
    logger.info("Recorded result: initial %.6f, final %.6f, passed=%s", first, last, success)
    if not success:
        message = "Training verification failed; see results.json."
        raise SystemExit(message)


if __name__ == "__main__":
    main()
