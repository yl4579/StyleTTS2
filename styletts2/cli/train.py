import os
import shutil
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from everyvoice.base_cli.interfaces import train_base_command_interface
from everyvoice.utils import spinner
from merge_args import merge_args


class Mode(str, Enum):
    first = "first"
    second = "second"
    finetune = "finetune"


@merge_args(train_base_command_interface)
def train(
    mode: Annotated[
        Mode,
        typer.Option(
            "-m",
            "--mode",
            help="Training mode: 'first' (acoustic pre-training with TMA), 'second' (joint diffusion+adversarial), or 'finetune'.",
        ),
    ] = Mode.first,
    precision: Annotated[
        str,
        typer.Option(
            help="Floating-point precision passed to Lightning Trainer (e.g. '32', '16-mixed', 'bf16-mixed').",
        ),
    ] = "32",
    **kwargs,
):
    """Train a StyleTTS2 end-to-end TTS model."""
    with spinner():
        import torch

        if not torch.cuda.is_available():
            # device="cuda" is assumed in multiple places, so let's just tell the user up front
            # It's also pointless to try on CPU if it takes around a week on GPU...
            sys.exit(
                "ERROR: StyleTTS2 training requires a GPU with the cuda accellerator"
            )

        import lightning as L
        from everyvoice.utils import update_config_from_cli_args
        from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
        from lightning.pytorch.loggers import TensorBoardLogger
        from lightning.pytorch.strategies import DDPStrategy

        from .ev_config import (
            StyleTTS2Config,
        )
        from .ev_config.translation import (
            to_native_config,
        )
        from .lightning import (
            StyleTTS2DataModule,
            StyleTTS2Module,
        )

    config_file: Path = kwargs["config_file"]
    config_args: list[str] = kwargs.get("config_args", [])

    ev_config = StyleTTS2Config.load_config_from_path(config_file)
    ev_config = update_config_from_cli_args(config_args, ev_config)

    config = to_native_config(ev_config)

    tr = ev_config.training
    max_epochs = (
        tr.epochs_1st
        if mode == Mode.first
        else tr.epochs_2nd if mode == Mode.second else tr.max_epochs
    )

    log_dir = config["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    shutil.copy(str(config_file), os.path.join(log_dir, config_file.name))

    tb_logger = TensorBoardLogger(save_dir=log_dir, name="tensorboard", version="")

    ckpt_filename = f"epoch_{mode.value[0]}_" + "{epoch:05d}"
    # Always keep the last checkpoint regardless of performance.
    last_ckpt_callback = ModelCheckpoint(
        dirpath=log_dir,
        filename=ckpt_filename,
        save_top_k=1,
        save_last=True,
        every_n_train_steps=tr.ckpt_steps,
        every_n_epochs=tr.ckpt_epochs,
        enable_version_counter=True,
        save_on_train_epoch_end=True,
    )
    # Keep only the top-k checkpoints ranked by val/mel (lower is better).
    monitored_ckpt_callback = ModelCheckpoint(
        dirpath=log_dir,
        filename=ckpt_filename,
        monitor="val/mel",
        mode="min",
        save_top_k=tr.save_top_k_ckpts,
        every_n_train_steps=tr.ckpt_steps,
        every_n_epochs=tr.ckpt_epochs,
        enable_version_counter=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    devices = kwargs.get("devices", "auto")
    strategy = kwargs.get("strategy", "ddp")

    # GAN training uses separate discriminator/generator backward passes,
    # so find_unused_parameters=True is required for DDP correctness.
    try:
        n_devices = int(devices)
        multi_gpu = n_devices > 1
    except (TypeError, ValueError):
        multi_gpu = devices not in ("auto", "1", 1)

    if strategy == "ddp" or (multi_gpu and strategy == "auto"):
        resolved_strategy = DDPStrategy(find_unused_parameters=True)
    else:
        resolved_strategy = strategy

    trainer = L.Trainer(
        max_epochs=max_epochs,
        devices=devices,
        num_nodes=kwargs.get("nodes", 1),
        accelerator=kwargs.get("accelerator", "auto"),
        strategy=resolved_strategy,
        precision=precision,
        logger=tb_logger,
        callbacks=[monitored_ckpt_callback, last_ckpt_callback, lr_monitor],
        log_every_n_steps=config.get("log_interval", 10),
        enable_progress_bar=True,
    )

    datamodule = StyleTTS2DataModule(config, load_for_everyvoice=True)
    model = StyleTTS2Module(config, mode=mode.value)

    resume_ckpt = (
        str(tr.finetune_checkpoint)
        if tr.finetune_checkpoint and os.path.exists(tr.finetune_checkpoint)
        else None
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt)
