import os
import warnings

import pytorch_lightning as pl
import torch
import yaml
from addict import Dict as Adict
from lightning_fabric.utilities.seed import seed_everything
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from core import DataModuleRoadField, Litmodel

warnings.simplefilter("ignore", UserWarning)
import torch._dynamo

torch._dynamo.config.suppress_errors = True


def main(cfg_yaml: str):
    with open(cfg_yaml, "r") as file:
        cfg = Adict(yaml.safe_load(file))

    seed_everything(seed=cfg.training.seed, workers=True)

    datamodule = DataModuleRoadField(cfg=cfg)

    litmodel = Litmodel(cfg=cfg)

    logger = TensorBoardLogger("logs/", name=cfg.training.logname)

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

    ckpt_loss = pl.callbacks.ModelCheckpoint(
        filename="best_loss_ep{epoch}",
        auto_insert_metric_name=False,
        monitor="val_loss",
        save_last=True,
        save_top_k=3,
        mode="min",
    )

    ckpt_f1 = pl.callbacks.ModelCheckpoint(
        filename="best_f1_ep{epoch}",
        auto_insert_metric_name=False,
        monitor="val_f1",
        save_last=False,
        save_top_k=3,
        mode="max",
    )


    torch.set_float32_matmul_precision("high")
    strategy = "ddp_find_unused_parameters_true"

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        callbacks=[ckpt_loss, ckpt_f1, lr_monitor],
        logger=logger,
        max_epochs=cfg.training.max_epochs,
        strategy=strategy,
        precision="16-mixed",
        fast_dev_run=False,
    )

    if cfg.training.resume:
        trainer.fit(
            model=litmodel,
            datamodule=datamodule,
            ckpt_path=cfg.training.weight_path,
        )
    else:
        trainer.fit(model=litmodel, datamodule=datamodule)


if __name__ == "__main__":
    print("=== pytorch lightening version", pl.__version__)
    main(cfg_yaml=os.getenv("CONFIG_FILE", "configs/config.yml"))
