import os
import warnings

import pytorch_lightning as pl
import torch
import yaml
from addict import Dict as Adict
from lightning_fabric.utilities.seed import seed_everything

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

    torch.set_float32_matmul_precision("high")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=cfg.training.max_epochs,
        precision="16-mixed",
        fast_dev_run=False,
    )

    litmodel = Litmodel.load_from_checkpoint(cfg.valid.weight_path)
    trainer.validate(model=litmodel, datamodule=datamodule)

if __name__ == "__main__":
    print("=== pytorch lightening version", pl.__version__)
    main(cfg_yaml=os.getenv("CONFIG_FILE", "configs/config.yml"))
