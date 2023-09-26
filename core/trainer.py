import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torchmetrics.classification import MultilabelFBetaScore

from .model import ClassifierRoadField


class Litmodel(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = ClassifierRoadField(
            self.cfg.model.backbone,
            self.cfg.model.pretrained,
            self.cfg.model.target_size,
        )

        self.bce_loss = nn.BCEWithLogitsLoss()

        self.f1 = MultilabelFBetaScore(
            beta=1.0, num_labels=2,
        )

        self.save_hyperparameters()

    
    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, batch_idx):
        (imgs,labels) = batch

        pred= self(imgs)

        loss = self.bce_loss(pred,labels)

        f1score = self.f1(pred,labels)

        self.log(
            "train_loss", loss, prog_bar=True, on_step=False, on_epoch=True
        )
        
        self.log(
            "train_f1", f1score, prog_bar=False, on_step=False, on_epoch=True
        )

        return loss

    
    def validation_step(self, batch, batch_idx):
        (imgs,labels) = batch

        pred= self(imgs)

        loss = self.bce_loss(pred,labels)

        f1score = self.f1(pred,labels)

        self.log(
            "val_loss", loss, prog_bar=True, on_step=False, on_epoch=True
        )
        
        self.log(
            "val_f1", f1score, prog_bar=False, on_step=False, on_epoch=True
        )

        return loss


    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.cfg.training.lr)

        # if resume from checkpoint, load the last_epoch
        last_epoch = self.cfg.training.resume_epoch if self.cfg.training.resume else 0

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.cfg.training.max_epochs, last_epoch
        )
        return [optimizer], [scheduler]
