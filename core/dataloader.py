import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import DatasetRoadField
from .transform import (
    train_transform,
    valid_transform,
)

###=====================================================================
def get_data_sampler(samples_weight):
    samples_weight = torch.from_numpy(np.array(samples_weight)).double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler


###=====================================================================
class DataModuleRoadField(LightningDataModule):
    def __init__(self, cfg):

        super().__init__()
        self.cfg = cfg

    
    def train_dataloader(self) -> DataLoader:

        train_dataset = DatasetRoadField(
            base_path=self.cfg.data.base_path,
            base_info=self.cfg.data.base_info,
            targets=self.cfg.model.targets,
            fold=self.cfg.training.kfold,
            input_size=self.cfg.training.input_size,
            train=True,
            transform=train_transform(
                input_size=self.cfg.training.input_size,
                reg_factor=0.9,
            ),
        )
        
        print("=== train dataset size",len(train_dataset))

        samples_weight = train_dataset.info["weight"]

        sampler = get_data_sampler(samples_weight)

        return DataLoader(
            train_dataset,
            batch_size=self.cfg.training.batch_size,
            num_workers=self.cfg.training.num_workers,
            sampler=sampler,
        )

    
    def val_dataloader(self) -> DataLoader:

        val_dataset = DatasetRoadField(
            base_path=self.cfg.data.base_path,
            base_info=self.cfg.data.base_info,
            targets=self.cfg.model.targets,
            fold=self.cfg.training.kfold,
            input_size=self.cfg.training.input_size,
            train=False,
            transform= valid_transform(
                input_size=self.cfg.model.model_size
            ),
        )
        
        print("=== valid dataset size",len(val_dataset))

        return DataLoader(
            val_dataset,
            batch_size=self.cfg.valid.batch_size,
            num_workers=self.cfg.training.num_workers,
            shuffle=False,
        )


###=====================================================================

def test_dataloader(dataloader):
    import matplotlib.pylab as plt

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    batch = next(iter(dataloader))

    for item in batch:
        print(item.shape)

    plt.figure(1, figsize=(64, 64))
    for i, img in enumerate(batch[0]):
        if i < 16:
            img = (img.permute(1, 2, 0)).numpy().astype(np.float32)

            # if not apply A.Normalize()
            # img = img/255.0 

            img = img*std+mean
            img = np.clip(img, 0, 1)

            print(img.shape, np.min(img), np.max(img))
            plt.subplot(4, 4, i + 1)
            plt.imshow(img)
    plt.show()


###=====================================================================

if __name__ == "__main__":
    import yaml
    from addict import Dict as Adict

    with open("configs/config.yml", "r") as file:
        cfg = Adict(yaml.safe_load(file))
    
    datamodule = DataModuleRoadField(cfg=cfg)

    test_dataloader(datamodule.train_dataloader())
    test_dataloader(datamodule.val_dataloader())
