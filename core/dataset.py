import os.path as osp
from typing import Dict,List,Tuple

import numpy as np
import pandas as pd
import torch
from albumentations.pytorch.transforms import ToTensorV2

from .utils import read_image

###============================================================================

class DatasetRoadField(torch.utils.data.Dataset):

    def __init__(
        self,
        base_path: str,
        base_info: str,
        targets: List[str],
        fold: int,
        transform,
        input_size: int,
        train=True,
    ):
        self.base_path = base_path
        self.targets = targets
        self.fold = fold
        self.transform = transform
        self.input_size = input_size
        self.train = train

        df = pd.read_excel(base_info)    

        if train:
            self.info = df[df["kfold"] != fold].reset_index(drop=True)
        else:
            self.info = df[df["kfold"] == fold].reset_index(drop=True)

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        try:
            image = read_image(
                osp.join(
                    self.base_path,
                    self.info.loc[idx, "file"],
                )
            ).astype(np.uint8)

            label= self.get_label(idx, self.info)
            augmented = self.transform(image=image)
            image_aug = augmented["image"]
            image_aug = ToTensorV2()(image=image_aug)["image"]

        except:
            print("=== error read/transform image:", self.info.loc[idx, "file"])
            image = np.zeros((self.input_size, self.input_size, 3)).astype(np.uint8)
            label = np.zeros(len(self.targets))
            image_aug = ToTensorV2()(image=image)["image"]
            
        return (
            image_aug,
            label.astype(np.float16, copy=False),
        )

    
    def get_label(self, index: int, info):

        label = np.zeros(len(self.targets))
    
        for feature in info.iloc[index]["target"].split("_"):
            if feature in self.targets:
                idx = self.targets.index(feature)
                label[idx] = 1

        return label


###============================================================================

def test_dataset(dataset):
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    batch = next(iter(dataloader))

    for item in batch:
        print(item.shape)
        
    plt.figure(1, figsize=(64, 64))
    for i, img in enumerate(batch[0]):
        img = (img.permute(1, 2, 0)).numpy().astype(np.float32)

        # if not apply A.Normalize()
        # img = img/255.0 

        img = img*std+mean
        img = np.clip(img, 0, 1)

        print(img.shape, np.min(img), np.max(img))
        plt.subplot(4, 4, i + 1)
        plt.imshow(img)
    plt.show()


###============================================================================
if __name__ == "__main__":
    from transform import valid_transform,train_transform

    dataset = DatasetRoadField(
        base_path="data/train",
        base_info="data.xlsx",
        targets=["fields","roads"],
        fold=0,
        transform=train_transform(input_size=384),
        input_size=384,
        train=True,
    )

    print(dataset.__len__())
    test_dataset(dataset)
