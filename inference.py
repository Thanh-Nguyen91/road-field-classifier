import os
import torch
import numpy as np
import pandas as pd
import yaml
from addict import Dict as Adict

from core import ClassifierRoadField, valid_transform
from albumentations.pytorch.transforms import ToTensorV2
from core.utils import get_files
from PIL import Image
import matplotlib.pyplot as plt


class Inference():
    def __init__(self,cfg,device,checkpoint_path=None):
        self.cfg = cfg
        self.device = device
        if checkpoint_path is not None:
            self.checkpoint_path = checkpoint_path
        else:
            self.checkpoint_path = cfg.valid.weight_path

        self.transform = valid_transform(input_size=384)
        self.model = self.load_model()
        self.inference_results = []

    
    def load_model(self):
        model = ClassifierRoadField(
                backbone=self.cfg.model.backbone,
                pretrained=self.cfg.model.pretrained,
                target_size=self.cfg.model.target_size,
        )

        # Load the checkpoint
        checkpoint = torch.load(self.checkpoint_path)
        checkpoint["state_dict"] = {
            k[6:]: v for k, v in checkpoint["state_dict"].items()
        }

        # Load model state_dict from the checkpoint
        model.load_state_dict(checkpoint['state_dict'])

        # Set the model to evaluation mode
        model.eval().to(self.device)
        return model

    
    def process_batch(self,batched_inputs):
        # Stack batched input tensors
        batch_tensor = torch.stack(batched_inputs, dim=0).to(self.device)

        # Perform inference
        with torch.no_grad():
            pred = self.model(batch_tensor)
        
        # apply sigmoid and convert to numpy array
        pred = torch.sigmoid(pred)
        pred=pred.cpu().numpy()

        # Append the batched outputs to the inference results list
        self.inference_results.extend(pred)


    def process_directory(self,input_directory:str,batch_size = 100,save_fig=False):
        # Get file paths for inference
        filepaths = get_files(input_directory)
        filepaths_shorten = [filepath.replace(input_directory,"").lstrip('/') for filepath in filepaths]

        batched_inputs = []

        # Loop through each file in the directory
        for filepath in filepaths:
            # Load and preprocess the image
            image = Image.open(filepath).convert("RGB")
            aug_image = self.transform(image=np.array(image))["image"]
            input_tensor = ToTensorV2()(image=aug_image)["image"]

            # Add the input tensor to the batched lists
            batched_inputs.append(input_tensor)


            # Process the batch when it reaches the desired batch size
            if len(batched_inputs) == batch_size:
                self.process_batch(batched_inputs)                

                # Clear the batched lists for the next batch
                batched_inputs = []

        # Process any remaining inputs in the last batch
        if batched_inputs:
            self.process_batch(batched_inputs)

        # Save inference result to excel file
        result = pd.DataFrame(
                self.inference_results,
                index=filepaths_shorten,
                columns=self.cfg.model.targets
            )

        result.to_excel("result.xlsx")

        # save image with prediction
        if save_fig:
            os.makedirs("result",exist_ok=True)
            for filepath,pred in zip(filepaths,self.inference_results):
                img = Image.open(filepath).convert("RGB")
                plt.imshow(img)
                plt.title(f'Field {pred[0]:.2f} Road {pred[1]:.2f}')
                plt.savefig(f'result/{filepath.split("/")[-1]}')


    def process_one_image(self,filepath:str):
        # Load and preprocess the image
        image = Image.open(filepath).convert("RGB")
        aug_image = self.transform(image=np.array(image))["image"]
        input_tensor = ToTensorV2()(image=aug_image)["image"].unsqueeze(0).to(self.device)

        # Perform inference
        with torch.no_grad():
            pred = self.model(input_tensor)
        
        # apply sigmoid and convert to numpy array
        pred = torch.sigmoid(pred)
        pred=pred.cpu().numpy()

        result = {feature:score for feature,score in zip(self.cfg.model.targets,pred[0].tolist())}
        return result


if __name__=="__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(os.getenv("CONFIG_FILE", "configs/config.yml"), "r") as file:
        cfg = Adict(yaml.safe_load(file))

    # checkpoint_path = "logs/baseline-bceloss-f1score-input384/version_3/checkpoints/best_loss_ep87.ckpt"
    # inference = Inference(cfg,device,checkpoint_path)
    
    inference = Inference(cfg,device)

    inference.process_directory("data/test_images",save_fig=True)

    # result = inference.process_one_image("data/test_images/2.jpeg")
    # print(result)
