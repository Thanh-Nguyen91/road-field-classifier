# Field road classification model

## Folder structure
```
├── configs
├── core
    ├── utils.py
    ├── data_preparation.py
    ├── model.py   
    ├── dataset.py 
    ├── dataloader.py 
    ├── transform.py    
    ├── trainer.py
├── data.xlsx   # training data information
├── train.py
├── valid.py
├── inference.py
├── test_images 
├── result
    ├── valid_result.txt    # validation result of trained models
    ├── test_images_result.txt    # prediction of test_images
```

## Environment setup
```
conda env create -f environment.yml
```
or
```
docker compose build
```

## Training
change configuration in `configs/config.yml` and run training with
```
python train.py
```
or
```
docker compose up --build -d
```

## Inference
change configuration in `configs/config.yml` and run
```
python inference.py
```
or
```
docker compose up inference
```
