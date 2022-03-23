
---

<div align="center">

# Poetry generation with GPT-2

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>


</div>

## Description

This repository contains code for training poetry generation model and training model for converting poetry into images.

## How to run

Install dependencies

```yaml
poetry install
```

Download data

    dvc pull

Train model with default configuration

```yaml
# train on CPU
python run.py trainer.gpus=0

# train on GPU
python run.py trainer.gpus=1
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```yaml
python run.py experiment=experiment_name
```

You can override any parameter from command line like this

```yaml
python run.py trainer.max_epochs=20 datamodule.batch_size=64
```


### S3 for DVC configuration
To download data from 4soft's S3 you need access to that S3 :)
Assuming you have it, the configuration steps are as follows:

1. ```dvc remote add``` *remote*-**name** *remote*-**address**
2. ```dvc remote default``` *remote*-**name**
3. Done.

<br>