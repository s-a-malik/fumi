# Improving Few-Shot Learning using Task-Informed Meta-Initialisation

Authors: Matthew Jackson*, Shreshth Malik*, Michael Matthews, and Yousuf Mohamed-Ahmed

## Overview

This repository provides a framework to train and evaluate multi-modal models for few-shot classification on our Zanim dataset. It also includes the code used for collating the class descriptions used to create our dataset. 


## Dataset 
iNat-Anim is a dataset specifically designed to benchmark few-shot and multi-modal image classification; each animal in the dataset has a succinct description of its appearance, and, as a result, it is desirable to create models that can leverage this information to improve few (or zero)-shot image-classification.
<img src="dataset-example.svg">

The dataset is hosted on Zenodo [here (TODO)]().

You can explore the dataset using the demo in the `notebooks` directory.

## Models

There currently is support for [AM3](https://proceedings.neurips.cc/paper/2019/hash/d790c9e6c0b5e02c87b375e782ac01bc-Abstract.html), [MAML](https://arxiv.org/abs/1703.03400), [CLIP](https://arxiv.org/abs/2103.00020) and our novel model, Fusion by Meta-Initialisation (FuMI).

## Usage

First download the requirements and login to [wandb](https://wandb.ai/) (for logging): `pip install -r requirements.txt; wandb login <YOUR_API_KEY>`.

All experimental conditions and hyperparameters are set via command line arguments. Full predictions on test tasks are saved to the path given by the `--log_dir` argument, and all metrics (accuracy/prec/rec/F1) are saved to wandb.

### Example

For example, to train FuMI:
```bash
python main.py --data_dir "./dataset" --log_dir "./fumi" --wandb_entity "YOUR_WANDB" \
   --model fumi --patience 10000 --dropout 0.25 --im_hid_dim 256 64 --lr 3e-5 \
   --experiment fumi-5-shots --num_shots 5 --seed 123 --num_test_adapt_steps 100
```


## Cite

If you find our work useful, please consider citing. (TBC)


## Disclaimer

This is research code shared without support or guarantee of quality. Please let us know, however, if there is anything wrong or that could be improved and we will try to solve it.

