# Improving Few-shot Image Classification using Class Descriptions

Authors: Matthew Jackson, Shreshth Malik, Michael Matthews, and Yousuf Mohamed-Ahmed

## Overview

This repository provides a framework to train and evaluate multi-modal models for few-shot classification on our Zanim dataset. It also includes the code used for collating the class descriptions used to create our dataset. 


## Dataset 
Zanim ('zero-shot animals') is a dataset specifically designed to benchmark few-shot and multi-modal image classification; each animal in the dataset has a succinct description with a description of its appearance. This lends itself to attempting to create models that can leverage this information to improve few (or zero)-shot image-classification.
<img src="dataset-example.svg">

There is a zip of the entire set of images [here](https://drive.google.com/file/d/1ZYXk71G_DPZqrbE3Obqw7wN2b8yF0ejq/view?usp=sharing); the [license](https://drive.google.com/file/d/1_XMWVk63t5szHEuW-D5NB3Gab2c9NWX_/view?usp=sharing) of the dataset; a [json](https://drive.google.com/file/d/1Wr9ZBk3SMHFqkm_H9NYqx-Y_zJB0Cnpu/view?usp=sharing) with annotations and class descriptions.

## Models

There currently is support for [AM3](https://proceedings.neurips.cc/paper/2019/hash/d790c9e6c0b5e02c87b375e782ac01bc-Abstract.html), [MAML](https://arxiv.org/abs/1703.03400), [CLIP](https://arxiv.org/abs/2103.00020) and our novel model, FuMI.

## Usage

First download the requirements and login to [wandb](https://wandb.ai/) (for logging): `pip install -r requirements.txt; wandb login`. The wandb entity will have to be changed in `main.py` to work with a different account.

All experimental conditions and hyperparameters are set via command line arguments. Full predictions on test tasks are saved to the path given by the `--log_dir` argument, and all metrics (accuracy/prec/rec/F1) are saved to wandb.

### Example

For example, to train FuMI:
```bash
python main.py --dataset zanim --data_dir "./Dataset" --log_dir "./fumi" \
    --model fumi --experiment label --seed 123 --patience 10000 --eval_freq 500 \
    --epochs 50000 --optim adam --lr 1e-4 --weight_decay 0.0005 --batch_size 4 \
    --num_shots 10 --num_ways 5 --num_shots_test 15 --num_ep_test 200 \
    --im_encoder precomputed --image_embedding_model resnet-152 \
    --im_emb_dim 2048 --im_hid_dim 64 --text_encoder glove --pooling_strat mean \
    --remove_stop_words --text_type common_name --text_emb_dim 768 --text_hid_dim 256 \
    --step_size 0.01 --num_train_adapt_steps 5 --num_test_adapt_steps 25 --shared_feats
```

## Cite

If you find our work useful, please consider citing. (TBC)


## Disclaimer

This is research code shared without support or guarantee of quality. Please let us know, however, if there is anything wrong or that could be improved and we will try to solve it.

