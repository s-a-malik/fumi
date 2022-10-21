# Improving Few-Shot Learning using Task-Informed Meta-Initialisation

Authors: Matthew Jackson*, Shreshth Malik*, Michael Matthews, and Yousuf Mohamed-Ahmed

Presented as a poster at FARSCOPE Robotics Conference 2022, Bristol, UK (Best Poster award). [Arxiv](https://arxiv.org/abs/2210.04843).

## Overview

This repository provides a framework to train and evaluate multi-modal models, including the proposed Fusion by Meta-Initialisation (FuMI) model, for few-shot classification on the iNat-Anim dataset.

## Dataset

iNat-Anim is a dataset specifically designed to benchmark few-shot and multi-modal image classification; each animal in the dataset has a succinct description of its appearance, and, as a result, it is desirable to create models that can leverage this information to improve few (or zero)-shot image-classification.
<img src="dataset-example.svg">

The dataset is hosted on Zenodo [here](https://zenodo.org/record/6703088#.Y1Lu4-xufAA).

You can explore the dataset using the demo in the `notebooks` directory, which can be opened in Google Colab.

## Models

There currently is support for [AM3](https://proceedings.neurips.cc/paper/2019/hash/d790c9e6c0b5e02c87b375e782ac01bc-Abstract.html), [MAML](https://arxiv.org/abs/1703.03400), [CLIP](https://arxiv.org/abs/2103.00020) and our novel model, Fusion by Meta-Initialisation (FuMI). See our [paper](https://arxiv.org/abs/2210.04843) for details.

## Usage

First download and unzip the [data](https://doi.org/10.5281/zenodo.6703088), install the requirements, and login to [wandb](https://wandb.ai/) (for logging): `pip install -r requirements.txt; wandb login <YOUR_API_KEY>`.

All experimental conditions and hyperparameters are set via command line arguments (e.g. you can change the text embedding model and use common species names instead of descriptions). Full predictions on test tasks are saved to the path given by the `--log_dir` argument, and all metrics (accuracy/prec/rec/F1) are saved to wandb.

### Example

For example, to train FuMI:

```bash
python fumi/main.py --data_dir "./data" \
--wandb_entity "YOUR_WANDB" \
--wandb_project "YOUR_PROJECT" \
--wandb_experiment "YOUR_EXPERIMENT_NAME" \
--model fumi \
--num_shots 5
```

Note, to train clip, you must also set the `--dataset supervised-inat-anim` flag as well.

## Cite

If you find our work useful, please consider citing.

```
@article{jackson2022multi,
  title={Multi-Modal Fusion by Meta-Initialization},
  author={Jackson, Matthew T and Malik, Shreshth A and Matthews, Michael T and Mohamed-Ahmed, Yousuf},
  journal={arXiv preprint arXiv:2210.04843},
  year={2022}
}
```

## Disclaimer

This is research code shared without support or guarantee of quality. Please let us know, however, if there is anything wrong or that could be improved and we will try to solve it.
