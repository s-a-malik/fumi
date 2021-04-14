# AM3-PyTorch

Reimplementation of [AM3](https://arxiv.org/pdf/1902.07104.pdf) ([original codebase](https://github.com/ElementAI/am3)) in PyTorch using the Torchmeta library.

## Data

The datasets are all loaded using the same dataloaders via Torchmeta so if we want to use a new dataset we just need to fit it into that framework.

Datasets:

- CUB (can get label names from torchmeta datasets?)
- iNaturalist
- toy

## Usage

Hyperparameters are set with argparse flags. Use `--evaluate` flag for testing a trained model on the test set. See [colab notebook](https://colab.research.google.com/drive/1LiisACQeuVdFOg57wYzWUC1Bz1L6prHI) on the Google Drive for how the experiments are run. All logs/checkpoints are saved to [wandb](https://wandb.ai/multimodal-image-cls/am3).

Example:

```bash
python main.py --dataset zanim --data_dir ../Dataset --log_dir ./am3 \
    --experiment debug --num_shots 5 --num_ways 3 --num_shots_test 32 \
    --im_encoder precomputed --text_encoder BERT --text_type label \
    --text_hid_dim 300 --prototype_dim 512 \
```

## Experiments to try

- [ ] AM3 baseline - GloVE class label embeddings. (they use resnet like us as well)
- [ ] BERT class label embeddings
- [ ] BERT class description embeddings
- [ ] Plots of lamdba with different number of shots/ways
- [ ] No text (just image prototype, force lamda = 1 is probs easiest way to do this)
- [ ] vary N, K etc.

## TODO

- [x] Implement the model and test end-to-end
- [x] Logs and model saving on wandb.
- [ ] further metrics e.g. f1, prec, rec, lamda etc. 
- [ ] add support to visualise batches (query/support sets)
- [ ] add support for multiple training runs (or can do manually)
- [ ] add training stuff to improve performance (lr decay, augmentation etc.)
- [ ] Reproduce results on their datasets - need to get class labels from CUB etc. 
- [ ] Run experiments

## Acknowledgements

Some functions were built on code in the Torchmeta examples [repo](https://github.com/tristandeleu/pytorch-meta).
