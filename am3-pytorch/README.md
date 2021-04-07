# AM3-PyTorch

Reimplementation of [AM3](https://arxiv.org/pdf/1902.07104.pdf) ([original codebase](https://github.com/ElementAI/am3)) in PyTorch using the Torchmeta library.

## Data

The datasets are all loaded using the same dataloaders via Torchmeta so if we want to use a new dataset we just need to fit it into that framework.

Datasets:

- CUB
- iNaturalist
- toy

## Usage

Hyperparameters are set with argparse flags. Use `--evaluate` flag for testing a trained model on the test set.

Example:
```bash
python train.py --arg1 {arg1} --arg2 {arg2}
```

## TODO

- [ ] Implement the model and test end-to-end
- [ ] Reproduce results on their datasets
- [ ] Use model on our datasets
- [ ] Extend to use BERT embeddings and class descriptions instead of class labels