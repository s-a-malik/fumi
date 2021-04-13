# Instantiating a data-loader
The path should be modified as appropriate, the `tokenisation_mode` can also be varied as per the code.
```python
from torchmeta.utils.data import BatchMetaDataLoader, CombinationSequentialSampler
from torchmeta.transforms import ClassSplitter
dataset = Zanim(root="/content/drive/My Drive/NLP project/Dataset", num_classes_per_task=5, meta_train=True, tokenisation_mode=TokenisationMode.BERT)
split_dataset = ClassSplitter(dataset, shuffle=True, num_test_per_class=10, num_train_per_class=10)
split_dataset.seed(0)
loader = BatchMetaDataLoader(split_dataset, shuffle=True, batch_size=16)
```

Retrieve the dictionary used to tokenize (in the case that `tokenisation_mode=TokenisationMode.STANDARD`) as follows:
```python
dataset.dictionary
```
