# Instantiating a data-loader
The path should be modified as appropriate, the `tokenisation_mode` can also be varied as per the code.
```python
from torchmeta.utils.data import BatchMetaDataLoader, CombinationSequentialSampler
from torchmeta.transforms import ClassSplitter
z = Zanim(root="/content/drive/My Drive/NLP project/Dataset", num_classes_per_task=5, meta_train=True, tokenisation_mode=TokenisationMode.BERT)
z = ClassSplitter(z, shuffle=True, num_test_per_class=10, num_train_per_class=10)
z.seed(0)
loader = BatchMetaDataLoader(z, shuffle=True, batch_size=16)
```
