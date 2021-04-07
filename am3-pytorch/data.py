"""Data processing functions and classes.
"""

import torchmeta.datasets as datasets


def get_dataset(dataset: str, data_dir: str, num_way: int):
    """Return the appropriate dataset, with preprocessing transforms
    """

    if dataset == "CUB":
        train, val, test = get_CUB(dataset, data_dir, num_way)
    elif dataset == "iNat":
        train, val, test = get_inat(dataset, data_dir, num_way)
    else:
        raise NotImplementedError()

    return train, val, test


def get_CUB(dataset: str, data_dir: str, num_way: int):
    # add transforms 
    train = datasets.CUB(data_dir, num_classes_per_task=num_way, meta_split="train",
                        download=True)

    val = datasets.CUB(data_dir, num_classes_per_task=num_way, meta_split="val", 
                        download=True)
    
    test = datasets.CUB(data_dir, num_classes_per_task=num_way, meta_split="test",
                        download=True)
    
    return train, val, test

def get_inat(dataset: str, data_dir: str, num_way: int):
    """loads up our data
    """
    raise NotImplementedError()