"""Data processing functions and classes.
"""

import torchmeta.datasets.helpers as datasets
from torchmeta.utils.data import BatchMetaDataLoader

def get_dataset(args):
    """Return the appropriate dataset, with preprocessing transforms
    """
    dataset = args.dataset
    data_dir = args.data_dir
    num_way = args.num_way
    num_shots = args.num_shots
    num_shots_test = args.num_shots_test
    text_encoder = args.text_encoder
    text_type = args.text_type

    if dataset == "CUB":
        train, val, test = get_CUB(dataset, data_dir, num_way, num_shots, num_shots_test)
    elif dataset == "iNat":
        train, val, test = get_inat(dataset, data_dir, num_way, num_shots, num_shots_test, text_encoder, text_type)
    else:
        raise NotImplementedError()
    
    train_loader = BatchMetaDataLoader(train,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=args.num_workers)
    val_loader = BatchMetaDataLoader(val,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=args.num_workers)
    test_loader = BatchMetaDataLoader(test,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=args.num_workers)

    return train_loader, val_loader, test_loader


def get_CUB(dataset: str, data_dir: str, num_way: int, num_shots: int, num_shots_test: int):
    # add transforms 
    train = datasets.cub(data_dir, ways=num_way, shots=num_shots, test_shots=num_shots_test,
                        meta_split="train", download=True)

    val = datasets.cub(data_dir, ways=num_way, shots=num_shots, test_shots=int(100/num_shots),
                        meta_split="val", download=True)

    test = datasets.CUB(data_dir, ways=num_way, shots=num_shots, test_shots=int(100/num_shots), meta_split="test",
                        download=True)
    
    return train, val, test

def get_inat(dataset: str, data_dir: str, num_way: int, num_shots: int, num_shots_test: int):
    """loads up our data
    """
    raise NotImplementedError()