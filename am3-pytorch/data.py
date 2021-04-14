
import json
import os
import numpy as np
import subprocess
from typing import List
import random
import h5py
from enum import Enum

import torch
from torchvision.transforms import Compose
from torchvision import transforms
from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
import torchmeta.datasets.helpers as datasets
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.transforms import ClassSplitter, Categorical

from transformers import BertTokenizer

import nltk
from nltk.corpus import stopwords
from gensim.utils import tokenize
from gensim import corpora


def get_dataset(args):
    """Return the appropriate dataset, with preprocessing transforms
    Returns:
    - train_loader (BatchMetaDataLoader): Train dataloader
    - val_loader (BatchMetaDataLoader): Validation dataloader
    - test_loader (BatchMetaDataLoader): Test dataloader
    - dictionary: token2id dict for word tokenisation if not BERT (else None)
    """
    dataset = args.dataset
    data_dir = args.data_dir
    num_way = args.num_ways
    num_shots = args.num_shots
    num_shots_test = args.num_shots_test
    text_encoder = args.text_encoder
    text_type = args.text_type
    remove_stop_words = args.remove_stop_words

    if dataset == "cub":
        train, val, test, dictionary = get_CUB(data_dir, num_way, num_shots, num_shots_test)
    elif dataset == "zanim":
        train, val, test, dictionary = get_zanim(data_dir, num_way, num_shots, num_shots_test, text_encoder, text_type, remove_stop_words)
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

    return train_loader, val_loader, test_loader, dictionary


def get_zanim(data_dir: str, num_way: int, num_shots: int, num_shots_test: int, text_encoder: str, text_type: str, remove_stop_words: bool):
    """loads up our data
    """
    if text_encoder == "BERT":
        token_mode = TokenisationMode.BERT
    else:
        token_mode = TokenisationMode.STANDARD

    if text_type == "description":
        full_description = True
    elif text_type == "label":
        full_description = False
    else:
        NameError(f"text type {text_type} not allowed")

    train = Zanim(root=data_dir, num_classes_per_task=num_way, meta_train=True, tokenisation_mode=token_mode, full_description=full_description, remove_stop_words=remove_stop_words)
    train_split = ClassSplitter(train, shuffle=True, num_test_per_class=num_shots_test, num_train_per_class=num_shots)
    train_split.seed(0)

    val = Zanim(root=data_dir, num_classes_per_task=num_way, meta_val=True, tokenisation_mode=token_mode, full_description=full_description, remove_stop_words=remove_stop_words)
    val_split = ClassSplitter(val, shuffle=True, num_test_per_class=int(100/num_shots), num_train_per_class=num_shots)
    val_split.seed(0) 

    test = Zanim(root=data_dir, num_classes_per_task=num_way, meta_test=True, tokenisation_mode=token_mode, full_description=full_description, remove_stop_words=remove_stop_words)
    test_split = ClassSplitter(test, shuffle=True, num_test_per_class=int(100/num_shots), num_train_per_class=num_shots)
    test_split.seed(0)

    if text_encoder != "BERT":
        # all the same dictionary anyway
        dictionary = train.dictionary
    else:
        dictionary = {}
    
    return train_split, val_split, test_split, dictionary


def get_CUB(data_dir: str, num_way: int, num_shots: int, num_shots_test: int):
    """Need to fix to get text as well
    """
    train = datasets.cub(data_dir, ways=num_way, shots=num_shots, test_shots=num_shots_test,
                        meta_split="train", download=True)

    val = datasets.cub(data_dir, ways=num_way, shots=num_shots, test_shots=int(100/num_shots),
                        meta_split="val", download=True)

    test = datasets.CUB(data_dir, ways=num_way, shots=num_shots, test_shots=int(100/num_shots), meta_split="test",
                        download=True)
    
    dictionary = {}

    return train, val, test, dictionary


class DefaultTransform(Compose):

    def __init__(self, image_size: int = 256, channel_means: List[float] = [0.485, 0.456, 0.406], channel_stds: List[float] = [0.229, 0.224, 0.225]):
        super().__init__([
            transforms.Resize(image_size),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(channel_means, channel_stds)
        ])


class TokenisationMode(Enum):
    BERT = 1
    STANDARD = 2


class Zanim(CombinationMetaDataset):

    def __init__(self, root, json_path="train.json", num_classes_per_task=None, meta_train=False, meta_val=False, meta_test=False, tokenisation_mode: TokenisationMode = TokenisationMode.BERT, full_description=True, remove_stop_words=True, target_transform=None):
        """
        :param root: the path to the root directory of the dataset
        :param json_path: the path to the json file containing the annotations
        """
        if target_transform is None:
            target_transform = Categorical(num_classes_per_task)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        dataset = ZanimClassDataset(root, json_path, meta_train=meta_train, meta_val=meta_val, meta_test=meta_test,
                                    tokenisation_mode=tokenisation_mode, full_description=full_description, remove_stop_words=remove_stop_words)
        super().__init__(dataset, num_classes_per_task)
        super().__init__(self.dataset, num_classes_per_task, target_transform=target_transform)

    @property
    def dictionary(self):
        return self.dataset.dictionary.token2id

class ZanimClassDataset(ClassDataset):

    def __init__(self, root: str, json_path: str, meta_train=False, meta_val=False, meta_test=False, tokenisation_mode=TokenisationMode.BERT, full_description=True, remove_stop_words=True):
        super().__init__(meta_train=meta_train, meta_val=meta_val, meta_test=meta_test)
        if not(root in json_path):
            json_path = os.path.join(root, json_path)
        self.root = root
        self.tokenisation_mode = tokenisation_mode
        print('Loading json annotations')
        with open(json_path) as annotations:
            annotations = json.load(annotations)
            self.annotations = annotations
        N = len(annotations['categories'])
        self.categories = np.arange(N)
        np.random.shuffle(self.categories)
        if meta_train:
            self.categories = self.categories[:int(0.6*N)]
        elif meta_val:
            self.categories = self.categories[int(0.6*N):int(0.8*N)]
        elif meta_test:
            self.categories = self.categories[int(0.8*N):]
        else:
            raise ValueError("One of meta_train, meta_val, meta_test must be true")

        self.image_ids = [i['id'] for i in annotations['images']
            if annotations['annotations'][i['id']]['category_id'] in self.categories]
        print("Building class id mapping")
        self.category_id = [annotations['annotations']
            [id]['category_id'] for id in self.image_ids]
        self.category_id_map = {}
        for id in range(len(self.image_ids)):
            cat_id = self.category_id[id]
            image_id = self.image_ids[id]
            if cat_id in self.category_id_map:
                self.category_id_map[cat_id].append(image_id)
            else:
                self.category_id_map[cat_id] = [image_id]
        for cat_id in self.category_id_map.keys():
            self.category_id_map[cat_id] = np.array(self.category_id_map[cat_id])

        if full_description:
            self.descriptions = [annotations['categories']
                [i]['description'] for i in self.category_id]
        else:
            self.descriptions = [annotations['categories'][i]['name']
                for i in self.category_id]

        print("Copying image embeddings to local disk")
        if not os.path.exists('/content/image-embedding.hdf5'):
            self._copy_image_embeddings()
        self.image_embeddings = h5py.File('image-embedding.hdf5', 'r')['images']
        self._num_classes = len(self.categories)

        if remove_stop_words:
            nltk.download('stopwords')
            stop_words = stopwords.words('english')
            self.descriptions = [" ".join(
                [w for w in s.split() if not(w in stop_words)]) for s in self.descriptions]

        if tokenisation_mode == TokenisationMode.BERT:
            tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
            tokens = tokenizer(self.descriptions, return_token_type_ids=False, return_tensors="pt", padding=True, truncation=True)
            self.descriptions = tokens['input_ids']
            self.mask = tokens['attention_mask']
        elif tokenisation_mode == TokenisationMode.STANDARD:
            # since using a generator can't take len(tokenize(d))
            lengths = [sum([1 for w in tokenize(d)]) for d in self.descriptions]
            max_length = max(lengths)
            self.descriptions = [d.lower() + " " + " ".join(["<PAD>" for _ in range(
                max_length-lengths[i])]) for (i, d) in enumerate(self.descriptions)]
            full_set_of_descriptions = [annotations['categories'][i]['description' if full_description else 'name'] for i in range(N)]
            self.dictionary = corpora.Dictionary([tokenize(d.lower()) for d in full_set_of_descriptions])
            self.dictionary.add_documents([tokenize("<PAD>")])
            self.descriptions = [[self.dictionary.token2id[z]
                for z in tokenize(d)] for d in self.descriptions]

    def _copy_image_embeddings(self):
        self._run_command(["cp", os.path.join(self.root, "image-embedding.hdf5"), "/content/"])

    def _run_command(self, command):
        pipes = subprocess.Popen(command, stderr=subprocess.PIPE)
        _, err = pipes.communicate()
        if pipes.returncode != 0:
            raise Exception(f"Error in running custom command {' '.join(command)}: {err.strip()}")

    def retrieve_missing(self):
        for i in self.image_files:
            if not(os.path.exists(os.path.join(self.root, i))):
                self._run_command(["cp", f'/content/drive/My Drive/NLP project/Dataset/{i}', f'/content/zanim/{i}'])

    def __len__(self):
        return self._num_classes

    @property
    def num_classes(self):
        return self._num_classes

    def __getitem__(self, index):
        indices = self.category_id_map[self.categories[index%self.num_classes]]
        mask = self.mask[index] if self.tokenisation_mode == TokenisationMode.BERT else None
        return ZanimDataset(index, indices, self.image_embeddings[indices], self.descriptions[index], index%self.num_classes, attention_mask=mask, target_transform=self.get_target_transform(index))

class ZanimDataset(Dataset):

    def __init__(self, index, image_ids, data, description, category_id, attention_mask=None, target_transform=None):
        super().__init__(index, target_transform=target_transform)
        self.data = data
        self.category_id = category_id
        self.description = description
        self.image_ids = image_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        target = self.category_id
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.attention_mask is None:
            return (self.image_ids[index], torch.tensor(self.description), self.data[index]), target
        else:
            return (self.image_ids[index], torch.tensor(self.description), torch.tensor(self.attention_mask), self.data[index]), target

