import json
import os
import random
import subprocess
from enum import Enum
from typing import List, Set

import numpy as np
from tqdm import tqdm
import torch
from torchvision import transforms
from torchvision.transforms import Compose

import h5py
import nltk
import torchmeta.datasets.helpers as datasets
from gensim import corpora
from gensim.utils import tokenize
from nltk.corpus import stopwords
from torchmeta.transforms import Categorical, ClassSplitter
from torchmeta.utils.data import (BatchMetaDataLoader, ClassDataset,
                                  CombinationMetaDataset, Dataset)
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
import torch


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
    json_path = args.json_path
    num_way = args.num_ways
    num_shots = args.num_shots
    num_shots_test = args.num_shots_test
    text_encoder = args.text_encoder
    text_type = args.text_type
    remove_stop_words = args.remove_stop_words

    if dataset == "cub":
        train, val, test, dictionary = get_CUB(data_dir, num_way, num_shots,
                                               num_shots_test)
    elif dataset == "zanim":
        train, val, test, dictionary = get_zanim(data_dir, json_path, num_way,
                                                 num_shots, num_shots_test,
                                                 text_encoder, text_type,
                                                 remove_stop_words,
                                                 args.image_embedding_model,
                                                 args.colab)
    elif dataset == "supervised-zanim":
        train, val, test = get_supervised_zanim(data_dir, json_path,
                                                text_encoder, text_type,
                                                remove_stop_words,
                                                args.image_embedding_model,
                                                args.device,
                                                args.colab)
        if text_encoder != 'BERT':
            raise NotImplementedError()
        dictionary = {}
        dataloaders = tuple(
            DataLoader(d,
                       batch_size=args.batch_size,
                       num_workers=args.num_workers,
                       shuffle=True) for d in [train, val, test])
        return tuple(x for x in [*dataloaders, dictionary])
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
                                      shuffle=True,
                                      num_workers=args.num_workers)

    return train_loader, val_loader, test_loader, dictionary


def _convert_zanim_arguments(text_encoder: str, text_type: List[str]):
    token_mode = TokenisationMode.BERT if text_encoder == "BERT" else TokenisationMode.STANDARD
    modes = {
        "description": DescriptionMode.FULL_DESCRIPTION,
        "label": DescriptionMode.LABEL,
        "common_name": DescriptionMode.COMMON_NAME
    }
    try:
        description_mode = set([modes[l] for l in text_type])
    except KeyError:
        raise NameError(f"Invalid text type used")

    return (token_mode, description_mode)


def get_supervised_zanim(data_dir: str, json_path: str, text_encoder: str,
                         text_type: str, remove_stop_words: bool,
                         image_embedding_model: str, device: str, colab: bool):
    splits = []
    _, description_mode = _convert_zanim_arguments(text_encoder, text_type)
    for (train, val, test) in [(True, False, False), (False, True, False),
                               (False, False, True)]:
        splits.append(
            SupervisedZanim(root=data_dir,
                            json_path=json_path,
                            train=train,
                            val=val,
                            test=test,
                            description_mode=description_mode,
                            remove_stop_words=remove_stop_words,
                            image_embedding_model=image_embedding_model,
                            device=device,
                            colab=colab))
    return tuple(splits)


def get_zanim(data_dir: str, json_path: str, num_way: int, num_shots: int,
              num_shots_test: int, text_encoder: str, text_type: str,
              remove_stop_words: bool, image_embedding_model: str, colab: bool):

    token_mode, description_mode = _convert_zanim_arguments(
        text_encoder,
        text_type,
    )
    train = Zanim(root=data_dir,
                  json_path=json_path,
                  num_classes_per_task=num_way,
                  meta_train=True,
                  tokenisation_mode=token_mode,
                  description_mode=description_mode,
                  remove_stop_words=remove_stop_words,
                  image_embedding_model=image_embedding_model,
                  colab=colab)
    train_split = ClassSplitter(train,
                                shuffle=True,
                                num_test_per_class=num_shots_test,
                                num_train_per_class=num_shots)
    train_split.seed(0)

    val = Zanim(root=data_dir,
                json_path=json_path,
                num_classes_per_task=num_way,
                meta_val=True,
                tokenisation_mode=token_mode,
                description_mode=description_mode,
                remove_stop_words=remove_stop_words,
                image_embedding_model=image_embedding_model,
                colab=colab)
    val_split = ClassSplitter(val,
                              shuffle=True,
                              num_test_per_class=int(100 / num_way),
                              num_train_per_class=num_shots)
    val_split.seed(0)

    test = Zanim(root=data_dir,
                 json_path=json_path,
                 num_classes_per_task=num_way,
                 meta_test=True,
                 tokenisation_mode=token_mode,
                 description_mode=description_mode,
                 remove_stop_words=remove_stop_words,
                 image_embedding_model=image_embedding_model,
                 colab=colab)
    test_split = ClassSplitter(test,
                               shuffle=True,
                               num_test_per_class=int(100 / num_way),
                               num_train_per_class=num_shots)
    test_split.seed(0)

    dictionary = {} if text_encoder == "BERT" else train.dictionary

    return train_split, val_split, test_split, dictionary


def get_CUB(data_dir: str, num_way: int, num_shots: int, num_shots_test: int):
    """Need to fix to get text as well
	"""
    train = datasets.cub(data_dir,
                         ways=num_way,
                         shots=num_shots,
                         test_shots=num_shots_test,
                         meta_split="train",
                         download=True)

    val = datasets.cub(data_dir,
                       ways=num_way,
                       shots=num_shots,
                       test_shots=int(100 / num_shots),
                       meta_split="val",
                       download=True)

    test = datasets.CUB(data_dir,
                        ways=num_way,
                        shots=num_shots,
                        test_shots=int(100 / num_shots),
                        meta_split="test",
                        download=True)

    dictionary = {}

    return train, val, test, dictionary


class TokenisationMode(Enum):
    BERT = 1
    STANDARD = 2


class DescriptionMode(Enum):
    FULL_DESCRIPTION = 1
    LABEL = 2
    COMMON_NAME = 3


class SupervisedZanim(torch.utils.data.Dataset):
    def __init__(self,
                 root,
                 json_path="train.json",
                 train=True,
                 val=False,
                 test=False,
                 description_mode=[DescriptionMode.FULL_DESCRIPTION],
                 remove_stop_words=False,
                 image_embedding_model="resnet-152",
                 device=None,
                 pooling=lambda x: torch.mean(x, dim=1),
                 colab=False):
        super().__init__()
        if (train + val + test > 1) or (train + val + test == 0):
            raise ValueError(
                "Only a single value of train, val, test can be true")
        self._zcd = ZanimClassDataset(
            root,
            json_path,
            meta_train=train,
            meta_val=val,
            meta_test=test,
            tokenisation_mode=TokenisationMode.BERT,
            description_mode=description_mode,
            remove_stop_words=remove_stop_words,
            image_embedding_model=image_embedding_model,
            colab=colab)
        self.model = BertModel.from_pretrained('bert-base-uncased')

        print("Precomputing BERT embeddings")
        if device is not None:
            self.model.to(device)
        batch_size = 64
        self._bert_embeddings = torch.zeros(len(self._zcd.descriptions),
                                            self.model.config.hidden_size)
        for start in range(0, len(self._zcd.descriptions), batch_size):
            with torch.no_grad():
                end = min(len(self._zcd.descriptions), start + batch_size)
                des, mas = (self._zcd.descriptions[start:end].to(device),
                            self._zcd.mask[start:end].to(device)
                            ) if device is not None else (
                                self._zcd.descriptions[start:end],
                                self._zcd.mask[start:end])
                self._bert_embeddings[start:end] = pooling(
                    self.model(input_ids=des,
                               attention_mask=mas,
                               output_attentions=False).last_hidden_state)

        print("Completed embedding computation")
        self._bert_embeddings = self._bert_embeddings.cpu()

    def __len__(self):
        return len(self._zcd.category_id)

    def __getitem__(self, index):
        category_id = self._zcd.category_id[index]
        image_id = self._zcd.image_ids[index]
        bert_index = np.where(self._zcd.categories == category_id)[0][0]
        return self._zcd.image_embeddings[image_id], self._bert_embeddings[
            bert_index], category_id


class Zanim(CombinationMetaDataset):
    def __init__(self,
                 root,
                 json_path="train.json",
                 num_classes_per_task=None,
                 meta_train=False,
                 meta_val=False,
                 meta_test=False,
                 tokenisation_mode: TokenisationMode = TokenisationMode.BERT,
                 description_mode: Set[DescriptionMode] = [
                     DescriptionMode.FULL_DESCRIPTION
                 ],
                 remove_stop_words=True,
                 image_embedding_model='resnet-152',
                 target_transform=None,
                 categories=None,
                 colab=False):
        """
		:param root: the path to the root directory of the dataset
		:param json_path: the path to the json file containing the annotations
		"""
        if target_transform is None:
            target_transform = Categorical(num_classes_per_task)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        self.dataset = ZanimClassDataset(
            root,
            json_path,
            meta_train=meta_train,
            meta_val=meta_val,
            meta_test=meta_test,
            tokenisation_mode=tokenisation_mode,
            description_mode=description_mode,
            image_embedding_model=image_embedding_model,
            remove_stop_words=remove_stop_words,
            categories=categories,
            colab=colab)
        super().__init__(self.dataset,
                         num_classes_per_task,
                         target_transform=target_transform)

    @property
    def dictionary(self):
        return self.dataset.dictionary.token2id


class ZanimClassDataset(ClassDataset):
    def __init__(self,
                 root: str,
                 json_path: str,
                 meta_train=False,
                 meta_val=False,
                 meta_test=False,
                 tokenisation_mode=TokenisationMode.BERT,
                 description_mode: Set[DescriptionMode] = [
                     DescriptionMode.FULL_DESCRIPTION
                 ],
                 remove_stop_words=True,
                 image_embedding_model: str = "resnet-152",
                 categories=None,
                 colab=False):
        super().__init__(meta_train=meta_train,
                         meta_val=meta_val,
                         meta_test=meta_test)
        if not (root in json_path):
            json_path = os.path.join(root, json_path)

        self.root = root
        self.tokenisation_mode = tokenisation_mode
        with open(json_path) as annotations:
            annotations = json.load(annotations)
            self.annotations = annotations

        N = len(annotations['categories'])
        self.categories = np.arange(N)
        np.random.shuffle(self.categories)
        if categories is None:
            if meta_train:
                self.categories = self.categories[:int(0.6 * N)]
            elif meta_val:
                self.categories = self.categories[int(0.6 * N):int(0.8 * N)]
            elif meta_test:
                self.categories = self.categories[int(0.8 * N):]
            else:
                raise ValueError(
                    "One of meta_train, meta_val, meta_test must be true")
        else:
            self.categories = categories

        np.sort(self.categories)

        self.image_ids = [
            i['id'] for i in annotations['images']
            if annotations['annotations'][i['id']]['category_id'] in
            self.categories
        ]
        self.category_id = [
            annotations['annotations'][id]['category_id']
            for id in self.image_ids
        ]
        self.category_id_map = {}
        for id in range(len(self.image_ids)):
            cat_id = self.category_id[id]
            image_id = self.image_ids[id]
            if cat_id in self.category_id_map:
                self.category_id_map[cat_id].append(image_id)
            else:
                self.category_id_map[cat_id] = [image_id]
        for cat_id in self.category_id_map.keys():
            self.category_id_map[cat_id] = np.array(
                self.category_id_map[cat_id])

        self.descriptions = self._get_descriptions(self.annotations,
                                                   self.categories,
                                                   description_mode)
        print("Copying image embeddings to local disk")

        image_embedding_file = f"image-embedding-{image_embedding_model}.hdf5"
        if colab:
            local_image_embedding_path = os.path.join('/content',
                                                    image_embedding_file)
            if not os.path.exists(local_image_embedding_path):
                self._copy_image_embeddings(image_embedding_file)
        else:
            local_image_embedding_path = os.path.join(self.root, image_embedding_file)
        self.image_embeddings = h5py.File(local_image_embedding_path,
                                          'r')['images']
        self._num_classes = len(self.categories)

        if remove_stop_words:
            nltk.download('stopwords')
            stop_words = stopwords.words('english')
            self.descriptions = [
                " ".join([w for w in s.split() if not (w in stop_words)])
                for s in self.descriptions
            ]

        if tokenisation_mode == TokenisationMode.BERT:
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            tokens = tokenizer(self.descriptions,
                               return_token_type_ids=False,
                               return_tensors="pt",
                               padding=True,
                               truncation=True)
            self.descriptions = tokens['input_ids']
            self.mask = tokens['attention_mask']
        elif tokenisation_mode == TokenisationMode.STANDARD:
            # since using a generator can't take len(tokenize(d))
            lengths = [
                sum([1 for w in tokenize(d)]) for d in self.descriptions
            ]
            max_length = max(lengths)
            self.descriptions = [
                d.lower() + " " +
                " ".join(["<PAD>" for _ in range(max_length - lengths[i])])
                for (i, d) in enumerate(self.descriptions)
            ]
            # the dictionary should be formed across all folds, so recompute set of 'descriptions' across all categories
            full_set_of_descriptions = self._get_descriptions(
                self.annotations, np.arange(N), description_mode)
            self.dictionary = corpora.Dictionary(
                [tokenize(d.lower()) for d in full_set_of_descriptions])
            self.dictionary.add_documents([tokenize("<PAD>")])
            self.descriptions = [[
                self.dictionary.token2id[z] for z in tokenize(d)
            ] for d in self.descriptions]
        print("Completed tokenisation")

    def _get_descriptions(self, annotations, categories, description_mode):
        descriptions = ["" for i in categories]
        description_json_key_map = {
            DescriptionMode.FULL_DESCRIPTION: 'description',
            DescriptionMode.LABEL: 'name',
            DescriptionMode.COMMON_NAME: 'common_name'
        }
        description_mode = [
            description_json_key_map[d] for d in description_mode
        ]
        descriptions = [
            " ".join(
                [annotations['categories'][i][d] for d in description_mode])
            for i in categories
        ]
        return descriptions

    def _copy_image_embeddings(self, image_file):
        self._run_command(
            ["cp", os.path.join(self.root, image_file), "/content/"])

    def _run_command(self, command):
        pipes = subprocess.Popen(command, stderr=subprocess.PIPE)
        _, err = pipes.communicate()
        if pipes.returncode != 0:
            raise Exception(
                f"Error in running custom command {' '.join(command)}: {err.strip()}"
            )

    def __len__(self):
        return self._num_classes

    @property
    def num_classes(self):
        return self._num_classes

    def __getitem__(self, index):
        indices = self.category_id_map[self.categories[index %
                                                       self.num_classes]]
        mask = self.mask[
            index] if self.tokenisation_mode == TokenisationMode.BERT else None
        return ZanimDataset(index,
                            indices,
                            self.image_embeddings[indices],
                            self.descriptions[index],
                            index % self.num_classes,
                            attention_mask=mask,
                            target_transform=self.get_target_transform(index))


class ZanimDataset(Dataset):
    def __init__(self,
                 index,
                 image_ids,
                 data,
                 description,
                 category_id,
                 attention_mask=None,
                 target_transform=None):
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
            return (self.image_ids[index], torch.tensor(self.description),
                    self.data[index]), target
        else:
            return (self.image_ids[index], torch.tensor(self.description),
                    torch.tensor(self.attention_mask),
                    self.data[index]), target


if __name__ == "__main__":

    import sys
    import argparse
    parser = argparse.ArgumentParser(description="data module test")
    parser.add_argument("--text_type", type=str, default="label")
    parser.add_argument("--json_path", type=str, default="train.json")
    parser.add_argument("--text_encoder", type=str, default="BERT")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/content/drive/My Drive/MSc ML/NLP/NLP project/Dataset")
    parser.add_argument('--remove_stop_words',
                        action='store_true',
                        help="whether to remove stop words")

    args = parser.parse_args(sys.argv[1:])
    args.device = torch.device("cuda")
    text_type = args.text_type
    text_encoder = args.text_encoder
    num_way = 3
    num_shots = 5
    num_shots_test = 32
    batch_size = 5
    remove_stop_words = True if args.remove_stop_words else False

    data_dir = args.data_dir
    train, val, test = get_supervised_zanim(data_dir,
                                            args.json_path,
                                            text_encoder,
                                            text_type,
                                            remove_stop_words,
                                            image_embedding_model='resnet-152',
                                            device=args.device,
                                            colab=args.colab)
    for batch_idx, batch in enumerate(DataLoader(train, batch_size=10)):
        image, text, cat = batch
        print(image.shape)
        print(text.shape)
        print(cat)
        if batch_idx > 10:
            break

    train, val, test, dictionary = get_zanim(
        data_dir,
        args.json_path,
        num_way,
        num_shots,
        num_shots_test,
        text_encoder,
        text_type,
        remove_stop_words,
        image_embedding_model="resnet-152",
        colab=args.colab)
    print("dictionary", len(dictionary), dictionary)
    train_loader = BatchMetaDataLoader(train,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=0)

    # check first couple batches
    for batch_idx, batch in enumerate(train_loader):
        train_inputs, train_targets = batch['train']
        print("train targets")
        print(train_targets.shape, train_targets)
        test_inputs, test_targets = batch['test']
        if text_encoder == "BERT":
            idx, text, attn_mask, im = train_inputs
            print("idx")
            print(idx.shape, idx)
            print("text")
            print(text.shape, text)
            print("attn_mask")
            print(attn_mask.shape, attn_mask)
            print("im")
            print(im.shape, im)
        else:
            idx, text, im = train_inputs
            print("idx")
            print(idx.shape, idx)
            print("text")
            print(text.shape, text)
            print("im")
            print(im.shape, im)
        if batch_idx > 1:
            break
