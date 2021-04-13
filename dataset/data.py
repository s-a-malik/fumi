import torch.utils.data as data
from torch.utils.data import Subset
import json
import os
import numpy as np
from PIL import Image
import subprocess
from typing import List
from torchvision.transforms import Compose
from torchvision import transforms
from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
from nltk.corpus import stopwords
import h5py
from enum import Enum
import nltk
from nltk.corpus import stopwords
from transformers import BertTokenizer
import random
from gensim.utils import tokenize
from gensim import corpora

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

    def __init__(self, root, json_path = "train.json", num_classes_per_task=None, meta_train=False, meta_val=False, meta_test=False, tokenisation_mode : TokenisationMode = TokenisationMode.BERT, full_description = True, remove_stop_words = True):
        """
        :param root: the path to the root directory of the dataset
        :param json_path: the path to the json file containing the annotations
        """
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        self.dataset = ZanimClassDataset(root, json_path, meta_train=meta_train, meta_val=meta_val, meta_test=meta_test, tokenisation_mode=tokenisation_mode, full_description=full_description, remove_stop_words=remove_stop_words)
        super().__init__(dataset, num_classes_per_task)

    @property
    def dictionary(self):
      return self.dataset.dictionary.token2id

class ZanimClassDataset(ClassDataset):

  def __init__(self, root: str, json_path: str, meta_train=False, meta_val=False, meta_test=False, tokenisation_mode = TokenisationMode.BERT, full_description=True, remove_stop_words=True):
    super().__init__(meta_train=meta_train, meta_val=meta_val, meta_test=meta_test)
    if not(root in json_path):
        json_path = os.path.join(root, json_path)
    self.root = root

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

    self.image_ids = [i['id'] for i in annotations['images'] if annotations['annotations'][i['id']]['category_id'] in self.categories]
    print("Building class id mapping")
    self.category_id = [annotations['annotations'][id]['category_id'] for id in self.image_ids]
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
      self.descriptions = [annotations['categories'][i]['description'] for i in self.category_id]
    else:
      self.descriptions = [annotations['categories'][i]['name'] for i in self.category_id]

    print("Copying image embeddings to local disk")
    if not os.path.exists('/content/image-embedding.hdf5'):
      self._copy_image_embeddings()
    self.image_embeddings = h5py.File('image-embedding.hdf5', 'r')['images']
    self._num_classes = len(self.categories)

    if remove_stop_words:
      nltk.download('stopwords')
      stop_words = stopwords.words('english')
      self.descriptions = [" ".join([w for w in s.split() if not(w in stop_words)]) for s in self.descriptions]

    if tokenisation_mode == TokenisationMode.BERT:
      tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
      self.descriptions = tokenizer(self.descriptions, return_token_type_ids=False, return_tensors="pt", padding=True, truncation=True)['input_ids']
    elif tokenisation_mode == TokenisationMode.STANDARD:
      # since using a generator can't take len(tokenize(d))
      lengths = [sum([1 for w in tokenize(d)]) for d in self.descriptions]
      max_length = max(lengths)
      self.descriptions = [d.lower() + " " + " ".join(["<PAD>" for _ in range(max_length-lengths[i])]) for (i,d) in enumerate(self.descriptions)]
      self.dictionary = corpora.Dictionary([tokenize(d) for d in self.descriptions])
      self.descriptions = [[self.dictionary.token2id[z] for z in tokenize(d)] for d in self.descriptions]


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
    return ZanimDataset(index, indices, self.image_embeddings[indices], self.descriptions[index], index%self.num_classes)

class ZanimDataset(Dataset):

    def __init__(self, index, image_ids, data, description, category_id):
        super().__init__(index)
        self.data = data
        self.category_id = category_id
        self.description = description
        self.image_ids = image_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (self.image_ids[index], torch.tensor(self.description), self.data[index]), self.category_id)
