from .demo_data_parser import DemoDataParser
import os
import wandb
from IPython.display import display
from functools import partial
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
import cv2
import torch
import numpy as np
from PIL import Image
import IPython
import time
import random
import argparse
from torchmeta.transforms import Categorical, ClassSplitter
from torchmeta.utils.data import (BatchMetaDataLoader, ClassDataset,
                                  CombinationMetaDataset, Dataset)
from ..utils import utils
from ..dataset.data import Zanim, TokenisationMode, DescriptionMode
from ..models import am3


class AM3Explorer():
    def __init__(self, data: DemoDataParser):
        args = [
            "--seed", "123", "--patience", "500", "--eval_freq", "30",
            "--epochs", "2000", "--optim", "adam", "--lr", "1e-4",
            "--momentum", "0.9", "--weight_decay", "0.005", "--dropout", "0.2",
            "--batch_size", "5", "--num_shots", "1", "--num_ways", "5",
            "--num_shots_test", "1", "--num_ep_test", "1000", "--im_encoder",
            "precomputed", "--image_embedding_model", "resnet-152",
            "--im_emb_dim", "2048", "--text_encoder", "glove", "--text_type",
            "description common_name", "--text_emb_dim", "768",
            "--text_hid_dim", "512", "--prototype_dim", "512"
        ]
        self.data = data
        model, checkpoint = "am3", "6ze2cjev"
        model_path = f"./checkpoints/{model}/{checkpoint}"
        # init model and optimiser
        os.makedirs(model_path, exist_ok=True)
        parser = utils.parser()
        args = parser.parse_args(args)
        args.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        args.max_test_batches = 1000
        args.num_ways = 5
        self.args = args
        os.environ['WANDB_SILENT'] = "true"
        self.run = wandb.init(
            entity="multimodal-image-cls",
            project=args.model,
            group=args.experiment,
            # mode='offline',
            save_code=True)
        self.checkpoint_file = wandb.restore(
            "best.pth.tar",
            run_path=f"multimodal-image-cls/{model}/{checkpoint}",
            root=model_path)
        self.model, self.optimizer = None, None
        ui = self.construct_interface()
        out = widgets.interactive_output(
            self.explore, {
                'c1': self.common_name_area_1,
                'c2': self.common_name_area_2,
                'c3': self.common_name_area_3,
                'c4': self.common_name_area_4,
                'c5': self.common_name_area_5
            })
        self.row, self.col = 5, 5
        self.base = 0
        display(ui, out)

    def gen_batch(self, species):
        test = Zanim(root="/content/",
                     json_path="train.json",
                     num_classes_per_task=5,
                     meta_test=True,
                     tokenisation_mode=TokenisationMode.STANDARD,
                     description_mode=[
                         DescriptionMode.FULL_DESCRIPTION,
                         DescriptionMode.COMMON_NAME
                     ],
                     remove_stop_words=False,
                     image_embedding_model="resnet-152",
                     categories=species)
        test_split = ClassSplitter(test,
                                   shuffle=True,
                                   num_test_per_class=int(100 / 5),
                                   num_train_per_class=5)
        test_loader = BatchMetaDataLoader(test_split,
                                          batch_size=2,
                                          shuffle=True,
                                          num_workers=1)
        return test, test_loader

    def colour_image(self, pred, y_true, image):
        size = 8
        col = np.array([0, 255, 0]) if pred == y_true else np.array(
            [255, 0, 0])
        hor = col * np.ones((size, image.shape[1], 3))
        image = np.vstack([hor, image, hor])
        ver = col * np.ones((image.shape[0], size, 3))
        return np.hstack([ver, image, ver])

    def run_am3(self, test, test_loader, button):
        if self.model is None:
            self.model = utils.init_model(self.args, test.dictionary)
            self.optimizer = utils.init_optim(self.args, self.model)

            print("Loading AM3 checkpoint")
            self.model, self.optimizer = utils.load_checkpoint(
                self.model, self.optimizer, self.args.device,
                self.checkpoint_file.name)

        print("Running AM3 on test species")
        test_loss, test_acc, test_f1, test_prec, test_rec, test_avg_lamda, test_preds, test_true, query_idx, support_idx, support_lamda = am3.test_loop(
            self.args, self.model, test_loader, self.args.max_test_batches)

        self.query_idx = np.array(query_idx).reshape(-1)
        self.test_true = np.array(test_true).reshape(-1)
        self.test_preds = np.array(test_preds).reshape(-1)

        cindxs = [
            self.data.cname_category_index_map[x] for x in [
                self.common_name_area_1.value, self.common_name_area_2.value,
                self.common_name_area_3.value, self.common_name_area_4.value,
                self.common_name_area_5.value
            ]
        ]
        fixed_test_preds = self.fix_mapping(cindxs, query_idx, test_preds)
        fixed_test_true = self.fix_mapping(cindxs, query_idx, test_true)
        accs = []
        for i in range(5):
            ids = fixed_test_true == i
            acc = np.mean(fixed_test_preds[ids] == fixed_test_true[i])
            accs.append(acc)
        self._show_am3_images()

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.bar(np.arange(0, 1, 5), accs, width=0.12)
        plt.show()

    def _show_am3_images(self):
        ims = []
        for i in range(
                self.base,
                min(self.base + (self.row * self.col),
                    self.query_idx.shape[0])):
            ims.append(
                self.colour_image(self.test_preds[i], self.test_true[i],
                                  self.data.images[self.query_idx[i]]))

        frames = [
            np.hstack(ims[start:min(start + self.col, len(ims))])
            for start in range(0, min(self.row * self.col, len(ims)), self.col)
        ]
        # frames = [np.hstack(images[support_idx[start:min(start+col, len(support_idx))]]) for start in range(0,row*col,col)]
        frame = np.vstack(frames)
        frame = np.flip(frame, 2)
        _, im_buf_arr = cv2.imencode(".jpg", frame)
        byte_im = im_buf_arr.tobytes()
        self.am3_image.value = byte_im

    def construct_interface(self):
        cnames_list = list(self.data.common_names)
        cs = list(self.data.common_names)

        self.common_name_area_1 = widgets.Combobox(options=cs,
                                                   font_size="10px",
                                                   value=random.choice(cs),
                                                   layout=widgets.Layout(
                                                       width='200%',
                                                       height='50%'),
                                                   ensure_option=True)
        self.common_name_area_2 = widgets.Combobox(options=cs,
                                                   font_size="10px",
                                                   value=random.choice(cs),
                                                   layout=widgets.Layout(
                                                       width='200%',
                                                       height='50%'),
                                                   ensure_option=True)
        self.common_name_area_3 = widgets.Combobox(options=cs,
                                                   font_size="10px",
                                                   value=random.choice(cs),
                                                   layout=widgets.Layout(
                                                       width='200%',
                                                       height='50%'),
                                                   ensure_option=True)
        self.common_name_area_4 = widgets.Combobox(options=cs,
                                                   font_size="10px",
                                                   value=random.choice(cs),
                                                   layout=widgets.Layout(
                                                       width='200%',
                                                       height='50%'),
                                                   ensure_option=True)
        self.common_name_area_5 = widgets.Combobox(options=cs,
                                                   font_size="10px",
                                                   value=random.choice(cs),
                                                   layout=widgets.Layout(
                                                       width='200%',
                                                       height='50%'),
                                                   ensure_option=True)
        common_name_box = widgets.VBox([
            self.common_name_area_1, self.common_name_area_2,
            self.common_name_area_3, self.common_name_area_4,
            self.common_name_area_5
        ])
        self.description = widgets.Textarea(layout=widgets.Layout(
            width='500px', font_size="10px", height='500px'))

        self.image = widgets.Image(width="100%", height="200%", format='raw')
        self.am3_image = widgets.Image(width="40%", height="40%", format='raw')

        ui = widgets.VBox([common_name_box, self.description])
        self.run_am3_button = widgets.Button(
            description="Run AM3 (with the above support set)",
            layout=widgets.Layout(width='100%'),
            disabled=False)
        self.back_button = widgets.Button(description="Back",
                                          layout=widgets.Layout(width='30%'),
                                          disabled=True)
        self.next_button = widgets.Button(description="Next",
                                          layout=widgets.Layout(width='30%'),
                                          disabled=False)

        self.next_button.on_click(self.on_next)
        self.back_button.on_click(self.on_back)
        buttons = widgets.HBox([self.back_button, self.next_button])
        ui = widgets.HBox(
            [ui, widgets.VBox([self.image, self.run_am3_button])])
        ui = widgets.VBox([ui, self.am3_image])
        ui = widgets.VBox([ui, buttons])
        return ui

    def on_next(self, b):
        self.base += self.row * self.col
        self.back_button.disabled = False
        self.update_gallery()

    def on_back(self, b):
        self.next_button.disabled = False
        self.base -= (self.row * self.col)
        self.update_gallery()

    def update_gallery(self):
        self._show_am3_images()

        if self.base == 0:
            self.back_button.disabled = True
        if self.base > self.test_preds.shape[0] - (2 * self.row * self.col):
            self.next_button.disabled = True

    def fix_mapping(self, cindxs, indxs, targets):
        ts = targets.copy()
        for ind, i in enumerate(indxs):
            ts[ind] = cindxs.index(
                self.data.annotations['annotations'][i]['category_id'])
        return ts

    def explore(self, c1, c2, c3, c4, c5):
        cindxs = [
            self.data.cname_category_index_map[x]
            for x in [c1, c2, c3, c4, c5]
        ]
        test, test_loader = self.gen_batch(cindxs)
        self.run_am3_button.on_click(partial(self.run_am3, test, test_loader))
        for z in test_loader:
            indxs = z['train'][0][0].numpy().reshape(-1)
            targets = z['train'][1][0].numpy().reshape(-1)
        sorted_indxs = np.sort(indxs)
        argsorted_indxs = np.argsort(indxs)
        ims = self.data.images[sorted_indxs]
        targets = self.fix_mapping(cindxs, indxs, targets)
        targets = targets[argsorted_indxs]
        frames = [np.hstack(ims[targets == i]) for i in range(5)]
        frame = np.vstack(frames)
        frame = np.flip(frame, 2)
        _, im_buf_arr = cv2.imencode(".jpg", frame)
        byte_im = im_buf_arr.tobytes()
        self.image.value = byte_im
        self.description.value = "\n\n".join(
            [self.data.cname_description_map[x] for x in [c1, c2, c3, c4, c5]])
