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
from utils import utils
from dataset.data import Zanim, TokenisationMode, DescriptionMode
from models import am3, fumi


class AM3Explorer():
    def __init__(self, data: DemoDataParser):
        plt.style.use('seaborn')
        plt.rc('font', size=15)
        plt.rc('xtick', labelsize='medium')
        plt.rc('ytick', labelsize='medium')
        plt.rc('xtick.major', size=5, width=1.5)
        plt.rc('ytick.major', size=5, width=1.5)
        plt.rc('axes', linewidth=2, labelsize='large', titlesize='large')
        plt.rcParams["lines.markeredgewidth"] = 2
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
        fumi_args = [
            '--model', 'fumi', '--seed', '123', '--patience', '10000',
            '--eval_freq', '500', '--epochs', '50000', '--optim', 'adam',
            '--lr', '1e-4', '--weight_decay', '0.0005', '--batch_size', '2',
            '--num_shots', '5', '--num_ways', '5', '--num_shots_test', '8',
            '--num_ep_test', '250', '--im_encoder', 'precomputed',
            '--image_embedding_model', 'resnet-152', '--im_emb_dim', '2048',
            '--im_hid_dim', '64', '--text_encoder', 'glove', '--pooling_strat',
            'mean', '--remove_stop_words', '--text_type', 'description',
            '--text_emb_dim', '768', '--text_hid_dim', '256', '--step_size',
            '0.01', '--num_train_adapt_steps', '5', '--num_test_adapt_steps',
            '25', '--checkpoint', '249ovl3w', '--evaluate'
        ]
        self.data = data
        models, checkpoints = ["am3", "fumi"], ["6ze2cjev", "249ovl3w"]
        model_paths = [
            f"./checkpoints/{m}/{c}" for (m, c) in zip(models, checkpoints)
        ]
        # init model and optimiser
        os.makedirs(model_paths[0], exist_ok=True)
        os.makedirs(model_paths[1], exist_ok=True)
        parser = utils.parser()
        args = parser.parse_args(args)
        args.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        args.max_test_batches = 1000
        args.num_ways = 5
        self.args = args

        fumi_args = utils.parser().parse_args(fumi_args)
        fumi_args.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        fumi_args.max_test_batches = 1000
        fumi_args.num_ways = 5
        self.fumi_args = fumi_args

        os.environ['WANDB_SILENT'] = "true"
        # self.am3_checkpoint_file = wandb.restore(
        #     "best.pth.tar",
        #     run_path=f"multimodal-image-cls/{models[0]}/{checkpoints[0]}",
        #     root=model_paths[0])
        # self.fumi_checkpoint_file = wandb.restore(
        #     "best.pth.tar",
        # run_path=f"multimodal-image-cls/{models[1]}/{checkpoints[1]}",
        #     root=model_paths[1])

        # generate a fake batch to get the description dictionary (i.e. the tokens)
        test, _ = self.gen_batch([1, 2, 3, 4, 5])
        # load AM3 and FUMI checkpoint
        print("Loading AM3 checkpoint")
        self.am3_model = utils.init_model(self.args,
                                          test.dictionary,
                                          watch=False)
        self.am3_optimizer = utils.init_optim(self.args, self.am3_model)

        self.am3_model, self.am3_optimizer = utils.load_checkpoint(
            self.am3_model, self.am3_optimizer, self.args.device,
            "am3.pth.tar")
        print("Finished loading AM3 checkpoint")
        print("Loading FUMI checkpoint")
        test, _ = self.gen_batch([1, 2, 3, 4, 5],
                                 stop_words=True,
                                 common_name=False)
        self.fumi_model = utils.init_model(fumi_args,
                                           test.dictionary,
                                           watch=False)
        self.fumi_optimizer = utils.init_optim(fumi_args, self.fumi_model)

        self.fumi_model, self.fumi_optimizer = utils.load_checkpoint(
            self.fumi_model, self.fumi_optimizer, self.fumi_args.device,
            "fumi.pth.tar")
        print("Finished loading FUMI checkpoint")
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

        self.run_am3_button.on_click(self.run_am3)
        display(ui, out)

    def gen_batch(self, species, stop_words=False, common_name=True):
        dmode = [DescriptionMode.FULL_DESCRIPTION]
        if common_name:
            dmode = dmode + [DescriptionMode.COMMON_NAME]
        test = Zanim(root="/content/",
                     json_path="train.json",
                     num_classes_per_task=5,
                     meta_test=True,
                     tokenisation_mode=TokenisationMode.STANDARD,
                     description_mode=dmode,
                     remove_stop_words=stop_words,
                     image_embedding_model="resnet-152",
                     categories=species,
                     device=args.device)
        test_split = ClassSplitter(test,
                                   shuffle=False,
                                   num_test_per_class=int(100 / 5),
                                   num_train_per_class=5)
        test_split.seed(0)
        test_loader = BatchMetaDataLoader(test_split,
                                          batch_size=2,
                                          shuffle=False,
                                          num_workers=1)
        return test, test_loader

    def colour_image(self, am3_pred, fumi_pred, y_true, y_true_targets, image):
        size = 8
        col_a = np.array([0, 255, 0]) if am3_pred == y_true else np.array(
            [255, 0, 0])
        col_f = np.array([
            0, 255, 0
        ]) if fumi_pred == y_true_targets else np.array([255, 0, 0])
        boundary_size = 4

        hor = np.ones((size, image.shape[1] // 2, 3))
        hor = np.hstack([col_a * hor, col_f * hor])
        image = np.vstack([hor, image, hor])
        ver_a = col_a * np.ones((image.shape[0], size, 3))
        ver_f = col_f * np.ones((image.shape[0], size, 3))
        image = np.hstack([ver_a, image, ver_f])
        hor_white = np.array([255, 255, 255]) * np.ones(
            (boundary_size, image.shape[1], 3))
        image = np.vstack([hor_white, image, hor_white])
        ver_white = np.array([255, 255, 255]) * np.ones(
            (image.shape[0], boundary_size, 3))
        image = np.hstack([ver_white, image, ver_white])
        return image

    def run_am3(self, button):

        common_names_selected = [
            self.common_name_area_1.value, self.common_name_area_2.value,
            self.common_name_area_3.value, self.common_name_area_4.value,
            self.common_name_area_5.value
        ]
        cindxs = [
            self.data.cname_category_index_map[x]
            for x in common_names_selected
        ]
        # test, test_loader = self.gen_batch(cindxs)
        print("Running AM3 on test species")
        test_loss, test_acc, test_f1, test_prec, test_rec, test_avg_lamda, test_preds, test_true, query_idx, support_idx, support_lamda = am3.test_loop(
            self.args, self.am3_model, self.am3_test_loader,
            self.args.max_test_batches)
        _, _, fumi_preds, fumi_targets = fumi.test_loop(
            self.fumi_args, self.fumi_model, self.fumi_test_loader,
            self.args.max_test_batches)
        self.query_idx = np.array(query_idx).reshape(-1)
        self.test_targets = np.array(test_true).reshape(-1)
        self.am3_test_preds = np.array(test_preds).reshape(-1)
        self.fumi_test_preds = fumi_preds[0].numpy().reshape(-1).astype(
            np.uint8)
        self.fumi_test_targets = fumi_targets[0].numpy().reshape(-1).astype(
            np.uint8)
        am3_accs = []
        fumi_accs = []
        for i in range(5):
            ids = (self.test_targets == i)
            am3_acc = np.mean(
                self.am3_test_preds[ids] == self.test_targets[ids])
            fumi_acc = np.mean(
                self.fumi_test_preds[self.fumi_test_targets == i] ==
                self.fumi_test_targets[self.fumi_test_targets == i])
            am3_accs.append(am3_acc)
            fumi_accs.append(fumi_acc)
        self._show_am3_images()

        mapping = []
        fumi_mapping = []
        for i in range(5):
            true_class = self.data.annotations['annotations'][self.query_idx[
                self.test_targets == i][0]]['category_id']
            true_class_fumi = self.data.annotations['annotations'][
                self.query_idx[self.fumi_test_targets == i][0]]['category_id']
            try:
                mapping.append(cindxs.index(true_class))
                fumi_mapping.append(cindxs.index(true_class_fumi))
            except:
                pass

        am3_accs = np.array(am3_accs)
        fumi_accs = np.array(fumi_accs)
        am3_accs_fixed = am3_accs.copy()
        fumi_accs_fixed = fumi_accs.copy()
        for ind, (j, k) in enumerate(zip(mapping, fumi_mapping)):
            am3_accs_fixed[j] = am3_accs[ind]
            fumi_accs_fixed[k] = fumi_accs[ind]

        fig, ax = plt.subplots(figsize=(15, 8))
        h1 = ax.bar(np.arange(5), am3_accs_fixed, width=0.35)
        h2 = ax.bar(np.arange(5) + 0.35, fumi_accs_fixed, width=0.35)
        ax.set_xticks(np.arange(5))
        ax.set_xticklabels(common_names_selected)
        ax.set_ylabel("Accuracy per species")
        plt.legend([h1, h2], ["AM3", "FuMI"])
        plt.show()

    def _show_am3_images(self):
        ims = []
        for i in range(
                self.base,
                min(self.base + (self.row * self.col),
                    self.query_idx.shape[0])):
            ims.append(
                self.colour_image(self.am3_test_preds[i],
                                  self.fumi_test_preds[i],
                                  self.test_targets[i],
                                  self.fumi_test_targets[i],
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
            description="Run AM3 and FuMI (with the above support set)",
            layout=widgets.Layout(width='100%'),
            disabled=False)
        self.back_button = widgets.Button(description="Back",
                                          layout=widgets.Layout(width='20%'),
                                          disabled=True)
        self.next_button = widgets.Button(description="Next",
                                          layout=widgets.Layout(width='20%'),
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
        if self.base > self.test_targets.shape[0] - (2 * self.row * self.col):
            self.next_button.disabled = True

    def fix_mapping(self, cindxs, indxs, targets):
        ts = targets.copy()
        for ind, i in enumerate(indxs):
            try:
                ts[ind] = cindxs.index(
                    self.data.annotations['annotations'][i]['category_id'])
            except:
                pass
        return ts

    def explore(self, c1, c2, c3, c4, c5):
        cindxs = [
            self.data.cname_category_index_map[x]
            for x in [c1, c2, c3, c4, c5]
        ]
        self.am3_test, self.am3_test_loader = self.gen_batch(cindxs)
        self.fumi_test, self.fumi_test_loader = self.gen_batch(
            cindxs, common_name=False, stop_words=True)

        # self.run_am3_button.on_click(partial(self.run_am3, test, test_loader))
        for z in self.am3_test_loader:
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
