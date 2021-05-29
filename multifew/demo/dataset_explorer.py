from __future__ import print_function
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from IPython.display import display
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
import cv2
import numpy as np
from PIL import Image
import IPython
import time
import random
from demo_data_parser import DemoDataParser
import io


class DatasetExplorer():
    def __init__(self, data: DemoDataParser):
        # todo: add species name
        self.data = data
        self.cnames_list = list(self.data.common_names)
        default_cname = random.choice(self.cnames_list)
        self.common_name = widgets.Combobox(options=list(
            self.data.common_names),
                                            value=default_cname,
                                            font_size="20px",
                                            layout=widgets.Layout(width='50%'),
                                            ensure_option=True)
        self.common_name.add_class('data_input')
        self.random_species_button = widgets.Button(
            description="Random animal",
            icon='fa-dice',
            button_style='',
            layout=widgets.Layout(width='50%'))
        common_name_box = widgets.HBox(
            [self.common_name, self.random_species_button])
        self.description = widgets.Textarea(layout=widgets.Layout(
            width='100%', font_size="20px", height='100%'))

        data_input_style = '''<style>
        .mytext .fa, .far, .fas {
            font-style: italic;
            color: blue;
            font-size: 100px;
        }
        .data_input input { background-color:#bede68 !important; font-size: 5; }
        .data_input text { background-color:#bede68 !important; font-size: 5; }</style>'''
        self.image = widgets.Image(width="100%", height="100%", format='raw')

        ui = widgets.VBox([
            widgets.HTML(data_input_style), common_name_box, self.description
        ])
        self.back_button = widgets.Button(description="Back",
                                          layout=widgets.Layout(width='50%'),
                                          disabled=True)
        self.next_button = widgets.Button(description="Next",
                                          layout=widgets.Layout(width='50%'),
                                          disabled=False)

        self.random_species_button.on_click(self.on_random_click)
        self.next_button.on_click(self.on_next)
        self.back_button.on_click(self.on_back)
        buttons = widgets.HBox([self.back_button, self.next_button])
        ui = widgets.HBox([ui, widgets.VBox([self.image, buttons])])

        self.output = widgets.Output()
        self.row, self.col = 4, 6
        self.base = 0
        out = widgets.interactive_output(self.explore,
                                         {'common_name': self.common_name})
        display(ui, out, self.output)

    def on_random_click(self, b):
        self.base = 0
        self.back_button.disabled = True
        self.next_button.disabled = False
        with self.output:
            b.disabled = True
            self.common_name.value = random.choice(self.cnames_list)
            self.explore(self.common_name.value)
            b.disabled = False

    def on_next(self, b):
        self.base += self.row * self.col
        self.back_button.disabled = False
        self.update_gallery()

    def on_back(self, b):
        self.next_button.disabled = False
        self.base -= (self.row * self.col)
        self.update_gallery()

    def update_gallery(self):
        indxs = self.data.cname_image_index_map[self.common_name.value]
        frames = [
            np.hstack(
                self.data.images[indxs[self.base +
                                       start:min(self.base + start +
                                                 self.col, len(indxs))]])
            for start in range(0, self.row * self.col, self.col)
        ]
        frame = np.vstack(frames)
        if self.base == 0:
            self.back_button.disabled = True
        if self.base > len(indxs) - (2 * self.row * self.col):
            self.next_button.disabled = True

        frame = np.flip(frame, 2)
        _, im_buf_arr = cv2.imencode(".jpg", frame)
        byte_im = im_buf_arr.tobytes()
        self.image.value = byte_im

    def explore(self, common_name):
        self.base = 0
        indxs = self.data.cname_image_index_map[common_name]
        start = time.time()
        frames = [
            np.hstack(self.data.images[indxs[start:min(start +
                                                       self.col, len(indxs))]])
            for start in range(0, self.row * self.col, self.col)
        ]
        frame = np.vstack(frames)
        start = time.time()
        frame = np.flip(frame, 2)
        _, im_buf_arr = cv2.imencode(".jpg", frame)
        byte_im = im_buf_arr.tobytes()
        self.image.value = byte_im
        self.description.value = self.data.cname_description_map[common_name]
