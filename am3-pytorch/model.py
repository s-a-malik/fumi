"""Model classes for AM3 in Pytorch.
"""

import torch.nn as nn
import torch.nn.functional as F


class AM3(nn.Module):
    def __init__(self, im_emb_dim, text_emb_dim, hid_dim):
        super(AM3, self).__init__()
        self.im_emb_dim = im_emb_dim
        self.text_emb_dim = text_emb_dim
        self.hid_dim = hid_dim

        # image encoder
        # if raw images
        # self.resnet = xxx
        self.image_encoder = nn.Linear(im_input_dim, hid_dim)

        # TODO fixed word embeddings or BERT
        self.text_encoder = nn.Identity()
        self.text_fc = nn.Linear(text_input_dim, hid_dim)

        self.text_weighting = nn.Linear(hid_dim, 1)

    def forward(self, inputs):
        # need to split inputs into image and text
        im, text = inputs
        im_embeddings = self.image_encoder(inputs.view(-1, *inputs.shape[2:]))
        
        text_embeddings = self.text_encoder(text)
        # add non-linearity?
        text_embeddings = self.text_fc(text_embeddings)

        lamda = F.sigmoid(self.text_weighting(text_embeddings))

        return im_embeddings.view(*inputs.shape[:2], -1), text_embeddings.view(*inputs.shape[:2], -1), lamda.view(*inputs.shape[:2], -1)

    def get_prototypes(self, train_loader, val_loader, task="train"):
        """Combine the text and image output to get prototypes per class 
        """
