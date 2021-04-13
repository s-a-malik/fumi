"""Model classes for AM3 in Pytorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel

import utils


class AM3(nn.Module):
    def __init__(self, im_encoder, im_emb_dim, text_encoder, text_emb_dim, text_hid_dim=300, prototype_dim=512, dropout=0.7, fine_tune=False):
        super(AM3, self).__init__()
        self.im_emb_dim = im_emb_dim
        self.text_encoder_type = text_encoder
        self.text_emb_dim = text_emb_dim
        self.text_hid_dim = text_hid_dim        # AM3 uses 300
        self.prototype_dim = prototype_dim      # AM3 uses 512 (resnet)
        self.dropout = dropout                  # AM3 uses 0.7 or 0.9 depending on dataset
        self.fine_tune = fine_tune

        if im_encoder == "precomputed":
            # if using precomputed embeddings
            self.image_encoder = nn.Identity()
        elif im_encoder == "resnet":
            # TODO image encoder if raw images
            self.image_encoder = nn.Linear(im_emb_dim, prototype_dim)

        # TODO be able to use any hf bert model (requires correct tokenisation)
        if text_encoder == "BERT":
            self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
            self.text_emb_dim = self.text_encoder.config.hidden_size
            if not fine_tune:
                for param in self.text_encoder.parameters():
                    param.requires_grad = False
        # TODO other embeddings
        elif text_encoder == "GloVe":
            self.text_encoder = nn.Linear(text_emb_dim, text_emb_dim)

        elif text_encoder == "RNN":
            self.text_encoder = nn.Linear(text_emb_dim, text_emb_dim)

        # text to prototype neural net
        self.g = nn.Sequential(
            nn.Linear(self.text_emb_dim, self.text_hid_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.text_hid_dim, self.prototype_dim)
        )

        # text prototype to lamda neural net
        self.h = nn.Sequential(
            nn.Linear(self.prototype_dim, self.text_hid_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.text_hid_dim, 1)
        )

    def forward(self, inputs, im_only=False):
        """
        Params:
        -------
        - inputs (tuple): 
            - idx:  image ids
            - text: padded tokenised sequence. (tuple for BERT of input_ids and attn_mask - TODO). 
            - im:   precomputed image embeddings
        - im_only (bool): flag to only use image input (for query set)

        Returns:
        -------
        - im_embeddings (torch.FloatTensor): image in prototype space (batch, NxK, hid_dim)
        - (if not im_only) text_embeddings (torch.FloatTensor): text in prototype space (batch, NxK, hid_dim)
        """
        idx, text, im = inputs
        im_embeddings = self.image_encoder(im)      # (b x N*K x 512)
        if not im_only:
            # text input is same shape as images
            if self.text_encoder_type == "BERT":
                B, NK, seq_len = text.shape
                # bert_output = self.text_encoder(**text)   # if text is with attn mask
                bert_output = self.text_encoder(text.view(-1, seq_len))
                print(bert_output)
                # get [CLS] token
                text_encoding = bert_output[1].view(B, NK, -1)        # (b x N*K x 768)
                print(text_encoding)
            else:
                text_encoding = self.text_encoder(text)
            text_embeddings = self.g(text_encoding)   # (b x N*K x 512)
            lamda = F.sigmoid(self.h(text_embeddings))  # (b x N*K x 1)
            return im_embeddings, text_embeddings, lamda
        else:
            return im_embeddings

    def evaluate(self, batch, optimizer, num_ways, device, task="train"):
        """Run one episode through model
        Params:
        - batch (dict): meta-batch of tasks
        - optimizer (nn.optim): 
        - num_ways (int): number 'N' of classes to choose from.
        - device (torch.device): cuda or cpu
        - task (str): train, val, test
        Returns:
        - loss, acc
        - if test: also return predictions (class for each query)
        """
        if task == "train":
            self.train()
        else:
            self.eval()

        # support set
        train_inputs, train_targets = batch['train']            # (b x N*K x 512) for images
        train_inputs = train_inputs.to(device=device)
        train_targets = train_targets.to(device=device)
        train_im_embeddings, train_text_embeddings, train_lamda = self(train_inputs)

        # query set
        test_inputs, test_targets = batch['test']
        # need to also get image idx from this
        test_inputs = test_inputs.to(device=device)             # this might not work with input tuples.
        test_targets = test_targets.to(device=device)
        test_im_embeddings = self(test_inputs, im_only=True)    # only get image prototype

        prototypes = self.get_prototypes(
            train_im_embeddings,
            train_text_embeddings,
            train_lamda,
            train_targets,
            num_ways)
        
        # this is cross entropy on euclidean distance between prototypes and embeddings
        loss = utils.prototypical_loss(prototypes, test_im_embeddings, test_targets)

        if task == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            preds, acc = utils.get_preds(prototypes, test_im_embeddings, test_targets)
        
        if task == "test":
            # TODO return the query/support images and text per task and lamdas to compare 
            # returning just the query set targets is not that helpful.
            test_idx = test_inputs[0]
            return loss, acc, preds, test_targets.detach().cpu().numpy(), test_idx.detach().cpu().numpy()
        else:
            return loss, acc
        
    # from pytorch meta (need to change to AM3)
    def get_prototypes(self, im_embeddings, text_embeddings, lamdas, targets, num_classes):
        """Compute the prototypes (the mean vector of the embedded training/support 
        points belonging to its class) for each classes in the task.
        Parameters
        ----------
        im_embeddings : `torch.FloatTensor` instance
            A tensor containing the image embeddings of the support points. This tensor 
            has shape `(batch_size, num_examples, embedding_size)`.
        text_embeddings: `torch.FloatTensor` instance
            A tensor containing the text embeddings of the support points. This tensor 
            has shape `(batch_size, num_examples, embedding_size)`.
        lamda: `torch.FloatTensor` instance
            A tensor containing the weighting of text for the prototype
            has shape `(batch_size, num_examples, 1)`.
        targets : `torch.LongTensor` instance
            A tensor containing the targets of the support points. This tensor has 
            shape `(batch_size, num_examples)`.
        num_classes : int
            Number of classes in the task.
        Returns
        -------
        prototypes : `torch.FloatTensor` instance
            A tensor containing the prototypes for each class. This tensor has shape
            `(batch_size, num_classes, embedding_size)`.
        """
        batch_size, embedding_size = im_embeddings.size(0), im_embeddings.size(-1)

        # num_samples common across all computations
        num_samples = get_num_samples(targets, num_classes, dtype=im_embeddings.dtype)
        num_samples.unsqueeze_(-1)  # (b x N x 1)
        num_samples = torch.max(num_samples, torch.ones_like(num_samples))      # prevents zero division error
        indices = targets.unsqueeze(-1).expand_as(im_embeddings)                # (b x N*K x 512)

        im_prototypes = im_embeddings.new_zeros((batch_size, num_classes, embedding_size))
        im_prototypes.scatter_add_(1, indices, im_embeddings).div_(num_samples)   # compute mean embedding of each class

        # should all be equal anyway. TODO check they are.
        text_prototypes = text_embeddings.new_zeros((batch_size, num_classes, embedding_size))
        text_prototypes.scatter_add_(1, indices, text_embeddings).div_(num_samples)

        # should all be equal anyway. TODO check they are.
        lamdas_per_class = lamda.new_zeros((batch_size, num_classes, 1))
        lamdas_per_class.scatter_add_(1, targets.unsqueeze(-1), lamdas).div_(num_samples)

        # convex combination
        prototypes = lamdas_per_class * im_prototypes + (1-lamdas_per_class) * text_prototypes
        return prototypes


def get_num_samples(targets, num_classes, dtype=None):
    """Returns a vector with the number of samples in each class.
    - num_samples (torch.LongTensor): (b x N)
    """
    batch_size = targets.size(0)
    with torch.no_grad():
        ones = torch.ones_like(targets, dtype=dtype)
        num_samples = ones.new_zeros((batch_size, num_classes))
        num_samples.scatter_add_(1, targets, ones)
    return num_samples


if __name__ == "__main__":

    model = AM3(im_encoder="precomputed", im_emb_dim=512, text_encoder="BERT", text_emb_dim=768, text_hid_dim=300, prototype_dim=512, dropout=0.7, fine_tune=False)
    print(model)
    N = 5
    K = 2
    B = 5
    idx = torch.ones(B, N*K)
    text = torch.ones(B, N*K, 512, dtype=torch.int64)
    im = torch.ones(B, N*K, 128)   
    inputs = (idx, text, im)
    output = model(inputs)
    print(output)
