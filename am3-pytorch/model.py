"""Model classes for AM3 in Pytorch.
"""

import torch
import torch.nn as nn
import numpy as np

import gensim.downloader as api
from transformers import BertModel

import utils


class AM3(nn.Module):
    def __init__(self, im_encoder, im_emb_dim, text_encoder, text_emb_dim=300, text_hid_dim=300, prototype_dim=512, dropout=0.7, fine_tune=False, dictionary=None, pooling_strat="mean"):
        super(AM3, self).__init__()
        self.im_emb_dim = im_emb_dim            # image embedding size
        self.text_encoder_type = text_encoder
        self.text_emb_dim = text_emb_dim        # only applicable if precomputed
        self.text_hid_dim = text_hid_dim        # AM3 uses 300
        self.prototype_dim = prototype_dim      # AM3 uses 512 (resnet)
        self.dropout = dropout                  # AM3 uses 0.7 or 0.9 depending on dataset
        self.fine_tune = fine_tune
        self.dictionary = dictionary            # for word embeddings
        self.pooling_strat = pooling_strat

        if im_encoder == "precomputed":
            # if using precomputed embeddings
            self.image_encoder = nn.Linear(im_emb_dim, prototype_dim)
        elif im_encoder == "resnet":
            # TODO image encoder if raw images
            self.image_encoder = nn.Linear(im_emb_dim, prototype_dim)

        if self.text_encoder_type == "BERT":
            # TODO be able to use any hf bert model (requires correct tokenisation)
            self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
            self.text_emb_dim = self.text_encoder.config.hidden_size
        elif self.text_encoder_type == "precomputed":
            self.text_encoder = nn.Identity()
        elif self.text_encoder_type == "w2v" or self.text_encoder_type == "glove":
            # load pretrained word embeddings as weights
            self.text_encoder = WordEmbedding(self.text_encoder_type, self.pooling_strat, self.dictionary)
            self.text_emb_dim = self.text_encoder.embedding_dim
        elif self.text_encoder_type == "RNN":
            # TODO RNN implementation
            self.text_encoder = nn.Linear(text_emb_dim, text_emb_dim)   
        else:
            raise NameError(f"{text_encoder} not allowed as text encoder")

        # fine tune set up
        if not self.fine_tune:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
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
        - inputs (tuple): 
            - idx:  image ids
            - text: padded tokenised sequence. (tuple for BERT of input_ids and attn_mask). 
            - attn_mask: attention mask for bert (only if BERT)
            - im:   precomputed image embeddings
        - im_only (bool): flag to only use image input (for query set)

        Returns:
        - im_embeddings (torch.FloatTensor): image in prototype space (b, NxK, emb_dim)
        - (if not im_only) text_embeddings (torch.FloatTensor): text in prototype space (b, NxK, emb_dim)
        """
        # unpack input
        if self.text_encoder_type == "BERT":
            idx, text, attn_mask, im = inputs
        else:
            idx, text, im = inputs

        # process
        im_embeddings = self.image_encoder(im)      # (b x N*K x 512)  
        if not im_only:
            if self.text_encoder_type == "BERT":
                # need to reshape batch for BERT input
                B, NK, seq_len = text.shape
                bert_output = self.text_encoder(text.view(-1, seq_len), attention_mask=attn_mask.view(-1, seq_len))
                # get [CLS] token
                text_encoding = bert_output[1].view(B, NK, -1)        # (b x N*K x 768)
            else:
                text_encoding = self.text_encoder(text)
            text_embeddings = self.g(text_encoding)   # (b x N*K x 512)
            lamda = torch.sigmoid(self.h(text_embeddings))  # (b x N*K x 1)
            return im_embeddings, text_embeddings, lamda
        else:
            return im_embeddings

    def evaluate(self, batch, optimizer, num_ways, device, task="train"):
        """Run one episode through model
        Params:
        - batch (dict): meta-batch of tasks
        - optimizer (nn.optim): optimizer tied to model weights.
        - num_ways (int): number 'N' of classes to choose from.
        - device (torch.device): cuda or cpu
        - task (str): train, val, test
        Returns:
        - loss: prototypical loss
        - acc: accuracy on query set
        - if test: also return predictions (class for each query)
        """
        if task == "train":
            self.train()
        else:
            self.eval()

        # support set
        train_inputs, train_targets = batch['train']            
        train_inputs = [x.to(device) for x in train_inputs]
        train_targets = train_targets.to(device)                          
        train_im_embeddings, train_text_embeddings, train_lamda = self(train_inputs)
        avg_lamda = torch.mean(train_lamda)

        # query set
        test_inputs, test_targets = batch['test']
        test_inputs = [x.to(device) for x in test_inputs]
        test_targets = test_targets.to(device)
        test_im_embeddings = self(test_inputs, im_only=True)    # only get image prototype

        # construct prototypes
        prototypes = utils.get_prototypes(
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
            return loss.detach().cpu().numpy(), acc, preds, test_targets.detach().cpu().numpy(), test_idx.detach().cpu().numpy(), avg_lamda.detach().cpu().numpy()
        else:
            return loss.detach().cpu().numpy(), acc, avg_lamda.detach().cpu().numpy()


class WordEmbedding(nn.Module):
    def __init__(self, text_encoder_type, pooling_strat, dictionary):
        """Embeds tokenised sequence into a fixed length word embedding
        """
        super(WordEmbedding, self).__init__()
        self.pooling_strat = pooling_strat
        self.dictionary = dictionary
        self.text_encoder_type = text_encoder_type

        # get pretrained word embeddings
        print("dictionary size: ", len(self.dictionary))
        print("loading pretrained word vectors...")
        if text_encoder_type == "glove":
            word_model = api.load("glove-wiki-gigaword-300")
        elif text_encoder_type == "w2v":
            word_model = api.load("word2vec-google-news-300")
        self.embedding_dim = word_model.vector_size

        OOV = []
        # randomly initialise OOV tokens between -1 and 1
        weights = 2*np.random.rand(len(self.dictionary), self.embedding_dim) - 1
        for word, token in self.dictionary.items():
            if word == "PAD":
                self.padding_token = token
                weights[token, :] = np.zeros(self.embedding_dim) 
            elif word in word_model.vocab:
                weights[token, :] = word_model[word]
            else:
                OOV.append(word)
        # print number out of vocab
        print(f"done. Embedding dim: {self.embedding_dim}. "
              f"Number of OOV tokens: {len(OOV)}, padding token: {self.padding_token}")
        
        # use to make embedding layer
        self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(weights))

    def forward(self, x):
        """Params:
        x (torch.LongTensor): tokenised sequence (b x N*K x max_seq_len)
        Returns:
        text_embedding (torch.FloatTensor): embedded sequence (b x N*K x emb_dim)
        """
        # embed
        text_embedding = self.embed(x)      # (b x N*K x max_seq_len x emb_dim)
        # pool
        if self.pooling_strat == "mean":
            padding_mask = torch.where(x != self.padding_token, 1, 0)  # (b x N*K x max_seq_len)
            seq_lens = torch.sum(padding_mask, dim=-1).unsqueeze(-1)        # (b x N*K x 1)
            return torch.sum(text_embedding, dim=2).div_(seq_lens)
        elif self.pooling_strat == "max":
            # TODO check how to max pool
            return torch.max(text_embedding, dim=2)[0]
        else:
            raise NameError(f"{self.pooling_strat} pooling strat not defined")


if __name__ == "__main__":

    model = AM3(im_encoder="precomputed", im_emb_dim=512, text_encoder="BERT", text_emb_dim=768, text_hid_dim=300, prototype_dim=512, dropout=0.7, fine_tune=False)
    print(model)
    N = 5
    K = 2
    B = 5
    idx = torch.ones(B, N*K)
    text = torch.ones(B, N*K, 128, dtype=torch.int64)
    im = torch.ones(B, N*K, 512)   
    targets = 2*torch.ones(B, N*K, dtype=torch.int64)
    inputs = (idx, text, im)
    im_embed, text_embed, lamdas = model(inputs)
    print("output shapes (im, text, lamda)", im_embed.shape, text_embed.shape, lamdas.shape)
    prototypes = utils.get_prototypes(im_embed, text_embed, lamdas, targets, N)
    print("prototypes", prototypes.shape)
    loss = utils.prototypical_loss(prototypes, im_embed, targets)   # test on train set
    print("loss", loss)
