import torch
import torch.nn as nn
import numpy as np
import gensim.downloader as api


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
        weights = 2*np.random.rand(len(self.dictionary),
                                   self.embedding_dim) - 1
        for word, token in self.dictionary.items():
            if word == "PAD":
                self.padding_token = token
                weights[token, :] = np.zeros(self.embedding_dim)
            elif word in word_model.key_to_index:
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
        - x (torch.LongTensor): tokenised sequence (b x N*K x max_seq_len)
        Returns:
        - text_embedding (torch.FloatTensor): embedded sequence (b x N*K x emb_dim)
        """
        # embed
        text_embedding = self.embed(x)      # (b x N*K x max_seq_len x emb_dim)
        # pool
        if self.pooling_strat == "mean":
            # (b x N*K x max_seq_len)
            padding_mask = torch.where(x != self.padding_token, 1, 0)
            seq_lens = torch.sum(padding_mask, dim=-
                                 1).unsqueeze(-1)        # (b x N*K x 1)
            return torch.sum(text_embedding, dim=2).div_(seq_lens)
        elif self.pooling_strat == "max":
            # TODO check how to max pool
            return torch.max(text_embedding, dim=2)[0]
        else:
            raise NameError(f"{self.pooling_strat} pooling strat not defined")
