import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import gensim.downloader as api


class WordEmbedding(nn.Module):
    def __init__(self, text_encoder_type, pooling_strat, dictionary):
        """Embeds tokenised sequence into a fixed length word embedding
        """
        super(WordEmbedding, self).__init__()
        self.pooling_strat = pooling_strat
        self.dictionary = dictionary
        self.padding_token = self.dictionary["PAD"]
        self.text_encoder_type = text_encoder_type

        # get pretrained word embeddings
        embedding_weights = get_embedding_weights(dictionary, text_encoder_type)
        self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_weights))

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
            return torch.max(text_embedding, dim=2)[0]
        else:
            raise NameError(f"{self.pooling_strat} pooling strat not defined")


class RNN(nn.Module):
    def __init__(self, embedding_type, pooling_strat, dictionary, rnn_hid_dim):
        """Embeds tokenised sequence into a fixed length encoding with an RNN.
        """
        super(RNN, self).__init__()
        self.pooling_strat = pooling_strat
        self.dictionary = dictionary
        self.embedding_type = embedding_type
        self.rnn_hid_dim = rnn_hid_dim // 2      # assuming bidirectional
        self.padding_token = self.dictionary["PAD"]

        # word embedding
        if embedding_type == "rand":
            self.embed = nn.Embedding(len(self.dictionary), rnn_hid_dim)
            self.text_emb_size = rnn_hid_dim
        else:
            embedding_weights = get_embedding_weights(dictionary, embedding_type)
            self.text_emb_size = embedding_weights.shape[-1]
            self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_weights))

        # RNN 
        self.rnn = nn.LSTM(
            input_size=self.text_emb_size,
            hidden_size=self.rnn_hid_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True)

    def forward(self, x):
        """Params:
        - x (torch.LongTensor): tokenised sequence (b x N*K x max_seq_len)
        Returns:
        - text_embedding (torch.FloatTensor): embedded sequence (b x N*K x emb_dim)
        """
        # process data
        B, NK, max_seq_len = x.shape
        # flatten batch
        x_flat = x.view(-1, max_seq_len)    # (B*N*K x max_seq_len)
        # padding_masks
        padding_mask = torch.where(x_flat != self.padding_token, 1, 0)
        seq_lens = torch.sum(padding_mask, dim=-1)        # (B*N*K)

        # embed
        text_embedding = self.embed(x_flat)      # (B*N*K x max_seq_len x emb_dim)
        
        # feed through RNN
        text_embedding_packed = pack_padded_sequence(text_embedding, seq_lens, batch_first=True, enforce_sorted=False)
        rnn_out_packed, _ = self.rnn(text_embedding_packed)
        rnn_out, _ = pad_packed_sequence(rnn_out_packed, batch_first=True)     # (B*N*K, max_seq_len, rnn_hid_dim*2)

        # concat forward and backward results (takes output states)
        rnn_out_forward = rnn_out[:, seq_lens-1, :self.rnn_hid_dim]     # last state of forward
        rnn_out_backward = rnn_out[:, 0, self.rnn_hid_dim:]     # last state of backward (= first timestep)
        seq_embed = torch.cat((rnn_out_forward, rnn_out_backward), -1)        # (B*N*K, rnn_hid_dim*2)
        # unsqueeze
        return seq_embed.view(B, NK, -1)        # (B, N*K, rnn_hid_dim*2)


def get_embedding_weights(dictionary, text_encoder_type):
    """Loads gensim word embedding weights into a matrix
    Params:
    - dictionary: dictionary for text encoding
    - text_encoder_type: type of word embedding
    Returns:
    - embedding_matrix: matrix of embedding weights
    """
    print("dictionary size: ", len(dictionary))
    print("loading pretrained word vectors...")
    if text_encoder_type == "glove":
        word_model = api.load("glove-wiki-gigaword-300")
    elif text_encoder_type == "w2v":
        word_model = api.load("word2vec-google-news-300")
    embedding_dim = word_model.vector_size

    OOV = []
    # randomly initialise OOV tokens between -1 and 1
    weights = 2*np.random.rand(len(dictionary),
                                embedding_dim) - 1
    for word, token in dictionary.items():
        if word == "PAD":
            padding_token = token
            weights[token, :] = np.zeros(embedding_dim)
        elif word in word_model.key_to_index:
            weights[token, :] = word_model[word]
        else:
            OOV.append(word)
    # print number out of vocab
    print(f"done. Embedding dim: {embedding_dim}. "
            f"Number of OOV tokens: {len(OOV)}, padding token: {padding_token}")
    
    return weights