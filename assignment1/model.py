import torch
import torch.nn as nn
import zipfile
import numpy as np

class BaseModel(nn.Module):
    def __init__(self, args, vocab, tag_size):
        super(BaseModel, self).__init__()
        self.args = args
        self.vocab = vocab
        self.tag_size = tag_size

    def save(self, path):
        # Save model
        print(f'Saving model to {path}')
        ckpt = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(ckpt, path)

    def load(self, path):
        # Load model
        print(f'Loading model from {path}')
        ckpt = torch.load(path)
        self.vocab = ckpt['vocab']
        self.args = ckpt['args']
        self.load_state_dict(ckpt['state_dict'])


def load_embedding(vocab, emb_file, emb_size):
    """
    Read embeddings for words in the vocabulary from the emb_file (e.g., GloVe, FastText).
    Args:
        vocab: (Vocab), a word vocabulary
        emb_file: (string), the path to the embdding file for loading
        emb_size: (int), the embedding size (e.g., 300, 100) depending on emb_file
    Return:
        emb: (np.array), embedding matrix of size (|vocab|, emb_size) 
    """
    with open(emb_file, "r", encoding='utf-8', newline='\n', errors='ignore') as emb_f:
        n, d = map(int, emb_f.readline().split())
        print(n,d)
        emb = [0] * vocab.__len__()
        for line in emb_f:
            tokens = line.rstrip().split(' ')
            if tokens[0] in vocab.word2id.keys():
                if emb_size == d:
                    emb[vocab.word2id[tokens[0]]] = list(map(float, tokens[1:]))
                else:
                    emb[vocab.word2id[tokens[0]]] = list(map(float, tokens[1:emb_size+1]))
                    # But can not capture the whole information Maybe PCA?

        for i in range(len(emb)):
            if emb[i] == 0:
                emb[i] = [0] * emb_size

    return np.array(emb)

    # raise NotImplementedError()


class DanModel(BaseModel):
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        # Use pre-trained word embeddings if emb_file exists
        if args.emb_file is not None:
            self.copy_embedding_from_numpy()

        self.define_model_parameters()
        self.init_model_parameters()


    def define_model_parameters(self):
        """
        Define the model's parameters, e.g., embedding layer, feedforward layer.
        """
        if self.args.emb_file is not None:
            self.embedding_layer = nn.Embedding.from_pretrained(self.emb)
        else:
            self.embedding_layer = nn.Embedding(self.vocab.__len__(), self.args.emb_size, padding_idx=self.vocab.pad_id,
                                                max_norm=True)
            self.embedding_layer.weight.data.uniform_(-self.args.initvalue, self.args.initvalue)
        self.emb_drop = nn.Dropout(self.args.emb_drop)
        self.h1layer = nn.Linear(self.args.emb_size, self.args.hid_size)
        self.h12layer = nn.Linear(self.args.hid_size, self.args.emb_size)
        self.h2layer = nn.Linear(self.args.emb_size, self.tag_size)
        self.ReLU = nn.ReLU()

        self.linears = nn.ModuleList(
            [nn.Linear(self.args.hid_size, self.args.hid_size) for _ in range(self.args.hid_layer - 3)])
        self.identity = nn.Identity()
        self.logsoftmax = nn.LogSoftmax()
        # raise NotImplementedError()

    def init_model_parameters(self):
        """
        Initialize the model's parameters by uniform sampling from a range [-v, v], e.g., v=0.08
        """
        for ind, layer in enumerate(self.linears):
            nn.init.xavier_normal_(layer.weight.data, gain=1.0)
        nn.init.xavier_normal_(self.h1layer.weight.data, gain=1.0)
        nn.init.xavier_normal_(self.h2layer.weight.data, gain=1.0)

        # raise NotImplementedError()

    def copy_embedding_from_numpy(self):
        """
        Load pre-trained word embeddings from numpy.array to nn.embedding
        """
        self.emb = torch.FloatTensor(load_embedding(self.vocab, self.args.emb_file, self.args.emb_size))
        # raise NotImplementedError()

    def forward(self, x):
        """
        Compute the unnormalized scores for P(Y|X) before the softmax function.
        E.g., feature: h = f(x)
              scores: scores = w * h + b
              P(Y|X) = softmax(scores)  
        Args:
            x: (torch.LongTensor), [batch_size, seq_length]
        Return:
            scores: (torch.FloatTensor), [batch_size, ntags]
        """
        emb_x = self.embedding_layer(x)  # [batch_size, seq_length, emb_size]
        emb_x = self.emb_drop(emb_x)
        avg = torch.mean(emb_x, dim=1)  # [batch_size, emb_size]
        residual = self.identity(avg)
        h1 = self.h1layer(avg)  # [batch_size, hid_size]
        h1 = self.ReLU(h1)  # [batch_size, hid_size]
        for ind, layer in enumerate(self.linears): # REPEAT n-3 Layers, Total n layers
            h1 = layer(h1)
            h1 = self.ReLU(h1)
        h12 = self.h12layer(h1)
        h12 += residual
        h2 = self.h2layer(h12)  # [batch_size, tag_size]
        scores = self.logsoftmax(h2)

        return scores
        # raise NotImplementedError()
