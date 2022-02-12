import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud

from collections import Counter
import numpy as np
import random
import math

import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

USE_CUDA = torch.cuda.is_available()

# Control Seed
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if USE_CUDA:
    torch.cuda.manual_seed(1)

# Skip-gram hyper-parameters
C = 3  # context window
K = 10  # Number of negative samples
NUM_EPOCHS = 2
MAX_VOCAB_SIZE = 3000
BATCH_SIZE = 32
EMBEDDING_SIZE = 25
LEARNING_RATE = 0.2

def word_tokenize(text):
    return text.split()

'''
UNK 代表所有不常见的单词
'''

with open("data\\text8.train.txt","r") as f:
    text = f.read()

text = text.split()
vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1))
vocab["<unk>"] = len(text) - np.sum(list(vocab.values()))

idx_to_word = [word for word in vocab.keys()]
word_to_idx = {word:i for i, word in enumerate(idx_to_word)}


## Frequency
word_counts = np.array([count for count in vocab.values()], dtype = np.float32)
word_freqs = word_counts / np.sum(word_counts)
word_freqs = word_freqs ** (3./4.)
word_freqs = word_freqs / np.sum(word_freqs)
VOCAB_SIZE = len(idx_to_word)

## Dataloader 轻松返回Batch内容

class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, text, word_to_idx, idx_to_word, word_freqs, word_counts):
        super(WordEmbeddingDataset, self).__init__()
        self.text_encoded = [word_to_idx.get(word, VOCAB_SIZE-1) for word in text]
        self.text_encoded = torch.Tensor(self.text_encoded).long()
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_counts = torch.Tensor(word_counts)

    def __len__(self):
        return len(self.text_encoded)

    def __getitem__(self, idx):
        center_word = self.text_encoded[idx]
        pos_indices = list(range(idx-C)) + list(range(idx+1, idx+C+1)) # 可能超出文本长度
        pos_indices = [i % len(self.text_encoded) for i in pos_indices]
        pos_words = self.text_encoded[pos_indices]
        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)

        return center_word, pos_words, neg_words

dataset = WordEmbeddingDataset(text, word_to_idx, idx_to_word, word_freqs, word_counts)
print(dataset.text_encoded[:100])

## 这一块有问题
dataloader = tud.DataLoader(dataset, batch_size = BATCH_SIZE, shuffle=True, num_workers = 0)

print("Done")

## 定义pytorch 模型

class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size

        initrange = 0.5 / self.embed_size
        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.in_embed.weight.data.uniform_(-initrange, initrange)
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.out_embed.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_labels, pos_labels, neg_labels):
        input_embedding = self.in_embed(input_labels) # [batch_size, embed_size]
        pos_embedding = self.imbed(pos_labels) # [batch_size, (window_size * 2), embed_size]
        neg_embedding = self.in_embed(neg_labels) # [batch_size, (window_size * 2 * K), embed_size]

        input_embedding.unsqueeze(2) # [batch_size, embed_size, 1]
        pos_dot = torch.bmm(pos_embedding, input_embedding).squeeze(2) # [batch_size, (window_size * 2), 1]
        neg_dot = torch.bmm(neg_embedding, -input_embedding).squeeze(2)  # [batch_size, (window_size * 2 * K), 1]

        log_pos = F.logsigmoid(pos_dot).sum(1)  # 不要写成F.los(F.sigmoid) pytorch优化容易爆炸
        log_neg = F.logsigmoid(neg_dot).sum(1)

        loss = log_pos + log_neg # 求max
        return -loss # 求min [batch_size, 1]

    def input_embeddings(self):
        return self.in_embed.weight.data.cpu().numpy()

## 定义模型并移动到GPU

model = EmbeddingModel(VOCAB_SIZE, EMBEDDING_SIZE)
if USE_CUDA:
    model = model.cuda()

print("Done?1")
## optimizer的定义
optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE)
for e in range(NUM_EPOCHS):
    for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
        # print(input_labels, pos_labels, neg_labels)
        # if i > 5:
        #     break
        input_labels = input_labels.long()
        pos_labels = pos_labels.long()
        neg_labels = neg_labels.long()
        if USE_CUDA:
            input_labels = input_labels.cuda()
            pos_labels = pos_labels.cuda()
            neg_labels = neg_labels.cuda()

            optimizer.zero_grad()
            loss = model(input_labels, pos_labels, neg_labels).mean()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print("Epoch", e, "iteration", i, loss.item())

print("Done?")
