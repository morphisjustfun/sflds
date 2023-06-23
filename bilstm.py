import javalang
import numpy as np
from javalang.tree import *
from queue import LifoQueue, Queue
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader, TensorDataset, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


# load model dist/model.pt
class BILSTM(nn.Module):
    def __init__(self, vocab_size):
        super(BILSTM, self).__init__()
        self.bfs_embedding = nn.Embedding(vocab_size, 30)
        self.bfs_lstm = nn.LSTM(30, 20)
        self.dfs_embedding = nn.Embedding(vocab_size, 30)
        self.dfs_lstm = nn.LSTM(30, 20)

        self.fc1 = nn.Linear(40, 60)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(60, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, bfs, dfs):
        bfs_lengths = (bfs != 0).sum(dim=1)
        dfs_lengths = (dfs != 0).sum(dim=1)

        bfs_embedded = self.bfs_embedding(bfs)
        dfs_embedded = self.dfs_embedding(dfs)

        bfs_packed = rnn_utils.pack_padded_sequence(bfs_embedded.permute(1, 0, 2), bfs_lengths.cpu(),
                                                    enforce_sorted=False)
        dfs_packed = rnn_utils.pack_padded_sequence(dfs_embedded.permute(1, 0, 2), dfs_lengths.cpu(),
                                                    enforce_sorted=False)

        bfs_lstm_out, _ = self.bfs_lstm(bfs_packed)
        dfs_lstm_out, _ = self.dfs_lstm(dfs_packed)

        bfs_lstm_out, _ = rnn_utils.pad_packed_sequence(bfs_lstm_out)
        dfs_lstm_out, _ = rnn_utils.pad_packed_sequence(dfs_lstm_out)

        bfs_lstm_out = bfs_lstm_out[bfs_lengths - 1, torch.arange(bfs_lengths.shape[0])]
        dfs_lstm_out = dfs_lstm_out[dfs_lengths - 1, torch.arange(dfs_lengths.shape[0])]

        # bfs_lstm_out = bfs_lstm_out[-1]
        # dfs_lstm_out = dfs_lstm_out[-1]

        merged = torch.cat((bfs_lstm_out, dfs_lstm_out), dim=1)
        merged = self.fc1(merged)
        merged = self.relu(merged)
        merged = self.fc2(merged)
        merged = self.sigmoid(merged)
        return merged
