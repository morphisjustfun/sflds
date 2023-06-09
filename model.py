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

np.random.seed(42)

trainData = pd.read_csv('dist/train_df.csv')
df_majority = trainData[trainData.bug == 0]
df_minority = trainData[trainData.bug == 1]

# Count how many samples for the majority class
majority_count = df_majority.shape[0]

# Upsample minority class
df_minority_upsampled = df_minority.sample(majority_count, replace=True)

# Combine majority class with upsampled minority class
trainData_balanced = pd.concat([df_majority, df_minority_upsampled], axis=0)
trainData_balanced = trainData_balanced.sample(frac=1)
trainData_balanced = trainData_balanced.reset_index(drop=True)

bfs_data = trainData_balanced.iloc[:, 0:2600]
dfs_data = trainData_balanced.iloc[:, 2600:5200]
labels_data = trainData_balanced['bug']
vocabSize = bfs_data.max().max() + 1


class Net(nn.Module):
    def __init__(self, vocab_size):
        super(Net, self).__init__()
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


bfs_data, bfs_test, dfs_data, dfs_test, labels_data, labels_test = train_test_split(bfs_data, dfs_data, labels_data,
                                                                                    test_size=0.1)

train_data = TensorDataset(torch.from_numpy(bfs_data.values), torch.from_numpy(dfs_data.values),
                           torch.from_numpy(labels_data.values))
val_data = TensorDataset(torch.from_numpy(bfs_test.values), torch.from_numpy(dfs_test.values),
                         torch.from_numpy(labels_test.values))
train_loader = DataLoader(train_data, batch_size=120, shuffle=True)
model = Net(vocabSize)
device = torch.device("mps")
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
criterion = nn.BCELoss()
for epoch in range(100):
    model.train()
    for i, data in enumerate(train_loader, 0):
        bfs, dfs, labels = data
        bfs, dfs, labels = bfs.to(device), dfs.to(device), labels.float().to(device)
        labels = labels.view(-1, 1)
        optimizer.zero_grad()
        outputs = model(bfs, dfs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        val_loss = criterion(outputs, labels)
        acc = (outputs.round() == labels).float().mean()
    print(f"Epoch {epoch + 1}: Train Loss = {val_loss:.4f} | Train Acc = {acc:.4f}")
    scheduler.step()

# save model
torch.save(model.state_dict(), 'dist/model.pt')
