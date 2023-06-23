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

from bilstm import BILSTM

trainData = pd.read_csv('dist/train_df.csv')
df_majority = trainData[trainData.bug == 0]
df_minority = trainData[trainData.bug == 1]

# Count how many samples for the majority class
majority_count = df_majority.shape[0]

# Upsample minority class
# use SMOTE
df_minority_upsampled = df_minority.sample(majority_count, replace=True)

# Combine majority class with upsampled minority class
trainData_balanced = pd.concat([df_majority, df_minority_upsampled], axis=0)
trainData_balanced = trainData_balanced.sample(frac=1)
trainData_balanced = trainData_balanced.reset_index(drop=True)

bfs_data = trainData_balanced.iloc[:, 0:2600]
dfs_data = trainData_balanced.iloc[:, 2600:5200]
labels_data = trainData_balanced['bug']
vocabSize = bfs_data.max().max() + 1

# oversampling SMOGN, SMOTER, None
# normalize use Z, min-max, None

bfs_data, bfs_test, dfs_data, dfs_test, labels_data, labels_test = train_test_split(bfs_data, dfs_data, labels_data,
                                                                                    test_size=0.1)

train_data = TensorDataset(torch.from_numpy(bfs_data.values), torch.from_numpy(dfs_data.values),
                           torch.from_numpy(labels_data.values))
val_data = TensorDataset(torch.from_numpy(bfs_test.values), torch.from_numpy(dfs_test.values),
                         torch.from_numpy(labels_test.values))
train_loader = DataLoader(train_data, batch_size=15, shuffle=True)
model = BILSTM(vocabSize)
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
