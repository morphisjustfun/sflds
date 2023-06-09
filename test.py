import javalang
import numpy as np
from javalang.tree import *
from queue import LifoQueue, Queue
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from bilstm import BILSTM

# load model dist/model.pt of type  BILSTM
device = torch.device("cpu")
model = BILSTM(8146)
model.load_state_dict(torch.load('dist/model.pt', map_location=device))

# load data
data = pd.read_csv('dist/train_df.csv')

# split data to test
bfs_data = data.iloc[:, 0:2600]
dfs_data = data.iloc[:, 2600:5200]
label = data.iloc[:, 5200]

# sample only 200
bfs_data, bfs_test, dfs_data, dfs_test, labels_data, labels_test = train_test_split(bfs_data, dfs_data, label,
                                                                                    test_size=0.1)

# convert to tensor

# load test data
# with all dataset calculate accuracy, precision, recall, f1
model.eval()
with torch.no_grad():
    bfs_data, dfs_data, labels = torch.from_numpy(bfs_data.values), torch.from_numpy(dfs_data.values), torch.from_numpy(
        label.values)
    bfs_data, dfs_data, labels = bfs_data.to(device), dfs_data.to(device), labels.float().to(device)
    labels = labels.view(-1, 1)
    outputs = model(bfs_data, dfs_data)
    outputs = outputs.cpu().numpy()
    outputs = np.where(outputs > 0.5, 1, 0)
    labels = labels.cpu().numpy()
    print("Accuracy: ", accuracy_score(labels, outputs))
    print("Precision: ", precision_score(labels, outputs))
    print("Recall: ", recall_score(labels, outputs))
    print("F1: ", f1_score(labels, outputs))
