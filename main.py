# camel -> https://github.com/apache/camel
# train with 1.4 and test with 1.6
# WPDP -> Within Project Defect Prediction
import javalang
import numpy as np
import pandas as pd
from javalang.tree import *
from queue import LifoQueue, Queue

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader, TensorDataset, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

maxTokens = 2500

"""
# GET TRAIN DATA

functionInstanceInvocationClasses = [MethodInvocation, SuperMethodInvocation, ClassCreator]
declarationClasses = [PackageDeclaration, InterfaceDeclaration, ClassDeclaration, ConstructorDeclaration,
                      MethodDeclaration, VariableDeclarator, VariableDeclaration, FormalParameter]
controlFlowClasses = [IfStatement, ForStatement, WhileStatement, DoStatement, AssertStatement, BreakStatement,
                      ContinueStatement, ReturnStatement, ThrowStatement, TryStatement, SynchronizedStatement,
                      SwitchStatement, BlockStatement, CatchClauseParameter, TryResource, CatchClause,
                      SwitchStatementCase, ForControl, EnhancedForControl]
otherClasses = [BasicType, MemberReference, ReferenceType, SuperMemberReference, StatementExpression]


def dfsAstNode(root):
    q = LifoQueue()
    q.put(root)
    while not q.empty():
        node = q.get()
        if isinstance(node, Node):
            children = node.children
            yield node
        else:
            children = node

        for child in children:
            if isinstance(child, (Node, list, tuple)):
                q.put(child)


def bfsAstNode(root):
    q = Queue()
    q.put(root)
    while not q.empty():
        node = q.get()
        if isinstance(node, Node):
            children = node.children
            yield node
        else:
            children = node

        for child in children:
            if isinstance(child, (Node, list, tuple)):
                q.put(child)


def fileExists(file) -> bool:
    try:
        with open(file, 'r') as _:
            return True
    except IOError:
        return False


def getDfsFilteredNodes(root):
    dfsSequence = []
    for node in dfsAstNode(root):
        try:
            if any(isinstance(node, classWanted) for classWanted in functionInstanceInvocationClasses):
                if isinstance(node, ClassCreator):
                    dfsSequence.append(node.type.name)
                else:
                    dfsSequence.append(node.member)
            elif any(isinstance(node, classWanted) for classWanted in declarationClasses):
                if isinstance(node, LocalVariableDeclaration) or isinstance(node, VariableDeclaration):
                    dfsSequence.append(node.type.name)
                else:
                    dfsSequence.append(node.name)

            elif any(isinstance(node, classWanted) for classWanted in controlFlowClasses):
                if isinstance(node, IfStatement):
                    dfsSequence.append('if')
                elif isinstance(node, ForStatement):
                    dfsSequence.append('for')
                elif isinstance(node, WhileStatement):
                    dfsSequence.append('while')
                elif isinstance(node, DoStatement):
                    dfsSequence.append('do')
                elif isinstance(node, AssertStatement):
                    dfsSequence.append('assert')
                elif isinstance(node, BreakStatement):
                    dfsSequence.append('break')
                elif isinstance(node, ContinueStatement):
                    dfsSequence.append('continue')
                elif isinstance(node, ReturnStatement):
                    dfsSequence.append('return')
                elif isinstance(node, ThrowStatement):
                    dfsSequence.append('throw')
                elif isinstance(node, TryStatement):
                    dfsSequence.append('try')
                elif isinstance(node, SynchronizedStatement):
                    dfsSequence.append('synchronized')
                elif isinstance(node, SwitchStatement):
                    dfsSequence.append('switch')
                elif isinstance(node, BlockStatement):
                    dfsSequence.append('block')
                elif isinstance(node, CatchClauseParameter) or isinstance(node, TryResource):
                    dfsSequence.append(node.name)
                elif isinstance(node, SwitchStatementCase):
                    dfsSequence.append(node.member)
                elif isinstance(node, ForControl):
                    dfsSequence.append(node.condition)
                elif isinstance(node, EnhancedForControl):
                    dfsSequence.append(node.var.type.name)

            elif any(isinstance(node, classWanted) for classWanted in otherClasses):
                if isinstance(node, BasicType) or isinstance(node, ReferenceType):
                    dfsSequence.append(node.name)
                elif isinstance(node, MemberReference) or isinstance(node, SuperMemberReference):
                    dfsSequence.append(node.member)
                elif isinstance(node, StatementExpression):
                    if isinstance(node.expression, Assignment):
                        if isinstance(node.expression.value, MemberReference):
                            dfsSequence.append(node.expression.value.member)
                        else:
                            dfsSequence.append(node.expression.value.value)
                    else:
                        dfsSequence.append(node.expression.member)
        except:
            pass
    return dfsSequence


def getBfsFilteredNodes(root):
    bfsSequence = []
    for node in bfsAstNode(root):
        try:
            if any(isinstance(node, classWanted) for classWanted in functionInstanceInvocationClasses):
                if isinstance(node, ClassCreator):
                    bfsSequence.append(node.type.name)
                else:
                    bfsSequence.append(node.member)
            elif any(isinstance(node, classWanted) for classWanted in declarationClasses):
                if isinstance(node, LocalVariableDeclaration) or isinstance(node, VariableDeclaration):
                    bfsSequence.append(node.type.name)
                else:
                    bfsSequence.append(node.name)

            elif any(isinstance(node, classWanted) for classWanted in controlFlowClasses):
                if isinstance(node, IfStatement):
                    bfsSequence.append('if')
                elif isinstance(node, ForStatement):
                    bfsSequence.append('for')
                elif isinstance(node, WhileStatement):
                    bfsSequence.append('while')
                elif isinstance(node, DoStatement):
                    bfsSequence.append('do')
                elif isinstance(node, AssertStatement):
                    bfsSequence.append('assert')
                elif isinstance(node, BreakStatement):
                    bfsSequence.append('break')
                elif isinstance(node, ContinueStatement):
                    bfsSequence.append('continue')
                elif isinstance(node, ReturnStatement):
                    bfsSequence.append('return')
                elif isinstance(node, ThrowStatement):
                    bfsSequence.append('throw')
                elif isinstance(node, TryStatement):
                    bfsSequence.append('try')
                elif isinstance(node, SynchronizedStatement):
                    bfsSequence.append('synchronized')
                elif isinstance(node, SwitchStatement):
                    bfsSequence.append('switch')
                elif isinstance(node, BlockStatement):
                    bfsSequence.append('block')
                elif isinstance(node, CatchClauseParameter) or isinstance(node, TryResource):
                    bfsSequence.append(node.name)
                elif isinstance(node, SwitchStatementCase):
                    bfsSequence.append(node.member)
                elif isinstance(node, ForControl):
                    bfsSequence.append(node.condition)
                elif isinstance(node, EnhancedForControl):
                    bfsSequence.append(node.var.type.name)

            elif any(isinstance(node, classWanted) for classWanted in otherClasses):
                if isinstance(node, BasicType) or isinstance(node, ReferenceType):
                    bfsSequence.append(node.name)
                elif isinstance(node, MemberReference) or isinstance(node, SuperMemberReference):
                    bfsSequence.append(node.member)
                elif isinstance(node, StatementExpression):
                    if isinstance(node.expression, Assignment):
                        if isinstance(node.expression.value, MemberReference):
                            bfsSequence.append(node.expression.value.member)
                        else:
                            bfsSequence.append(node.expression.value.value)
                    else:
                        bfsSequence.append(node.expression.member)
        except:
            pass
    return bfsSequence


componentsList = ['camel-amqp', 'camel-atom', 'camel-bam', 'camel-csv', 'camel-cxf', 'camel-flatpack', 'camel-ftp',
                  'camel-groovy', 'camel-hamcrest', 'camel-http', 'camel-ibatis', 'camel-irc', 'camel-jaxb',
                  'camel-jcr', 'camel-jdbc', 'camel-jetty', 'camel-jhc', 'camel-jing', 'camel-jms', 'camel-josql',
                  'camel-jpa', 'camel-juel', 'camel-jxpath', 'camel-mail', 'camel-mina', 'camel-msv', 'camel-ognl',
                  'camel-osgi', 'camel-quartz', 'camel-rmi', 'camel-ruby', 'camel-saxon', 'camel-script',
                  'camel-spring', 'camel-spring-integration', 'camel-sql', 'camel-stream',
                  'camel-stringtemplate', 'camel-supercsv', 'camel-swing', 'camel-testng', 'camel-uface',
                  'camel-velocity', 'camel-xmlbeans', 'camel-xmpp', 'camel-xstream']

trainData = pd.read_csv('datasets/camel/train/labels.csv')
trainData = trainData.loc[:, ['name', 'bug']]
trainData['bug'] = trainData['bug'].apply(lambda x: 1 if x > 0 else 0)
trainDataNames = trainData.loc[:, ['name']]

trainDataNamesModified = []
for file in trainDataNames['name']:
    name = file.replace('.', '/') + '.java'
    exists = False
    nameToTest = 'datasets/camel/train/source/camel-core/src/main/java/' + name
    if fileExists(nameToTest):
        trainDataNamesModified.append(
            nameToTest)
        continue
    for component in componentsList:
        nameToTest = 'datasets/camel/train/source/components/' + component + '/src/main/java/' + name
        if fileExists(nameToTest):
            trainDataNamesModified.append(nameToTest)
            exists = True
            break
    if not exists:
        trainData = trainData[trainData.name != file]

trainData['name'] = trainDataNamesModified
trainData.reset_index(drop=True, inplace=True)

# max tokens would be 7000
trainData = pd.concat([trainData, pd.DataFrame(columns=[str(i) for i in range(maxTokens * 2)])])

for i in range(len(trainData)):
    name = trainData['name'][i]
    rootParsed = javalang.parse.parse(open(name).read())
    bfsSequence = getBfsFilteredNodes(rootParsed)
    dfsSequence = getDfsFilteredNodes(rootParsed)
    bfsSequenceFiltered = list(filter(lambda x: isinstance(x, str), bfsSequence))
    dfsSequenceFiltered = list(filter(lambda x: isinstance(x, str), dfsSequence))
    uniqueTokens = set(bfsSequenceFiltered)
    token2int = {token: ii for ii, token in enumerate(uniqueTokens, 1)}
    bfsEncoded = [token2int[token] for token in bfsSequenceFiltered]
    dfsEncoded = [token2int[token] for token in dfsSequenceFiltered]
    for j in range(len(bfsEncoded)):
        trainData.at[i, str(j)] = bfsEncoded[j]
    for j in range(len(dfsEncoded)):
        trainData.at[i, str(j + maxTokens)] = dfsEncoded[j]

trainData.drop(['name'], axis=1, inplace=True)
trainData.fillna(0, inplace=True)
trainData.to_csv('dist/train_df.csv', index=False)
"""

trainData = pd.read_csv('dist/train_df.csv')
df_majority = trainData[trainData.bug==0]
df_minority = trainData[trainData.bug==1]

# Count how many samples for the majority class
majority_count = df_majority.shape[0]

# Upsample minority class
df_minority_upsampled = df_minority.sample(majority_count, replace=True, random_state=42)

# Combine majority class with upsampled minority class
trainData_balanced = pd.concat([df_majority, df_minority_upsampled], axis=0)
trainData_balanced = trainData_balanced.sample(frac=1, random_state=42)

# randomly
# use mps macos
device = torch.device("cpu")
# bfs if first
# dfs is second
bfs = trainData_balanced.iloc[:, 1:2501]
dfs = trainData_balanced.iloc[:, 2501:5001]
labels = trainData_balanced['bug']

# X1 is bfs
# X2 is dfs
X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(bfs, dfs, labels, test_size=0.2)

class CustomDataset(Dataset):
    def __init__(self, data1, data2, labels):
        self.data1 = torch.tensor(data1.values).long()
        self.data2 = torch.tensor(data2.values).long()
        self.labels = torch.tensor(labels.values).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data1[idx], self.data2[idx], self.labels[idx]


train_dataset = CustomDataset(X1_train, X2_train, y_train)
val_dataset = CustomDataset(X1_val, X2_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding1 = nn.Embedding(2500, 128)
        self.embedding2 = nn.Embedding(2500, 128)
        self.lstm1 = nn.LSTM(128, 64, batch_first=True)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, x1, x2):
        x1 = self.embedding1(x1)
        x2 = self.embedding2(x2)

        x1, _ = self.lstm1(x1)
        x2, _ = self.lstm2(x2)

        x = torch.cat((x1[:, -1, :], x2[:, -1, :]), dim=1)
        x = self.fc(x)
        return torch.sigmoid(x)



model = Model().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(10):
    model.train()
    for x1, x2, labels in train_loader:
        x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(x1, x2).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = 0
        for x1, x2, labels in val_loader:
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            outputs = model(x1, x2).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()

        print(f'Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_loader)}')

# save
torch.save(model.state_dict(), 'dist/model.pt')
