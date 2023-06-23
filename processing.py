# camel1.4 -> https://github.com/apache/camel
# train with 1.4 and test with 1.6
# WPDP -> Within Project Defect Prediction
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

maxTokens = 2600

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


componentsList = ['camel-amqp', 'camel-atom', 'camel-bam', 'camel-csv', 'camel-cxf', 'camel-flatpack',
                  'camel-ftp',
                  'camel-groovy', 'camel-hamcrest', 'camel-http', 'camel-ibatis', 'camel-irc',
                  'camel-jaxb',
                  'camel-jcr', 'camel-jdbc', 'camel-jetty', 'camel-jhc', 'camel-jing', 'camel-jms',
                  'camel-josql',
                  'camel-jpa', 'camel-juel', 'camel-jxpath', 'camel-mail', 'camel-mina', 'camel-msv',
                  'camel-ognl',
                  'camel-osgi', 'camel-quartz', 'camel-rmi', 'camel-ruby', 'camel-saxon',
                  'camel-script',
                  'camel-spring', 'camel-spring-integration', 'camel-sql', 'camel-stream',
                  'camel-stringtemplate', 'camel-supercsv', 'camel-swing', 'camel-testng', 'camel-uface',
                  'camel-velocity', 'camel-xmlbeans', 'camel-xmpp', 'camel-xstream']


def getCammel(camel_version):
    trainData = pd.read_csv(f'datasets/camel{camel_version}/labels.csv')
    trainData = trainData.loc[:, ['name', 'bug']]
    trainData['bug'] = trainData['bug'].apply(lambda x: 1 if x > 0 else 0)
    trainDataNames = trainData.loc[:, ['name']]

    trainDataNamesModified = []
    for file in trainDataNames['name']:
        name = file.replace('.', '/') + '.java'
        exists = False
        nameToTest = f'datasets/camel{camel_version}/source/camel-core/src/main/java/' + name
        if fileExists(nameToTest):
            trainDataNamesModified.append(
                nameToTest)
            continue
        for component in componentsList:
            nameToTest = f'datasets/camel{camel_version}/source/components/' + component + '/src/main/java/' + name
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

    totalUniqueTokens = set()
    totalBfsSequences = []
    totalDfsSequences = []

    for i in range(len(trainData)):
        name = trainData['name'][i]
        rootParsed = javalang.parse.parse(open(name).read())
        bfsSequence = getBfsFilteredNodes(rootParsed)
        dfsSequence = getDfsFilteredNodes(rootParsed)
        bfsSequenceFiltered = list(filter(lambda x: isinstance(x, str), bfsSequence))
        dfsSequenceFiltered = list(filter(lambda x: isinstance(x, str), dfsSequence))
        totalBfsSequences.append(bfsSequenceFiltered)
        totalDfsSequences.append(dfsSequenceFiltered)
        totalUniqueTokens.update(bfsSequenceFiltered)

    trainData.drop(['name'], axis=1, inplace=True)
    trainData.fillna(0, inplace=True)
    trainData['bug'] = trainData['bug'].astype(int)
    return trainData, totalUniqueTokens, totalBfsSequences, totalDfsSequences


def merge():
    versions = ['1.0', '1.2', '1.4', '1.6']
    dfs = []
    totalUniqueTokens = set()
    totalBfsSequences = []
    totalDfsSequences = []
    for version in versions:
        df, uniqueTokens, bfsSequences, dfsSequences = getCammel(version)
        dfs.append(df)
        totalUniqueTokens.update(uniqueTokens)
        totalBfsSequences.append(bfsSequences)
        totalDfsSequences.append(dfsSequences)
    word2int = {token: ii for ii, token in enumerate(totalUniqueTokens, 1)}

    for i in range(len(dfs)):
        for j in range(len(dfs[i])):
            bfsEncoded = [word2int[token] for token in totalBfsSequences[i][j]]
            dfsEncoded = [word2int[token] for token in totalDfsSequences[i][j]]
            for k in range(len(bfsEncoded)):
                dfs[i].at[j, str(k)] = bfsEncoded[k]
            for k in range(len(dfsEncoded)):
                dfs[i].at[j, str(k + maxTokens)] = dfsEncoded[k]

    df = pd.concat(dfs)
    df = df[[c for c in df if c not in ['bug']] + ['bug']]
    df.to_csv('dist/train_df.csv', index=False)


if __name__ == '__main__':
    merge()
