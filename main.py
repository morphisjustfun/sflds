# camel -> https://github.com/apache/camel
# train with 1.4 and test with 1.6
# WPDP -> Within Project Defect Prediction

import javalang
import pandas as pd
from javalang.ast import Node
from queue import LifoQueue, Queue


## PREPROCESSING


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

## BUILD S-AST

test = javalang.parse.parse(open(trainData['name'][0]).read())
test

for node in dfsAstNode(test):
    print(node)

for node in bfsAstNode(test):
    print(node)
