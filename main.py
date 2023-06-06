# camel -> https://github.com/apache/camel
# train with 1.4 and test with 1.6
# WPDP -> Within Project Defect Prediction

import javalang
import pandas as pd
from javalang.ast import Node
from javalang.tree import *
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

test = javalang.parse.parse(open(trainData['name'][381]).read())

functionInstanceInvocationClasses = [MethodInvocation, SuperMethodInvocation, ClassCreator]
declarationClasses = [PackageDeclaration, InterfaceDeclaration, ClassDeclaration, ConstructorDeclaration,
                      MethodDeclaration, VariableDeclarator, VariableDeclaration, FormalParameter]
controlFlowClasses = [IfStatement, ForStatement, WhileStatement, DoStatement, AssertStatement, BreakStatement,
                      ContinueStatement, ReturnStatement, ThrowStatement, TryStatement, SynchronizedStatement,
                      SwitchStatement, BlockStatement, CatchClauseParameter, TryResource, CatchClause,
                      SwitchStatementCase, ForControl, EnhancedForControl]
otherClasses = [BasicType, MemberReference, ReferenceType, SuperMemberReference, StatementExpression]

dfsSequence = []
for node in dfsAstNode(test):
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
            dfsSequence.append(node.case)
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
                pass
# bfs

# bfsSequence = []
#
# for node in bfsAstNode(test):
#     if any(isinstance(node, classWanted) for classWanted in classesWanted):
#         bfsSequence.append(node)

# dfsSequence
# bfsSequence
