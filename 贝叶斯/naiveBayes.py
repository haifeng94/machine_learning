#coding = utf-8

import numpy as np

'''参考机器学习实战'''

"""
1、给定x，可以通过直接建模P(c|x)来预测c，这样得到的是“判别式模型”
2、先对联合分布P(x,c)建模，然后再有此获得P(c|x),这样得到的是“生成模型”
决策树、BP神经网络、支持向量机等，都可以纳入判别式模型的范畴
对生成模型来说，必然考虑：P(c|x) = P(x,c)/P(x) = P(x|c)*P(c)/P(x)
其中P(c)是“先验概率”,P(x|c)是样本x对于类标记c的类条件概率，或称为“似然”,P(x)是用于归一化的“证据”因子
给定样本x，证据因子P(x)与类标记无关(对所有的类标记均相同)，因此估计P(c|x)的问题就转化为如何基于训练数据D来估计先验概率P(c)和似然P(x|c)
"""

#数据集
def loadDataSet():
    #文本转化为词向量
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    #1表示侮辱类，0表示非侮辱类
    classVec = [0,1,0,1,0,1]
    return postingList, classVec

#创建包含所有词向量的list(去重后的)
def createVocabList(dataSet):
    #创建空集合，
    vocabSet = set([])
    for document in dataSet:
        #两集合的并集
        vocabSet = vocabSet | set(document)
    #print(vocabSet)
    return list(vocabSet)

#将每个数据集转化为数据形式(0表示"没有",1表示"有")，vocabList,inputSet分别表示词向量表以及某个example
def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
    #print(returnVec)
    return returnVec

#高级词袋模型，判断词出现次数。词袋模型主要修改上面的第三个步骤，因为有的词可能出现多次，所以在单个样本映射到词库的时候需要多次统计。
def bagOfWordsVecMN(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

#条件概率以及类标签概率的计算。trainMatrix为经过setOfWords2Vec()处理的数据集，trainCategory为类标签classVec
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    #计算某个类发生的概率(文档中属于侮辱类的概率，等于1才能算，0是非侮辱类)
    pAbusive = np.sum(trainCategory) / float(numTrainDocs) # 先验概率
    print(pAbusive)
    #初始样本个数为1，防止条件概率为0，影响结果
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0

    #遍历每个数据集
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            print(np.sum(trainMatrix[i]))
            print(p0Num)
            p0Denom += sum(trainMatrix[i])
            print(p0Denom)

    #计算类标签为1时的其它属性发生的条件概率
    p1Vect = np.log(p1Num / p1Denom)
    print(p1Vect)
    #计算标签为0时的其它属性发生的条件概率
    p0Vect = np.log(p0Num / p0Denom)
    print(p0Vect)

    return p0Vect,p1Vect,pAbusive

#给定词向量,判断类别
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    '''
    vec2Classify表示待分类的样本在词库中的映射集合
    p0Vec表示条件概率P(wi|c=0)
    p1Vec表示条件概率P(wi|c=1)
    pClass1表示类标签为1时的概率P(c=1)
    其中p1和p0表示的是lnp(w1|c=1)p(w2|c=1)...p(wn|c=1)∗p(c=1)
    和lnp(w1|c=0)p(w2|c=0)...p(wn|c=0)∗p(c=0)
    取对数是因为防止p(w_1|c=1)p(w_2|c=1)p(w_3|c=1)…p(w_n|c=1)多个小于1的数相乘结果值下溢。
    '''
    p1 = np.sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = np.sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    print(p1)
    print(p0)
    if p1 > p0:
        return 1
    else:
        return 0

#测试函数
def testingNB():
    #step1：加载数据集和类标号
    listOPosts, listClasses = loadDataSet()
    #step2：创建词库
    myVocabList = createVocabList(listOPosts)
    #step3：计算每个样本在词库中的出现情况
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    #step4：调用第四步函数，计算条件概率
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))
    #step5
    #测试1
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    result1 = classifyNB(thisDoc,p0V,p1V,pAb)
    print("result1=:",result1)

    #测试2
    testEntry2 = ['stupid', 'garbage']
    thisDoc2 = np.array(setOfWords2Vec(myVocabList, testEntry2))
    result2 = classifyNB(thisDoc2, p0V, p1V, pAb)
    print("result2=:",result2)

#文本解析，把大写转小写，且去掉长度小于2的字符
def textParse(bigString):
    listOfTokens = bigString.split()
    print(listOfTokens)
    #import re
    #listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []
    classList = []
    fullTest = []
    for i in range(1,26):
        # 垃圾邮件
        #wordList = textParse(open('email/spam/%d.txt' %i).read())
        wordList = textParse(open('E:/Python27 Document/ML/贝叶斯/email/spam/%d.txt' % i, "rb").read().decode('GBK', 'ignore'))
        docList.append(wordList) # 词向量
        fullTest.extend(wordList) # 文档向量
        classList.append(1)
        # 非垃圾邮件
        #wordList = textParse(open('email/ham/%d.txt' %i).read())
        wordList = textParse(open('email/ham/%d.txt' % i, "rb").read().decode('GBK', 'ignore'))
        docList.append(wordList)  # 词向量
        fullTest.extend(wordList)  # 文档向量
        classList.append(0)

    # 创建词列表
    vocabList = createVocabList(docList)
    trainingSet = list(range(50)) # spam+ham=50 eamils
    testSet = []
    # 随机选择10封作为测试集
    for i in range(10):
        randIndex = int(np.random.uniform(0,len(trainingSet))) # 生成随机数
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex])) # 0和1处理
        trainClasses.append(classList[docIndex]) #类别
    p0V,p1V,pSpam = trainNB0(np.array(trainMat),np.array(trainClasses))

    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is:', float(errorCount) / len(testSet))

if __name__ == "__main__":
    #testingNB()
    spamTest()
