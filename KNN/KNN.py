# coding = utf-8

import numpy as np
import operator
import matplotlib.pyplot as plt

# 数据集及其标签
def createDataSet():
    group = np.array([[1.0,2.0],[1.2,0.1],[0.1,1.4],[0.3,3.5]])
    labels = ['A','A','B','B']
    return group,labels

def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines) # 读出数据行数
    returnMat = np.zeros((numberOfLines,3)) # 创建返回矩阵
    classLabelVector = []
    index = 0

    for line in arrayOfLines:
        listFromLine = line.strip().split('\t')
        returnMat[index,:] = listFromLine[0:3] # 选取前3个元素（特征）存储在返回矩阵中
        classLabelVector.append(listFromLine[-1])
        #classLabelVector.append(int(listFromLine[-1])) #label信息存储在classLabelVector
        index += 1

    return returnMat,classLabelVector

# 归一化特征值,归一化公式 :（当前值-最小值）/ranges     (其中，ranges等于最大值减去最小值）
def autoNorm(dataSet):
    minVals = dataSet.min(0) # 存放每列最小值，参数0使得可以从列中选取最小值，而不是当前行
    maxVals = dataSet.max(0) # 存放每列最大值,axis=0表示列，axis=1表示行
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet)) # 初始化归一化矩阵为读取的dataSet
    m = dataSet.shape[0]  # dataSet的行数
    normDataSet = dataSet - np.tile(minVals,(m,1)) # 当前值-最小值
    normDataSet = normDataSet / np.tile(ranges,(m,1))

    return normDataSet,ranges,minVals

# knn算法进行分类
def knnClassify(inX,dataSet,label,k):
    dataSetSize = dataSet.shape[0] # shape读取数据矩阵第一维度的长度
    # 计算欧式距离
    #print(np.tile(inX,(dataSetSize,1)))  # tile(A，rep)重复A的各个维度，A: Array类的都可以，rep：A沿着各个维度重复的次数
    diffMat = np.tile(inX,(dataSetSize,1)) - dataSet # tile重复数组inX，有dataSet行 1个dataSet列，减法计算差值
    #print(diffMat)
    sqDiffMat = diffMat ** 2  # **是幂运算的意思，这里用的欧式距离
    #print(sqDiffMat)
    sqDisttances = sqDiffMat.sum(axis=1) # 普通sum默认参数为axis=0为普通相加，axis=1为一行的行向量相加
    #print(sqDisttances)
    distances = sqDisttances ** 0.5
    #print(distances)
    sortedDistIndicies = np.argsort(distances) # argsort返回数值从小到大的索引值（数组索引0,1,2,3）
    #print(sortedDistIndicies)

    # 选择距离最小的k个点
    classCount = {}
    for i in range(k):
        voteLabel = label[sortedDistIndicies[i]] # 根据排序结果的索引值返回靠近的前k个标签
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1 #各个标签出现频率
    print(classCount)

    # reverse=True表示降序排序，key关键字排序itemgetter（1）按照第一维度排序(0,1,2,3)
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)

    return sortedClassCount[0][0]

#测试约会网站分类结果
def datingClassTest():
    hoRatio = 0.10 #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt') # datingTestSet与datingTestSet2其实是一样的，其实datingTestSet2标签被序列化成1,2,3
    normDataSet, ranges, minVals = autoNorm(datingDataMat)
    m = normDataSet.shape[0]
    testNum = int(hoRatio * m) #选取10%测试
    print(testNum)
    errorCount = 0.0
    for i in range(testNum):
        classifierResult = knnClassify(normDataSet[i,:],normDataSet[testNum:m,:],datingLabels[testNum:m],3)
        print(classifierResult)
        print(datingLabels[i])
        print("the classifier came back with: %s, the real answer is: %s" % (classifierResult, datingLabels[i]))  # datingTestSet
        #print("the classifier came back with: %d, the real answer is: %d" % (int(classifierResult), int(datingLabels[i]))) # datingTestSet2
        if classifierResult != datingLabels[i]:
            errorCount += 1.0

    print("the total error rate is: %f" % (errorCount / float(testNum)))

if __name__ == "__main__":
    '''
    dataSet,label = createDataSet()
    print(dataSet)
    print(label)
    inX = np.array([1.1,0.3])
    print(inX)
    K = 3
    output = knnClassify(inX,dataSet,label,K)
    print("测试数据为:%s,分类结果为：%s" %(inX,output))
    '''
    datingClassTest()