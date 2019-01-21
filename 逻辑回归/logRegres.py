#coding = utf-8

import numpy as np
import matplotlib.pyplot as plt

'''
参考机器学习实战
'''

# 数据集及标签
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split() # 逐行读入并切分，每行的前两个值为X1，X2
        # print(lineArr)
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])]) # X0设为1.0，保存X1，X2
        labelMat.append(int(lineArr[2])) # 类别标签
    # print(dataMat)
    # print(labelMat)
    return dataMat,labelMat

# 定义sigmoid函数
def sigmoid(inX):
    return 1.0/(1+np.exp(-inX))

# 梯度下降算法，每次参数迭代时都需要遍历整个数据集
def gradAscent(dataMatIn,classLabels):
    dataMat = np.mat(dataMatIn)  # 转成矩阵
    #print(dataMat)
    labelMat = np.mat(classLabels).transpose() # 矩阵转置
    m,n = np.shape(dataMat)
    alpha = 0.001 # 步长
    maxCycles = 500 # 迭代次数
    weights = np.ones((n,1))
    for i in range(maxCycles):
        error = sigmoid(dataMat*weights) - labelMat
        weights = weights - alpha*dataMat.transpose()*error

    return weights

# 随机梯度下降算法的实现，对于数据量较多的情况下计算量小，但分类效果差,每次参数迭代时通过一个数据进行运算
def stocGradAscent0(dataMatIn,classLabels):
    dataMat = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(dataMat)

    weights = np.ones((n,1))
    alpha = 0.001
    maxCycles = 20
    for i in range(maxCycles):
        for j in range(m):
            error = sigmoid(dataMat[j]*weights) - labelMat[j]
            weights = weights - alpha*dataMat[j].transpose()*error

    return weights

# 改进后的随机梯度上升算法
# 从两个方面对随机梯度上升算法进行了改进,正确率确实提高了很多
# 改进一：对于学习率alpha采用非线性下降的方式使得每次都不一样
# 改进二：每次使用一个数据，但是每次随机的选取数据，选过的不在进行选择
def stocGradAscent1(dataMatIn,classLabels):
    dataMat = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(dataMat)

    weights = np.ones((n,1))
    maxCycles = 100
    for i in range(maxCycles):
        dataIndex = list(range(m))
        np.random.shuffle(dataIndex) # 打乱列表的排序
        for j in range(m):
            alpha = 4/(i+j+1) + 0.01
            randIndex = dataIndex[j]
            # print(dataMat[randIndex])
            # print(weights)
            # print(dataMat[randIndex]*weights)
            error = sigmoid(dataMat[randIndex]*weights) - labelMat[randIndex]
            weights = weights - alpha*dataMat[randIndex].transpose()*error

    return weights

# 绘制图像
def plotBestFit(weights):
    x0List = []
    x1List = []
    y0List = []
    y1List = []
    fr = open('testSet.txt','r')
    for line in fr.readlines():
        lineList = line.strip().split()
        if lineList[2] == '0':
            x0List.append(float(lineList[0]))
            y0List.append(float(lineList[1]))
        else:
            x1List.append(float(lineList[0]))
            y1List.append(float(lineList[1]))

    fig = plt.figure()
    ax = fig.add_subplot(111) # 349表示的是将画布分割成3行4列，图像画在从左到右从上到下的第9块。或者ax = fig.add_subplot(1,1,1)
    ax.scatter(x0List,y0List,s=10,c='red')
    ax.scatter(x1List,y1List,s=10,c='green')
    # plt.show()

    xList = []
    yList = []
    # [-3,3)步长为0.1
    x = np.arange(-3,3,0.1) # numpy中的arange非常类似range函数，只不过arange()返回的是array对象，而range()返回的是list
    # print(x)
    # print(x.shape)
    for i in range(len(x)):
        xList.append(x[i])
    y = (-weights[0]-weights[1]*x) / weights[2]
    # print(y)
    # print(y.shape)
    for j in np.arange(y.shape[1]):
        yList.append(y[0,j])
        # print(y[0,j])

    ax.plot(xList,yList)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split()
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
        #trainingLabels.append(lineArr)
    print(trainingSet)
    print(trainingLabels)

    trainWeights = stocGradAscent1(trainingSet,trainingLabels)
    print(trainWeights)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split()
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr),trainWeights)) != int(currLine[-1]):
            errorCount += 1

    errorRate = float(errorCount)/numTestVec
    print("the error rate of this test is:%f" % errorRate)
    return errorRate

def classifyVector(intX,weights):
    prob = sigmoid(np.sum(intX*weights))
    if prob>0.5:
        return 1.0
    else:
        return 0.0

def multiTest():
    numTests = 6
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iteration the average error rate is:%f" % (numTests, errorSum / float(numTests)))

if __name__ == "__main__":
    dataMatIn, classLabels = loadDataSet()
    # print(sigmoid(2))
    w = gradAscent(dataMatIn,classLabels)
    # w = stocGradAscent0(dataMatIn,classLabels)
    w = stocGradAscent1(dataMatIn,classLabels)
    print(w)
    # print(w.shape)
    # print(w[0].shape)
    plotBestFit(w)
    #multiTest()
