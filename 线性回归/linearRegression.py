#coding = utf-8

import numpy as np
import matplotlib.pyplot as plt

'''
线性回归：来自机器学习实战
'''

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split()) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curline = line.strip().split()
        for i in range(numFeat):
            lineArr.append(float(curline[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curline[-1]))

    return dataMat,labelMat

def standRegres(xArr,yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0: # 矩阵行列式|A|=0,则矩阵不可逆;np.linalg.det()：矩阵求行列式（标量）
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I*(xMat.T * yMat) # 矩阵A.I表示A的逆矩阵，矩阵A.T表示A的转置矩阵
    return ws # 回归系数w

# 线性回归拟合直线
def plotGrap():
    '''
    线性回归拟合直线
    '''
    dataMat, labelMat = loadDataSet('ex0.txt')
    xMat = np.mat(dataMat)
    yMat = np.mat(labelMat)
    ws = standRegres(dataMat, labelMat)
 
    yHat = xMat * ws

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # flatten()表示返回一个折叠成一维的数组。但是该函数只能适用于numpy对象，即array或者mat，普通的list列表是不行的
    ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0]) # xMat[:,1]表示取xMat的第一列，A[0],矩阵.A（等效于矩阵.getA()）变成了数组(Array)

    xcopy = xMat.copy()
    xcopy.sort(0)
    yHat = xcopy * ws
    ax.plot(xcopy[:,1],yHat)
    plt.show()

# 局部加权线性回归（Local Weighted Linear Regression，LWLR）
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m))) # 生成对角矩阵(方阵),对角线为1，其余全为0
    #print(weights)
    for i in range(m):
        diffMat = testPoint - xMat[i,:] # 计算样本点与预测值的距离
        #print(diffMat*diffMat.T)
        weights[i,i] = np.exp(diffMat*diffMat.T/(-2.0*k**2)) # 计算高斯核函数W
        #print(weights)
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0: # 判断是否可逆
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws

# 测试
def lwlrTest(testArr,xArr,yArr,k=1.0): #loops over all the data points and applies lwlr to each one
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)

    return yHat

def plotGrap2():
    '''
    局部加权线性回归
    Local Weighted Linear Regression，LWLR
    '''
    dataMat, labelMat = loadDataSet('ex0.txt')
    yHat = lwlrTest(dataMat, dataMat, labelMat, 0.01)
    #print(yHat)
    xMat = np.mat(dataMat)
    yMat = np.mat(labelMat)
    srtInd = xMat[:,1].argsort(0) # argsort函数返回的是数组值从小到大的索引值,0表示列，1表示行
    print(srtInd)      # xMat[:,1] shape(200,1)
    xSort = xMat[srtInd][:,0,:] # shape(200,1,2)降维
    print(xSort)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:,1],yHat[srtInd])
    ax.scatter(xMat[:,1].flatten().A[0],yMat.T.flatten().A[0],s=2,c='red')

    plt.show()

# 计算误差
def rssError(yArr,yHatArr):
    return ((yArr - yHatArr)**2).sum()

# 缩减系数之“岭”回归
def ridgeRegres(xMat,yMat,lam=0.2):
    '''
    当数据特征比样本点还多的时候,输入数据的矩阵X不是满秩矩阵,无法求逆矩阵
    针对非满秩矩阵无法求逆的问题，岭回归加入了λI，其中I是一个m*m的单位矩阵，其对角线元素为1，其余元素为0
    λ被称为惩罚项，是一个用户定义的数值。通过引用惩罚项λ限制了所有w的和，该过程能够减少不重要的参数，这在统计学中也被称为缩减。
    缩减之所以能够达到更好的预测效果，是因为它能够去掉不重要的参数，从而更好地理解数据
    '''
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1])*lam # np.eye()创造一个对角线为1，其余元素为0的m*m单位矩阵

    if np.linalg.det(denom) == 0.0:
        print("This matrix is singular,cannot do inverse")
        return

    #print(denom.I.shape)
    #print((xMat.T*yMat).shape)
    ws = denom.I * (xMat.T * yMat)
    return ws

# 测试
def ridgeTest(xArr,yArr):
    '''
    numpy.mean()函数功能：
    axis 不设置值，对 m*n 个数求均值，返回一个实数
    axis = 0：压缩行，对各列求均值，返回 1* n 矩阵
    axis =1 ：压缩列，对各行求均值，返回 m *1 矩阵
    '''
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    print(xMat.shape)
    print(yMat.shape)

    # 数据标准化（特征标准化处理），减去均值，除以方差
    yMean = np.mean(yMat,0)
    yMat = yMat - yMean
    xMean = np.mean(xMat,0)
    xVar = np.var(xMat,0) # 求方差
    xMat = (xMat - xMean) / xVar

    # 设置i的步长，确定lambda的变化范围为[0, 30]
    numTestPts = 30

    # 设置0矩阵wMat用于置放每一个样本，对8个特征每一次进行计算时的ws权值
    wMat = np.zeros((numTestPts,np.shape(xMat)[1]))
    print(wMat)
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,np.exp(i-10))
        wMat[i,:] = ws.T

    return wMat

def plotGrap3():
    '''
    缩减系数之“岭”回归
    '''
    dataMat,labelMat = loadDataSet('abalone.txt')
    ridgeWeights = ridgeTest(dataMat,labelMat)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()

# 定义标准化函数regularize()
def regularize(xMat):
    inMat = xMat.copy()
    inMean = np.mean(inMat,0)
    inVar = np.var(inMat,0)
    inMat = (inMat-inMean) / inVar
    return inMat

# 前向逐步线性回归,eps为迭代的步长，numIt为迭代次数
def stageWise(xArr, yArr, eps=0.01, numIt=100):
    '''
    数据标准化，使其分布满足0均值和单位方差
    在每轮迭代过程中：
	    设置当前最小误差lowestError为正无穷
	    对每个特征：
		    增大或减小：
			    改变一个系数得到一个新的W
			    计算新W下的误差
			    如果误差Error小于当前最小误差lowestError：设置Wbest等于当前的W
		    将W设置为新的Wbest
    '''
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat,0)
    yMat = yMat - yMean
    xMat = regularize(xMat)
    m,n = np.shape(xMat)
    returnMat = np.zeros((numIt,n))

    #为实现贪心算法，建立了两份ws的副本
    ws = np.zeros((n,1))
    wsTest = ws.copy()
    wsMax = ws.copy()

    for i in range(numIt):
        #print(ws.T)
        lowestError = float('inf')  # inf的意思是正无穷
        #print(lowestError)

        '''
        开始进行贪心算法，针对整个数据集迭代一次，即上面的for循环中的一次i,
        输出结果为returnMat矩阵中的一行，1x8的数组,
        迭代8个特征，因为此数据集中n=8
        '''
        for j in range(n):
            for sign in [-1,1]: # 两次循环，计算增加或者减少该特征对误差的影响
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                print('wsTest=',wsTest[j])
                yTest = xMat * wsTest
                print('yTest=',yTest)
                rssE = rssError(yMat.A, yTest.A) # 计算平方误差；矩阵.A表示的是将矩阵转成array
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
                    print('wsMax=',wsMax)

        ws = wsMax.copy()
        returnMat[i,:] = ws.T

    # 整合全部迭代次数numIt之后，输出returnMat矩阵，规格为[numIt * 8]
    return returnMat

def stageTest():
    xArr, yArr = loadDataSet('abalone.txt')
    st = stageWise(xArr, yArr, 0.001, 200)
    print(st)

if __name__ == "__main__":
    # plotGrap()
    #plotGrap2()
    #plotGrap3()
    stageTest()
