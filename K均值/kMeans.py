# coding = utf-8

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

'''
K-Means:
当任意一个点的簇分配结果发生改变时：
        对数据集中的每个点：
                对每个质心：
                计算质心与数据点之间的距离
         将数据点分配到距离其最近的簇
    对每一个簇，计算簇中所有点的均值并将均值作为质心。
'''

#load data
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines(): #for each line
        curLine = line.strip().split()
        # python3不适用：fltLine = map(float,curLine) 修改为：
        fltLine = list(map(float,curLine)) # 将每行映射成浮点数，python3返回值改变,python3返回的是object对象，所以需要转list
        dataMat.append(fltLine)

    return dataMat

#distance function
def distEclud(vecA,vecB):
    return np.sqrt(np.sum(np.power(vecA-vecB,2)))  # 计算A，B两点的欧式距离，其中，np.power(a,b)表示a**b,可以用np.linalg.norm(vec1 - vec2)

#initialize K points randomly
def randCent(dataSet, k):
    n = np.shape(dataSet)[1]  # 列的数量
    centroids = np.mat(np.zeros((k,n)))  # 创建k个质心矩阵
    for j in range(n):  # 创建随机簇质心，并且在每一维的边界内
        minJ = min(dataSet[:,j]) # 最小值
        rangeJ = float(max(dataSet[:,j]) - minJ)  # 范围 = 最大值 - 最小值
        centroids[:,j] = np.mat(minJ + rangeJ * np.random.rand(k,1)) # np.random.rand(k,1)表示随机生成[0,1)范围内的shape(k,1)的array(数组）

    return centroids

#K-均值算法
def kMeans(dataSet,k,distMeas=distEclud,createCent=randCent):
    '''
    dataSet: 数据集
    k: num of cluster(簇数)
    distMeas=distEclud: 距离函数
    createCent=randCent：随机初始化簇类中心函数
    '''

    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m,2)))  #创建一个与 dataSet 行数一样，但是有两列的矩阵，用来保存簇分配结果(2 cols for index and error)
    centroids = createCent(dataSet,k)  #创建质心，随机k个质心
    clusterChanged = True

    while clusterChanged:
        clusterChanged = False
        for i in range(m):  #循环每一个数据点并分配到最近的质心中去
            minDist = float('inf')
            minIndex = -1  # init
            for j in range(k): #for every k centers，find the nearest center
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:  #if distance is shorter than minDist, 更新 minDist（最小距离）和最小质心的 index（索引）
                    minDist = distJI
                    minIndex = j  #update distance and index(类别)

            if clusterAssment[i,0] != minIndex:  #此处判断数据点所属类别与之前是否相同（是否变化，只要有一个点变化就重设为True，再次迭代）
                clusterChanged = True

            clusterAssment[i,:] = minIndex, minDist**2

        for cent in range(k):  # 更新质心
            ptsInClust = dataSet[np.nonzero(clusterAssment[:,0]==cent)[0]]  # 获取该簇中的所有点
            centroids[cent,:] = np.mean(ptsInClust,axis=0) # 将质心修改为簇中所有点的平均值，mean 就是求平均值的

    return centroids,clusterAssment

#二分K-均值聚类
def biKmeans(dataSet,k,distMeas=distEclud):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m,2)))  # 保存每个数据点的簇分配结果和平方误差
    centroid0 = np.mean(dataSet,axis=0).tolist()[0]  # 质心初始化为所有数据点的均值, numpy tolist()将数组或者矩阵转换成列表
    centList = [centroid0]  #初始化只有 1 个质心的 list

    for j in range(m):   #计算所有数据点到初始质心的距离平方误差
        clusterAssment[j,1] = distMeas(np.mat(centroid0),dataSet[j,:]) ** 2

    while len(centList) < k:
        lowestSSE = float('inf')  # init SSE
        for i in range(len(centList)): # for every centroid
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:,0].A == i)[0],:]  # 获取当前簇i下的所有数据点
            centroidMat,splitClustAss = kMeans(ptsInCurrCluster,2,distMeas) # 将当前簇 i 进行二分kMeans处理
            sseSplit = np.sum(splitClustAss[:,1])  # 将二分kMeans结果中的平方和的距离进行求和
            sseNotSplit = np.sum(clusterAssment[np.nonzero(clusterAssment[:,0].A != i)[0], 1]) # 将未参与二分 kMeans 分配结果中的平方和的距离进行求和
            print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:  #总的（未拆分和已拆分）误差和越小，越相似，效果越优化，划分的结果更好
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit

        # 找出最好的簇分配结果
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 1)[0], 0] = len(centList) # 调用二分 kMeans 的结果，默认簇是 0,1. 当然也可以改成其它的数字
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 0)[0], 0] = bestCentToSplit # 更新为最佳质心
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]  # 更新原质心位置 list 中的第 i 个质心为使用二分 kMeans 后 bestNewCents 的第一个质心
        centList.append(bestNewCents[1,:].tolist()[0])  # 添加 bestNewCents 的第二个质心
        clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0],:] = bestClustAss # 重新分配最好簇下的数据（质心）以及SSE

    return np.mat(centList),clusterAssment

#distance calc function：结合两个点经纬度（用角度做单位），返回地球表面两点之间距离
def distSLC(vecA,vecB): # Spherical Law of Cosines, 余弦球面定理
    a = np.sin(vecA[0,1] * np.pi / 180) * np.sin(vecB[0,1] * np.pi / 180)
    b = np.cos(vecA[0,1] * np.pi / 180) * np.cos(vecB[0,1] * np.pi / 180) * np.cos(np.pi * (vecB[0,0] - vecA[0,0]) / 180)
    return np.arccos(a + b) * 6371.0  # 6371.0为地球半径

#draw function
def clusterClubs(numClust=5): # 参数numClust，希望得到的簇数目
    datList = []
    for line in open('places.txt').readlines(): # 获取地图数据
        lineArr = line.strip().split()
        datList.append([float(lineArr[-1]), float(lineArr[-2])])  # 逐个获取第四列和第五列的经纬度信息

    dataMat = np.mat(datList)
    myCentroids, clustAssing = biKmeans(dataMat, numClust, distMeas=distSLC)

    # draw
    fig = plt.figure()
    rect = [0.1,0.1,0.8,0.8] # 创建矩形
    # 创建不同标记图案
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[],yticks=[]) # {'yticks': [], 'xticks': []}
    ax0 = fig.add_axes(rect, label = 'ax0', **axprops)
    imgP = plt.imread('Portland.png') # 导入地图
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label = 'ax1', frameon = False)

    for i in range(numClust):
        ptsInCurrCluster = dataMat[np.nonzero(clustAssing[:,0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s = 90)

    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s = 300)
    plt.show()

if __name__ == "__main__":
    clusterClubs()
