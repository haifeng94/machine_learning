# coding = utf-8

import numpy as np
import copy
import matplotlib.pyplot as plt

'''
支持向量机：来自机器学习实战
'''

def loadDataSet(filename):
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))

    return dataMat,labelMat

def selectJrand(i,m): # i表示alpha的下标，m表示alpha的总数
    j = i
    while j==i:
        j = int(np.random.uniform(0,m)) # 简化版SMO，alpha随机选择
    return j

# 辅助函数，用于调整alpha范围
def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

# SMO simple algorithm
def smoSimple(dataMatIn,classLabels,C,toler,maxIter):# toler表示容错率 常数C
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m,n = np.shape(dataMatrix)
    b = 0
    alphas = np.mat(np.zeros((m,1)))
    iter = 0
    while iter < maxIter:
        alphaPairsChanged = 0  # 标记alpha是否被优化
        for i in range(m):
            # fXi是预测的类别
            fXi = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b
            # Ei表示误差
            Ei = fXi - float(labelMat[i]) # 预测结果和真实结果比对，计算误差
            # 对alpha进行优化，同时检查alpha的值满足两个条件：if
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m) # 随机选择第二个alpha
                fXj = float(np.multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                # 分配内存 稍后比较误差
                #alphaIold = alphas[i].copy()
                #alphaJold = alphas[j].copy()
                alphaIold = copy.deepcopy(alphas[i])
                alphaJold = copy.deepcopy(alphas[j])
                # 计算L H用于将alpha[j]调整到0—C之间
                if labelMat[i] != labelMat[j]:
                    L = max(0,alphas[j] - alphas[i])
                    H = min(C,C + alphas[j] + alphas[i])
                else:
                    L = max(0,alphas[j] + alphas[i] - C)
                    H = min(C,alphas[j] + alphas[i])
                if L == H:
                    print('L==H')
                    continue
                # eta为alpha[j]的最优修改量
                eta = 2.0 * dataMatrix[i,:] * dataMatrix[j,:].T - dataMatrix[i,:] * dataMatrix[i,:].T - dataMatrix[j,:] * dataMatrix[j,:].T
                if eta >= 0:
                    print("eta>=0")
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                # 检查alpha[j]是否有轻微改变
                if abs(alphas[j] - alphaJold) < 0.00001:
                    print("j not moving enough")
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j]) # update i by the same amount as j , the update is in the oppostie direction
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i,:] * dataMatrix[i,:].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i,:] * dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaJold) * dataMatrix[i,:] * dataMatrix[j,:].T - labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j,:] * dataMatrix[j,:].T
                if alphas[i] > 0 and alphas[i] < C:
                    b = b1
                elif alphas[j] > 0 and alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print("iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
        if alphaPairsChanged == 0:
            iter += 1
        else:
            iter = 0
        print("iteration number: %d" % iter)
    return b, alphas

def showClassifer(dataMat,labelMat):
    # 绘制样本点
    data_plus = [] # 正样本
    data_minus = [] # 负样本
    for i in range(len(dataMat)):
        if labelMat[i] > 0:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    # 转换为numpy矩阵
    data_plus_np = np.array(data_plus)
    data_minus_np = np.array(data_minus)
    # 正样本散点图
    plt.scatter(np.transpose(data_plus_np)[0],np.transpose(data_plus_np)[1],s=30,alpha=0.7)
    # 负样本散点图
    plt.scatter(np.transpose(data_minus_np)[0], np.transpose(data_minus_np)[1], s=30, alpha=0.7)
    plt.show()

if __name__ == "__main__":
    dataMat,labelMat = loadDataSet('testSet.txt')
    print(dataMat)
    print(labelMat)
    showClassifer(dataMat,labelMat)
