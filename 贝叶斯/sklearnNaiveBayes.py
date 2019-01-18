#coding = utf-8

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

'''利用sklearn模块进行贝叶斯学习'''

#高斯朴素贝叶斯
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
clf = GaussianNB().fit(X,Y)
print(clf.predict([[-0.8,-1]]))

"""
partial_fit说明：增量的训练一批样本
这种方法被称为连续几次在不同的数据集，从而实现核心和在线学习，这是特别有用的，当数据集很大的时候，不适合在内存中运算
该方法具有一定的性能和数值稳定性的开销，因此最好是作用在尽可能大的数据块（只要符合内存的预算开销）
"""
clf_pf = GaussianNB().partial_fit(X,Y,np.unique(Y))
print(clf_pf.predict([[-0.8,-1]]))

############################################################################################################

#多项式朴素贝叶斯
#numpy.random.randint(low, high=None, size=None, dtype='l')
x = np.random.randint(5,size=(6,100))
#print(x)
y = np.array([1, 2, 3, 4, 5, 6])
clf = MultinomialNB().fit(x,y)
t  = x[2:3]
#print(t)
print(clf.predict(t))
print(clf.predict(np.random.randint(5,size=(1,100))))

#############################################################################################################

#伯努利朴素贝叶斯
X = np.random.randint(2, size=(6, 100))
Y = np.array([1, 2, 3, 4, 4, 5])
clf = BernoulliNB().fit(X,Y)
print(clf.predict(X[2:3]))
