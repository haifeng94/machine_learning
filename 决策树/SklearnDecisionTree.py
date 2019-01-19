# -*- coding: UTF-8 -*-

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.externals.six import StringIO
from sklearn import tree
import pandas as pd
import numpy as np
import pydotplus

if __name__ == '__main__':
	#加载数据
	with open('lense.txt', 'r') as fr:
		# 处理文件
		lenses = [inst.strip().split('\t') for inst in fr.readlines()]
		print('lenses:',lenses)
	# 提取每组数据的类别，保存在列表里
	lenses_target = []
	for each in lenses:
		lenses_target.append(each[-1])
	print('lenses_target:',lenses_target)

	# 特征标签
	lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
	# 保存lenses数据的临时列表
	lenses_list = []
	# 保存lenses数据的字典，用于生成pandas
	lenses_dict = {}
	# 提取信息，生成字典
	for each_label in lensesLabels:
		for each in lenses:
			lenses_list.append(each[lensesLabels.index(each_label)])
		lenses_dict[each_label] = lenses_list
		lenses_list = []
	# 打印字典信息
	print('lenses_dict:',lenses_dict)
	# 生成pandas.DataFrame
	lenses_pd = pd.DataFrame(lenses_dict)
	# 打印pandas.DataFrame
	print('lenses_pd:',lenses_pd)
	# 创建LabelEncoder()对象，用于序列化
	le = LabelEncoder()
	# 序列化
	for col in lenses_pd.columns:
		lenses_pd[col] = le.fit_transform(lenses_pd[col])
	# 打印编码信息
	print('lenses_pd:',lenses_pd)

	# 创建DecisionTreeClassifier()类
	clf = tree.DecisionTreeClassifier(max_depth = 4)
	# 使用数据，构建决策树
	clf = clf.fit(lenses_pd.values.tolist(), lenses_target)
	print('clf:',clf)

	dot_data = StringIO()
	# 绘制决策树
	tree.export_graphviz(clf, out_file = dot_data,
						feature_names = lenses_pd.keys(),
						class_names = clf.classes_,
						filled=True, rounded=True,
						special_characters=True)
	graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
	# 保存绘制好的决策树，以PDF的形式存储。
	graph.write_pdf("tree.pdf")
	# 预测
	print(clf.predict([[1,1,1,0]]))
