# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 22:25:50 2018
training decision trees
@author: JimQ1

get data of flowers from iris database

"""
import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris()

test_idx = [0, 50, 100]

"""
training data
"""

train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

"""
test data
"""

test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

"""
train classifier on training data
"""

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

"""
will print out [0 1 2]
"""

print (test_target)

"""
what the tree predicts
"""

print (clf.predict(test_data))

"""
visualizing the tree

renders and outputs data onto graph

"""
from sklearn.externals.six import StringIO
import pydotplus

dot_data = StringIO()
tree.export_graphviz(clf,
                     out_file=dot_data,
                     feature_names=iris.feature_names,
                     class_names=iris.target_names,
                     filled=True, rounded=True,
                     impurity=False
                     )
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

graph.write_pdf("iris.pdf")

print (test_data[1], test_target[0])

print (iris.feature_names, iris.target_names)