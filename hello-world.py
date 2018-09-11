# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 21:29:33 2018
machine learning 101
use anaconda with spyder 
on python 3.6
@author: JimQ1
"""

from sklearn import tree

"""
0 = bumpy
1 = smooth
"""

features = [
        [140, 1],[130, 1],
        [150, 0],[170, 0]
]

"""
0 = apple
1 = orange
"""
labels = [
        0,0,
        1,1
        ]
"""
tree classifier crux of machine learning
"""

clf = tree.DecisionTreeClassifier()

"""
tree classifier takes the above qualities and it's parameters
'fit' takes the fine details 

"""

clf = clf.fit(features, labels)

"""
classifier will print out a predictions
[0] being an apple
[1] being an orange from above
"""
print (clf.predict([[150, 0]]))

