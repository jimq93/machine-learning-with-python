# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 2018
training feature sets
@author: JimQ1

doges

"""
import numpy as np
import matplotlib.pyplot as plt

greyhounds = 500
labs = 500

"""
random set of doggos

"""

grey_height = 28 + 4 * np.random.randn(greyhounds)
lab_height = 24 + 4 * np.random.randn(labs)

"""
plot height of doggos and compare

"""

plt.hist([grey_height, lab_height], stacked=True, color=['r','b'])
plt.show()