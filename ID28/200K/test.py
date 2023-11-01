#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Created on Wed Oct 25 13:08:49 2023

#@author: stevengebel
import numpy as np
import matplotlib.pyplot as plt

x, y = np.linspace(-1, 1, 1000), np.linspace(-1, 1, 1000)
X2d, Y2d = np.meshgrid(x, y)
X2d, Y2d = X2d - np.random.uniform(-np.max(x), np.max(x))*np.ones(len(x), len(x)),\
            Y2d - np.random.uniform(-np.max(x), np.max(x))*np.ones(len(x), len(x))
sigma = 20
Z = np.exp(- (X2d**2 + Y2D**2) ) + 1e-1 * np.random.randn(len(x), len(x))
plt.imshow(Z)
plt.colorbar()
plt.show()