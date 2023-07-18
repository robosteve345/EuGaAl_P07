"""
Created on Thu Apr  6 21:36:28 2023

@author: stevengebel
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
plt.rcParams["figure.autolayout"] = True
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1.50
A = 6  # Want figures to be A6
plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)])

#############################################
"""Plot1: x vs. T_CDW"""
#############################################
# x = [0, 0.5, 0.58, 0.71, 0.9, 1.0]
# y = [0, 52, 25, 0, 72, 140]
# plt.xlabel(r"\textbf{T$_{CDW}$}", fontsize=10)
# plt.ylabel(r"\textbf{x}", fontsize=10)
# plt.scatter(x, y, c='tab:blue', marker='x')
# plt.tick_params(bottom=True, top=True, left=True, right=True)
# plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
# plt.tick_params(axis='both', labelcolor='k', labelsize=10,
#                 size=5, direction='in', length=20, width=1.5)
# plt.show()
# # plt.savefig("x.jpg", dpi=300)