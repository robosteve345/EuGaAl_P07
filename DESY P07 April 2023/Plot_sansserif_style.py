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

#############################################
"""Plot2: I vs. T"""
#############################################
x90, y90 = np.loadtxt("gaal_0p9.txt", unpack=True, usecols=(0, 1), skiprows=0)
x50, y50 = np.loadtxt("ga2al2.txt", unpack=True, usecols=(0, 1), skiprows=0)
y50 = (y50 - 6e5)
y50 = (y50)/np.max(y50)
y90 = (y90 - 7.4e4)
y90 = (y90)/np.max(y90)
plt.xlabel(r"\textbf{Temperature T(K)}", fontsize=21)
plt.ylabel(r"\textbf{Intensity I (arb. units)}", fontsize=21)
"""x=0.90"""
plt.text(s=r"\textbf{SC-XRD}", x=44, y=0.90, fontsize=21)
plt.text(s=r"\textbf{Eu(Ga$_{0.1}$Al$_{0.9}$)$_4$}", x=44, y=0.80, fontsize=21)

#plt.vlines(x=[20, 15.6, 11.5, 53.5], ymin=[0,0,0,0],
#           ymax=[0.98,0.237,0.30,0.051], colors=np.array(['tab:blue','tab:blue','tab:blue','tab:orange']))
# plt.text(x=12, y=0.18, s=r'\textbf{II}', fontsize=21)
# plt.text(x=17, y=0.18, s=r'\textbf{I}', fontsize=21)
# plt.text(x=24, y=0.4, s=r'\textbf{CDW only}', fontsize=21)
plt.xlim(5,75)
plt.ylim(0,1.1)
# plt.fill_between(x= x50[0:13], y1= y50[0:13], color= "b", alpha= 0.5)
# plt.fill_between(x= x50[12:21], y1= y50[12:21], color= "b", alpha= 0.4)
# plt.fill_between(x= x50[20:30], y1= y50[20:30], color= "b", alpha= 0.3)
# plt.fill_between(x= x50[29:-35], y1= y50[29:-35], color= "orange", alpha= 0.2)
plt.plot(x90, y90, marker='.', lw=0.5, ms=11, c='k', label='CDW Order parameter')
plt.tick_params(bottom=True, top=True, left=True, right=True)
plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
plt.tick_params(axis='both', labelcolor='k', labelsize=21,
                size=5, direction='in', length=20, width=1.5)
plt.legend(fontsize=20, loc='lower left')
plt.savefig("CDW0.9.jpg", dpi=300)


"""x=0.50"""
# plt.text(s=r"\textbf{SC-XRD}", x=42, y=0.80, fontsize=21)
# plt.text(s=r"\textbf{EuGa$_2$Al$_2$}", x=42, y=0.70, fontsize=21)
#
# plt.vlines(x=[20, 15.6, 11.5, 53.5], ymin=[0,0,0,0],
#            ymax=[0.98,0.237,0.30,0.051], colors=np.array(['tab:blue','tab:blue','tab:blue','tab:orange']))
# plt.text(x=7, y=0.18, s=r'\textbf{III}', fontsize=21)
# plt.text(x=12, y=0.18, s=r'\textbf{II}', fontsize=21)
# plt.text(x=17, y=0.18, s=r'\textbf{I}', fontsize=21)
# plt.text(x=24, y=0.4, s=r'\textbf{CDW only}', fontsize=21)
# plt.xlim(5.6,60)
# plt.ylim(0,1.1)
# plt.fill_between(x= x50[0:13], y1= y50[0:13], color= "b", alpha= 0.5)
# plt.fill_between(x= x50[12:21], y1= y50[12:21], color= "b", alpha= 0.4)
# plt.fill_between(x= x50[20:30], y1= y50[20:30], color= "b", alpha= 0.3)
# plt.fill_between(x= x50[29:-35], y1= y50[29:-35], color= "orange", alpha= 0.2)
# plt.plot(x50, y50, marker='.', lw=0.5, ms=11, c='k', label='CDW Order parameter')
# plt.tick_params(bottom=True, top=True, left=True, right=True)
# plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
# plt.tick_params(axis='both', labelcolor='k', labelsize=21,
#                 size=5, direction='in', length=20, width=1.5)
# plt.legend(fontsize=20, loc='upper right')
# plt.savefig("CDW1.jpg", dpi=300)
plt.show()

