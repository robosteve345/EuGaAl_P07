#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 10:04:16 2023

@author: stevengebel
"""

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import fabio as fabio
import os.path
import re
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.pyplot import *
from scipy.ndimage import center_of_mass

img = "al2ga2_0KL_0.5A.img"

obj = fabio.open(img)
NX = obj.header["NX"]
NY = obj.header["NY"]
data = obj.data
resolution = 0.5
lat_a = 4.218
lat_c = 10.907
qmaxa = lat_a/(resolution)
qmaxc = lat_c/(resolution)

qpixelx = 2*qmaxa/NX
qpixely = 2*qmaxc/NY
qxrange = np.arange(-qpixelx*(NX/2.), qpixelx*(NX/2.), qpixelx)
qyrange = np.arange(-qpixely*(NY/2.), qpixely*(NY/2.), qpixely)
qx, qy = np.meshgrid(qxrange, qyrange)

H = -5
ind_H = int(np.abs(H/qpixelx))# int(NX - np.abs(H/qpixelx))
print("ind_H={}".format(ind_H))
# IMSHOW MAP
plt.title("{}".format(img))
plt.imshow(obj.data, cmap='viridis', norm = LogNorm(vmin=1, vmax=np.max(data)))
# INTENSITY MAP
fig,ax = plt.subplots(figsize=(5, 3))
# ax.vlines(x=H, ymin=-14, ymax=14, ls='-', lw=1.5, color='black')
im = ax.pcolormesh(qx, qy, data, cmap='inferno',
                   norm= LogNorm(vmin = 1, vmax = np.max(data))
                   )
ax.grid(True, which='both',axis='both',linestyle='-', color='black', lw=0.10)
#ax.set_yticks(np.arange(-ylim,ylim,1))
#ax.set_xticks(np.arange(xlim,0,1))
ax.set_xlim(1,8)
ax.set_ylim(-10,10)
#ax.set_ylim([-1, 10])
#ax.set_xlim([2, 8]) # qpixelx*(NX/2.)
ax.set_xlabel('H (rlu)')
ax.set_ylabel('L (rlu)')
ax.set_title("{}".format(img))

fig.colorbar(im, ax=ax)
plt.savefig("0p5_110_0KL_0.5A.jpg", dpi=300)
# plt.figure()

# # LINECUTS
# figure(figsize=(15, 6), dpi=100)
# for i in range(-62,-59,1):
#     I = data[:, ind_H+i]
#     print("max(I)={}, ind={}, i={}".format(np.max(I), int(ind_H+i), i))
#     plt.plot(qyrange, I, label='ind_H={}'.format(int(ind_H+i)), marker='.', 
#              lw=0.5)
#     plt.xlabel('L (rlu)')
#     plt.ylabel('Intensity I')
#     plt.title('H={}'.format(H))
#     plt.xlim(-ylim, ylim)
#     plt.legend()
#     plt.show()
plt.show()

