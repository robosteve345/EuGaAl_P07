#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 10:04:16 2023

@author: stevengebel
"""

# fabians code

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import fabio as fabio
import os.path
from matplotlib.colors import LogNorm

############################################################
# INPUT
img = "0p5_159_H0L_0.5A_1.img"
resolution = 0.5
lat_a = 4.3262
lat_c = 10.996
############################################################

obj = fabio.open(img)
data = obj.data
NX = obj.header["NX"]
NY = obj.header["NY"]
qmaxa = lat_a/(resolution)
qmaxc = lat_c/(resolution)

#  Momentum resolution in x and y
qpixelx, qpixely = 2*qmaxa/NX, 2*qmaxc/NY
print("qpixelx = {}, qpixely = {}".format(qpixelx, qpixely))

qxrange = np.arange(-qpixelx*(NX/2.), qpixelx*(NX/2.), qpixelx)
qyrange = np.arange(-qpixely*(NY/2.), qpixely*(NY/2.), qpixely)
QX2d, QY2d = np.meshgrid(qxrange, qyrange)
xlim_pos, xlim_neg, ylim_pos, ylim_neg = int(qmaxa), -int(qmaxa), int(qmaxc), -int(qmaxc)

# LINECUTS
fig = plt.figure(figsize=(10, 4))
plt.title("{}".format(img))
indices = np.arange(0, int(2*qmaxa + 1), 1) * int(1/qpixelx)
for i in indices:
    I = data[:, i]
    plt.plot(qyrange, I , label='ind_H={}'.format(i), lw=0.5)
    plt.xlabel('L (r.l.u.)')
    plt.ylabel('Intensity I')
    plt.xlim(xlim_neg, xlim_pos) 
    plt.legend()
#     plt.savefig("0p5_001_H0L_0.5A_Linecuts_H={}.jpg".format(H), dpi=300)

# INTENSITY MAP
plt.title("{}".format(img))
fig = plt.figure(figsize=(4, 10))
for i in range(-int(qmaxa), int(qmaxa), 1):
    plt.vlines(x=i, ymin=ylim_neg, ymax=ylim_pos, ls='--', lw=0.2, color='tab:red')
plt.imshow(data, cmap='viridis',
                    norm = LogNorm(vmin=0.1, vmax=np.max(data)),
                    extent=(qxrange[0], qxrange[-1], qyrange[0], qyrange[-1])
                )
plt.grid(True, which='both',axis='both',linestyle='-', color='black', lw=0.5)
plt.yticks(np.arange(ylim_neg, ylim_pos, 1))
plt.xticks(np.arange(xlim_neg, xlim_pos, 1))
plt.ylim(int(ylim_neg), int(ylim_pos))
plt.xlim(int(xlim_neg), int(xlim_pos)) 
plt.xlabel('H (r.l.u.)')
plt.ylabel('L (r.l.u.)')
plt.colorbar()
# plt.set_aspect('equal')
# plt.savefig("0p5_159_H0L_0.5A.jpg", dpi=300)

# plt.figure()
plt.show()



