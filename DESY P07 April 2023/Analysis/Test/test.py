#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 20:56:45 2023

@author: stevengebel
"""
import fabio
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LogNorm

img = [spectrum for spectrum in os.listdir() if spectrum.endswith(".img")]

obj = fabio.open(img)
NX = obj.header["NX"]
NY = obj.header["NY"]
data = obj.data
resolution = 0.5
lat_a = 4.3262
lat_c = 10.996
qmaxa = lat_a/(resolution)
qmaxc = lat_c/(resolution)

qpixelx = 2*qmaxa/NX
qpixely = 2*qmaxc/NY
qxrange = np.arange(-qpixelx*(NX/2.), qpixelx*(NX/2.), qpixelx)
qyrange = np.arange(-qpixely*(NY/2.), qpixely*(NY/2.), qpixely)
qx, qy = np.meshgrid(qxrange, qyrange)

H = -5
xlim, ylim = 0, 14
ind_H = int(np.abs(H/qpixelx))# int(NX - np.abs(H/qpixelx))
print("ind_H={}".format(ind_H))
# INTENSITY MAP
fig,ax = plt.subplots(figsize=(9, 7))
ax.vlines(x=H, ymin=-14, ymax=14, ls='-', lw=1.5, color='black')
im = ax.pcolormesh(qx, qy, data,cmap='viridis',
                   norm=LogNorm(vmin = 10, vmax = np.max(data)))
ax.grid(True, which='both',axis='both',linestyle='-', color='black', lw=0.25)
ax.set_yticks(np.arange(-ylim,ylim,1))
ax.set_xticks(np.arange(xlim,0,1))
ax.set_ylim([-ylim, ylim])
ax.set_xlim([xlim, 0]) # qpixelx*(NX/2.)
ax.set_xlabel('H (rlu)')
ax.set_ylabel('L (rlu)')
#ax.set_aspect('equal')
fig.colorbar(im, ax=ax)
# plt.savefig("0p5_159_H0L_0.5A.jpg", dpi=300)

plt.show()

