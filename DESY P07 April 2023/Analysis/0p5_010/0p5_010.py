#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Created on Wed Jul 26 10:04:16 2023

#@author: stevengebel
"""Visualization of integrated Intensity maps with linecutting for I4/mmm Eu(Ga,Al)4

Input: .txt-file with lattice parameters a,c and resolution of integrated maps
Output: Visualization of the intensity and linecuts

"""

"""TO DO:
- Summation over k-points, sum over ~k_nominal +- ∆k
- Add noise, s.t. plot is without I~0 artifacts for LogNorm (DONE)
- Find way to (a) Subtract spurious intensity distribution (b) extract only 
  good peak intensities
- Create non-isotrpic gaussian convolution
"""

import numpy as np
import matplotlib.pyplot as plt
import fabio as fabio
import matplotlib as mpl
mpl.rc('text', usetex=True)
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['font.family'] = "sans-serif"
import os
from matplotlib.colors import LogNorm

#########################################################################
# INPUT
img = "al2ga2_Hm1L_0.5A.img"
lat_a, lat_c, resolution = np.loadtxt("0p5_010.txt", usecols=(0, 1, 2), 
                                      skiprows=1, unpack=True)
#########################################################################
print(__doc__)
print("Elements in this directory:{}".format(os.listdir()))
obj = fabio.open(img)
data = obj.data
NX = obj.header["NX"]
NY = obj.header["NY"]
qmaxa, qmaxc = lat_a/(resolution), lat_c/(resolution)

#  Momentum resolution in x and y
qpixelx, qpixely = 2*qmaxa/NX, 2*qmaxc/NY
print("qpixelx(1/Angstrom) = {}, qpixely(1/Angstrom) = {}".format(qpixelx, qpixely))

qxrange = np.arange(-qpixelx*(NX/2.), qpixelx*(NX/2.), qpixelx)
qyrange = np.arange(-qpixely*(NY/2.), qpixely*(NY/2.), qpixely)
QX2d, QY2d = np.meshgrid(qxrange, qyrange)
xlim_pos, xlim_neg, ylim_pos, ylim_neg = int(qmaxa), -int(qmaxa), \
                                         int(qmaxc), -int(qmaxc)

# LINECUTS
fig = plt.figure(figsize=(10, 4))
plt.title("{}".format(img))
# PIXELZAHL DES ERSTEN INTEGER-K-WERTES FÜR LINECUTS
important = 63 # Für P07
indices = np.arange(important, int(2*qmaxa)*int(1/qpixelx), int(1/qpixelx)) 
# indices = [1428]
for i in indices:
    I = data[:, i]
    plt.plot(qyrange, I, label='ind(H)={}, H={}'.format(i, 
                            np.round(qxrange[i], 1)), lw=0.5)
    plt.xlabel('L (r.l.u.)')
    plt.ylabel('Intensity I')
    plt.xlim(ylim_neg, ylim_pos) 
    plt.legend()
    #plt.savefig("{}_linecuts.jpg".format(img), dpi=300)
    


# INTENSITY MAP
# Create additional noise for visibility reasons
noise = np.abs(np.random.randn(NX, NY)) * 3
data = noise + data
fig = plt.figure(figsize=(4, 10))
plt.title("{}".format(img))
for i in range(-int(qmaxa), int(qmaxa) + 1, 1):
    plt.vlines(x=i, ymin=ylim_neg, ymax=ylim_pos, ls='--', lw=0.2, color='tab:red')
# =============================================================================
# The problem is that bins with 0 can not be properly log normalized so they are 
# flagged as 'bad', which are mapped to differently. The default behavior is to 
# not draw anything on those pixels. You can also specify what color to draw 
# pixels that are over or under the limits of the color map (the default is to 
# draw them as the highest/lowest color).
# =============================================================================
plt.imshow(data, cmap='viridis',
                    norm = LogNorm(vmin=0.1, vmax=np.max(data)),
                    extent=(qxrange[0], qxrange[-1], qyrange[0], qyrange[-1])
                )
# plt.grid(True, which='both',axis='both',linestyle='-', color='black', lw=0.5)
plt.yticks(np.arange(ylim_neg, ylim_pos + 1, 1))
plt.xticks(np.arange(xlim_neg, xlim_pos + 1, 1))
plt.ylim(int(ylim_neg), int(ylim_pos))
plt.xlim(int(xlim_neg), int(xlim_pos)) 
plt.xlabel(r'H (r.l.u.)')
plt.ylabel(r'L (r.l.u.)}')
plt.colorbar()
# plt.savefig("{}.jpg".format(img), dpi=300)


plt.show()



