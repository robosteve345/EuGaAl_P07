#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 10:04:16 2023

@author: stevengebel
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import fabio as fabio
import os
import re
from matplotlib.colors import LogNorm
from matplotlib.pyplot import *
from scipy.ndimage import center_of_mass


# assign directory
directory = '/Users/stevengebel/PycharmProjects/EuGaAl_P07/ID28/200K'
 
# iterate over files in
# that directory
for item in os.listdir(directory):
    if not item.startswith('.') and item.endswith('.img') and os.path.isfile(os.path.join(directory, item)):
        print(item)
        obj = fabio.open(item)
        NX = obj.header["NX"]
        NY = obj.header["NY"]
        data = obj.data
        resolution = 0.5
        lat_a = 4.379
        lat_c = 11.165
        qmaxa = lat_a/(resolution)
        qmaxc = lat_c/(resolution)

        qpixelx = 2*qmaxa/NX
        qpixely = 2*qmaxc/NY
        qxrange = np.arange(-qpixelx*(NX/2.), qpixelx*(NX/2.), qpixelx)
        qyrange = np.arange(-qpixely*(NY/2.), qpixely*(NY/2.), qpixely)
        qx, qy = np.meshgrid(qxrange, qyrange)

        # H = -5
        # xlim ylim = 0, 8
        # ind_H = int(np.abs(H/qpixelx))# int(NX - np.abs(H/qpixelx))
        # print("ind_H={}".format(ind_H))
        # IMSHOW MAP
        plt.title("{}".format(item))
        plt.imshow(obj.data, cmap='viridis', norm = LogNorm(vmin=1, vmax=0.1*np.max(data)))
        # INTENSITY MAP
        fig,ax = plt.subplots(figsize=(5, 3))
        ax.set_xlim(-5,5)
        ax.set_ylim(-5,5)
        # ax.vlines(x=H, ymin=-14, ymax=14, ls='-', lw=1.5, color='black')
        im = ax.pcolormesh(qx, qy, data, cmap='inferno', vmax=100
                           # norm= LogNorm(vmin = 1, vmax = 100)
                           )
        ax.grid(False, which='both',axis='both',linestyle='-', color='black', lw=0.1)
        ax.set_xlabel('H (rlu)')
        ax.set_ylabel('L (rlu)')
        ax.set_title("{}".format(item))

        fig.colorbar(im, ax=ax)
        plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/ID28/200K/{}.jpg".format(item), dpi=400)
        # plt.savefig("0p5_110_0KL_0.5A.jpg", dpi=300)

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
# plt.show(block=False)

