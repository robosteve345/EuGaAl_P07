#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Created on Wed Nov  1 14:15:18 2023

#@author: stevengebel

import numpy as np
import matplotlib.pyplot as plt
import fabio as fabio
import matplotlib as mpl
import os
import time
from matplotlib.colors import LogNorm
mpl.rcParams.update(mpl.rcParamsDefault)
useMathText=True

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

"""Visualization of integrated Intensity KL-maps with linecutting for I4/mmm
Eu(Ga,Al)4 on P07, DESY.

Input: Lattice parameters a, c and resolution of integrated map from CrysAlisPro
Output: Visualization & Linecuts for integer K-values of every .img-file in dir
"""

"""Data manipulations: 
    For Linecuts
    - Summing over linecuts for ±nmax next-neighbour pixels (and dividing by 
                                                             (2*nmax + 1))
    - Setting negative intensities to 0
    For Imshow
    - Manually set negative intensities to 0
    - Min. value = 1
    - Added noise of max. value |1|
    """

###############################################################################
# =============================================================================
#               GENERAL INPUT
# =============================================================================
directory = os.getcwd()  #  Get current directory-path
#  Get information on crystal-structure from manually created .txt-file in dir.
for item in os.listdir(directory):
    if item.endswith('.txt') and os.path.isfile(os.path.join(directory, item)):
        latticeparam1, latticeparam2, resolution = np.loadtxt(str(item), 
                                                   usecols=(0, 1, 2), 
                                                   skiprows=1, unpack=True)
# latticeparam1 = input("Lattice parameter a (in Angstrom)  [a]: ")
# latticeparam2 = input("Lattice parameter c (in Angstrom)  [c]: ")
# assign directory
# directory = input("Directory []: ") # "{}".format(directory)
directory = os.getcwd()
resolution = 0.5
cmap = "viridis"
nmax = 0  #  Maximum next neighbor pixels for mean value of Linecuts
###############################################################################
print(__doc__)
print("Elements in this directory:{}".format(os.listdir()))
# =============================================================================
#     INPUT FOR LINECUT
# =============================================================================
xzoomindex = 15  #  Linecut integer H value  -->indecesx
yzoomindex = 22  #  Zoom-window centered L-cut value   -->indecesy
st = time.time()
for item in os.listdir(directory):
# =============================================================================
#     HL MAPS
# =============================================================================
    if not item.startswith('.') and item.startswith('0p5_159_H') and item.endswith('L_0.5A.img') and os.path.isfile(os.path.join(directory, item)):
            print("CURRENT FILE: {}".format(item))
            #########################################################################
            # INPUT FOR HL-maps
            img = item
            lat_a, lat_c = float(latticeparam1), float(latticeparam2)
            #########################################################################
            obj = fabio.open(img)
            data = obj.data
            NX = obj.header["NX"]
            NY = obj.header["NY"]
            qmaxa, qmaxc = lat_a/(resolution), lat_c/(resolution)
            #  Momentum resolution in x and y
            qpixelx, qpixely = 2*qmaxa/NX, 2*qmaxc/NY
            # print("qpixelx(1/Angstrom) = {}, qpixely(1/Angstrom) = {}".format(qpixelx, qpixely))
            qxrange = np.arange(-qpixelx*(NX/2.), qpixelx*(NX/2.), qpixelx)
            qyrange = np.arange(-qpixely*(NY/2.), qpixely*(NY/2.), qpixely)
            QX2d, QY2d = np.meshgrid(qxrange, qyrange)
            xlim_pos, xlim_neg, ylim_pos, ylim_neg = int(qmaxa), -int(qmaxa), \
                                                      int(qmaxc), -int(qmaxc)
    # =============================================================================
    #         Compute integer-indices for x,y
    # =============================================================================
            fig = plt.figure(figsize=(6, 8))
            plt.title("{}, sum of ±{}-pixel nearest-neighbors".format(img, nmax))
            # # PIXEL INDICE FOR FIRST INTEGER X-VALUE (manually set)
            # important = 46 # for ID28
            # x-axis indices for integers values
            indicesx = []  
            for i in range(int(xlim_neg), int(xlim_pos)):
                indicesx.append(find_nearest(qxrange, i))
            # y-axis indices for integers values
            indicesy = []
            for i in range(int(ylim_neg), int(ylim_pos)):
                indicesy.append(find_nearest(qyrange, i)) 
    # =========================================================================
    #         Manually set negative Intensities to 0
    # =========================================================================
            # for i in range(0, data.shape[0]):
            #     for j in range(0, data.shape[1]):
            #         if (data[i][j] < 0):
            #             data[i][j] = 0
    # =========================================================================
    #         Compute Linecuts
    # =========================================================================
            for i in indicesx[xzoomindex:xzoomindex+1]:
                # i is centered H-value (integer)
                # Overcome "byte overflow": use .astype(np.uint16), uint64,...
                # https://stackoverflow.com/questions/68653746/sum-of-positive-arrays-yields-negative-results
                I = []
                I0 = data[:,i].astype(np.uint64)  #  centered intensity
                I.append(I0)
                #  Compute summation over nmax nearest neighbor pixels
                for j in range(1, nmax + 1):
                    I.append(data[:,i+j].astype(np.uint64))
                    I.append(data[:,i-j].astype(np.uint64))    
                Imean = sum(I) / (2*nmax + 1)
    # =========================================================================
    #           Plot Linecuts
    # =========================================================================
                plt.plot(qyrange, Imean, label='Sum around H={}'.format( 
                                    round(qxrange[i])),
                                    lw=0.5, marker='s', ms=3)
                plt.xlabel('L (r.l.u.)')
                plt.ylabel('Intensity I')
                plt.legend(loc='upper left')
                plt.xlim(ylim_neg + 6, ylim_pos - 6) 
                plt.ticklabel_format(axis='both', useMathText=True)
                plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    # =========================================================================
    #           Generate data for the zoomed portion
    # =========================================================================
                X_detail = qyrange[indicesy[yzoomindex]-30:indicesy[yzoomindex]+30]# qyrange[int(indicesy[yzoomindex]-100):int(indicesy[yzoomindex]+100)]
                Y_detail = Imean[indicesy[yzoomindex]-30:indicesy[yzoomindex]+30]
                # location for the zoomed portion 
                sub_axes = plt.axes([.6, .6, .25, .25]) 
                # plot the zoomed portion
                sub_axes.plot(X_detail, Y_detail, marker='s', ms=1, lw=0.5, 
                    label="[{}K0{}]".format(round(qxrange[i]),
                    round(qyrange[indicesy[yzoomindex]]))
                                                            )
                sub_axes.legend()
                # insert the zoomed figure
                plt.show(block=False)
                plt.ticklabel_format(axis='both', useMathText=True)
                plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
                plt.savefig("{}_linecuts.jpg".format(img), dpi=300)
    # =========================================================================
    #        Add noise with amplitude 1 to Intensity map
    # =========================================================================
            noise = np.abs(np.random.randn(NX, NY)) * 1
            data = noise + data
    # =========================================================================
    #         Plot Intensity map
    # =========================================================================
            fig = plt.figure(figsize=(6, 8))
            plt.title("{}".format(img))
            plt.imshow(data, cmap=cmap,  origin='lower',
                                    norm = LogNorm(vmin=1, vmax=np.max(data)),
                                    extent=(qxrange[0], qxrange[-1],qyrange[0], 
                                            qyrange[-1]), 
                                    aspect='auto'
                            )
            #  #  Plot Linecut-lines
            # for i in qxrange[indicesx]: # qxrange[indices]
            #     plt.vlines(x=i, ymin=ylim_neg, ymax=ylim_pos, ls='--', 
            #                 lw=0.5, color='tab:red')
            #   #  Plot grid
            # plt.grid(True, which='both',axis='both',linestyle='-', 
            # color='black', lw=0.25)
            # plt.yticks(np.arange(ylim_neg, ylim_pos + 1, 1))
            # plt.xticks(np.arange(xlim_neg, xlim_pos + 1, 1))
            plt.ylim(qyrange[indicesy[6]], qyrange[indicesy[-6]])
            plt.xlim(qxrange[indicesx[8]], qxrange[-1]) 
            plt.xlabel('H (r.l.u.)')
            plt.ylabel('L (r.l.u.)')
            plt.colorbar()
            plt.savefig("{}.jpg".format(img), dpi=300, bbox_inches='tight')
            plt.show(block=False)
    else:
        pass
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')