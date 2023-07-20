
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import fabio as fabio
import os.path
import re
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.pyplot import *
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rcParams['pcolor.shading']
matplotlib.rc('text', usetex=True)


def imgplot(imgpath, resolution, lat_a, lat_c, ylim, xlim=0):
    """
    :input:
    lat_a, lat_c: tetragonal unit cell parameters
    resolution: resolution of the reciprocal space intensity maps
    :return:
    """
    obj = fabio.open(imgpath)
    NX = obj.header["NX"]
    print("NX={}".format(NX))
    NY = obj.header["NY"]
    data = obj.data
    qmaxa = lat_a / (resolution)
    qmaxc = lat_c / (resolution)
    qpixelx = 2 * qmaxa / NX
    qpixely = 2 * qmaxc / NY
    # qpixel = qmax/NX
    qxrange = np.arange(-qpixelx * (NX / 2.), qpixelx * (NX / 2.), qpixelx)
    qyrange = np.arange(-qpixely * (NY / 2.), qpixely * (NY / 2.), qpixely)
    qx, qy = np.meshgrid(qxrange, qyrange)
    # PLOTTING
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = cm.get_cmap("viridis")
    im = ax.pcolormesh(qx, qy, data, cmap=cmap,
                       norm=LogNorm(vmin=10, vmax=np.max(data) * 0.5)
                       )
    ax.set_ylim([-ylim, ylim])
    ax.set_xlim([xlim, qpixelx * (NX / 2.)])
    ax.set_xlabel("H (rlu)")
    ax.set_ylabel("L (rlu)")
    fig.colorbar(im, ax=ax)
    plt.show()


def main():
    list = ["0p5_159_H0L_0.5A.img"]  #For batch-processing, define list with img-names
    imgplot(list[0], 0.5, 4.3486, 10.9766, xlim=-1, ylim=14)  #Process by iterating through given list with imgplot-function

if __name__ == "__main__":
    main()





# steven = "0p5_159_H0L_0.5A.img"
# obj = fabio.open(steven)
# NX = obj.header["NX"]
# print("NX={}".format(NX))
# NY = obj.header["NY"]
# data = obj.data
# resolution = 0.5
# lat_a = 4.3486
# lat_c = 10.9766
# qmaxa = lat_a/(resolution)
# qmaxc = lat_c/(resolution)
# print(qmaxa)
# print(qmaxc)
#
# qpixelx = 2*qmaxa/NX
# qpixely = 2*qmaxc/NY
# #qpixel = qmax/NX
# qxrange = np.arange(-qpixelx*(NX/2.), qpixelx*(NX/2.), qpixelx)
# qyrange = np.arange(-qpixely*(NY/2.), qpixely*(NY/2.), qpixely)
# qx, qy = np.meshgrid(qxrange, qyrange)
#
# fig,ax = plt.subplots(figsize=(9, 7))
# cmap = cm.get_cmap("viridis")
# im = ax.pcolormesh(qx, qy, data, cmap=cmap,
#                    norm=LogNorm(vmin = 10, vmax = np.max(data)*0.5)
#                    )
# # ax.grid(True, which='both',axis='both',linestyle='-', color='black')
# #plt.clim([0,1000])
# #plt.axis('image')
# # ax.set_yticks(np.arange(-22,22,1))
# # ax.set_xticks(np.arange(-10,10,1))
# ax.set_ylim([-14, 14])
# ax.set_xlim([-1, qpixelx*(NX/2.)])
# ax.set_xlabel("H (rlu)")
# ax.set_ylabel("L (rlu)")
# # ax.set_aspect('equal')
# #fig.colorbar(im, ax=ax)
# # plt.show()
# plt.show()

