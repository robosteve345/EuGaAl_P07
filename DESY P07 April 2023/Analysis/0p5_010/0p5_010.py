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

"""Print unwarped CrysAlisPro intensity maps for EuGa2Al2 run 10"""

def imgplot(imgpath, resolution, lat_a, lat_c, ylim, title, xlim=0):
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
    plt.subplot(2, 1, 1)
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = cm.get_cmap("viridis")
    im = ax.pcolormesh(qx, qy, data, cmap=cmap,
                       norm=LogNorm(vmin=10, vmax=np.max(data))
                       )
    fig.suptitle("a={}, c={}".format(lat_a, lat_c))
    # ax.set_ylim([-ylim, ylim])
    #ax.set_xlim([xlim, qpixelx * (NX / 2.)])
    # ax.set_yticks(np.arange(-ylim, ylim, 1))
    # ax.set_xticks(np.arange(xlim, 10, 1))
    ax.set_xlabel("H (rlu)")
    ax.set_ylabel("L (rlu)")
    # ax.grid(True, which='both', axis='both', linestyle='-', lw=1, color='red')
    fig.colorbar(im, ax=ax)
    print("FILE: {}".format(title))
    print("Resolution: {}, a={}Å, c={}Å".format(resolution, lat_a, lat_c))
    plt.savefig("{}.jpg".format(title), dpi=300)
    plt.show()
    plt.subplot(2, 1, 2)
    print(2/qpixelx)
    y = data[:, 210]
    plt.plot(np.arange(-qpixely * (NY / 2.), qpixely * (NY / 2.), qpixely), y, ls='', marker='.')
    plt.show()


def main():
    print(__doc__)
    # For batch-processing, define list with img-names
    list = ["al2ga2_H0L_0.5A.img", "al2ga2_H1L_0.5A.img", "al2ga2_Hm1L_0.5A.img",
            "al2ga2_HK0_0.5A.img", "al2ga2_HK1_0.5A.img", "al2ga2_HKm1.img",
            "al2ga2_0KL_0.5A.img", "al2ga2_1KL_0.5A.img", "al2ga2_m1KL_0.5A.img",
            ]
    ##########  EXAMPLE .cbf imshow-PLOT
    # obj = fabio.open("ga2al2_01_atten_20_00010_00001.cbf")
    # plt.imshow(obj.data, norm=LogNorm(vmin=1, vmax=np.max(obj.data)*0.1))
    # plt.show()
    ##########

    imgplot(list[6], 0.5, 4.2726, 10.792, xlim=-1, ylim=14, title='al2ga2_010_0KL_0.5A')  #Process by iterating through given list with imgplot-function

if __name__ == "__main__":
    main()


