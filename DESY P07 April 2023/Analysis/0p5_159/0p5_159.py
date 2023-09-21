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
matplotlib.rcParams['text.usetex'] = False

"""Print unwarped CrysAlisPro intensity maps"""

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
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = cm.get_cmap("viridis")
    im = ax.pcolormesh(qx, qy, data, cmap=cmap, shading='nearest',
                       norm=LogNorm(vmin=1.1, vmax=np.max(data))
                       )
    ax.set_ylim([-ylim, ylim])
    fig.suptitle("a={}, c={}".format(lat_a, lat_c))
    ax.set_xlim([xlim, qpixelx * (NX / 2.)])
    ax.set_xlabel("H (rlu)")
    ax.set_ylabel("L (rlu)")
    ax.grid(True, which='both', axis='both', linestyle='-', lw=1, color='red')
    ax.set_yticks(np.arange(-ylim, ylim,1))
    ax.set_xticks(np.arange(xlim,10,1))
    fig.colorbar(im, ax=ax)
    print("FILE: {}".format(title))
    print("Resolution: {}, a={}Å, c={}Å".format(resolution, lat_a, lat_c))
    plt.savefig("{}.jpg".format(title), dpi=300)
    plt.show()


def main():
    print(__doc__)
    imgplot("0p5_159_H0L_0.5A_1.img", 0.5, 4.3262, 10.996, xlim=-1, ylim=14, title='0p5_159_H0L_0.5A_1')  #Process by iterating through given list with imgplot-function
if __name__ == "__main__":
    main()


