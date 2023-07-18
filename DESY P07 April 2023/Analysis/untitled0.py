
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import fabio as fabio
import os.path
import re
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.pyplot import *
from scipy.ndimage import center_of_mass



steven = "0p5_159_H0L_0.5A.img"

obj = fabio.open(steven)
NX = obj.header["NX"]
NY = obj.header["NY"]
data = obj.data
resolution = 0.5
lat_a = 4.3486
lat_c = 10.9766
qmaxa = lat_a/(resolution)
qmaxc = lat_c/(resolution)
print(qmaxa)
print(qmaxc)

qpixelx = 2*qmaxa/NX
qpixely = 2*qmaxc/(NY)
#qpixel = qmax/NX
qxrange = np.arange(-qpixelx*(NX/2.), qpixelx*(NX/2.), qpixelx)
qyrange = np.arange(-qpixely*(NY/2.), qpixely*(NY/2.), qpixely)
qx, qy = np.meshgrid(qxrange, qyrange)



fig,ax = plt.subplots(figsize=(12, 8))
cmap = cm.get_cmap("viridis")
im = ax.pcolormesh(qx, qy, data,cmap=cmap,norm=LogNorm(vmin = 10, vmax = np.max(data)))
# ax.grid(True, which='both',axis='both',linestyle='-', color='black')
#plt.clim([0,1000])
#plt.axis('image')
ax.set_yticks(np.arange(-22,22,1))
ax.set_xticks(np.arange(-10,10,1))
ax.set_ylim([-14, 14])
ax.set_xlim([-1, qpixelx*(NX/2.)])
#ax.set_aspect('equal')
fig.colorbar(im, ax=ax)
plt.show()
