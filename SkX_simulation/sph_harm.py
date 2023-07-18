import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm
import mpl_toolkits.mplot3d.axes3d as axes3d
import matplotlib.colors as mcolors
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
#######
l, m = 3, 1
########
# Define grid of unit sphere coordinates
theta = np.linspace(0, np.pi, 50)
phi = np.linspace(0, 2 * np.pi, 50)
Theta, Phi = np.meshgrid(theta, phi)
X = 1 * np.sin(Theta) * np.cos(Phi)
Y = 1 * np.sin(Theta) * np.sin(Phi)
Z = 1 * np.cos(Theta)
# Calculate real spherical harmonics
R = np.abs(sph_harm(abs(m), l, Phi, Theta))
if m < 0:
    R = np.sqrt(2) * (-1)**m * np.imag(sph_harm(np.abs(m), l, Phi, Theta))
elif m > 0:
    R = np.sqrt(2) * (-1)**m * np.real(sph_harm(m, l, Phi, Theta))
x = np.abs(R) * np.sin(Theta) * np.cos(Phi)
y = np.abs(R) * np.sin(Theta) * np.sin(Phi)
z = np.abs(R) * np.cos(Theta)
ax.scatter(x, y, z, marker='x', c='g')
cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap('PRGn'))
cmap.set_clim(-0.5, 0.5)
ax.plot_surface(X, Y, Z, facecolors=cmap.to_rgba(np.abs(R)))
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# plt.show()
theta = np.linspace(0,2*np.pi,1000)
r = (0.5*(5*np.cos(theta)**3 - 3*np.cos(theta)))**2
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta, r)
plt.show()