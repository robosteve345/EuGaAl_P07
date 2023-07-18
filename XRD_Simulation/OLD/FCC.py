"""XRD_Simulation of CDW pattern in k-space: FCC
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy.ndimage import gaussian_filter
from scipy import signal
from scipy.stats import gaussian_kde
from scipy import ndimage as ndi

Xe = np.array([[0,0,0], [0,0.5,0.5]]).T
Cr = np.array([[0.5,0,0.5], [0.5,0.5,0]]).T
boundary = 4
h = 0 # set h to a constant value
start = -2
k = np.arange(start, start+boundary, 1)#np.linspace(-boundary, boundary, n)#
l = np.arange(start, start+boundary, 1)#np.linspace(-boundary, boundary, n)# np.arange(start, start+boundary, 1)
n =  len(k) # Dimension of the reciprocal space:
k2d, l2d = np.meshgrid(k, l) # or np.mgrid[-boundary:boundary, -boundary:boundary]
f_Xe, f_Cr = 1, 1
F_Xe_list, F_Cr_list = [], []
for i in range(2): # 2 atoms for each sort Xe, Cr
    F_Xe_list.append(f_Xe * np.exp(1j * 2 * np.pi * (h * np.ones((n, n)) * Xe[0,i] + k2d * Xe[1,i] + l2d * Xe[2,i])))
    F_Cr_list.append(f_Cr * np.exp(1j * 2 * np.pi * (h * np.ones((n, n)) * Cr[0,i] + k2d * Cr[1,i] + l2d * Cr[2,i])))
F_FCC = np.zeros((n, n), dtype=complex) # n dimension of k-space
pre_F_FCC = np.add(F_Xe_list, F_Cr_list) # hand-made
for i in range(len(pre_F_FCC)):
    F_FCC = F_FCC + pre_F_FCC[i]
I = np.absolute(F_FCC) ** 2
print("min(I)={}".format(np.min(I)))
blurry = gaussian_filter(np.array([[1,2,3],[4,3,2],[1,0,1]]), 0.1)
plt.imshow(blurry, cmap=cm.coolwarm)
plt.show()

def gaussian(theta, lamb, sigma):
    gauss = (1 / (2 * np.pi * sigma)) * np.exp(-(theta ** 2 + lamb ** 2) / (2 * sigma))
    return gauss


def convolute2d(I):
    """Faltung eines 2d-Datensatzes mit Gauß-peaks"""
    k2d_conv, l2d_conv = np.meshgrid(np.linspace(-2,2,10), np.linspace(-1,2,10))
    sigma=0.1
    g = gaussian(k2d_conv, l2d_conv, sigma)
    blurred = scipy.signal.convolve2d(I, g,  boundary='symm', mode='full')
    plt.imshow(blurred, cmap=cm.coolwarm, interpolation='nearest')
    plt.imshow(I)
    plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(X,Y,Z)
convolute2d()








def scatter3d(I, x, y):
    ax = plt.axes(projection='3d')
    ax.scatter(x, y, I, c=I, cmap='plasma', linewidth=0.5)
    plt.show()


def d2plot(I):
    fig, ax = plt.subplots()
    levels = np.linspace(np.min(I), np.max(I), 5)
    # plt.imshow(I, cmap=cm.coolwarm, interpolation='nearest')
    cp = ax.contourf(k2d, l2d, I, levels=levels)
    fig.colorbar(cp)  # Add a colorbar to a plot
    plt.show()
    ax.set_title('HKL-plot')
    ax.set_xlabel('k')
    ax.set_ylabel('l')
    plt.show()


def d3plot(I):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.set_aspect('auto')
    # Plot the surface.
    surf = ax.plot_surface(k2d, l2d, I, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')
    # Add a color bar which maps values to colors: 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
    fig.colorbar(surf, shrink=10, aspect=5)


def main():
    print(__doc__)
    sigma=1
    #convolute2d(I, sigma)
    #scatter3d(I, k2d, l2d)
    d2plot(I)
    # d3plot(I)
    plt.show()

if __name__ == '__main__':
    main()