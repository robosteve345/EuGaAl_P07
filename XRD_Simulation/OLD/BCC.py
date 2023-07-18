import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import signal
import matplotlib as mpl

"""
First principle XRD intensity map calculation for BCC of Fe (ground state)
Credit: Steven Gebel, Dresden 25.11.2022
"""

def gaussian2d(A, X, Y, s):
 return A/s * np.exp(-(X**2 + Y**2)/(2*s))

def kspacecreator(boundary, diff):
    """Create kspace with integer points evenly distributed with periodic boundaries in KL
    """
    # BENCHMARKING: b=2, diff=0.5
    n = np.intc(boundary / diff * 2 + 1)
    print("n={}".format(n))
    #k = np.arange(-boundary, boundary + 1, 1)
    k = np.linspace(-boundary, boundary, n)
    # l = np.arange(-boundary, boundary + 1, 1)
    l = np.linspace(-boundary, boundary, n)
    k2d, l2d = np.meshgrid(k, l)
    print("k2d[0][i]={}".format(k2d[0,:]))
    print("l2d[i][0]={}".format(l2d[:,0]))
    # print("k2d={}".format(k2d), "l2d={}".format(l2d))
    return n, k2d, l2d


def plotfunction(k2d, l2d, I, boundary, diff, A, s, levels):
    """Plot all this shit
    """
    cmap = "viridis"
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(1, 1, 1)
    fig.suptitle('XRD intensity map simulation')
    # color = 'binary_r' # 'inferno', 'magma', 'viridis'
    # ax.set_title('HKL-plot')
    ax.set_xlabel('K (r.l.u.)')
    ax.set_ylabel('L (r.l.u.)')

    Kernel = gaussian2d(A, k2d, l2d, s)
    print("Kernel dim: {}x{}".format(Kernel.shape[0], Kernel.shape[1]))

    # im.set_interpolation('gaussian')
    # interpolation methods: 'nearest', 'bilinear', 'bicubic', 'spline16',
    #            'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
    #            'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
    #################
    #Before pattern
    #ax.imshow(I, cmap=color, vmin=abs(I).min(), vmax=abs(I).max(),
     #              extent=[-boundary, boundary, -boundary, boundary])
    #plt.show()
    #################
    # Manually set F(Q) elements to zero, that are forbidden by crystal symmetry.
    integers = np.arange(-boundary, boundary + 1, 1)
    print("integers={}".format(integers))
    indices = 1 / diff * np.arange(0, 2 * boundary + 1, 1) + 0
    print("newindices={}".format(np.intc(indices)))
    print("I_before={}".format(I))
    for i in range(len(k2d)):
        if i in np.intc(indices):
            I[i, :] = I[i, :]
        else:
            I[i, :] = 0
            for j in range(len(l2d)):
                if j in np.intc(indices):
                    I[:, j] = I[:, j]
                else:
                    I[:, j] = 0
    print("I_after={}".format(I))
    print("I dim: {}x{}".format(I.shape[0], I.shape[1]))
    #################
    #After pattern
    #Convolution
    # conv = sp.signal.convolve2d(I, Kernel, mode='same')
    # print("conv={}".format(conv))
    # print("Conv dim: {}x{}".format(conv.shape[0], conv.shape[1]))
    #ax.scatter(k2d, l2d, s=scatterfactor * I, linewidth=1.0, c='k')
    # ax.imshow(I, cmap=color, vmin=abs(I).min(), vmax=abs(I).max(),
    #           extent=[-boundary, boundary, -boundary, boundary])
    ax.imshow(I, cmap=cmap, interpolation='gaussian', origin='lower',
              extent=[-boundary, boundary, -boundary, boundary],
              vmax=np.max(I)/1)
    # ax.contourf(k2d, l2d, I, levels=levels, cmap=cmap)
    # ax.contourf(k2d, l2d, conv, levels=levels, cmap=cmap)
    plt.show()
    #################

    # # 2d contourplot
    # ax = fig.add_subplot(1, 2, 1)
    # levels = np.linspace(np.min(I), np.max(I), 1000)
    # contourplot = ax.contourf(k2d, l2d, I, levels=levels, cmap=color)
    # fig.colorbar(contourplot)  # Add a colorbar to a plot
    # ax.set_title('HKL-plot')
    # ax.set_xlabel('k')
    # ax.set_ylabel('l')

    # 2d scatter plot
    # ax = fig.add_subplot(1, 2, 2)
    # ax.scatter(k2d, l2d, s=scatterfactor * I, linewidth=1.0, c='k')
    # ax.set_title('HKL-plot')
    # ax.set_xlabel('k')
    # ax.set_ylabel('l')

    # # 3d surface plot
    # ax = fig.add_subplot(4, 1, 3, projection='3d')
    # surface = ax.plot_surface(k2d, l2d, I, rstride=1, cstride=1,
    #                        linewidth=0, antialiased=False, cmap='plasma')
    # fig.colorbar(surface, shrink=1, aspect=5)
    #
    # # 3d scatter plot
    # ax = fig.add_subplot(4, 1, 4, projection='3d')
    # ax.scatter(k2d, l2d, I, c=I, cmap='plasma', linewidth=0.5)

    # plt.savefig('{}kl'.format(h))
    plt.show()


def main():
    print(__doc__)
    boundary = 5 # Symmetric boundaries for K, L
    print("k_max = {}".format(boundary))
    print("l_max = {}".format(boundary))
    diff = 0.08
    print("delta_k = {}".format(diff))
    h = 0
    print("h={}".format(h))
    n, k2d, l2d = kspacecreator(boundary, diff)
    # Structure parameters
    a = 2.86  # in angstrom
    scatterfactor = 0.1
    sigma=0.1

    x, y, z = 0.5, 0.5, 0.5
    fe1 = np.transpose([0, 0, 0])
    fe2 = np.transpose([x, y, z])

    # Debye-Waller factor
    lamb = 1 # Wellenlänge
    # Compute all distances corresponding to lattice diffractions with hkl
    d_hkl = a/(np.sqrt( (h * np.ones((n, n)) )**2 + k2d**2 + l2d**2))
    # 1. compute all angles corresponding to the k points according to braggs law
    theta = np.arcsin(lamb/(2 * d_hkl))
    B_iso_Fe = 0 # isotropic movements around equilibrium, inverse proportional to the mass of the atom

    # Atomic Form factor, a_fe,b_Fe from tugraz website, f(q)=exp(
    a_fe  = [6.4202, 3.0387, 1.9002, 0.7426]
    b_fe = [1.5936, 31.5472, 1.9646, 85.0886]
    c_fe = 1.1151
    f_fe1 = a_fe[0] * np.exp(- b_fe[0] * (((h * np.ones((n, n)) )**2 + k2d**2 + l2d**2) / (4 * np.pi)**2  ))
    f_fe2 = a_fe[1] * np.exp(- b_fe[1] * (((h * np.ones((n, n))) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    f_fe3 = a_fe[2] * np.exp(- b_fe[2] * (((h * np.ones((n, n))) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    f_fe4 = a_fe[3] * np.exp(- b_fe[3] * (((h * np.ones((n, n))) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    f_fe = f_fe1+f_fe2+f_fe3+f_fe4
    F_fe1_list, F_fe2_list = [], []
    #np.exp(-B_iso_Fe / lamb * (np.sin(theta))**2)
    F_fe1_list.append(
        f_fe  * np.exp(1j * 2 * np.pi * (h * np.ones((n, n)) * fe1[0] + k2d * fe1[1] + l2d * fe1[2])))
    F_fe2_list.append(
        f_fe  * np.exp(1j * 2 * np.pi * (h * np.ones((n, n)) * fe2[0] + k2d * fe2[1] + l2d * fe2[2])))
    atoms_list = [F_fe1_list, F_fe2_list]  # create list with components corresponding to F evaluated for all atomic positions of element X with all k-space points
    F_str = np.zeros((n, n), dtype=complex)  # Structure factor with dimensions nxn
    pre_F_str = [np.zeros((n, n))]  # Hilfsgröße, in nächstem loop beschrieben

    # Zusammenfügen der Formfaktoren für die Atomsorten
    for i in atoms_list:  # put together the lists in a ndarray for each atom with each N positions, to get rid of 3rd dimension (better workaround probably possible...)
        pre_F_str = np.add(pre_F_str, i)
    for i in range(len(pre_F_str)):  # add together
        F_str = F_str + pre_F_str[i]
    I = np.absolute(F_str) ** 2  # Calculate intensity |F|^2
    print("I_before={}".format(I))

    # Plot everything
    plotfunction(k2d, l2d, I, boundary, diff, A=1, s=0.001, levels=10)



if __name__ == '__main__':
    main()


















def gaussian(theta, lamb, sigma):
    gauss = (1 / (2 * np.pi * sigma)) * np.exp(-(theta ** 2 + lamb ** 2) / (2 * sigma))
    return gauss


def convolute2d(matrix, sigma, len_k2d):
    """Faltung eines 2d-Datensatzes mit Gauß-peaks"""
    boundary = len_k2d # muss gleiche dimension, wie k space haben, dehsalb len(k2d)=len(l2d)
    x_conv, y_conv = np.meshgrid(np.linspace(-boundary,  boundary, 2* boundary+1), np.linspace(-boundary,  boundary, 2* boundary+1)) # np.meshgrid(np.arange(-boundary,  boundary, 0.1), np.arange(- boundary,  boundary, 0.1))
    g = gaussian(x_conv, y_conv, sigma)
    convolved = scipy.signal.convolve2d(matrix, g,  boundary='wrap', mode='same') #boundary='symm', mode='same')
    plt.imshow(convolved, cmap='viridis', interpolation='gaussian')
    plt.show()












# F_SC = np.zeros((n, n), dtype=complex) # n dimension of k-space
# pre_F_SC = np.add(F_atom1_list, F_atom2_list) # hand-made
# for i in range(len(pre_F_SC)):
#     F_SC = F_SC + pre_F_SC[i]
# I = np.absolute(F_SC) ** 2
# print("min(I)={}".format(np.min(I)))
#
#
# def gaussian(theta, lamb, sigma):
#     gauss = (1 / (2 * np.pi * sigma)) * np.exp(-(theta ** 2 + lamb ** 2) / (2 * sigma))
#     return gauss
#
#
# def convolute2d(I, sigma):
#     """Faltung eines 2d-Datensatzes mit Gauß-peaks"""
#     k2d_conv, l2d_conv = np.meshgrid(np.linspace(-1,1,100), np.linspace(-1,1,100))
#     g = gaussian(k2d_conv, l2d_conv, sigma)
#     N_neighbors = scipy.signal.convolve2d(I, g, boundary='symm', mode='same')
#     plt.imshow(N_neighbors, interpolation="nearest", cmap=plt.cm.coolwarm)
#     plt.show()
#
#
# def scatter3d(I, x, y):
#     ax = plt.axes(projection='3d')
#     ax.scatter(x, y, I, c=I, cmap='plasma', linewidth=0.5)
#     plt.show()
#
#
# def d2plot(I):
#     fig, ax = plt.subplots()
#     # ax.scatter(k2d, l2d, color="green")
#     levels = np.linspace(np.min(I), np.max(I), 1000)
#     cp = ax.contourf(k2d, l2d, I, levels=levels)
#     fig.colorbar(cp)  # Add a colorbar to a plot
#     ax.set_title('HKL-plot')
#     ax.set_xlabel('k')
#     ax.set_ylabel('l')
#     plt.show()
#
#
# def d3plot(I):
#     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#     #ax.set_aspect('auto')
#     # Plot the surface.
#     surf = ax.plot_surface(k2d, l2d, I, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#     ax.zaxis.set_major_locator(LinearLocator(10))
#     ax.zaxis.set_major_formatter('{x:.02f}')
#     # Add a color bar which maps values to colors: 'viridis', 'plasma', 'inferno', 'magma', 'cividis'
#     fig.colorbar(surf, shrink=10, aspect=5)
#
#
# def main():
#     print(__doc__)
#     sigma = 0.1
#     convolute2d(I, sigma)
#     #scatter3d(I, k2d, l2d)
#     #d2plot(I)
#     # d3plot(I)
#     #scatter3d(I, k2d, l2d)
#
#     plt.show()
#
# if __name__ == '__main__':
#     main()