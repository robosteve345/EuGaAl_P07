#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 17:52:02 2023

@author: stevengebel
"""
"""XRD EuGa2Al2 simulation"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.pyplot import figure
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rc('text', usetex=True)
import matplotlib as mpl
from scipy import ndimage
mpl.rcParams.update(mpl.rcParamsDefault)


def ztranslationpos(x, storage_x, storage_y, storage_z, i):
    """Translate atomic positions one given direction x,y,z
    """
    x_transl = x + np.array([0, 0, i])
    storage_x.append(x_transl[0])
    storage_y.append(x_transl[1])
    storage_z.append(x_transl[2])
    
def ztranslationneg(x, storage_x, storage_y, storage_z, i):
    """Translate atomic positions one given direction x,y,z
    """
    x_transl = x + np.array([0, 0, -i])
    storage_x.append(x_transl[0])
    storage_y.append(x_transl[1])
    storage_z.append(x_transl[2])


def kspacecreator(k0, l0, kmax, lmax, deltak):
    """K-space creator"""
    k = np.arange(k0-kmax, k0+kmax + deltak, deltak)
    l = np.arange(l0-lmax, l0+lmax + deltak, deltak)
    k2d, l2d = np.meshgrid(k, l)
    Unitary = np.ones((len(k2d), len(l2d)))  # Unitary matrix
    return k2d, l2d, k, Unitary
                                                        
                                                        
def structurefactorandplotting(k0, l0, k2d, k, kmax, lmax, l2d, h, z_Ga, deltak, Unitary, noiseamplitude, f_list, sigma, kernelsize, savefig=False):
    print("CENTERED PEAK: [HKL] = [{}{}{}]".format(h, k0, l0))
    """EuGa2Al2 atomic positions"""
    Eu, EuT, Al1, Al1T, Al2, Al2T, Ga1, Ga1T, Ga2, Ga2T = np.array([[0],[0],[0]]), np.array([[1/2],[1/2],[1/2]]), \
                                                            np.array([[0],[1/2],[1/4]]), np.array([[1/2],[1],[3/4]]),\
                                                            np.array([[1/2],[0],[1/4]]), np.array([[1],[1/2],[3/4]]), \
                                                            np.array([[0],[0],[z_Ga]]), np.array([[1/2],[1/2],[1/2+z_Ga]]), \
                                                            np.array([[0],[0],[-z_Ga]]), np.array([[1/2],[1/2],[1/2-z_Ga]])
    
                                                        
    """Scattering amplitudes F"""
    # Form factors
    f_Eu, f_Al, f_Ga = 1, 1, 1
    # Scattering Amplitudes
    F_Eu = f_Eu * (np.exp(-2*np.pi*1j*(h*Unitary*Eu[0] + k2d*Eu[1] + l2d*Eu[2])) + 
                       np.exp(-2*np.pi*1j*(h*Unitary*EuT[0] + k2d*EuT[1] + l2d*EuT[2]))
                       )
    F_Al = f_Al * (np.exp(-2*np.pi*1j*(h*Unitary*Al1[0] + k2d*Al1[1] + l2d*Al1[2])) + 
                       np.exp(-2*np.pi*1j*(h*Unitary*Al2[0] + k2d*Al2[1] + l2d*Al2[2])) + 
                       np.exp(-2*np.pi*1j*(h*Unitary*Al1T[0] + k2d*Al1T[1] + l2d*Al1T[2])) + 
                       np.exp(-2*np.pi*1j*(h*Unitary*Al2T[0] + k2d*Al2T[1] + l2d*Al2T[2]))
                   )
    F_Ga = f_Ga * (np.exp(-2*np.pi*1j*(h*Unitary*Ga1[0] + k2d*Ga1[1] + l2d*Ga1[2])) + 
                       np.exp(-2*np.pi*1j*(h*Unitary*Ga2[0] + k2d*Ga2[1] + l2d*Ga2[2])) + 
                       np.exp(-2*np.pi*1j*(h*Unitary*Ga1T[0] + k2d*Ga1T[1] + l2d*Ga1T[2])) + 
                       np.exp(-2*np.pi*1j*(h*Unitary*Ga2T[0] + k2d*Ga2T[1] + l2d*Ga2T[2]))
                       )
    F = F_Eu + F_Al + F_Ga + noiseamplitude*np.random.rand(len(k2d), len(k2d)) 
    """Intensity I"""
    I = np.absolute(F)**2  # I \propto F(Q)^2, F complex
    ##############################################################################    
    #  Table for extracting unwanted K,L points
    #  q_cdw     0.1   0.1    0.2   0.2    0.5   0.5   0.125  0.125
    #  ∆k        0.1   0.01   0.1   0.01   0.1   0.01
    #  Kfactor1  1     1      1
    #  Kfactor2  9     99     9
    #  Lfactor1  1     10     1
    #  Lfactor2  9     9      9
    ########################################################################
    Kfactor1, Kfactor2, Lfactor1, Lfactor2 = 1, 99, 1, 9
    # #  Excluding unallowed K-points (ONLY FOR deltak=/1)
    k_intlist = np.arange(0, len(k2d), 1)  # erstelle indices aller k-Werte
    print("k_integer_initial={}".format(k_intlist))
    for i in range(0, (2 * kmax*Kfactor1 +1)):  # LEAVES ONLY INTEGER K-values
            # print(range(0,2*kmax+1))
            k_intlist = np.delete(k_intlist, i * Kfactor2)  #  n*9, since the list gets one less each time
            print("k_intlist={}".format(k_intlist))
    for i in k_intlist:  # Set unallowed K-values for intensities to 0
            I[:, i] = 0
    # #  Exluding unallowed L-points
    # l_intlist = np.arange(0, len(l2d), 1)  # erstelle indices aller l-Werte
    # print("l_integer_initial={}".format(l_intlist))
    # for i in range(0, 2 * kmax * Lfactor1 + 1):
    #     l_intlist = np.delete(l_intlist, i * Lfactor2)  # Lösche jeden zehnten index
    #     print("l_intlist={}".format(l_intlist))
    # for i in l_intlist:  # Set unallowed L-values for intensities to 0
    #     I[i, :] = 0
    # if deltak == 0.1:
    #     for i in range(0, 2 * kmax + 1):
    #         l_intlist = np.delete(l_intlist, i * Lfactor1)  # Lösche jeden zehnten index
    #     for i in l_intlist:  # Set unallowed L-values for intensities to 0
    #         I[i, :] = 0
    # else:
    #     for i in range(0, 2 * kmax * 10 + 1):
    #         l_intlist = np.delete(l_intlist, i * Lfactor1)  # Lösche jeden zehnten index
    #     for i in l_intlist:  # Set unallowed L-values for intensities to 0
    #         I[i, :] = 0
    ########################################################################
    I = I + noiseamplitude * np.random.rand(len(k2d), len(k2d))  # Add random noise with maximum 1
    #  Plotabschnitt
    fig = plt.figure(figsize=(15,4), dpi=100)
    plt.suptitle("EuGa2Al2, H={}".format(h))
    # LINECUTS
    plt.title("")
    plt.plot(l2d[:,0], I[:,0], ls='--', marker='.', label='K={}'.format(np.round(k[0], 2)), lw=0.5, ms=1.5)
    plt.plot(l2d[:,0], I[:,-1], ls='--', marker='.', label='K={}'.format(np.round(k[-1], 2)), lw=0.5, ms=1.5)
    plt.legend(loc='upper right')
    plt.ylabel(r"Intensity $I\propto F(\mathbf{Q})^2$")
    plt.xlabel("L(rlu)")
    
    fig = plt.figure(figsize=(15,4), dpi=100)
    #  INTERPOLATION
    plt.subplot(1, 3, 1)
    plt.title("Gaussian interpolation")
    plt.imshow(I, cmap='inferno',
               interpolation='gaussian',
               extent=(k0 - kmax, k0 + kmax, l0 - lmax, l0 + lmax),
               origin='lower',
               norm=LogNorm(vmin = 10, vmax = np.max(I))
               )
    plt.colorbar()
    plt.xlabel("K(rlu)")
    plt.ylabel("L(rlu)")
    # CONVOLUTION
    plt.subplot(1, 3, 3)
    plt.title("Gaussian convolution")
    x, y = np.linspace(-1,1,kernelsize), np.linspace(-1,1,kernelsize)
    X, Y = np.meshgrid(x, y)
    kernel = 1/(2*sigma**2)*np.exp(-(X**2+Y**2)/(2*sigma**2))
    Iconv = ndimage.convolve(I, kernel, mode='constant', cval=0.0)
    plt.imshow(Iconv, cmap='inferno', extent=(k0-kmax, k0+kmax, l0-lmax, l0+lmax), 
                   origin='lower',
                   #vmin=0, vmax=0.1*np.max(Iconv)
                   norm=LogNorm(vmin = 1, vmax = np.max(Iconv))
                   )
    plt.colorbar()
    plt.ylabel("L(rlu)")
    plt.xlabel("K(rlu)")     
    #  3D ATOMIC PLOT
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    # print("Eu={}, Al={}".format(Eu, Al))
    ax.scatter(Eu[0], Eu[1], Eu[2], label='Eu, Wyckoff 2a (000)', c='yellow')
    ax.scatter(EuT[0], EuT[1], EuT[2], label='Eu_T, Wyckoff 2a T(000)', facecolors='none', edgecolors='yellow')
    ax.scatter(Al1[0], Al1[1], Al1[2], label='Al1, Wyckoff 4d (1/2 0 1/4), (0 1/2 1/4)', c='blue')
    ax.scatter(Al1T[0], Al1T[1], Al1T[2], facecolors='none', edgecolors='blue')
    ax.scatter(Al2[0], Al2[1], Al2[2], c='blue')
    ax.scatter(Al2T[0], Al2T[1], Al2T[2], facecolors='none', edgecolors='blue')
    ax.scatter(Ga1[0], Ga1[1], Ga1[2], label='Ga1, Wyckoff 4e (0 0 z0), (0 0 -z0)', c='green')
    ax.scatter(Ga1T[0], Ga1T[1], Ga1T[2], facecolors='none', edgecolors='green')
    ax.scatter(Ga2[0], Ga2[1], Ga2[2], c='green')
    ax.scatter(Ga2T[0], Ga2T[1], Ga2T[2], facecolors='none', edgecolors='green')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    # #  2D Scatter plot
    # plt.figure()
    # plt.scatter(k2d, l2d, s=I/np.max(I), cmap='inferno', 
    #             label=r'$I \propto F(\mathbf{Q})^2$'
    #             )
    # plt.colorbar()
    # plt.legend()
    # plt.ylabel("L(rlu)")
    # plt.xlabel("K(rlu)")
    # plt.tight_layout()
    # plt.savefig("Map_BCC_{}UC_{}SC_A={}_q={}_H={}_center{}{}{}.jpg".format(n, int(n/int(q_cdw**(-1))), Amplitude, q_cdw, h, h, k0, l0), dpi=300)
    # plt.subplots_adjust(wspace=0.3)
        
        
    plt.show(block=False)
        

def main():
    ########################################################################
    #  General input
    #  --> Need to define atoms at unique wyckoff positions in Structurefactor
    k0, l0, kmax, lmax = 0, 0, 5, 5  # boundaries of K and L for the intensity maps
    deltak = 0.01  # k-space point distance
    h = 2  # H value in k space
    a, c = 4, 11  # Unit cell parameters in Å
    z_Ga = 0.38
    f_list = [1, 1]  # Atomic form factor list --> later as Q-dependent quantity
    #  Other
    noiseamplitude = 0.2  # Noise amplitude
    sigma, kernelsize = 0.1, int(0.1/deltak) # Gaussian Kernel parameters
    #  PROGRAM
    k2d, l2d, k, Unitary = kspacecreator(k0, l0, kmax, lmax, deltak)
    structurefactorandplotting(k0=k0, l0=l0, k2d=k2d, k=k, kmax=kmax, lmax=lmax, 
                               l2d=l2d, h=h, z_Ga=z_Ga, deltak=deltak, 
                               Unitary=Unitary, noiseamplitude=noiseamplitude, 
                               f_list=f_list, sigma=sigma, kernelsize=kernelsize,
                               savefig=False)

if __name__ == '__main__':
    main()