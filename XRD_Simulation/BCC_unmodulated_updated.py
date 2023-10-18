#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 20:39:47 2023

@author: stevengebel
"""

import numpy as np
import matplotlib
import time
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.colors import LogNorm
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rc('text', usetex=True)
# mpl.rcParams.update(mpl.rcParamsDefault)
from XRD_simulation_functions import ztranslationpos, kspacecreator, excludekspacepoints, debyewallerfactor
from XRD_simulation_functions import atomicformfactorBCC

"""TO DO
-Implement deltak possibilities for 0.1&0.01 in L for q_cdw=0.2, 0.1 & 0.5
-Implement possibility for HKL_0 & HK_0L maps
-Implement input-interface: kß, lß, deltak, Amplitude, q_cdw, kmax
"""


def structurefactorandplotting(a, c, k0, l0, k2d, k, kmax, lmax, l2d, h, 
                               deltak, Unitary, u_list, kernelsize, 
                               noiseamplitude, sigma, lognorm=True, 
                               normalization=False, DBW=False, savefig=False,
                               properatomicformfactor=False):
    # print("SIMULATION FOR N={} unmodulated unit cells".format(n))
    print("CENTERED PEAK: [HKL] = [{}{}{}]".format(h, k0, l0))
    
    # """BCC unit cell Wyckoff positions"""
    # x_Atom1, y_Atom1, z_Atom1 = [0], [0], [0]
    # x_Atom2, y_Atom2, z_Atom2 = [0.5], [0.5], [0.5]
    
    """Atomic positions"""
    # #  Full translation of Atom1 & Atom2 for N unit cells
    # for i in range(1, n):  # begins at 1 because unit cell 0 is already given 1, to n-1
    #     ztranslationpos(np.array([x_Atom1[0], y_Atom1[0], z_Atom1[0]]), x_Atom1, y_Atom1, z_Atom1, i)
    #     ztranslationpos(np.array([x_Atom2[0], y_Atom2[0], z_Atom2[0]]), x_Atom2, y_Atom2, z_Atom2, i)

    # #  Final positions
    # Atom1, Atom2 = np.array([x_Atom1, y_Atom1, z_Atom1]), np.array([x_Atom2, y_Atom2, z_Atom2])
    
    # SECOND METHOD
    Nxy_neg, Nxy_pos = 0, 1
    Nz = 1000
    
    one_directionxy = np.arange(Nxy_neg, Nxy_pos, 1.0)
    one_directionz = np.arange(0, Nz, 1.0)
    # Create the NumPy array using a combination of meshgrid and reshape
    x, y, z = np.meshgrid(one_directionxy, one_directionxy, one_directionz, indexing='ij')
    baseposition1 = np.vstack((x.ravel(), y.ravel(), z.ravel()))
    baseposition2 = np.vstack((x.ravel(), y.ravel(), z.ravel()))
    
    # Atom1
    position1, position2, = baseposition1, baseposition2
    # Add a constant factor 'a' to every element in the array
    a = 0.5  # Translation für Atom2-Position

    for i in range(position2.shape[0]):
        for j in range(position2.shape[1]):
            position2[i, j] = position2[i, j] + a
    
    # Modulation
    A, q_cdw  = 0.2, 0.2
    for i in range(0, len(position2[2])):  # begins with 1 because it evaluates every unit cell, 0 to n-1, BLEIBT SO
        position2[2][i] = position2[2][i] + A * np.cos(q_cdw * 2 * np.pi * position2[2][i]) # damit i: 1...n und nicht 0...n-1
    
    # Füge zusammen zu ndarrays
    Atom2 = np.array([position2[0], position2[1], position2[2]])   # Transpose to match the desired shape (3, 8)
    Atom1 = np.array([position1[0], position1[1], position1[2]])
    
    #  Compute Debye-Waller-Factor
    if DBW == True:
        DBW_list = debyewallerfactor(k2d, l2d, Unitary, h, a, c, u_list)
    else:
        DBW_list = np.ones(3)
        
    """Scattering amplitudes F"""
    #  Atomic form factors
    f_Atom1, f_Atom2 = atomicformfactorBCC(h, k2d, l2d, Unitary, properatomicformfactor)
    
    #  Crystal structure factor for each atom
    F_Atom1, F_Atom2 = [], []
    print("# of Unit cells = {}".format(len(Atom1[0])))
    for i in range(0, len(Atom1[0])):
        F_Atom1.append(
            f_Atom1 * DBW_list[0] * np.exp(-2 * np.pi * 1j * 
        (h * Unitary * Atom1[0, i] + k2d * Atom1[1, i] + l2d * Atom1[2, i])))
        F_Atom2.append(
            f_Atom2 * DBW_list[1] * np.exp(-2 * np.pi * 1j * 
        (h * Unitary * Atom2[0, i] + k2d * Atom2[1, i] + l2d * Atom2[2, i])))
    F_init = np.zeros((len(k2d), len(k2d)), dtype=complex)  # quadratic 0-matrix with dimensions of k-space
    F_list = [F_Atom1, F_Atom2]
    pre_F_init = [np.zeros((len(k2d), len(k2d)))]
    
    #  Zusammenfügen der Formfaktoren für die Atomsorten
    for i in F_list:  # put together the lists in a ndarray for each atom with each N positions, to get rid of 3rd dimension (better workaround probably possible...)
        pre_F_init = np.add(pre_F_init, i)
    for i in range(len(pre_F_init)):
        F_init = F_init + pre_F_init[i]
    # print(F_init)
    #  Compute Intensity
    I = np.abs(np.round(F_init, 3)) ** 2  # I \propto F(Q)^2, F complex
    # I = I + noiseamplitude * np.random.rand(len(k2d), len(k2d))  # Add random noise with maximum 1
    
    # Compute normalization
    if normalization == True:
        I = I/np.max(I)
    else:
        I = I
        
    # # Exclude k-space points (OBSOLETE)
    # I = excludekspacepoints(kspacefactors, k2d, l2d, deltak, I, noiseamplitude, 
    #                         kmax, lmax, Lexclude)

    
    # # PLOTTING
    fig = plt.figure(figsize=(15,2), dpi=100)
    plt.suptitle("Body centered cubic (bcc), Centered around [{}{}{}]".format(h, k0, l0))
    #  LINECUTS
    plt.subplot(1, 1, 1)
    plt.plot(l2d[:, 0], I[:, len(k) // 2], ls='-', lw=0.5, marker='.', 
             ms=1, label='K={}'.format(np.round(k[len(k) // 2], 2)))
    plt.plot(l2d[:, 0], I[:, 0], ls='-', marker='.', lw=0.5, 
             ms=1, label='K={}'.format(np.round(k[0], 2)))
    plt.legend()
    plt.xlim(l0-kmax, l0+kmax)
    plt.ylabel(r"Intensity $I\propto F(\mathbf{Q})^2$")
    plt.xlabel("L(rlu)")
    if savefig == True:
        plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/XRD_Simulation/BCC_unmodulated/Linecuts_BCC_H={}_center{}{}{}_updated.jpg".format(h, h, k0, l0), dpi=300)
    else:
        pass
    plt.subplots_adjust(wspace=0.3)
    
    fig = plt.figure(figsize=(8, 4))
    plt.suptitle("Body centered cubic (bcc)")

    # CONVOLUTION
    plt.subplot(1, 2, 1)
    plt.title("Gaussian convolution, H={}".format(h))
    x, y = np.linspace(-1,1,kernelsize), np.linspace(-1,1,kernelsize)
    X, Y = np.meshgrid(x, y)
    kernel = 1/(2*sigma**2)*np.exp(-(X**2+Y**2)/(2*sigma**2))
    Iconv = ndimage.convolve(I, kernel, mode='constant', cval=0.0)
    if lognorm == True:
        plt.imshow(Iconv, cmap='inferno', extent=(k0-kmax, k0+kmax, l0-lmax, l0+lmax), 
               origin='lower',
               #vmin=0, vmax=0.1*np.max(Iconv)
               norm=LogNorm(vmin = 1, vmax = np.max(Iconv))
               )
    else:
        plt.imshow(Iconv, cmap='inferno', extent=(k0-kmax, k0+kmax, l0-lmax, l0+lmax), 
               origin='lower',
               )
            
    plt.colorbar()
    plt.ylabel("L(rlu)")
    plt.xlabel("K(rlu)")     
    #  3D ATOM PLOT
    ax = fig.add_subplot(1,2,2, projection='3d')
    # Atomlist = np.concatenate((Atom1, Atom2), axis=1)
    # print("Atom1={}, Atom2={}".format(Atom1, Atom2))
    # print("Atomlist={}".format(Atomlist))
    ax.scatter(position1[0], position1[1], position1[2], label='Atom1')
    ax.scatter(position2[0], position2[1], position2[2], label='Atom2')
    # ax.set_xlim(-0.5, 0.5)
    # ax.set_ylim(-0.5, 0.5)
    # ax.set_zlim(-2 * kmax, 2 * kmax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    
    plt.show(block=False)


def main():
    ##########################################################################
    ########################## GENERAL INPUT #################################
    ##########################################################################
    k0, l0, kmax, lmax = 10, 5, 1, 1  # boundaries of K and L 
    deltak = 0.01  # k-space point distance
    h = 5  # H value in k space
    a, c = 1, 1  # Unit Cell Parameters
    ##########################################################################
    #  INPUT FOR DBW
    u_list = [1e-3, 1e-3]  # Isotropic displacements <u^2> in Å^2, ~1/m. For more detail take bonding angles, distances into account.
    ##########################################################################
    noiseamplitude = 1e-4  # Noise amplitude
    sigma, kernelsize = 0.5, int(0.1/deltak) # Gaussian Kernel parameters
    
    
    #  PROGRAM
    st = time.time()
    k2d, l2d, k, Unitary = kspacecreator(k0, l0, kmax, lmax, deltak)
    structurefactorandplotting(a, c, k0, l0, k2d, k, kmax, lmax, l2d, h, deltak=deltak, 
                               Unitary=Unitary, u_list=u_list,
                               kernelsize = kernelsize, 
                               noiseamplitude=noiseamplitude, sigma=sigma,
                               normalization=False, DBW=False, 
                               lognorm=False, savefig=False,
                               properatomicformfactor=False)

    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

if __name__ == '__main__':
        main()
    
