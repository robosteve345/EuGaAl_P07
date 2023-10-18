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
                               deltak, Unitary, u_list, Nz, kernelsize, 
                               noiseamplitude, sigma, kspacefactors, Lexclude=False, lognorm=True, 
                               normalization=False, DBW=False, savefig=False,
                               properatomicformfactor=False):
    for n in Nz:
        print("SIMULATION FOR N={} unmodulated unit cells".format(n))
        print("CENTERED PEAK: [HKL] = [{}{}{}]".format(h, k0, l0))
        
        """BCC unit cell Wyckoff positions"""
        x_Atom1, y_Atom1, z_Atom1 = [0], [0], [0]
        x_Atom2, y_Atom2, z_Atom2 = [0.5], [0.5], [0.5]
        
        """Atomic positions"""
        # #  Full translation of Atom1 & Atom2 for N unit cells
        for i in range(1, n):  # begins at 1 because unit cell 0 is already given 1, to n-1
            ztranslationpos(np.array([x_Atom1[0], y_Atom1[0], z_Atom1[0]]), x_Atom1, y_Atom1, z_Atom1, i)
            ztranslationpos(np.array([x_Atom2[0], y_Atom2[0], z_Atom2[0]]), x_Atom2, y_Atom2, z_Atom2, i)

        #  Final positions
        Atom1, Atom2 = np.array([x_Atom1, y_Atom1, z_Atom1]), np.array([x_Atom2, y_Atom2, z_Atom2])
        
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
        for i in range(0, len(Atom1[0])):
            F_Atom1.append(
                f_Atom1 * DBW_list[0] * np.exp(-2 * np.pi * 1j * (h * Unitary * 
                        Atom1[0, i] + k2d * Atom1[1, i] + l2d * Atom1[2, i])))
            F_Atom2.append(
                f_Atom2 * DBW_list[1] * np.exp(-2 * np.pi * 1j * (h * Unitary * 
                        Atom2[0, i] + k2d * Atom2[1, i] + l2d * Atom2[2, i])))
        F_init = np.zeros((len(k2d), len(k2d)), dtype=complex)  # quadratic 0-matrix with dimensions of k-space
        F_list = [F_Atom1, F_Atom2]
        pre_F_init = [np.zeros((len(k2d), len(k2d)))]
        
        #  Zusammenfügen der Formfaktoren für die Atomsorten
        for i in F_list:  # put together the lists in a ndarray for each atom with each N positions, to get rid of 3rd dimension (better workaround probably possible...)
            pre_F_init = np.add(pre_F_init, i)
        for i in range(len(pre_F_init)):
            F_init = F_init + pre_F_init[i]
            
        #  Compute Intensity
        I = np.abs(np.round(F_init, 3)) ** 2  # I \propto F(Q)^2, F complex
        # I = I + noiseamplitude * np.random.rand(len(k2d), len(k2d))  # Add random noise with maximum 1
        
        # Compute normalization
        if normalization == True:
            I = I/np.max(I)
        else:
            I = I
            
        # Exclude k-space points 
        I = excludekspacepoints(kspacefactors, k2d, l2d, deltak, I, noiseamplitude, 
                                kmax, lmax, Lexclude)

        
        # # PLOTTING
        fig = plt.figure(figsize=(15,4), dpi=100)
        plt.suptitle("BCC, {} unit cells".format(n))
        #  LINECUTS
        plt.subplot(1, 1, 1)
        plt.title(r'Centered around [{}{}{}]'.format(h, k0, l0))
        plt.plot(l2d[:, 0], I[:, len(k) // 2], ls='-', lw=0.5, marker='.', 
                 ms=1, label='K={}'.format(np.round(k[len(k) // 2], 2)))
        plt.plot(l2d[:, 0], I[:, 0], ls='-', marker='.', lw=0.5, 
                 ms=1, label='K={}'.format(np.round(k[0], 2)))
        plt.legend()
        plt.ylabel(r"Intensity $I\propto F(\mathbf{Q})^2$")
        plt.xlabel("L(rlu)")
        if savefig == True:
            plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/XRD_Simulation/BCC_unmodulated/Linecuts_BCC_{}UC_H={}_center{}{}{}.jpg".format(n, h, h, k0, l0), dpi=300)
        else:
            pass
        
        plt.subplots_adjust(wspace=0.3)
        
        fig = plt.figure(figsize=(15,4), dpi=100)
        plt.suptitle("BCC, {} unit cells".format(n))
        #  INTERPOLATION
        plt.subplot(1, 3, 1)
        plt.title("Gaussian interpolation, H={}".format(h))
        if lognorm == True:
            plt.imshow(I, cmap='inferno',
                       interpolation='gaussian',
                       extent=(k0 - kmax, k0 + kmax, l0 - lmax, l0 + lmax),
                       origin='lower',
                       norm=LogNorm(vmin = 10, vmax = np.max(I))
                       )
        else:
            plt.imshow(I, cmap='inferno',
                       interpolation='gaussian',
                       extent=(k0 - kmax, k0 + kmax, l0 - lmax, l0 + lmax),
                       origin='lower',
                       )
            
        plt.colorbar()
        plt.xlabel("K(rlu)")
        plt.ylabel("L(rlu)")

        # CONVOLUTION
        plt.subplot(1, 3, 3)
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
        ax = fig.add_subplot(1,3,2, projection='3d')
        ax.scatter(Atom1[0], Atom1[1], Atom1[2], label='Atom1')
        ax.scatter(Atom2[0], Atom2[1], Atom2[2], label='Atom2')
        # ax.set_xlim(-0.5, 0.5)
        # ax.set_ylim(-0.5, 0.5)
        # ax.set_zlim(-2 * kmax, 2 * kmax)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.legend()
        
        plt.show(block=False)


def main():
    ########################################################################
    #  GENERAL INPUT
    k0, l0, kmax, lmax = 0, 0, 5, 5  # boundaries of K and L for the intensity maps
    deltak = 0.1  # k-space point distance
    h = 1  # H value in k space
    a, c = 1, 1  # Unit Cell Parameters
    #####################################
    #  INPUT FOR DBW
    u_list = [1e-3, 1e-3]  # Isotropic displacements <u^2> in Å^2, ~1/m. For more detail take bonding angles, distances into account.
    #####################################
    noiseamplitude = 1e-4  # Noise amplitude
    sigma, kernelsize = 0.5, int(0.5/deltak) # Gaussian Kernel parameters
    
    #  OTHER
    kspacefactors = [1, 9, 1, 9]
    Nz = [20]  #  # of unmodulated unit cells in x direction
    #  PROGRAM
    st = time.time()
    k2d, l2d, k, Unitary = kspacecreator(k0, l0, kmax, lmax, deltak)
    structurefactorandplotting(a, c, k0, l0, k2d, k, kmax, lmax, l2d, h, deltak=deltak, 
                               Unitary=Unitary, u_list=u_list,
                               Nz=Nz, kernelsize = kernelsize, 
                               noiseamplitude=noiseamplitude, sigma=sigma,
                               kspacefactors=kspacefactors, Lexclude=False,
                               normalization=True, DBW=True, 
                               lognorm=False, savefig=False,
                               properatomicformfactor=False)

    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

if __name__ == '__main__':
        main()
    
