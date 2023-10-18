#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 18:13:00 2023

@author: stevengebel
"""

import numpy as np
import matplotlib
import time
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.colors import LogNorm
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
# matplotlib.rcParams['font.family'] = "sans-serif"
# matplotlib.rc('text', usetex=True)
import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)
from XRD_simulation_functions import ztranslationpos, kspacecreator, excludekspacepoints, debyewallerfactor
from XRD_simulation_functions import atomicformfactorEuGa2Al2

"""TO DO
-Implement deltak possibilities for 0.1&0.01 in L for q_cdw=0.2, 0.1 & 0.5
-Implement possibility for HKL_0 & HK_0L maps
-Implement input-interface: kß, lß, deltak, Amplitude, q_cdw, kmax
"""


def structurefactorandplotting(a, c, k0, l0, k2d, k, kmax, lmax, l2d, h, deltak, 
                               Unitary, u_list, z0, N, 
                               kernelsize, noiseamplitude, sigma, kspacefactors, 
                               Lexclude=False, lognorm=True, properatomicformfactor=False, 
                               normalization=False, DBW=False, savefig=False):
    for n in N:
        print("SIMULATION FOR N={} cells".format(n))
        print("CENTERED PEAK: [HKL] = [{}{}{}]".format(h, k0, l0))
        """EuGa2Al2 unit cell Wyckoff positions"""
        #  Europium:
        x_Eu, y_Eu, z_Eu = [0], [0], [0]
        x_Eu_T, y_Eu_T, z_Eu_T = [0.5], [0.5], [0.5]
        #  Aluminium:
        x_Al1, y_Al1, z_Al1 = [0], [0.5], [0.25]
        x_Al1_T, y_Al1_T, z_Al1_T = [0.5], [1], [0.75]
        x_Al2, y_Al2, z_Al2 = [0.5], [0], [0.25]
        x_Al2_T, y_Al2_T, z_Al2_T = [1], [0.5], [0.75]
        #  Gallium:
        x_Ga1, y_Ga1, z_Ga1 = [0], [0], [z0]
        x_Ga1_T, y_Ga1_T, z_Ga1_T = [0.5], [0.5], [0.5 + z0]
        x_Ga2, y_Ga2, z_Ga2 = [0], [0], [-z0]
        x_Ga2_T, y_Ga2_T, z_Ga2_T = [0.5], [0.5], [0.5 - z0]

        """Atomic positions modulation"""
        # Full translation of Wyckoff 2a(Eu), 4d(Al) & 4e(Ga) for N unit cells
        for i in range(1, n):
            ztranslationpos(np.array([x_Eu[0], y_Eu[0], z_Eu[0]]), x_Eu, y_Eu, z_Eu, i)
            ztranslationpos(np.array([x_Eu_T[0], y_Eu_T[0], z_Eu_T[0]]), x_Eu_T, y_Eu_T, z_Eu_T, i)
            ztranslationpos(np.array([x_Al1[0], y_Al1[0], z_Al1[0]]), x_Al1, y_Al1, z_Al1, i)
            ztranslationpos(np.array([x_Al1_T[0], y_Al1_T[0], z_Al1_T[0]]), x_Al1_T, y_Al1_T, z_Al1_T, i)
            ztranslationpos(np.array([x_Al2[0], y_Al2[0], z_Al2[0]]), x_Al2, y_Al2, z_Al2, i)
            ztranslationpos(np.array([x_Al2_T[0], y_Al2_T[0], z_Al2_T[0]]), x_Al2_T, y_Al2_T, z_Al2_T, i)
            ztranslationpos(np.array([x_Ga1[0], y_Ga1[0], z_Ga1[0]]), x_Ga1, y_Ga1, z_Ga1, i)
            ztranslationpos(np.array([x_Ga1_T[0], y_Ga1_T[0], z_Ga1_T[0]]), x_Ga1_T, y_Ga1_T, z_Ga1_T, i)
            ztranslationpos(np.array([x_Ga2[0], y_Ga2[0], z_Ga2[0]]), x_Ga2, y_Ga2, z_Ga2, i)
            ztranslationpos(np.array([x_Ga2_T[0], y_Ga2_T[0], z_Ga2_T[0]]), x_Ga2_T, y_Ga2_T, z_Ga2_T, i)

        #  Final atomic positions
        Eu, Eu_T, Al1, Al1_T, Al2, Al2_T, Ga1, Ga1_T, Ga2, Ga2_T = np.array([x_Eu, y_Eu, z_Eu]), np.array([x_Eu_T, y_Eu_T, z_Eu_T]),\
                                       np.array([x_Al1, y_Al1, z_Al1]), np.array([x_Al1_T, y_Al1_T, z_Al1_T]),\
                                       np.array([x_Al2, y_Al2, z_Al2]), np.array([x_Al2_T, y_Al2_T, z_Al2_T]), \
                                       np.array([x_Ga1, y_Ga1, z_Ga1]), np.array([x_Ga1_T, y_Ga1_T, z_Ga1_T]), \
                                       np.array([x_Ga2, y_Ga2, z_Ga2]), np.array([x_Ga2_T, y_Ga2_T, z_Ga2_T])
        
        #  Compute Debye-Waller-Factor
        if DBW == True:
            DBW_list = debyewallerfactor(k2d, l2d, Unitary, h, a, c, u_list)
        else:
            DBW_list = np.ones(3)
        
        """Scattering amplitudes F"""
        #  Crystal structure factor for each atom at each Wyckoff position with respect to the tI symmtry +1/2 transl.
        F_Eu, F_Eu_T, F_Al1, F_Al1_T, F_Al2, F_Al2_T, F_Ga1, F_Ga1_T, \
        F_Ga2, F_Ga2_T = [], [], [], [], [], [], [], [], [], []
        # Calculate atomic form factors
        f_Eu, f_Ga, f_Al = atomicformfactorEuGa2Al2(h, k2d, l2d, Unitary, properatomicformfactor)
        for i in range(0, n):
            F_Eu.append(f_Eu * DBW_list[0] * np.exp(-2 * np.pi * 1j * (h * Unitary * Eu[0, i] + 
                                                      k2d * Eu[1, i] + 
                                                      l2d * Eu[2, i]))
            )
            F_Eu_T.append(f_Eu * DBW_list[0] * np.exp(-2 * np.pi * 1j * (h * Unitary * Eu_T[0, i] + 
                                                      k2d * Eu_T[1, i] + 
                                                      l2d * Eu_T[2, i]))
            )
            F_Al1.append(f_Al * DBW_list[1] * np.exp(-2 * np.pi * 1j * (h * Unitary * Al1[0, i] + 
                                                      k2d * Al1[1, i] + 
                                                      l2d * Al1[2, i]))
            )
            F_Al1_T.append(f_Al * DBW_list[1] *np.exp(-2 * np.pi * 1j * (h * Unitary * Al1_T[0, i] + 
                                                      k2d * Al1_T[1, i] + 
                                                      l2d * Al1_T[2, i]))
            )
            F_Al2.append(f_Al * DBW_list[1] *np.exp(-2 * np.pi * 1j * (h * Unitary * Al2[0, i] + 
                                                      k2d * Al2[1, i] + 
                                                      l2d * Al2[2, i]))
            )
            F_Al2_T.append(f_Al * DBW_list[1] *np.exp(-2 * np.pi * 1j * (h * Unitary * Al2_T[0, i] + 
                                                      k2d * Al2_T[1, i] + 
                                                      l2d * Al2_T[2, i]))
            )
            F_Ga1.append(f_Ga * DBW_list[2] * np.exp(-2 * np.pi * 1j * (h * Unitary * Ga1[0, i] + 
                                                      k2d * Ga1[1, i] + 
                                                      l2d * Ga1[2, i]))
            )
            F_Ga1_T.append(f_Ga * DBW_list[2] * np.exp(-2 * np.pi * 1j * (h * Unitary * Ga1_T[0, i] + 
                                                      k2d * Ga1_T[1, i] + 
                                                      l2d * Ga1_T[2, i]))
            )
            F_Ga2.append(f_Ga * DBW_list[2] * np.exp(-2 * np.pi * 1j * (h * Unitary * Ga2[0, i] + 
                                                      k2d * Ga2[1, i] + 
                                                      l2d * Ga2[2, i]))
            )
            F_Ga2_T.append(f_Ga * DBW_list[2] * np.exp(-2 * np.pi * 1j * (h * Unitary * Ga2_T[0, i] + 
                                                      k2d * Ga2_T[1, i] + 
                                                      l2d * Ga2_T[2, i]))
            )
        F_init = np.zeros((len(k2d), len(k2d)), dtype=complex)  # quadratic 0-matrix with dimensions of k-space
        F_list = [F_Eu, F_Eu_T, F_Al1, F_Al1_T, F_Al2, F_Al2_T, F_Ga1, F_Ga1_T, F_Ga2, F_Ga2_T]
        pre_F_init = [np.zeros((len(k2d), len(k2d)))]
        #  Zusammenfügen der Formfaktoren für die Atomsorten
        for i in F_list:  # put together the lists in a ndarray for each atom with each N positions, to get rid of 3rd dimension (better workaround probably possible...)
            pre_F_init = np.add(pre_F_init, i)
        for i in range(len(pre_F_init)):
            F_init = F_init + pre_F_init[i]
            
        #  Compute Intensity 
        I = np.abs(np.round(F_init, 3)) ** 2  # I \propto F(Q)^2, F complex
        
        # Normalize Intensity
        if normalization == True:
            I = I/np.max(I)
        else:
            I = I
            
        # Exclude k-space points
        I = excludekspacepoints(kspacefactors, k2d, l2d, deltak, I, noiseamplitude, 
                                kmax, lmax, Lexclude)
        
        # # PLOTTING
        fig = plt.figure(figsize=(15, 4), dpi=100)
        plt.suptitle("EuGa2Al2, {} unit cells".format(n))
        # #  INTERPOLATION
        # plt.subplot(2, 2, 2)
        # plt.title("Gaussian interpolation")
        # if lognorm == True:
        #     plt.imshow(I, cmap='inferno',
        #                interpolation='gaussian',
        #                extent=(k0 - kmax, k0 + kmax, l0 - lmax, l0 + lmax),
        #                origin='lower',
        #                norm=LogNorm(vmin = 10, vmax = np.max(I))
        #                )
        # else:
        #     plt.imshow(I, cmap='inferno',
        #                interpolation='gaussian',
        #                extent=(k0 - kmax, k0 + kmax, l0 - lmax, l0 + lmax),
        #                origin='lower',
        #                )
        # plt.colorbar()
        # plt.xlabel("K(rlu)")
        # plt.ylabel("L(rlu)")
        #  2D SCATTER PLOT
        plt.subplot(1, 3, 1)
        plt.scatter(k2d, l2d, label=r'$I \propto F(\mathbf{Q})^2$'
                    , s=I / np.max(I),
                    # , c = I / np.max(I),
                    )
        plt.colorbar()
        plt.legend()
        plt.ylabel("L(rlu)")
        plt.xlabel("K(rlu)")
        plt.tight_layout()
        
        # CONVOLUTION
        plt.subplot(1, 3, 3)
        plt.title("Gaussian convolution")
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
        #  3D ATOMIC PLOT
        ax = fig.add_subplot(1, 3, 2, projection='3d')
        # ax.title(r"(Modulated) unit cell(s) with Wyckoff 4e z-parameter z0={} and $T=(1/2 1/2 1/2)$".format(z0))
        ax.scatter(Eu[0], Eu[1], Eu[2], label='Eu, Wyckoff 2a (000)', c='yellow')
        ax.scatter(Eu_T[0], Eu_T[1], Eu_T[2], label='Eu_T, Wyckoff 2a T(000)', facecolors='none', edgecolors='yellow')
        ax.scatter(Al1[0], Al1[1], Al1[2], label='Al1, Wyckoff 4d (1/2 0 1/4), (0 1/2 1/4)', c='blue')
        ax.scatter(Al1_T[0], Al1_T[1], Al1_T[2], facecolors='none', edgecolors='blue')
        ax.scatter(Al2[0], Al2[1], Al2[2], c='blue')
        ax.scatter(Al2_T[0], Al2_T[1], Al2_T[2], facecolors='none', edgecolors='blue')
        ax.scatter(Ga1[0], Ga1[1], Ga1[2], label='Ga1, Wyckoff 4e (0 0 z0), (0 0 -z0)', c='green')
        ax.scatter(Ga1_T[0], Ga1_T[1], Ga1_T[2], facecolors='none', edgecolors='green')
        ax.scatter(Ga2[0], Ga2[1], Ga2[2], c='green')
        ax.scatter(Ga2_T[0], Ga2_T[1], Ga2_T[2], facecolors='none', edgecolors='green')
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.legend()
        if savefig == True:
            plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/XRD_Simulation/EuGa2Al2_unmodulated/{}UC_H={}_center{}{}{}_diffpattern.jpg".format(n, h, h, k0, l0), dpi=300)
        else:
            pass
        
        fig = plt.figure(figsize=(15, 2), dpi=100)
        plt.suptitle("EuGa2Al2, {} unit cells, centered around [{}{}{}]".format(n, h, k0, l0))
        #  LINECUTS
        plt.plot(l2d[:, 0], I[:, len(k) // 2], ls='--', lw=0.5, marker='.', 
                 ms=1.5, label='K={}'.format(np.round(k[len(k) // 2], 2)))
        plt.plot(l2d[:, 0], I[:, 0], ls='--', marker='.', lw=0.5, ms=1.5, 
                 label='K={}'.format(np.round(k[0], 2)))
        plt.legend()
        plt.ylabel(r"Intensity $I\propto F(\mathbf{Q})^2$")
        plt.xlabel("L(rlu)")
        if savefig == True:
            plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/XRD_Simulation/EuGa2Al2_unmodulated/{}UC_H={}_center{}{}{}.jpg".format(n, h, h, k0, l0), dpi=300)
        else:
            pass
        
        plt.show(block=False)


def euga2al2modulation1():
    ########################################################################
    """INPUT PARAMETERS"""
    #  --> Need to define atoms at unique wyckoff positions in Structurefactor
    k0, l0, kmax, lmax = 0, 0, 10, 10  # boundaries of K and L for the intensity maps
    z0 = 0.38  #  Wyckoff 4e z-parameter
    a, c = 4, 11 #  Unit cell parameters in Angstrom
    deltak = 0.1  #  k-space point distance
    h = 9  #  H value in k space
    u_Eu, u_Ga, u_Al = 1e-1, 1, 1 # Isotropic displacements <u^2> in Å^2, ~1/m. For more detail take bonding angles, distances into account.
    u_list = [u_Eu, u_Eu, u_Eu]
    #####################################
    #  OTHER
    kspacefactors = [1, 9, 1, 9]
    sigma, kernelsize = 0.2, int(0.5/deltak) # Gaussian Kernel parameters
    N = [1, 2, 10]  
    noiseamplitude = 1e-5  #  Noise amplitude
    ########################################################################
    #  PROGRAM
    st = time.time()
    k2d, l2d, k, Unitary = kspacecreator(k0, l0, kmax, lmax, deltak)
    structurefactorandplotting(a, c, k0=k0, l0=l0, k2d=k2d, k=k, kmax=kmax, 
                               lmax=lmax, l2d=l2d, h=h, deltak=deltak,
                               Unitary=Unitary, u_list=u_list,
                               z0=z0, N=N, kernelsize=kernelsize, 
                               noiseamplitude=noiseamplitude, sigma=sigma,
                               kspacefactors=kspacefactors, properatomicformfactor=True,
                               DBW=True, savefig=True, Lexclude=False)
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

if __name__ == '__main__':
    euga2al2modulation1()
    # TO DO: calculate modulation just with z_Ga1...2_T and add up at the end + compare with mareins definition, plot dz values vs sin#8z_Ga1...2_T) and compare

