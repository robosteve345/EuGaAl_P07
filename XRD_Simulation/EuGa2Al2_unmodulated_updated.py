#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Created on Thu Oct 19 14:02:00 2023

#@author: stevengebel
"""XRD Structure Factor F(Q) simulation for Eu(Ga,Al)4"""

import numpy as np
import matplotlib as mpl
import time
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.colors import LogNorm
from XRD_simulation_functions import kspacecreator, debyewallerfactor
from XRD_simulation_functions import atomicformfactorEuGaAl, excludekspacepoints


def structurefactorandplotting(a, c, k0, l0, k2d, k, l, kmax, lmax, l2d, h, 
                               deltak, Unitary, u_list, kernelsize, Nz, kspacefactors, z0,
                               noiseamplitude, sigma, lognorm=False, 
                               normalization=False, DBW=False, savefig=False,
                               Lexclude=False,
                               properatomicformfactor=False, EuAl4=False):
    print("CENTERED PEAK: [HKL] = [{}{}{}]".format(h, k0, l0))
    print("K-space point density ∆k = {}".format(deltak))
    print("H = {}, K, L in {} to {}".format(h, -kmax, kmax))
    """Atomic positions"""
    # Europium Z=63:
    x_Eu, y_Eu, z_Eu = 0, 0, 0
    #  Aluminium, Z=13:
    x_Al1, y_Al1, z_Al1 = 0, 0.5, 0.25
    x_Al2, y_Al2, z_Al2 = 0.5, 0, 0.25
    #  Gallium, Z=31:
    x_Ga1, y_Ga1, z_Ga1 = 0, 0, z0
    x_Ga2, y_Ga2, z_Ga2 = 0, 0, -z0
    # Create multiple unit cells
    if Lexclude == True:
        Nz = 1
        Nxy_neg, Nxy_pos = 0, 1
        
    else:
        Nz = Nz
        Nxy_neg, Nxy_pos = 0, int(np.sqrt(Nz/0.20))
    Nz_neg, Nz_pos = 0, Nz
    
    print(r"Ratio of unit cells Nz/Nxy = {}".format(np.round(Nz/Nxy_pos**2, 2)))
    print(r"Stacked Unit Cells in z-direction Nz = {}".format(Nz))
    print(r"Stacked Unit Cells in xy-direction Nxy=Nx*Ny = {}".format(Nxy_pos**2))
    print(r"For a sufficient convolution, Nxy is given by Nz: Nxy = int(np.sqrt(Nz/0.20))")
    print(r"Debye-Waller-Factor = {}".format(DBW))
    print(r"Q-dependent atomic form factors f(Q) = {}".format(properatomicformfactor))
    print(r"Simulation for EuAl4 = {}".format(EuAl4))

    # Europium
    init_Eu_x = np.arange(Nxy_neg + x_Eu, Nxy_pos + x_Eu, 1.0)
    init_Eu_y = np.arange(Nxy_neg + y_Eu, Nxy_pos + y_Eu, 1.0)
    init_Eu_z = np.arange(Nz_neg + z_Eu, Nz_pos + z_Eu, 1.0)
    x, y, z = np.meshgrid(init_Eu_x, init_Eu_y, init_Eu_z, indexing='ij')
    bpEu = np.vstack((x.ravel(), y.ravel(), z.ravel()))
    bpEu_T = np.vstack((x.ravel(), y.ravel(), z.ravel()))
    # Aluminum
    init_Al1_x = np.arange(Nxy_neg + x_Al1, Nxy_pos + x_Al1, 1.0)
    init_Al1_y = np.arange(Nxy_neg + y_Al1, Nxy_pos + y_Al1, 1.0)
    init_Al1_z = np.arange(Nz_neg + z_Al1, Nz_pos + z_Al2, 1.0)
    x, y, z = np.meshgrid(init_Al1_x, init_Al1_y, init_Al1_z, indexing='ij')
    bpAl1 = np.vstack((x.ravel(), y.ravel(), z.ravel()))
    bpAl1_T = np.vstack((x.ravel(), y.ravel(), z.ravel()))
    
    init_Al2_x = np.arange(Nxy_neg + x_Al2, Nxy_pos + x_Al2, 1.0)
    init_Al2_y = np.arange(Nxy_neg + y_Al2, Nxy_pos + y_Al2, 1.0)
    init_Al2_z = np.arange(Nz_neg + z_Al2, Nz_pos + z_Al2, 1.0)
    x, y, z = np.meshgrid(init_Al2_x, init_Al2_y, init_Al2_z, indexing='ij')
    bpAl2 = np.vstack((x.ravel(), y.ravel(), z.ravel()))
    bpAl2_T = np.vstack((x.ravel(), y.ravel(), z.ravel()))
    # Gallium
    init_Ga1_x = np.arange(Nxy_neg + x_Ga1, Nxy_pos + x_Ga1, 1.0)
    init_Ga1_y = np.arange(Nxy_neg + y_Ga1, Nxy_pos + y_Ga1, 1.0)
    init_Ga1_z = np.arange(Nz_neg + z_Ga1, Nz_pos + z_Ga2, 1.0)
    x, y, z = np.meshgrid(init_Ga1_x, init_Ga1_y, init_Ga1_z, indexing='ij')
    bpGa1 = np.vstack((x.ravel(), y.ravel(), z.ravel()))
    bpGa1_T = np.vstack((x.ravel(), y.ravel(), z.ravel()))
    
    init_Ga2_x = np.arange(Nxy_neg + x_Ga2, Nxy_pos + x_Ga2, 1.0)
    init_Ga2_y = np.arange(Nxy_neg + y_Ga2, Nxy_pos + y_Ga2, 1.0)
    init_Ga2_z = np.arange(Nz_neg + z_Ga2, Nz_pos + z_Ga2, 1.0)
    x, y, z = np.meshgrid(init_Ga2_x, init_Ga2_y, init_Ga2_z, indexing='ij')
    bpGa2 = np.vstack((x.ravel(), y.ravel(), z.ravel()))
    bpGa2_T = np.vstack((x.ravel(), y.ravel(), z.ravel()))
    
    # Basepositions
    positionEu, positionEu_T, positionAl1, positionAl1_T, positionAl2, positionAl2_T, \
    positionGa1, positionGa1_T, positionGa2, positionGa2_T = bpEu, bpEu_T, bpAl1,  \
    bpAl1_T, bpAl2, bpAl2_T, bpGa1, bpGa1_T, bpGa2, bpGa2_T
    
    # Add a constant factor 'T' to every element in the array
    T = 0.5  # Translation für (Eu,Ga,Al)_T Positionen
    for i in range(positionEu.shape[0]):
        for j in range(positionEu.shape[1]):
            positionEu_T[i, j] = positionEu_T[i, j] + T
            positionAl1_T[i, j] = positionAl1_T[i, j] + T
            positionAl2_T[i, j] = positionAl2_T[i, j] + T
            positionGa1_T[i, j] = positionGa1_T[i, j] + T
            positionGa2_T[i, j] = positionGa2_T[i, j] + T
    
    # Füge zusammen zu ndarrays
    Eu = np.array([positionEu[0], positionEu[1], positionEu[2]])   # Transpose to match the desired shape (3, 8)
    Eu_T = np.array([positionEu_T[0], positionEu_T[1], positionEu_T[2]])
    Al1 = np.array([positionAl1[0], positionAl1[1], positionAl1[2]])    # Transpose to match the desired shape (3, 8)
    Al1_T = np.array([positionAl1_T[0], positionAl1_T[1], positionAl1_T[2]]) 
    Al2 = np.array([positionAl2[0], positionAl2[1], positionAl2[2]])    # Transpose to match the desired shape (3, 8)
    Al2_T = np.array([positionEu[0], positionAl2_T[1], positionAl2_T[2]]) 
    Ga1 = np.array([positionGa1[0], positionGa1[1], positionGa1[2]])    # Transpose to match the desired shape (3, 8)
    Ga1_T = np.array([positionGa1_T[0], positionGa1_T[1], positionGa1_T[2]]) 
    Ga2 = np.array([positionGa2[0], positionGa2[1], positionGa2[2]])    # Transpose to match the desired shape (3, 8)
    Ga2_T = np.array([positionEu[0], positionGa2_T[1], positionGa2_T[2]]) 
    print("# of Unit Cells Nz*Nxy = {}".format(Nz_pos*Nxy_pos))
    
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
    f_Eu, f_Ga, f_Al = atomicformfactorEuGaAl(h, k2d, l2d, Unitary, 
                                              properatomicformfactor)
    for i in range(0, len(Eu[0])):
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
    I = np.abs(F_init) ** 2  # I \propto F(Q)^2, F complex
    
    # Exclude unwanted kspace-points
    if Lexclude == True:
        I = excludekspacepoints(kspacefactors, k2d, l2d, deltak, I, 
                                noiseamplitude, kmax, lmax, Lexclude)
    else:
        pass
    
    # Normalize Intensity
    if normalization == True:
        I = I/np.max(I)
    else:
        I = I
    
    # PLOTTING
    fig = plt.figure(figsize=(15, 3))
    plt.suptitle(r"EuGa$_2$Al$_2$, Centered around [{} {} {}], {} Unit Cells".format(h, k0, l0, len(Eu[0])))
    """LINECUTS"""
    plt.subplot(1, 1, 1)
    plt.plot(l2d[:, 0], I[:, len(k) // 2], ls='-', lw=0.5, marker='.', 
             ms=1, label='K={}'.format(np.round(k[len(k) // 2], 2)))
    plt.plot(l2d[:, 0], I[:, 0], ls='-', marker='.', lw=0.5, 
             ms=1, label='K={}'.format(np.round(k[0], 2)))
    plt.legend()
    plt.xlim(l0-kmax, l0+kmax)
    plt.ylabel(r"Intensity $I\propto F(\mathbf{Q})^2$")
    plt.xlabel("L (r.l.u.)") 
    if savefig == True:
        plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/XRD_Simulation/EuGa2Al2_unmodulated/L-cuts_N={}{}{}_H={}_center{}{}{}_updated.jpg".format(Nxy_pos, Nxy_pos, Nz, h, h, k0, l0), dpi=300)
    else:
        pass
    plt.subplots_adjust(wspace=0.3)
    
    fig = plt.figure(figsize=(15, 5))
    plt.suptitle("X-ray form factor simulation for Eu(Ga,Al)4 I4/mmm, {} Unit Cells".format(len(Eu[0])))
    """CONVOLUTION"""
    plt.subplot(1, 2, 1)
    plt.title("Gaussian Convolution, [{}KL]-map".format(h))
    x, y = np.linspace(-1,1,kernelsize), np.linspace(-1,1,kernelsize)

    X, Y = np.meshgrid(x, y)
    kernel = 1/(2*np.pi*sigma**2)*np.exp(-(X**2+Y**2)/(2*sigma**2))
    Iconv = ndimage.convolve(I, kernel, mode='constant', cval=0.0)
    if lognorm == True:
        plt.imshow(Iconv, cmap='inferno', extent=(k0-kmax, k0+kmax, l0-lmax, l0+lmax), 
                origin = 'lower',
                norm = LogNorm(vmin=0.06*np.max(Iconv), vmax=np.max(Iconv))
                )
    else:
        plt.imshow(Iconv, cmap='inferno', extent=(k0-kmax, k0+kmax, l0-lmax, l0+lmax), 
                origin='lower', vmin= 0, vmax= np.max(Iconv)
                )          
    plt.colorbar()
    plt.ylabel("L (r.l.u.)")
    plt.xlabel("K (r.l.u.)")   
    """3D ATOM PLOT"""
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    # ax.title(r"(Modulated) unit cell(s) with Wyckoff 4e z-parameter z0={} and $T=(1/2 1/2 1/2)$".format(z0))
    ax.scatter(Eu[0][0], Eu[1][0], Eu[2][0], label='Eu, Wyckoff 2a (000)', c='purple')
    ax.scatter(Eu_T[0][0], Eu_T[1][0], Eu_T[2][0], label='Eu_T, Wyckoff 2a T(000)', facecolors='none', edgecolors='purple')
    ax.scatter(Al1[0][0], Al1[1][0], Al1[2][0], label='Al1, Wyckoff 4d (1/2 0 1/4), (0 1/2 1/4)', c='blue')
    ax.scatter(Al1_T[0][0], Al1_T[1][0], Al1_T[2][0], facecolors='none', edgecolors='blue')
    ax.scatter(Al2[0][0], Al2[1][0], Al2[2][0], c='blue')
    ax.scatter(Al2_T[0][0], Al2_T[1][0], Al2_T[2][0], facecolors='none', edgecolors='blue')
    ax.scatter(Ga1[0][0], Ga1[1][0], Ga1[2][0], label='Ga1, Wyckoff 4e (0 0 z0), (0 0 -z0)', c='green')
    ax.scatter(Ga1_T[0][0], Ga1_T[1][0], Ga1_T[2][0], facecolors='none', edgecolors='green')
    ax.scatter(Ga2[0][0], Ga2[1][0], Ga2[2][0], c='green')
    ax.scatter(Ga2_T[0][0], Ga2_T[1][0], Ga2_T[2][0], facecolors='none', edgecolors='green')
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_zlim(0, 1.0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.legend()
    if savefig == True:
        plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/XRD_Simulation/EuGa2Al2_unmodulated/Map_N={}{}{}_H={}_center{}{}{}_updated.jpg".format(Nxy_pos, Nxy_pos, Nz, h, h, k0, l0), dpi=300)
    else:
        pass
    plt.show(block=False)


def main():
    ##########################################################################
    ########################## GENERAL INPUT #################################
    ##########################################################################
    print(__doc__)
    k0, l0, kmax = 0, 0, 30  # boundaries of K and L 
    lmax = kmax
    deltak = 0.10  # k-space point distance
    h = 5 # H value in k space
    a, c, z0 = 4, 11, 0.38  # Unit Cell Parameters in Angstrom
    Nz = 1  #  Number of stacked unmodulated unit cells in z-direction
    kspacefactors = [1, 9, 1, 9]
    ######################################
    #  INPUT FOR DBW
    u_list = [1e-3, 1e-3, 1e-3]  # Isotropic displacements <u^2> in Å^2, ~1/m. 
    ######################################
    # OTHER
    noiseamplitude = 1e-4  # Noise amplitude
    ######################################
    # GAUSSIAN KERNEL -- DEPENDENT ON RESOLUTION FUNCTION
    # ID28
    # P07
    sigma, kernelsize = 0.5, 5  #int(0.1/deltak) 
    
    #  PROGRAM
    st = time.time()
    k2d, l2d, k, l, Unitary = kspacecreator(k0, l0, kmax, lmax, deltak)
    print("Memory occupancy factor (2) = {}".format(kmax*deltak*Nz)) 
    structurefactorandplotting(a, c, k0, l0, k2d, k, l, kmax, lmax, l2d, h, 
                               deltak=deltak, Unitary=Unitary, u_list=u_list,
                               kernelsize = kernelsize, Nz=Nz, kspacefactors=kspacefactors, 
                               z0=z0,  
                               noiseamplitude=noiseamplitude, sigma=sigma,
                               normalization=False, DBW=False, 
                               lognorm=False, savefig=True, Lexclude=True,
                               properatomicformfactor=True, EuAl4=False)
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

if __name__ == '__main__':
        main()
    

