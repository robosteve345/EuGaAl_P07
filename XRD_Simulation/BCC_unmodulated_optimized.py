#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Created on Tue Oct 24 09:02:27 2023

#@author: stevengebel

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import time
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.colors import LogNorm
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rc('text', usetex=True)
from XRD_simulation_functions import kspacecreator, excludekspacepoints, debyewallerfactor
from XRD_simulation_functions import atomicformfactorBCC, kspacemask


def calculate_F(f, DBW, h, Unitary, coords, k2d, l2d, result):
    for i in range(coords.shape[1]):
        result += f * DBW * np.exp(-2 * np.pi * 1j * (h * Unitary * coords[0, i] + k2d * coords[1, i] + l2d * coords[2, i]))


def structurefactorandplotting_optimized(a, c, k0, l0, k2d, k, l, kmax, lmax, l2d, h, 
                               deltak, Unitary, u_list, kernelsize, Nz, kspacefactors,
                               noiseamplitude, sigma, lognorm=True,
                               normalization=False, DBW=False, savefig=False,
                               properatomicformfactor=False, Lexclude=False):
    print("CENTERED PEAK: [HKL] = [{} {} {}]".format(h, k0, l0))
    print("∆k = {}".format(deltak))
    print("H = {}, K, L in {} to {}".format(h, -kmax, kmax))
    """Atomic positions"""
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
    one_directionxy = np.arange(Nxy_neg, Nxy_pos, 1.0)
    one_directionz = np.arange(Nz_neg, Nz_pos, 1.0)
    # Create the NumPy array using a combination of meshgrid and reshape
    x, y, z = np.meshgrid(one_directionxy, one_directionxy, one_directionz, indexing='ij')
    baseposition1 = np.vstack((x.ravel(), y.ravel(), z.ravel()))
    baseposition2 = np.vstack((x.ravel(), y.ravel(), z.ravel()))
    
    # Atom1, Atom2 basepositions
    position1, position2, = baseposition1, baseposition2
    # Add a constant factor 'T' to every element in the array
    T = 0.5  # Translation für Atom2-Position

    for i in range(position2.shape[0]):
        for j in range(position2.shape[1]):
            position2[i, j] = position2[i, j] + T
    
    # Füge zusammen zu ndarrays
    Atom2 = np.array([position2[0], position2[1], position2[2]])   # Transpose to match the desired shape (3, 8)
    Atom1 = np.array([position1[0], position1[1], position1[2]])
    print("# of Unit Cells = {}".format(Nz_pos*Nxy_pos))
    
    #  Compute Debye-Waller-Factor
    if DBW==True:
        DBW_list = debyewallerfactor(k2d, l2d, Unitary, h, a, c, u_list)
    else:
        DBW_list = np.ones(2)
    
    """Scattering amplitudes F"""
    f_Atom1, f_Atom2 = atomicformfactorBCC(h, k2d, l2d, Unitary, properatomicformfactor)
    
    F_Atom1 = np.zeros((len(k2d), len(k2d)), dtype=complex)
    F_Atom2 = np.zeros((len(k2d), len(k2d)), dtype=complex)
    
    calculate_F(f_Atom1, DBW_list[0], h, Unitary, Atom1, k2d, l2d, F_Atom1)
    calculate_F(f_Atom2, DBW_list[1], h, Unitary, Atom2, k2d, l2d, F_Atom2)
    
    F = F_Atom1 + F_Atom2
    
    #  Compute Intensity
    I = np.abs(F) ** 2  # I \propto F(Q)^2, F complex 
    
    # Exclude unwanted kspace-points
    I = excludekspacepoints(kspacefactors, k2d, l2d, deltak, I, noiseamplitude, kmax, lmax, Lexclude)
    
    # Compute normalization
    if normalization:
        I = I / np.max(I)
    
    # PLOTTING
    fig = plt.figure(figsize=(15, 3))
    plt.suptitle("BCC, Centered around [{} {} {}]".format(h, k0, l0))
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
        plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/XRD_Simulation/BCC_unmodulated/L-cuts_N={}{}{}_H={}_center{}{}{}_updated.jpg".format(len(one_directionxy), len(one_directionxy), 
                        len(one_directionz), h, h, k0, l0), dpi=300)
    else:
        pass
    plt.subplots_adjust(wspace=0.3)
    
    fig = plt.figure(figsize=(15, 10))
    plt.suptitle("BCC")
    """CONVOLUTION"""
    plt.subplot(1, 2, 1)
    plt.title("Gaussian conv., H={}".format(h))
    x, y = np.linspace(-1,1,kernelsize), np.linspace(-1,1,kernelsize)

    X, Y = np.meshgrid(x, y)
    # Kernel with random noise with maximum of noiseamplitude
    kernel = 1/(2*np.pi*sigma**2)*np.exp(-(X**2+Y**2)/(2*sigma**2)) 
    Iconv = ndimage.convolve(I, kernel, mode='constant', cval=0.0)
    if lognorm == True:
        plt.imshow(Iconv, cmap='viridis', extent=(k0-kmax, k0+kmax, l0-lmax, l0+lmax), 
                origin='lower',
                norm=LogNorm(vmin = 1, vmax = np.max(Iconv))
                )
    else:
        plt.imshow(Iconv, cmap='viridis', extent=(k0-kmax, k0+kmax, l0-lmax, l0+lmax), 
                origin='lower', vmin=0, vmax = np.max(Iconv)
                )          
    plt.colorbar()
    plt.ylabel("L (r.l.u.)")
    plt.xlabel("K (r.l.u.)")   
    """3D ATOM PLOT"""
    ax = fig.add_subplot(1,2,2, projection='3d')
    ax.scatter(Atom1[0,:], Atom1[1,:], Atom1[2,:], label='Atom1')
    ax.scatter(Atom2[0,:], Atom2[1,:], Atom2[2,:], label='Atom2')
    # ax.set_xlim(-0.5, 0.5)
    # ax.set_ylim(-0.5, 0.5)
    # ax.set_zlim(-2 * kmax, 2 * kmax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    if savefig == True:
        plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/XRD_Simulation/BCC_unmodulated/Map_N={}{}{}_H={}_center{}{}{}_updated.jpg".format(len(one_directionxy), len(one_directionxy), 
                        len(one_directionz), h, h, k0, l0), dpi=300)
    else:
        pass
    plt.show(block=False)


def main():
    ##########################################################################
    ########################## GENERAL INPUT #################################
    ##########################################################################
    print(__doc__)
    k0, l0, kmax = 0, 0, 15 # boundaries of K and L 
    lmax = kmax
    deltak = 0.01  # k-space point distance
    h = 1  # H value in k space
    a, c = 1, 1  # Unit Cell Parameters
    Nz = 1
    kspacefactors = [1, 99, 10, 9]
    ######################################
    #  INPUT FOR DBW
    u_list = [1e-3, 1e-3]  # Isotropic displacements <u^2> in Å^2, ~1/m. For more detail take bonding angles, distances into account.
    ######################################
    # OTHER
    noiseamplitude = 10  # Noise amplitude
    ######################################
    # GAUSSIAN KERNEL
    sigma, kernelsize = 0.5, 5  #int(0.1/deltak) 
    
    st = time.time()
    k2d, l2d, k, l, Unitary = kspacecreator(k0, l0, kmax, lmax, deltak)
    print("(2) = {}".format(kmax * deltak * Nz)) 
    structurefactorandplotting_optimized(a, c, k0, l0, k2d, k, l, kmax, lmax, l2d, h, deltak=deltak, 
                               Unitary=Unitary, u_list=u_list,
                               kernelsize = kernelsize, Nz=Nz, kspacefactors=kspacefactors,
                               noiseamplitude=noiseamplitude, sigma=sigma,
                               normalization=False, DBW=False, 
                               lognorm=False, savefig=False, Lexclude=False,
                               properatomicformfactor=False)
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

if __name__ == '__main__':
    main()
