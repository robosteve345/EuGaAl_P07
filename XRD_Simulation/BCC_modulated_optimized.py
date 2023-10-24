#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Created on Tue Oct 24 09:25:36 2023

#@author: stevengebel

"""BCC propermodulation updated
"""

import numpy as np
import matplotlib
import time
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.colors import LogNorm
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rc('text', usetex=True)
from XRD_simulation_functions import ztranslationpos, kspacecreator, excludekspacepoints, debyewallerfactor
from XRD_simulation_functions import atomicformfactorBCC

"""TO DO
-Implement possibility for HKL_0 & HK_0L maps
-Implement input-interface: kß, lß, deltak, Amplitude, q_cdw, kmax
"""


def calculate_F(f, DBW, h, Unitary, coords, k2d, l2d, result):
    for i in range(coords.shape[1]):
        result += f * DBW * np.exp(-2 * np.pi * 1j * (h * Unitary * coords[0, i] + k2d * coords[1, i] + l2d * coords[2, i]))



def structurefactorandplotting(a, c, k0, l0, k2d, k, l, kmax, lmax, l2d, h, 
                               deltak, Unitary, u_list, kernelsize,
                               A, q_cdw, Nsc, kspacefactors, noiseamplitude, sigma,
                               lognorm=False, normalization=False, DBW=False, 
                               savefig=False, properatomicformfactor=False,
                               Lexclude=False):
    print("CENTERED PEAK: [HKL] = [{} {} {}]".format(h, k0, l0))
    print("∆k = {}".format(deltak))
    print("A = {}, q_cdw = {}".format(A, q_cdw))
    print("H = {}, kmax=lmax = {}".format(h, kmax))
    print("# of Supercells = {}".format(Nsc))
    """Atomic positions"""
    ##########################################################################
    Nxy_neg, Nxy_pos = 0, 1 #int(np.sqrt(Nsc/(q_cdw*0.25)))
    Nz_neg, Nz_pos = 0, int(q_cdw**(-1)) * Nsc
    # print("Nz/Nxy = {}".format(np.round(q_cdw**(-1) * Nsc/Nxy_pos**2, 2)))
    # print("Nxy_neg, Nxy_pos = {}, {}".format(Nxy_neg, Nxy_pos))
    # print("Nz_neg, Nz_pos = {}, {}".format(Nz_neg, Nz_pos))
    ##########################################################################
    one_directionxy = np.arange(Nxy_neg, Nxy_pos, 1.0)
    one_directionz = np.arange(Nz_neg, Nz_pos, 1.0)
    # Create the NumPy array using a combination of meshgrid and reshape
    x, y, z = np.meshgrid(one_directionxy, one_directionxy, one_directionz, indexing='ij')
    baseposition1 = np.vstack((x.ravel(), y.ravel(), z.ravel()))
    baseposition2 = np.vstack((x.ravel(), y.ravel(), z.ravel()))
    # print("one_directionxy = {}".format(one_directionxy))
    # print("one_directionz = {}".format(one_directionz))
    
    # Atom1, Atom2 basepositions
    position1, position2, = baseposition1, baseposition2
    # Add a constant factor 'T' to every element in the array
    T = 0.5  # Translation für Atom2-Position

    for i in range(position2.shape[0]):
        for j in range(position2.shape[1]):
            position2[i, j] = position2[i, j] + T
    
    # Modulation
    for i in range(0, len(position2[2])):  # begins with 1 because it evaluates every unit cell, 0 to n-1, BLEIBT SO
        for j in range(1,2):
            B = 0
            position2[2][i] = position2[2][i] + \
                A * np.sin(q_cdw * 2 * np.pi * j * position2[2][i])  + \
                B * np.cos(q_cdw * 2 * np.pi * j * position2[2][i])
                
    # Füge zusammen zu ndarrays
    Atom2 = np.array([position2[0], position2[1], position2[2]])   # Transpose to match the desired shape (3, 8)
    Atom1 = np.array([position1[0], position1[1], position1[2]])
    print("# of unmodulated Unit Cells = {}".format(Nz_pos*Nxy_pos))
    
    #  Compute Debye-Waller-Factor
    if DBW == True:
        DBW_list = debyewallerfactor(k2d, l2d, Unitary, h, a, c, u_list)
    else:
        DBW_list = np.ones(3)
        
    f_Atom1, f_Atom2 = atomicformfactorBCC(h, k2d, l2d, Unitary, properatomicformfactor)
    
    F_Atom1 = np.zeros((len(k2d), len(k2d)), dtype=complex)
    F_Atom2 = np.zeros((len(k2d), len(k2d)), dtype=complex)
    
    calculate_F(f_Atom1, DBW_list[0], h, Unitary, Atom1, k2d, l2d, F_Atom1)
    calculate_F(f_Atom2, DBW_list[1], h, Unitary, Atom2, k2d, l2d, F_Atom2)
    
    F = F_Atom1 + F_Atom2
        
    #  Compute Intensity
    I = np.abs(np.round(F, 3)) ** 2  # I \propto F(Q)^2, F complex

    ############################################################################
    # EXCLUDE K-SPACE POINTS
    I = excludekspacepoints(kspacefactors, k2d, l2d, deltak, I, noiseamplitude, kmax, lmax, Lexclude)

    # Compute normalization
    if normalization == True:
        I = I/np.max(I)
    else:
        I = I
    
    # # PLOTTING
    fig = plt.figure(figsize=(15, 3))
    plt.suptitle("BCC, Centered around [{}{}{}]".format(h, k0, l0))
    """MODULATED ATOMIC POSITIONS"""
    plt.subplot(2, 1, 1)
    print("# one_directionxy = {}".format(len(one_directionxy)))
    print("# one_directionz = {}".format(len(one_directionz)))
    plt.scatter(1 / 2 * np.ones(len(one_directionz)) + one_directionz, np.ones(len(one_directionz)),
                label=r'Atom2=$(\frac{1}{2},\frac{1}{2},\frac{1}{2})$ equilibrium', facecolors='none',
                edgecolors='orange', s=100)
    plt.xlabel('z')
    plt.ylabel('')

    plt.scatter(Atom2[2, :], np.ones(len(one_directionz) * len(one_directionxy)**2),
                label='Atom2=$(0.5,0.5,0.5 + {} \sin({} \cdot 2\pi L))$ distorted'.format(A, q_cdw),
                marker='o')
    plt.legend()
    """LINECUTS"""
    plt.subplot(2, 1, 2)
    plt.plot(l2d[:, 0], I[:, len(k) // 2], ls='-', lw=0.5, marker='.', 
             ms=1, label='K={}'.format(np.round(k[len(k) // 2], 2)))
    plt.plot(l2d[:, 0], I[:, 0], ls='-', marker='.', lw=0.5, 
             ms=1, label='K={}'.format(np.round(k[0], 2)))
    plt.legend()
    plt.xlim(l0-kmax, l0+kmax)
    plt.ylabel(r"Intensity $I\propto F(\mathbf{Q})^2$")
    plt.xlabel("L (r.l.u.)") 
    if savefig == True:
        plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/XRD_Simulation/BCC_modulated/L-cuts_N={}{}{}_q={}_H={}_center{}{}{}_updated.jpg".format(len(one_directionxy), len(one_directionxy), 
                        len(one_directionz), q_cdw, h, h, k0, l0), dpi=300)
    else:
        pass
    plt.subplots_adjust(wspace=0.3)
    
    fig = plt.figure(figsize=(15, 5))
    plt.suptitle("BCC")
    """CONVOLUTION"""
    plt.subplot(1, 2, 1)
    plt.title("Gaussian conv., [{}KL]-map".format(h))
    x, y = np.linspace(-1,1,kernelsize), np.linspace(-1,1,kernelsize)

    X, Y = np.meshgrid(x, y)
    kernel = 1/(2*np.pi*sigma**2)*np.exp(-(X**2+Y**2)/(2*sigma**2))
    Iconv = ndimage.convolve(I, kernel, mode='constant', cval=0.0)
    if lognorm == True:
        plt.imshow(Iconv, cmap='inferno', extent=(k0-kmax, k0+kmax, l0-lmax, l0+lmax), 
                origin='lower',
                #vmin=0, vmax=0.1*np.max(Iconv)
                norm=LogNorm(vmin=1e-1, vmax = np.max(Iconv))
                )
    else:
        plt.imshow(Iconv, cmap='inferno', extent=(k0-kmax, k0+kmax, l0-lmax, l0+lmax), 
                origin='lower', vmin=0, vmax=np.max(Iconv)
                )          
    plt.colorbar()
    plt.ylabel("L (r.l.u.)")
    plt.xlabel("K (r.l.u.)")   
    """3D ATOM PLOT"""
    ax = fig.add_subplot(1,2,2, projection='3d')
    ax.scatter(Atom1[0,:], Atom1[1,:], Atom1[2,:], label='Atom1')
    ax.scatter(Atom2[0,:], Atom2[1,:], Atom2[2,:], label='Atom2')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    # ax.set_zlim(-2 * kmax, 2 * kmax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    if savefig == True:
        plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/XRD_Simulation/BCC_modulated/Map_N={}{}{}_q={}_H={}_center{}{}{}_updated.jpg".format(len(one_directionxy), len(one_directionxy), 
                        len(one_directionz), q_cdw, h, h, k0, l0), dpi=300)
    else:
        pass
    plt.show(block=False)


def main():
    ##########################################################################
    ########################## GENERAL INPUT #################################
    ##########################################################################
    print(__doc__)
    k0, l0, kmax = 5, 5, 4 # boundaries of K and L 
    lmax=kmax
    deltak = 0.01  # k-space point distance
    h = 1  # H value in k space
    a, c = 1, 1  # Unit Cell Parameters
    ######################################
    #  INPUT FOR CDW
    A, q_cdw = -0.10, 0.1
    ###############################
    # If Lexclude=True: Manually set unwanted intensities to zero for one Cell
    # Nsc = 1 & define kspacefactors acc. to deltak
    Nsc = 1 #  Number of supercells
    kspacefactors = [1, 99, 10, 9]
    ######################################
    #  INPUT FOR DBW
    u_list = [1e-3, 1e-3]  # Isotropic displacements <u^2> in Å^2, ~1/m. For more detail take bonding angles, distances into account.
    ######################################
    # OTHER
    noiseamplitude = 1e-4  # Noise amplitude
    ######################################
    # GAUSSIAN KERNEL
    sigma, kernelsize = 0.5, 10  #int(0.1/deltak) 
    
    
    #  PROGRAM
    st = time.time()
    k2d, l2d, k, l, Unitary = kspacecreator(k0, l0, kmax, lmax, deltak)
    print("(2) = {}".format(kmax*deltak*Nsc/q_cdw)) 
    structurefactorandplotting(a, c, k0, l0, k2d, k, l, kmax, lmax, l2d, h, deltak=deltak, 
                               Unitary=Unitary, u_list=u_list,
                               kernelsize = kernelsize, Nsc=Nsc,
                               kspacefactors=kspacefactors,
                               A=A, q_cdw=q_cdw, 
                               noiseamplitude=noiseamplitude, sigma=sigma,
                               normalization=True, DBW=False, 
                               lognorm=False, savefig=True,
                               properatomicformfactor=False,
                               Lexclude=True)
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

if __name__ == '__main__':
        main()
    
