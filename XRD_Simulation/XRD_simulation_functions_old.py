#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 15:28:35 2023

@author: stevengebel
"""

"""XRD simulation functions (OLD and DISCARDED functions)"""

import numpy as np

def debyewallerfactor(k2d, l2d, Unitary, h, a, c, u_list):
    """theta-input for DBW and DBW calculation
    DBW = exp(- (B_iso * sin(theta)**2) / lambda )
    :param lamb: x-ray wavelength
    :param k2d, l2d: kspace 2D parametrization
    :param a, c: crystal parameters
    :return: B_iso parameters, theta: size as l2d, k2d
    """
    lamb = 1e-10  # x-ray wavelength in m (Mo Ka)
    # Compute all distances corresponding to lattice diffractions with hkl
    d_hkl = a / (np.sqrt((h * Unitary) ** 2 + k2d ** 2 + (a / c) ** 2 * l2d ** 2))
    # 1. compute all angles corresponding to the k points according to braggs law
    theta = np.arcsin(lamb / (2 * d_hkl))
    B_iso_list = 8 * np.pi ** 2 / 3 * np.array(u_list)
    DBW_list = []
    for i in range(0, len(B_iso_list)):     
        DBW_list.append(np.exp(-B_iso_list[i] / lamb **2 * (np.sin(theta)) ** 2))
    return DBW_list


def ztranslationpos(x, storage_x, storage_y, storage_z, i):
    """Translate atomic positions one given direction x,y,z
    """
    x_transl = x + np.array([0, 0, i])
    storage_x.append(x_transl[0])
    storage_y.append(x_transl[1])
    storage_z.append(x_transl[2])
    

def kspacemask(k, l, k2d, l2d, q_cdw, I):
    """
    Masks Intensity I and excludes kspace points that would be minimized by 
    summing over infinitely many perfect (modulated) crystal unit cells
    
    Parameters
    ----------
    q_cdw : float
        Periodicity of the CDW
    I : 2d-Array
    k2d : 
    l2d : 
    k : 
     

    Returns
    -------
    I: Masked Intensity 2d-array

    """
    # Set columns of I to zero where k-values are not integers
    print(I)
    column_indices = np.arange(k2d.shape[1])  # Create an array of column indices
    non_integer_columns = ~np.isclose(k[column_indices] % 1, 0)  # Check if k-values are not integers
    I[:, non_integer_columns] = 0  # Set columns to zero

    # Set rows of I to zero where l-values are not multiples of q_cdw
    non_multiple_rows = ~np.isclose(l % q_cdw, 0)  # Check if l-values are not multiples of q_cdw
    I[non_multiple_rows, :] = 0  # Set rows to zero
    print(I)
    return I
    

def atomicformfactorBCC(h, k2d, l2d, Unitary, properatomicformfactor):
    """Atomic form factors according to de Graed, structure of materials, 
    chapter 12: Eu2+. Ga1+, Al3+"""

    if properatomicformfactor == True:
        f_Atom1 = 1
        f_Atom2 = 1/f_Atom1
    
    else:
        f_Atom1, f_Atom2 = 1, 1
    return f_Atom1, f_Atom2


def excludekspacepoints_OLD(kspacefactors, k2d, l2d, deltak, I, noiseamplitude, kmax, lmax, Lexclude):
    """Exclude k-space points from calculated Intensity array that should 
    average out after summation over N->\inf unit cells manually"""
    ########################################################################
    #  Benchmarked table for extracting unwanted K,L points
    #  q_cdw     0.1   0.1    0.2   0.2    
    #  ∆k        0.1   0.01   0.1   0.01   
    #  Kfactor1  1     1      1
    #  Kfactor2  9     99     9
    #  Lfactor1  1     10     1
    #  Lfactor2  9     9      2
    ########################################################################
    Kfactor1, Kfactor2, Lfactor1, Lfactor2 = kspacefactors[0], kspacefactors[1], \
    kspacefactors[2], kspacefactors[3]
    
    # #  Excluding unallowed K-points 
    k_intlist = np.arange(0, len(k2d), 1)  # erstelle indices aller k-Werte
    for i in range(0, (2 * kmax*Kfactor1 + 1)):  # LEAVES ONLY INTEGER K-values
        k_intlist = np.delete(k_intlist, i * Kfactor2)  #  n*9, since the list gets one less each time
        # print("k_intlist={}".format(k_intlist))
    for i in k_intlist:  # Set unallowed K-values for intensities to 0
        I[:, i] = 0
    
    # #  Excluding unallowed L-points 
    if Lexclude == True:
        l_intlist = np.arange(0, len(l2d), 1)  # erstelle indices aller l-Werte
        
        # for i in range(0, 2 * kmax * Lfactor1 + 1):
        #     l_intlist = np.delete(l_intlist, i * Lfactor2)  # Lösche jeden zehnten index
        #     print("l_intlist={}".format(l_intlist))
        # for i in l_intlist:  # Set unallowed L-values for intensities to 0
        #     I[i, :] = 0

        if deltak == 0.01:
            for i in range(0, 2 * kmax * Lfactor1 + 1):
                l_intlist = np.delete(l_intlist, i * Lfactor2)  # Lösche jeden zehnten index
            for i in l_intlist:  # Set unallowed L-values for intensities to 0
                I[i, :] = 0
        else:
            for i in range(0, 2 * kmax * Lfactor1 + 1):
                l_intlist = np.delete(l_intlist, i * Lfactor2)  # Lösche jeden zehnten index
            for i in l_intlist:  # Set unallowed L-values for intensities to 0
                I[i, :] = 0
    return I

    
#############################################
# PLOTTING
    # #  INTERPOLATION
    # plt.subplot(1, 3, 1)
    # plt.title("Gaussian interpolation, H={}".format(h))
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
        # #  2D SCATTER PLOT
        # plt.subplot(1, 3, 1)
        # plt.scatter(k2d, l2d, label=r'$I \propto F(\mathbf{Q})^2$'
        #             , s=I / np.max(I),
        #             # , c = I / np.max(I),
        #             )
        # plt.colorbar()
        # plt.legend()
        # plt.ylabel("L(rlu)")
        # plt.xlabel("K(rlu)")
        # plt.tight_layout()
    

    
    
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
        # plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/XRD_Simulation/BCC_modulation1/Map_BCC_{}UC_{}SC_A={}_q={}_H={}_center{}{}{}.jpg".format(n, int(n/int(q_cdw**(-1))), Amplitude, q_cdw, h, h, k0, l0), dpi=300)
        # plt.subplots_adjust(wspace=0.3)
    
    
    
    # """MODULATED ATOMIC POSITIONS"""
    # plt.subplot(2, 1, 1)
    # plt.scatter(init_Ga1_z, np.ones(len(Ga1[0])),
    #             label=r'4e = $(0,0,{})$ equilibrium'.format(z0), facecolors='none',
    #             edgecolors='orange', s=100)
    # plt.xlabel('z')
    # plt.ylabel('')
    # plt.scatter(Ga1[2, :], np.ones(len(init_Ga1_z)),
    #             label=r'4e = $(0, 0, {} + {} sin({} 2 pi L))$ distorted'.format(z0, A, q_cdw),
    #             marker='o')
    # plt.legend()