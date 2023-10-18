#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 10:11:21 2023

@author: stevengebel
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.colors import LogNorm
from matplotlib.pyplot import figure
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rc('text', usetex=True)
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)


"""XRD Simulation functions package"""

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


def xtranslationpos(x, storage_x, storage_y, storage_z, i):
    """Translate atomic positions one given direction x,y,z
    """
    x_transl = x + np.array([i, 0, 0])
    storage_x.append(x_transl[0])
    storage_y.append(x_transl[1])
    storage_z.append(x_transl[2])

def xtranslationneg(x, storage_x, storage_y, storage_z, i):
    """Translate atomic positions one given direction x,y,z
    """
    x_transl = x + np.array([-i, 0, 0])
    storage_x.append(x_transl[0])
    storage_y.append(x_transl[1])
    storage_z.append(x_transl[2])


def ytranslationpos(x, storage_x, storage_y, storage_z, i):
    """Translate atomic positions one given direction x,y,z
    """
    x_transl = x + np.array([0, i, 0])
    storage_x.append(x_transl[0])
    storage_y.append(x_transl[1])
    storage_z.append(x_transl[2])


def ytranslationneg(x, storage_x, storage_y, storage_z, i):
    """Translate atomic positions one given direction x,y,z
    """
    x_transl = x + np.array([0, -i, 0])
    storage_x.append(x_transl[0])
    storage_y.append(x_transl[1])
    storage_z.append(x_transl[2])
    
    
def atomicformfactorEuGa2Al2(h, k2d, l2d, Unitary, properatomicformfactor=False):
    """Atomic form factors according to de Graed, structure of materials, 
    chapter 12: Eu2+. Ga1+, Al3+"""

    if properatomicformfactor == True:
        a_eu, a_ga, a_al  = [24.0063, 19.9504, 11.8034, 3.87243], \
        [15.2354, 6.7006, 4.3591, 2.9623], [4.17448, 3.3876, 1.20296, 0.528137]
        b_eu, b_ga, b_al = [2.27783, 0.17353, 11.6096, 26.5156], \
        [3.0669, 0.2412, 10.7805, 61.4135], [1.93816, 4.14553, 0.228753, 8.28524]
    
    else:
        a_eu, a_ga, a_al  = np.ones(4), np.ones(4), np.ones(4)
        b_eu, b_ga, b_al =  np.ones(4), np.ones(4), np.ones(4)
        
    # c_eu, c_ga, c_al = 1.36389, 1.7189, 0.706786
    f_eu1 = a_eu[0] * np.exp(- b_eu[0] * (((h * Unitary )**2 + k2d**2 + l2d**2) / (4 * np.pi)**2  ))
    f_eu2 = a_eu[1] * np.exp(- b_eu[1] * (((h * Unitary) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    f_eu3 = a_eu[2] * np.exp(- b_eu[2] * (((h * Unitary) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    f_eu4 = a_eu[3] * np.exp(- b_eu[3] * (((h * Unitary) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    f_Eu = f_eu1+f_eu2+f_eu3+f_eu4 
    f_ga1 = a_ga[0] * np.exp(- b_ga[0] * (((h * Unitary )**2 + k2d**2 + l2d**2) / (4 * np.pi)**2  ))
    f_ga2 = a_ga[1] * np.exp(- b_ga[1] * (((h * Unitary) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    f_ga3 = a_ga[2] * np.exp(- b_ga[2] * (((h * Unitary) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    f_ga4 = a_ga[3] * np.exp(- b_ga[3] * (((h * Unitary) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    f_Ga = f_ga1+f_ga2+f_ga3+f_ga4 
    f_al1 = a_al[0] * np.exp(- b_al[0] * (((h * Unitary )**2 + k2d**2 + l2d**2) / (4 * np.pi)**2  ))
    f_al2 = a_al[1] * np.exp(- b_al[1] * (((h * Unitary) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    f_al3 = a_al[2] * np.exp(- b_al[2] * (((h * Unitary) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    f_al4 = a_al[3] * np.exp(- b_al[3] * (((h * Unitary) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    f_Al = f_al1+f_al2+f_al3+f_al4 
    
    return f_Eu, f_Ga, f_Al


def atomicformfactorBCC(h, k2d, l2d, Unitary, properatomicformfactor):
    """Atomic form factors according to de Graed, structure of materials, 
    chapter 12: Eu2+. Ga1+, Al3+"""

    if properatomicformfactor == True:
        f_Atom1 = 1
        f_Atom2 = 1/f_Atom1
    
    else:
        f_Atom1, f_Atom2 = 1, 1
    
    return f_Atom1, f_Atom2
   
    
def kspacecreator(k0, l0, kmax, lmax, deltak):
    """K-space creator"""
    k = np.arange(k0-kmax, k0+kmax + deltak, deltak)
    l = np.arange(l0-lmax, l0+lmax + deltak, deltak)
    k2d, l2d = np.meshgrid(k, l)
    Unitary = np.ones((len(k2d), len(l2d)))  # Unitary matrix
    return k2d, l2d, k, Unitary


def debyewallerfactor(k2d, l2d, Unitary, h, a, c, u_list):
    """theta-input for DBW and DBW calculation
    DBW = exp(- (B_iso * sin(theta)**2) / lambda )
    :param lamb: x-ray wavelength
    :param k2d, l2d: kspace 2D parametrization
    :param a, c: crystal parameters
    :return: B_iso parameters, theta: size as l2d, k2d
    """
    lamb = 1e-1  # x-ray wavelength in m (Mo Ka)
    # Compute all distances corresponding to lattice diffractions with hkl
    d_hkl = a / (np.sqrt((h * Unitary) ** 2 + k2d ** 2 + (a / c) ** 2 * l2d ** 2))
    # 1. compute all angles corresponding to the k points according to braggs law
    theta = np.arcsin(lamb / (2 * d_hkl))
    B_iso_list = 8 * np.pi ** 2 / 3 * np.array(u_list)
    DBW_list = []
    for i in range(0, len(B_iso_list)):     
        DBW_list.append(np.exp(-B_iso_list[i] / lamb **2 * (np.sin(theta)) ** 2))
    
    return DBW_list


def excludekspacepoints(kspacefactors, k2d, l2d, deltak, I, noiseamplitude, kmax, lmax, Lexclude):
    """Exclude k-space points from calculated Intensity array that should 
    average out after summation over N->\inf unit cells manually"""
    ########################################################################
    #  Benchmarked table for extracting unwanted K,L points
    #  q_cdw     0.1   0.1    0.2   0.2    0.5   0.5   0.125  0.125
    #  ∆k        0.1   0.01   0.1   0.01   0.1   0.01
    #  Kfactor1  1     1      1
    #  Kfactor2  9     99     9
    #  Lfactor1  1     10     1
    #  Lfactor2  9     9      2
    ########################################################################
    Kfactor1, Kfactor2, Lfactor1, Lfactor2 = kspacefactors[0], kspacefactors[1], \
    kspacefactors[2], kspacefactors[3]
    
    # #  Excluding unallowed K-points 
    k_intlist = np.arange(0, len(k2d), 1)  # erstelle indices aller k-Werte
    # print("k_integer={}".format(k_intlist))
    for i in range(0, (2 * kmax*Kfactor1 + 1)):  # LEAVES ONLY INTEGER K-values
        # print(range(0,2*kmax+1))
        k_intlist = np.delete(k_intlist, i * Kfactor2)  #  n*9, since the list gets one less each time
        # print("k_intlist={}".format(k_intlist))
    for i in k_intlist:  # Set unallowed K-values for intensities to 0
        I[:, i] = 0
    
    # #  Excluding unallowed L-points 
    if Lexclude == True:
        # #  Exluding unallowed L-points
        l_intlist = np.arange(0, len(l2d), 1)  # erstelle indices aller l-Werte
        for i in range(0, 2 * kmax * Lfactor1 + 1):
            l_intlist = np.delete(l_intlist, i * Lfactor2)  # Lösche jeden zehnten index
            print("l_intlist={}".format(l_intlist))
        for i in l_intlist:  # Set unallowed L-values for intensities to 0
            I[i, :] = 0
        if deltak == 0.1:
            for i in range(0, 2 * kmax + 1):
                l_intlist = np.delete(l_intlist, i * Lfactor2)  # Lösche jeden zehnten index
            for i in l_intlist:  # Set unallowed L-values for intensities to 0
                I[i, :] = 0
        else:
            for i in range(0, 2 * kmax * 10 + 1):
                l_intlist = np.delete(l_intlist, i * Lfactor2)  # Lösche jeden zehnten index
            for i in l_intlist:  # Set unallowed L-values for intensities to 0
                I[i, :] = 0
    # I = I + noiseamplitude * np.random.rand(len(k2d), len(k2d))  
    # Add random noise with maximum of noiseamplitude
    
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
    
    
    
    