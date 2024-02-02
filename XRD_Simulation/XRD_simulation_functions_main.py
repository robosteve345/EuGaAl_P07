#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Created on Sun Oct 29 09:37:59 2023

#@author: stevengebel

"""XRD simulation function pacakge: Main functions that compute |F(Q)|^2-maps
specified by cuts [HKL] in rec. space and include plotting and other 
features for modulated unit cells of EuGa2Al2 and EuAl4"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import ndimage
from matplotlib.colors import LogNorm
mpl.rc('text', usetex=True)
mpl.rcParams.update(mpl.rcParamsDefault) 
mpl.rcParams['font.family'] = "sans-serif"
from XRD_simulation_functions import dbw_H0KL, dbw_HK0L
from XRD_simulation_functions import dbw_HKL0
from XRD_simulation_functions import fatom_calc_H0KL, fatom_calc_HK0L, fatom_calc_HKL0
from XRD_simulation_functions import Fstructure_calc_H0KL, Fstructure_calc_HK0L
from XRD_simulation_functions import Fstructure_calc_HKL0, excludekspacepoints
from XRD_simulation_functions import excludekspacepoints_HKmap

# =============================================================================
# GENERAL VARIABLES (SECONDARY IMPORTANCE TO CODE-PERFORMANCE)
#  Intensity map cmap color scheme
cmap = "inferno_r"
#  Order m of bessel functions
besselorder = 1
# =============================================================================

def main_H0KL(a, c, k0, l0, k2d, k, l, kmax, lmax, l2d, H, 
                                deltak, Unitary, u_list, kernelsize,
                                q_cdw, Nsc, z0, A, B, 
                                noiseamplitude, sigma, lognorm=False, 
                                normalization=False, DBW=False, savefig=False,
                                fatom=False, EuAl4=False, EuGa4=False):
    print("SIMULATION OF {}KL-MAP".format(H))
    print("Centered peak: [HKL] = [{}{}{}]".format(H, k0, l0))
    print("Modulation vector q_z = {}".format(q_cdw))
    print("Resolution in x and y: {}r.l.u.".format(deltak))
    print("H = {}, K, L in {} to {}".format(H, -kmax, kmax))
    """Nominal atomic positions"""
    # # Europium Z=63:
    # x_Eu, y_Eu, z_Eu = 0, 0, 0
    # #  Aluminium, Z=13:
    # x_Al1, y_Al1, z_Al1 = 0, 0.5, 0.25
    # x_Al2, y_Al2, z_Al2 = 0.5, 0, 0.25
    # #  Gallium, Z=31:
    # x_Ga1, y_Ga1, z_Ga1 = 0, 0, z0
    # x_Ga2, y_Ga2, z_Ga2 = 0, 0, -z0
# =============================================================================
#     
# =============================================================================
    # Immm No. 71
    # Europium Z=63, 2a:
    x_Eu, y_Eu, z_Eu = 0, 0, 0
    #  Aluminium, Z=13, 4j:
    x_Al1, y_Al1, z_Al1 = 0, 0.5, 0.254
    x_Al2, y_Al2, z_Al2 = 0, 0.5, -0.254
    #  Gallium, Z=31:
    x_Ga1, y_Ga1, z_Ga1 = 0, 0, z0
    x_Ga2, y_Ga2, z_Ga2 = 0, 0, -z0
    print("DBW-Factor = {}".format(DBW))
    print("Q-dependent atomic form factors f(Q) = {}".format(fatom))
    print("Simulation for EuAl4 = {}".format(EuAl4))
    
    """Atomic positions"""
    ##########################################################################
    Nxy_neg, Nxy_pos = 0, 1  #int(np.sqrt(Nsc/(q_cdw*0.3)))
    Nz_neg, Nz_pos = 0, 1# int(q_cdw**(-1)) * Nsc
    ##########################################################################
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
    
    """Nominal atomic positions"""
    posEu, posEu_T, posAl1, posAl1_T, posAl2, posAl2_T, \
    posGa1, posGa1_T, posGa2, posGa2_T = bpEu, bpEu_T, bpAl1,  \
    bpAl1_T, bpAl2, bpAl2_T, bpGa1, bpGa1_T, bpGa2, bpGa2_T
    
    # Add a constant factor 'T' to every element in the array
    T = 0.5  # Translation für (Eu,Ga,Al)_T posen
    for i in range(posEu.shape[0]):
        for j in range(posEu.shape[1]):
            posEu_T[i, j] = posEu_T[i, j] + T
            posAl1_T[i, j] = posAl1_T[i, j] + T
            posAl2_T[i, j] = posAl2_T[i, j] + T
            posGa1_T[i, j] = posGa1_T[i, j] + T
            posGa2_T[i, j] = posGa2_T[i, j] + T
    ##########################################################################    
    """Modulation"""
    #  Sum over N=1/q_cdw atoms per Wyckoff position
    if q_cdw == 1:
        pass
    else:
        for i in range(0, len(posAl1[2])):
            #  Order of Bessel functions B_j: simplest model j=1
            for j in range(1, besselorder + 1):
                # Create a list of position arrays
                position_arrays = [[posAl1, posAl1_T, posAl2, posAl2_T], 
                                    [posGa1, posGa1_T, posGa2, posGa2_T], 
                                    [posEu, posEu_T]]
                # Iterate over the positions and apply the calculations
                #  4d
                for pos in position_arrays[0]:
                    pos[0][i] += A[0][0] * np.sin(q_cdw * 2 * np.pi * j * pos[2][i]) + \
                        B[0][0] * np.cos(q_cdw * 2 * np.pi * j * pos[2][i])
                    pos[1][i] += A[0][1] * np.sin(q_cdw * 2 * np.pi * j * pos[2][i]) + \
                        B[0][1] * np.cos(q_cdw * 2 * np.pi * j * pos[2][i])
                    pos[2][i] += A[0][2] * np.sin(q_cdw * 2 * np.pi * j * pos[2][i]) + \
                        B[0][2] * np.cos(q_cdw * 2 * np.pi * j * pos[2][i])
                #  4e
                for pos in position_arrays[1]:
                    pos[0][i] += A[1][0] * np.sin(q_cdw * 2 * np.pi * j * pos[2][i]) + \
                        B[1][0] * np.cos(q_cdw * 2 * np.pi * j * pos[2][i])
                    pos[1][i] += A[1][1] * np.sin(q_cdw * 2 * np.pi * j * pos[2][i]) + \
                        B[1][1] * np.cos(q_cdw * 2 * np.pi * j * pos[2][i])
                    pos[2][i] += A[1][2] * np.sin(q_cdw * 2 * np.pi * j * pos[2][i]) + \
                        B[1][2] * np.cos(q_cdw * 2 * np.pi * j * pos[2][i])
                #  2a
                for pos in position_arrays[2]:
                    pos[0][i] += A[2][0] * np.sin(q_cdw * 2 * np.pi * j * pos[2][i]) + \
                        B[2][0] * np.cos(q_cdw * 2 * np.pi * j * pos[2][i])
                    pos[1][i] += A[2][1] * np.sin(q_cdw * 2 * np.pi * j * pos[2][i]) + \
                        B[2][1] * np.cos(q_cdw * 2 * np.pi * j * pos[2][i])
                    pos[2][i] += A[2][2] * np.sin(q_cdw * 2 * np.pi * j * pos[2][i]) + \
                        B[2][2] * np.cos(q_cdw * 2 * np.pi * j * pos[2][i])
    # Appending the ndarrays
    Eu = np.array([posEu[0], posEu[1], posEu[2]])   # Transpose to match the desired shape (3, 8)
    Eu_T = np.array([posEu_T[0], posEu_T[1], posEu_T[2]])
    Al1 = np.array([posAl1[0], posAl1[1], posAl1[2]])    # Transpose to match the desired shape (3, 8)
    Al1_T = np.array([posAl1_T[0], posAl1_T[1], posAl1_T[2]]) 
    Al2 = np.array([posAl2[0], posAl2[1], posAl2[2]])    # Transpose to match the desired shape (3, 8)
    Al2_T = np.array([posAl2_T[0], posAl2_T[1], posAl2_T[2]]) 
    Ga1 = np.array([posGa1[0], posGa1[1], posGa1[2]])    # Transpose to match the desired shape (3, 8)
    Ga1_T = np.array([posGa1_T[0], posGa1_T[1], posGa1_T[2]]) 
    Ga2 = np.array([posGa2[0], posGa2[1], posGa2[2]])    # Transpose to match the desired shape (3, 8)
    Ga2_T = np.array([posGa2_T[0], posGa2_T[1], posGa2_T[2]]) 
    print("# of Unit Cells = {}".format(Nz_pos*Nxy_pos))    
# =============================================================================
#     #  Compute Debye-Waller-Factor
# =============================================================================
    if DBW == True:
        DBW_list = dbw_H0KL(k2d, l2d, Unitary, H, a, c, u_list)
    else:
        DBW_list = np.ones(3)   
# =============================================================================
#     Calculate the structure factor F(Q)
# =============================================================================
    #  Crystal structure factor for each atom at each Wyckoff position with 
    #  respect to the tI symmtry +1/2 transl.
    f_Eu, f_Ga, f_Al = fatom_calc_H0KL(H, k2d, l2d, Unitary, fatom, EuAl4, EuGa4)    
    F_Eu = np.zeros((len(k2d), len(k2d)), dtype=complex)
    F_Eu_T = np.zeros((len(k2d), len(k2d)), dtype=complex)
    F_Al1 = np.zeros((len(k2d), len(k2d)), dtype=complex)
    F_Al1_T = np.zeros((len(k2d), len(k2d)), dtype=complex)
    F_Al2 = np.zeros((len(k2d), len(k2d)), dtype=complex)
    F_Al2_T = np.zeros((len(k2d), len(k2d)), dtype=complex)
    F_Ga1 = np.zeros((len(k2d), len(k2d)), dtype=complex)
    F_Ga1_T = np.zeros((len(k2d), len(k2d)), dtype=complex)
    F_Ga2 = np.zeros((len(k2d), len(k2d)), dtype=complex)
    F_Ga2_T = np.zeros((len(k2d), len(k2d)), dtype=complex)   
    Fstructure_calc_H0KL(f_Eu, DBW_list[0], H, Unitary, Eu, k2d, l2d, F_Eu)
    Fstructure_calc_H0KL(f_Eu, DBW_list[0], H, Unitary, Eu_T, k2d, l2d, F_Eu_T)
    Fstructure_calc_H0KL(f_Al, DBW_list[2], H, Unitary, Al1, k2d, l2d, F_Al1)
    Fstructure_calc_H0KL(f_Al, DBW_list[2], H, Unitary, Al1_T, k2d, l2d, F_Al1_T)
    Fstructure_calc_H0KL(f_Al, DBW_list[2], H, Unitary, Al2, k2d, l2d, F_Al2)
    Fstructure_calc_H0KL(f_Al, DBW_list[2], H, Unitary, Al2_T, k2d, l2d, F_Al2_T)
    Fstructure_calc_H0KL(f_Ga, DBW_list[1], H, Unitary, Ga1, k2d, l2d, F_Ga1)
    Fstructure_calc_H0KL(f_Ga, DBW_list[1], H, Unitary, Ga1_T, k2d, l2d, F_Ga1_T)
    Fstructure_calc_H0KL(f_Ga, DBW_list[1], H, Unitary, Ga2, k2d, l2d, F_Ga2)
    Fstructure_calc_H0KL(f_Ga, DBW_list[1], H, Unitary, Ga2_T, k2d, l2d, F_Ga2_T)   
    F = F_Eu + F_Eu_T + F_Al1 + F_Al1_T + F_Al2 + F_Al2_T + F_Ga1 + F_Ga1_T + \
        F_Ga2 + F_Ga2_T
# =============================================================================
#     # Exclude unwanted kspace-points
# =============================================================================
    F, l_indices, k_indices = excludekspacepoints(k2d, l2d, deltak, F, q_cdw, 
                                                  kmax, lmax)
# =============================================================================
#     #  Compute Intensity 
# =============================================================================
    I = np.abs(F) ** 2  # I \propto F(Q)^2, F complex
# =============================================================================
#     # Normalize Intensity
# =============================================================================
    if normalization == True:
        I = I/np.max(I)
    else:
        I = I
# =============================================================================
#     # PLOTTING
# =============================================================================
    fig = plt.figure(figsize=(8, 3))
    if EuAl4 == True:
        plt.suptitle("EuAl4 I4/mmm, q={}rlu, [{}KL]".format(q_cdw, H))
    else:
        plt.suptitle("EuGa2Al2 I4/mmm, q={}rlu, [{}KL]".format(q_cdw, H))
    """LINECUTS"""
    k_indices = np.arange(0, int(2*kmax + 1), 1) * int(1/deltak)
    #  Only plot 1/4 of k-space, only plot for multiples of q_cdw in L
    for i in k_indices[len(k_indices) // 2:-int(len(k_indices)/4)]:
        plt.plot(l, I[:, i], ls='-', lw=0.5 , marker='x', 
                  ms=0.5, label='K={}'.format(np.round(k[i], 1)))
    plt.legend()
    plt.xlim(l0-kmax, l0+kmax)
    plt.ylabel("Intensity $I\propto F(\mathbf{Q})^2$")
    plt.xlabel("L (r.l.u.)") 
    if savefig == True:
        if EuAl4 == True:
            plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/XRD_Simulation/Eu(Ga,Al)4_modulation1/L-cuts_EuAl4_q={}_H={}_center{}{}{}_optimized.jpg".format(q_cdw, H, H, k0, l0), dpi=300, bbox_inches='tight')
        else:
            if EuGa4 == True:
                plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/XRD_Simulation/Eu(Ga,Al)4_modulation1/L-cuts_EuGa4_q={}_H={}_center{}{}{}_optimized.jpg".format(q_cdw, H, H, k0, l0), dpi=300, bbox_inches='tight')
            else:
                plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/XRD_Simulation/Eu(Ga,Al)4_modulation1/L-cuts_EuGa2Al2_q={}_H={}_center{}{}{}_optimized.jpg".format(q_cdw, H, H, k0, l0), dpi=300, bbox_inches='tight')
    else:
        pass
    
    fig = plt.figure()
    if EuAl4 == True:
        plt.suptitle("EuAl4 I4/mmm, q={}rlu, [{}KL]".format(q_cdw, H))
    else:
        if EuGa4 == True:
            plt.suptitle("EuGa4 I4/mmm, q={}rlu, [{}KL]".format(q_cdw, H))
        else:
            plt.suptitle("EuGa2Al2 I4/mmm, q={}rlu, [{}KL]".format(q_cdw, H))
    """CONVOLUTION"""
    plt.title(r"$|F(Q)|^2$-[{}KL]-map".format(H))
    x, y = np.linspace(-1, 1, kernelsize), np.linspace(-1, 1, kernelsize)
    X, Y = np.meshgrid(x, y)     
    kernel = 1/(2*np.pi*sigma**2)*np.exp(-(X**2+Y**2)/(2*sigma**2)) 
    Iconv = ndimage.convolve(I, kernel, mode='constant', cval=0.0)
    # Add noise
    Iconv = Iconv + np.abs(np.random.randn(len(k), len(k))) * noiseamplitude 
    if lognorm == True:
        plt.imshow(Iconv, cmap=cmap, 
                    extent=(k0-kmax,k0+kmax,l0-lmax,l0+lmax), 
                    origin='lower',
                    norm=LogNorm(vmin = 1, vmax = np.max(Iconv))
                )
    else:
        plt.imshow(Iconv, cmap=cmap, 
                    extent=(k0-kmax,k0+kmax,l0-lmax,l0+lmax), 
                    origin='lower', 
                    vmin=0, vmax = np.max(Iconv)
                )          
    plt.colorbar()
    plt.ylabel("L (r.l.u.)")
    plt.xlabel("K (r.l.u.)")   
    if savefig == True:
        if EuAl4 == True:
            plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/XRD_Simulation/Eu(Ga,Al)4_modulation1/{}KL_EuAl4_q={}_optimized.jpg".format(H, q_cdw), dpi=300, bbox_inches='tight')
        else:
            if EuGa4 == True:
                plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/XRD_Simulation/Eu(Ga,Al)4_modulation1/{}KL_EuGa4_q={}_optimized.jpg".format(H, q_cdw), dpi=300, bbox_inches='tight')
            else:
                plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/XRD_Simulation/Eu(Ga,Al)4_modulation1/{}KL_EuGa2Al2_q={}_optimized.jpg".format(H, q_cdw), dpi=300, bbox_inches='tight')
    else:
        pass
    """3D ATOM PLOT"""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # ax.title("Supercell with Wyckoff 4e z-parameter z0={}".format(z0))
    ax.scatter(Eu[0], Eu[1], Eu[2], label='2a (000)', c='yellow')
    ax.scatter(Eu_T[0], Eu_T[1], Eu_T[2], label='2a T(000)', facecolors='none', edgecolors='yellow')
    ax.scatter(Al1[0], Al1[1], Al1[2], label='4d (1/2 0 1/4), (0 1/2 1/4)', c='blue')
    ax.scatter(Al1_T[0], Al1_T[1], Al1_T[2], facecolors='none', edgecolors='blue')
    ax.scatter(Al2[0], Al2[1], Al2[2], c='blue')
    ax.scatter(Al2_T[0], Al2_T[1], Al2_T[2], facecolors='none', edgecolors='blue')
    ax.scatter(Ga1[0], Ga1[1], Ga1[2], label='4e (0 0 z0), (0 0 -z0)', c='green')
    ax.scatter(Ga1_T[0], Ga1_T[1], Ga1_T[2], facecolors='none', edgecolors='green')
    ax.scatter(Ga2[0], Ga2[1], Ga2[2], c='green')
    ax.scatter(Ga2_T[0], Ga2_T[1], Ga2_T[2], facecolors='none', edgecolors='green')
    # ax.set_xlim(-0.5, 1.5)
    # ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/XRD_Simulation/Eu(Ga,Al)4_modulation1/1412_Structure_q={}.jpg".format(q_cdw), dpi=300, bbox_inches='tight')
    plt.show(block=False)
    return Iconv


def main_HK0L(a, c, h0, l0, h2d, h, l, hmax, lmax, l2d, K, 
                                deltak, Unitary, u_list, kernelsize,
                                q_cdw, Nsc, z0, A, B, 
                                noiseamplitude, sigma, lognorm=False, 
                                normalization=False, DBW=False, savefig=False,
                                fatom=False, EuAl4=False, EuGa4=False):
    print("SIMULATION OF H{}L-MAP".format(K))
    print("Centered peak: [HKL] = [{}{}{}]".format(h0, K, l0))
    print("Modulation vector q_z = {}".format(q_cdw))
    print("Resolution in x and y: {}r.l.u.".format(deltak))
    print("K = {}, H, L in {} to {}".format(K, -hmax, hmax))
    """Nominal atomic positions"""
    # # Europium Z=63:
    # x_Eu, y_Eu, z_Eu = 0, 0, 0
    # #  Aluminium, Z=13:
    # x_Al1, y_Al1, z_Al1 = 0, 0.5, 0.25
    # x_Al2, y_Al2, z_Al2 = 0.5, 0, 0.25
    # #  Gallium, Z=31:
    # x_Ga1, y_Ga1, z_Ga1 = 0, 0, z0
    # x_Ga2, y_Ga2, z_Ga2 = 0, 0, -z0
    # Immm No. 71
    # Europium Z=63, 2a:
    x_Eu, y_Eu, z_Eu = 0, 0, 0
    #  Aluminium, Z=13, 4j:
    x_Al1, y_Al1, z_Al1 = 0, 0.5, 0.254
    x_Al2, y_Al2, z_Al2 = 0, 0.5, -0.254
    #  Gallium, Z=31, 4i:
    x_Ga1, y_Ga1, z_Ga1 = 0, 0, z0
    x_Ga2, y_Ga2, z_Ga2 = 0, 0, -z0
    print("DBW-Factor = {}".format(DBW))
    print("Q-dependent atomic form factors f(Q) = {}".format(fatom))
    print("Simulation for EuAl4 = {}".format(EuAl4))
    
    """Atomic positions"""
    ##########################################################################
    Nxy_neg, Nxy_pos = 0, 2  #int(np.sqrt(Nsc/(q_cdw*0.3)))
    Nz_neg, Nz_pos = 0, 5 #int(q_cdw**(-1)) * Nsc
    ##########################################################################
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
    
    """Nominal atomic positions"""
    posEu, posEu_T, posAl1, posAl1_T, posAl2, posAl2_T, \
    posGa1, posGa1_T, posGa2, posGa2_T = bpEu, bpEu_T, bpAl1,  \
    bpAl1_T, bpAl2, bpAl2_T, bpGa1, bpGa1_T, bpGa2, bpGa2_T
    
    # Add a constant factor 'T' to every element in the array
    T = 0.5  # Translation für (Eu,Ga,Al)_T posen
    for i in range(posEu.shape[0]):
        for j in range(posEu.shape[1]):
            posEu_T[i, j] = posEu_T[i, j] + T
            posAl1_T[i, j] = posAl1_T[i, j] + T
            posAl2_T[i, j] = posAl2_T[i, j] + T
            posGa1_T[i, j] = posGa1_T[i, j] + T
            posGa2_T[i, j] = posGa2_T[i, j] + T
    ##########################################################################    
    """Modulation"""
    #  Sum over N=1/q_cdw atoms per Wyckoff position
    if q_cdw == 1:
        pass
    else:
        for i in range(0, len(posAl1[2])):
            #  Order of Bessel functions B_j: simplest model j=1
            for j in range(1, besselorder + 1):
                # Create a list of position arrays
                position_arrays = [[posAl1, posAl1_T, posAl2, posAl2_T], 
                                    [posGa1, posGa1_T, posGa2, posGa2_T], 
                                    [posEu, posEu_T]]
                # Iterate over the positions and apply the calculations
                #  4d
                for pos in position_arrays[0]:
                    pos[0][i] += A[0][0] * np.sin(q_cdw * 2 * np.pi * j * pos[2][i]) + \
                        B[0][0] * np.cos(q_cdw * 2 * np.pi * j * pos[2][i])
                    pos[1][i] += A[0][1] * np.sin(q_cdw * 2 * np.pi * j * pos[2][i]) + \
                        B[0][1] * np.cos(q_cdw * 2 * np.pi * j * pos[2][i])
                    pos[2][i] += A[0][2] * np.sin(q_cdw * 2 * np.pi * j * pos[2][i]) + \
                        B[0][2] * np.cos(q_cdw * 2 * np.pi * j * pos[2][i])
                #  4e
                for pos in position_arrays[1]:
                    pos[0][i] += A[1][0] * np.sin(q_cdw * 2 * np.pi * j * pos[2][i]) + \
                        B[1][0] * np.cos(q_cdw * 2 * np.pi * j * pos[2][i])
                    pos[1][i] += A[1][1] * np.sin(q_cdw * 2 * np.pi * j * pos[2][i]) + \
                        B[1][1] * np.cos(q_cdw * 2 * np.pi * j * pos[2][i])
                    pos[2][i] += A[1][2] * np.sin(q_cdw * 2 * np.pi * j * pos[2][i]) + \
                        B[1][2] * np.cos(q_cdw * 2 * np.pi * j * pos[2][i])
                #  2a
                for pos in position_arrays[2]:
                    pos[0][i] += A[2][0] * np.sin(q_cdw * 2 * np.pi * j * pos[2][i]) + \
                        B[2][0] * np.cos(q_cdw * 2 * np.pi * j * pos[2][i])
                    pos[1][i] += A[2][1] * np.sin(q_cdw * 2 * np.pi * j * pos[2][i]) + \
                        B[2][1] * np.cos(q_cdw * 2 * np.pi * j * pos[2][i])
                    pos[2][i] += A[2][2] * np.sin(q_cdw * 2 * np.pi * j * pos[2][i]) + \
                        B[2][2] * np.cos(q_cdw * 2 * np.pi * j * pos[2][i])
    # Appending the ndarrays
    Eu = np.array([posEu[0], posEu[1], posEu[2]])   # Transpose to match the desired shape (3, 8)
    Eu_T = np.array([posEu_T[0], posEu_T[1], posEu_T[2]])
    Al1 = np.array([posAl1[0], posAl1[1], posAl1[2]])    # Transpose to match the desired shape (3, 8)
    Al1_T = np.array([posAl1_T[0], posAl1_T[1], posAl1_T[2]]) 
    Al2 = np.array([posAl2[0], posAl2[1], posAl2[2]])    # Transpose to match the desired shape (3, 8)
    Al2_T = np.array([posAl2_T[0], posAl2_T[1], posAl2_T[2]]) 
    Ga1 = np.array([posGa1[0], posGa1[1], posGa1[2]])    # Transpose to match the desired shape (3, 8)
    Ga1_T = np.array([posGa1_T[0], posGa1_T[1], posGa1_T[2]]) 
    Ga2 = np.array([posGa2[0], posGa2[1], posGa2[2]])    # Transpose to match the desired shape (3, 8)
    Ga2_T = np.array([posGa2_T[0], posGa2_T[1], posGa2_T[2]]) 
    print("# of Unit Cells = {}".format(Nz_pos*Nxy_pos))    
# =============================================================================
#     #  Compute Debye-Waller-Factor
# =============================================================================
    if DBW == True:
        DBW_list = dbw_HK0L(h2d, l2d, Unitary, K, a, c, u_list)
    else:
        DBW_list = np.ones(3)   
# =============================================================================
#     Calculate the structure factor F(Q)
# =============================================================================
    #  Crystal structure factor for each atom at each Wyckoff position with 
    #  respect to the tI symmtry +1/2 transl.
    f_Eu, f_Ga, f_Al = fatom_calc_HK0L(K, h2d, l2d, Unitary, 
                                              fatom, EuAl4)    
    F_Eu = np.zeros((len(h2d), len(h2d)), dtype=complex)
    F_Eu_T = np.zeros((len(h2d), len(h2d)), dtype=complex)
    F_Al1 = np.zeros((len(h2d), len(h2d)), dtype=complex)
    F_Al1_T = np.zeros((len(h2d), len(h2d)), dtype=complex)
    F_Al2 = np.zeros((len(h2d), len(h2d)), dtype=complex)
    F_Al2_T = np.zeros((len(h2d), len(h2d)), dtype=complex)
    F_Ga1 = np.zeros((len(h2d), len(h2d)), dtype=complex)
    F_Ga1_T = np.zeros((len(h2d), len(h2d)), dtype=complex)
    F_Ga2 = np.zeros((len(h2d), len(h2d)), dtype=complex)
    F_Ga2_T = np.zeros((len(h2d), len(h2d)), dtype=complex)   
    Fstructure_calc_HK0L(f_Eu, DBW_list[0], K, Unitary, Eu, h2d, l2d, F_Eu)
    Fstructure_calc_HK0L(f_Eu, DBW_list[0], K, Unitary, Eu_T, h2d, l2d, F_Eu_T)
    Fstructure_calc_HK0L(f_Al, DBW_list[2], K, Unitary, Al1, h2d, l2d, F_Al1)
    Fstructure_calc_HK0L(f_Al, DBW_list[2], K, Unitary, Al1_T, h2d, l2d, F_Al1_T)
    Fstructure_calc_HK0L(f_Al, DBW_list[2], K, Unitary, Al2, h2d, l2d, F_Al2)
    Fstructure_calc_HK0L(f_Al, DBW_list[2], K, Unitary, Al2_T, h2d, l2d, F_Al2_T)
    Fstructure_calc_HK0L(f_Ga, DBW_list[1], K, Unitary, Ga1, h2d, l2d, F_Ga1)
    Fstructure_calc_HK0L(f_Ga, DBW_list[1], K, Unitary, Ga1_T, h2d, l2d, F_Ga1_T)
    Fstructure_calc_HK0L(f_Ga, DBW_list[1], K, Unitary, Ga2, h2d, l2d, F_Ga2)
    Fstructure_calc_HK0L(f_Ga, DBW_list[1], K, Unitary, Ga2_T, h2d, l2d, F_Ga2_T)   
    F = F_Eu + F_Eu_T + F_Al1 + F_Al1_T + F_Al2 + F_Al2_T + F_Ga1 + F_Ga1_T + \
        F_Ga2 + F_Ga2_T
# =============================================================================
#     # Exclude unwanted kspace-points
# =============================================================================
    F, l_indices, h_indices = excludekspacepoints(h2d, l2d, deltak, F, q_cdw, 
                                                  hmax, lmax)
# =============================================================================
#     #  Compute Intensity 
# =============================================================================
    I = np.abs(F) ** 2  # I \propto F(Q)^2, F complex
# =============================================================================
#     # Normalize Intensity
# =============================================================================
    if normalization == True:
        I = I/np.max(I)
    else:
        I = I
# =============================================================================
#     # PLOTTING
# =============================================================================
    # fig = plt.figure(figsize=(8, 3))
    # if EuAl4 == True:
    #     plt.suptitle("EuAl4 I4/mmm, q={}rlu, [H{}L]".format(q_cdw, K))
    # else:
    #     if EuGa4 == True:
    #         plt.suptitle("EuGa4 I4/mmm, q={}rlu, [H{}L]".format(q_cdw, K))
    #     else:
    #         plt.suptitle("EuGa2Al2 I4/mmm, q={}rlu, [H{}L]".format(q_cdw, K))
    # """LINECUTS"""
    # h_indices = np.arange(0, int(2*hmax + 1), 1) * int(1/deltak)
    # #  Only plot 1/4 of k-space, only plot for multiples of q_cdw in L
    # for i in h_indices[len(h_indices) // 2:-int(len(h_indices)/4)]:
    #     plt.plot(l, I[:, i], ls='-', lw=0.5 , marker='s', 
    #               ms=2, label='H={}'.format(np.round(h[i], 1)))
    # plt.legend()
    # plt.xlim(l0-lmax, l0+lmax)
    # plt.ylabel("Intensity $I\propto F(\mathbf{Q})^2$")
    # plt.xlabel("L (r.l.u.)") 
    # if savefig == True:
    #     if EuAl4 == True:
    #         plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/XRD_Simulation/Eu(Ga,Al)4_modulation1/L-cuts_EuAl4_q={}_K={}_center{}{}{}_optimized.jpg".format(q_cdw, K, h0, K, l0), dpi=300)
    #     else:
    #         if EuGa4 == True:
    #             plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/XRD_Simulation/Eu(Ga,Al)4_modulation1/L-cuts_EuGa4_q={}_K={}_center{}{}{}_optimized.jpg".format(q_cdw, K, h0, K, l0), dpi=300, bbox_inches='tight')
    #         else:
    #             plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/XRD_Simulation/Eu(Ga,Al)4_modulation1/L-cuts_EuGa2Al2_q={}_K={}_center{}{}{}_optimized.jpg".format(q_cdw, K, h0, K, l0), dpi=300, bbox_inches='tight')
    # else:
    #     pass
    
    # fig = plt.figure()  
    # if EuAl4 == True:
    #     plt.suptitle("EuAl4 I4/mmm, q={}rlu, [H{}L]".format(q_cdw, K))
    # else:
    #     if EuGa4 == True:
    #         plt.suptitle("EuGa4 I4/mmm, q={}rlu, [H{}L]".format(q_cdw, K))
    #     else:
    #         plt.suptitle("EuGa2Al2 I4/mmm, q={}rlu, [H{}L]".format(q_cdw, K))
    # """CONVOLUTION"""
    # plt.title("[H{}L], q={}rlu".format(K, q_cdw))
    # x, y = np.linspace(-1, 1, kernelsize), np.linspace(-1, 1, kernelsize)
    # X, Y = np.meshgrid(x, y)     
    # kernel = 1/(2*np.pi*sigma**2)*np.exp(-(X**2+Y**2)/(2*sigma**2)) 
    # Iconv = ndimage.convolve(I, kernel, mode='constant', cval=0.0)
    # # Add noise
    # Iconv = Iconv + np.abs(np.random.randn(len(h), len(h))) * noiseamplitude 
    # if lognorm == True:
    #     plt.imshow(Iconv, cmap=cmap, 
    #                extent=(h0-hmax,h0+hmax,l0-lmax,l0+lmax),
    #                origin='lower', aspect='auto',
    #                norm=LogNorm(vmin=1, vmax=np.max(Iconv))
    #             )
    # else:
    #     plt.imshow(Iconv, cmap=cmap, 
    #                extent=(h0-hmax,h0+hmax,l0-lmax,l0+lmax), 
    #                origin='lower', 
    #                vmin=0, vmax = np.max(Iconv)
    #             )          
    # plt.colorbar()
    # # plt.xlim(2, 8)     
    # # plt.ylim(-13, 8)    
    # plt.ylabel("L (r.l.u.)")      
    # plt.xlabel("H (r.l.u.)")   
    # if savefig == True:
    #     if EuAl4 == True:
    #         plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/XRD_Simulation/Eu(Ga,Al)4_modulation1/H{}L_EuAl4_q={}_optimized.jpg".format(K, q_cdw), dpi=300, bbox_inches='tight')
    #     else:
    #         if EuGa4 == True:
    #             plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/XRD_Simulation/Eu(Ga,Al)4_modulation1/H{}L_EuGa4_q={}_optimized.jpg".format(K, q_cdw), dpi=300, bbox_inches='tight')
    #         else:
    #             plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/XRD_Simulation/Eu(Ga,Al)4_modulation1/H{}L_EuGa2Al2_q={}_optimized.jpg".format(K, q_cdw), dpi=300, bbox_inches='tight')
    # else:
    #     pass
    # # """3D ATOM PLOT"""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
# =============================================================================
#     I4/mmm
# =============================================================================
# 3D Plot
    # ax.set_title("Immm supercell with Wyckoff 4e z-parameter z={}".format(z0))
    # usable_pointsx = (x < 1) 
    # usable_pointsx = (y < 1) 
    # usable_pointsz  = (z < 5)
    # Eu[0], Eu[1], Eu[2] = Eu[0].flatten(), Eu[1].flatten(), Eu[2].flatten()
    # usable_points = (Eu[0] < 1) & (Eu[1] < 1) & (Eu[2] < 5)
    # Eu[0], Eu[1], Eu[2] = Eu[0][usable_points], Eu[1][usable_points], Eu[2][usable_points]
    # Eu[0], Eu[1], Eu[2] = Eu[0].flatten(), Eu[1].flatten(), Eu[2].flatten()
    # Eu[0], Eu[1], Eu[2] = Eu[0][usable_pointsxy], Eu[1][usable_pointsxy], Eu[2][usable_pointsz]
    # Eu_T[0], Eu_T[1], Eu_T[2] = Eu_T[0].flatten(), Eu_T[1].flatten(), Eu_T[2].flatten()
    # Eu_T[0], Eu_T[1], Eu_T[2] = Eu_T[0][usable_pointsxy], Eu_T[1][usable_pointsxy], Eu_T[2][usable_pointsz]
    # x, y, z = Al1[0].flatten(), Al1[1].flatten(), Al1[2].flatten()
    # Al1[0], Al1[1], Al1[2] = x[usable_points], y[usable_points], z[usable_points]
    # x, y, z = Al1_T[0].flatten(), Al1_T[1].flatten(), Al1_T[2].flatten()
    # Al1_T[0], Al1_T[1], Al1_T[2] = x[usable_points], y[usable_points], z[usable_points]
    # x, y, z = Al2[0].flatten(), Al2[1].flatten(), Al2[2].flatten()
    # Al2[0], Al2[1], Al2[2] = x[usable_points], y[usable_points], z[usable_points]
    # x, y, z = Al2_T[0].flatten(), Al2_T[1].flatten(), Al2_T[2].flatten()
    # Al2_T[0], Al2_T[1], Al2_T[2] = x[usable_points], y[usable_points], z[usable_points]
    # x, y, z = Ga1[0].flatten(), Ga1[1].flatten(), Ga1[2].flatten()
    # Ga1[0], Ga1[1], Ga1[2] = x[usable_points], y[usable_points], z[usable_points]
    # x, y, z = Ga1[0].flatten(), Ga1[1].flatten(), Ga1[2].flatten()
    # Ga1_T[0], Ga1_T[1], Ga1_T[2] = x[usable_points], y[usable_points], z[usable_points]
    # x, y, z = Ga1_T[0].flatten(), Ga1_T[1].flatten(), Ga1_T[2].flatten()
    # Ga2[0], Ga2[1], Ga2[2] = x[usable_points], y[usable_points], z[usable_points]
    # x, y, z = Ga2[0].flatten(), Ga2[1].flatten(), Ga2[2].flatten()
    # Ga2_T[0], Ga2_T[1], Ga2_T[2] = x[usable_points], y[usable_points], z[usable_points]
    ax.scatter(Eu[0], Eu[1], Eu[2], 
               c='yellow', 
               s=100)
    ax.scatter(Eu_T[0], Eu_T[1], Eu_T[2], 
               c='yellow',
               s=100)
    ax.scatter(Al1[0], Al1[1], Al1[2], label='4d ', 
               c='blue', 
               s=20)
    ax.scatter(Al1_T[0], Al1_T[1], Al1_T[2], 
               c='blue', 
               s=20)
    ax.scatter(Al2[0], Al2[1], Al2[2], 
               c='blue', 
               s=20)
    ax.scatter(Al2_T[0], Al2_T[1], Al2_T[2], 
               c='blue', 
               s=20)
    ax.scatter(Ga1[0], Ga1[1], Ga1[2], label='4e', 
               c='green', 
               s=20)
    ax.scatter(Ga1_T[0], Ga1_T[1], Ga1_T[2], 
               c='green', 
               s=20)
    ax.scatter(Ga2[0], Ga2[1], Ga2[2], 
               c='green', 
               s=20)
    ax.scatter(Ga2_T[0], Ga2_T[1], Ga2_T[2], 
               c='green', 
               s=20)
    # ax.set_xlim(-5, 5.0)
    # ax.set_ylim(-5, 5.0)
    # ax.set_zlim(0, 0.01)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.axis('off')
    # ax.set_proj_type('persp', focal_length=1)
    ax.view_init(elev=0, azim=-90)
    ax.set_aspect('equal', adjustable='box')
    plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/XRD_Simulation/Eu(Ga,Al)4_modulation1/5UC_T>Tcdw_q={}.jpg".format(q_cdw), transparent = True, bbox_inches= 'tight', dpi=600, edgecolor= None)
    plt.legend()
    # # 2D Plot
    # fig = plt.figure(figsize=(3, 7))
    # plt.scatter(Eu[0], Eu[2], label='Eu', c='k', s=50)
    # # plt.scatter(Eu_T[0], Eu_T[2], c='k', s=50)
    # plt.scatter(Al1[0], Al1[2], label='Al1', c='g', s=20)
    # # plt.scatter(Al1_T[0], Al1_T[2], c='g', s=20)
    # plt.scatter(Al2[0], Al2[2], label='Al1',c ='g', s=20)
    # # plt.scatter(Al2_T[0], Al2_T[2],c ='g', s=20)
    # plt.scatter(Ga1[0], Ga1[2], label='Al2', c='b', s=20)
    # # plt.scatter(Ga1_T[0], Ga1_T[2], c='b', s=20)
    # plt.scatter(Ga2[0], Ga2[2], label='Al2',c ='b', s=20)
    # # plt.scatter(Ga2_T[0], Ga2_T[2],c ='b', s=20)
    # plt.legend()
    # plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/XRD_Simulation/Eu(Ga,Al)4_modulation1/2D_1412_Structure_q={}.jpg".format(q_cdw), dpi=300)
    plt.show(block=False)
   # return Iconv


def main_HKL0(a, c, h0, k0, h2d, h, k, hmax, kmax, k2d, L, 
                                deltak, Unitary, u_list, kernelsize,
                                q_cdw, Nsc, z0, A, B, 
                                noiseamplitude, sigma, lognorm=False, 
                                normalization=False, DBW=False, savefig=False,
                                fatom=False, EuAl4=False, EuGa4=False):
    print("SIMULATION OF HK{}-MAP".format(L))
    print("Centered peak: [HKL] = [{}{}{}]".format(h0, k0, L))
    print("Modulation vector q_z = {}".format(q_cdw))
    print("Resolution in x and y: {}r.l.u.".format(deltak))
    print("L = {}, H, K in {} to {}".format(L, -hmax, hmax))
    """Nominal atomic positions"""
    # Europium Z=63:
    x_Eu, y_Eu, z_Eu = 0, 0, 0
    #  Aluminium, Z=13:
    x_Al1, y_Al1, z_Al1 = 0, 0.5, 0.25
    x_Al2, y_Al2, z_Al2 = 0.5, 0, 0.25
    #  Gallium, Z=31:
    x_Ga1, y_Ga1, z_Ga1 = 0, 0, z0
    x_Ga2, y_Ga2, z_Ga2 = 0, 0, -z0
    print("DBW-Factor = {}".format(DBW))
    print("Q-dependent atomic form factors f(Q) = {}".format(fatom))
    print("Simulation for EuAl4 = {}".format(EuAl4))
    
    """Atomic positions"""
    ##########################################################################
    Nxy_neg, Nxy_pos = 0, 1  #int(np.sqrt(Nsc/(q_cdw*0.3)))
    Nz_neg, Nz_pos = 0, int(q_cdw**(-1)) * Nsc
    ##########################################################################
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
    """Nominal atomic positions"""
    posEu, posEu_T, posAl1, posAl1_T, posAl2, posAl2_T, \
    posGa1, posGa1_T, posGa2, posGa2_T = bpEu, bpEu_T, bpAl1,  \
    bpAl1_T, bpAl2, bpAl2_T, bpGa1, bpGa1_T, bpGa2, bpGa2_T
    # Add a constant factor 'T' to every element in the array
    T = 0.5  # Translation für (Eu,Ga,Al)_T posen
    for i in range(posEu.shape[0]):
        for j in range(posEu.shape[1]):
            posEu_T[i, j] = posEu_T[i, j] + T
            posAl1_T[i, j] = posAl1_T[i, j] + T
            posAl2_T[i, j] = posAl2_T[i, j] + T
            posGa1_T[i, j] = posGa1_T[i, j] + T
            posGa2_T[i, j] = posGa2_T[i, j] + T
    ##########################################################################    
    """Modulation"""
    #  Sum over N=1/q_cdw atoms per Wyckoff position
    if q_cdw == 1:
        pass
    else:
        for i in range(0, len(posAl1[2])):
            #  Order of Bessel functions B_j: simplest model j=1
            for j in range(1, besselorder + 1):
                # Create a list of position arrays
                position_arrays = [[posAl1, posAl1_T, posAl2, posAl2_T], 
                                    [posGa1, posGa1_T, posGa2, posGa2_T], 
                                    [posEu, posEu_T]]
                # Iterate over the positions and apply the calculations
                #  4d
                for pos in position_arrays[0]:
                    pos[0][i] += A[0][0] * np.sin(q_cdw * 2 * np.pi * j * pos[2][i]) + \
                        B[0][0] * np.cos(q_cdw * 2 * np.pi * j * pos[2][i])
                    pos[1][i] += A[0][1] * np.sin(q_cdw * 2 * np.pi * j * pos[2][i]) + \
                        B[0][1] * np.cos(q_cdw * 2 * np.pi * j * pos[2][i])
                    pos[2][i] += A[0][2] * np.sin(q_cdw * 2 * np.pi * j * pos[2][i]) + \
                        B[0][2] * np.cos(q_cdw * 2 * np.pi * j * pos[2][i])
                #  4e
                for pos in position_arrays[1]:
                    pos[0][i] += A[1][0] * np.sin(q_cdw * 2 * np.pi * j * pos[2][i]) + \
                        B[1][0] * np.cos(q_cdw * 2 * np.pi * j * pos[2][i])
                    pos[1][i] += A[1][1] * np.sin(q_cdw * 2 * np.pi * j * pos[2][i]) + \
                        B[1][1] * np.cos(q_cdw * 2 * np.pi * j * pos[2][i])
                    pos[2][i] += A[1][2] * np.sin(q_cdw * 2 * np.pi * j * pos[2][i]) + \
                        B[1][2] * np.cos(q_cdw * 2 * np.pi * j * pos[2][i])
                #  2a
                for pos in position_arrays[2]:
                    pos[0][i] += A[2][0] * np.sin(q_cdw * 2 * np.pi * j * pos[2][i]) + \
                        B[2][0] * np.cos(q_cdw * 2 * np.pi * j * pos[2][i])
                    pos[1][i] += A[2][1] * np.sin(q_cdw * 2 * np.pi * j * pos[2][i]) + \
                        B[2][1] * np.cos(q_cdw * 2 * np.pi * j * pos[2][i])
                    pos[2][i] += A[2][2] * np.sin(q_cdw * 2 * np.pi * j * pos[2][i]) + \
                        B[2][2] * np.cos(q_cdw * 2 * np.pi * j * pos[2][i])
    # Appending the ndarrays
    Eu = np.array([posEu[0], posEu[1], posEu[2]])   # Transpose to match the desired shape (3, 8)
    Eu_T = np.array([posEu_T[0], posEu_T[1], posEu_T[2]])
    Al1 = np.array([posAl1[0], posAl1[1], posAl1[2]])    # Transpose to match the desired shape (3, 8)
    Al1_T = np.array([posAl1_T[0], posAl1_T[1], posAl1_T[2]]) 
    Al2 = np.array([posAl2[0], posAl2[1], posAl2[2]])    # Transpose to match the desired shape (3, 8)
    Al2_T = np.array([posAl2_T[0], posAl2_T[1], posAl2_T[2]]) 
    Ga1 = np.array([posGa1[0], posGa1[1], posGa1[2]])    # Transpose to match the desired shape (3, 8)
    Ga1_T = np.array([posGa1_T[0], posGa1_T[1], posGa1_T[2]]) 
    Ga2 = np.array([posGa2[0], posGa2[1], posGa2[2]])    # Transpose to match the desired shape (3, 8)
    Ga2_T = np.array([posGa2_T[0], posGa2_T[1], posGa2_T[2]]) 
    print("# of Unit Cells = {}".format(Nz_pos*Nxy_pos))    
# =============================================================================
#     #  Compute Debye-Waller-Factor
# =============================================================================
    if DBW == True:
        DBW_list = dbw_HKL0(h2d, k2d, Unitary, L, a, c, u_list)
    else:
        DBW_list = np.ones(3)   
# =============================================================================
#     Calculate the structure factor F(Q)
# =============================================================================
    #  Crystal structure factor for each atom at each Wyckoff position with 
    #  respect to the tI symmtry +1/2 transl.
    f_Eu, f_Ga, f_Al = fatom_calc_HKL0(L, h2d, k2d, Unitary, 
                                              fatom, EuAl4)    
    F_Eu = np.zeros((len(h2d), len(h2d)), dtype=complex)
    F_Eu_T = np.zeros((len(h2d), len(h2d)), dtype=complex)
    F_Al1 = np.zeros((len(h2d), len(h2d)), dtype=complex)
    F_Al1_T = np.zeros((len(h2d), len(h2d)), dtype=complex)
    F_Al2 = np.zeros((len(h2d), len(h2d)), dtype=complex)
    F_Al2_T = np.zeros((len(h2d), len(h2d)), dtype=complex)
    F_Ga1 = np.zeros((len(h2d), len(h2d)), dtype=complex)
    F_Ga1_T = np.zeros((len(h2d), len(h2d)), dtype=complex)
    F_Ga2 = np.zeros((len(h2d), len(h2d)), dtype=complex)
    F_Ga2_T = np.zeros((len(h2d), len(h2d)), dtype=complex)   
    Fstructure_calc_HKL0(f_Eu, DBW_list[0], L, Unitary, Eu, h2d, k2d, F_Eu)
    Fstructure_calc_HKL0(f_Eu, DBW_list[0], L, Unitary, Eu_T, h2d, k2d, F_Eu_T)
    Fstructure_calc_HKL0(f_Al, DBW_list[2], L, Unitary, Al1, h2d, k2d, F_Al1)
    Fstructure_calc_HKL0(f_Al, DBW_list[2], L, Unitary, Al1_T, h2d, k2d, F_Al1_T)
    Fstructure_calc_HKL0(f_Al, DBW_list[2], L, Unitary, Al2, h2d, k2d, F_Al2)
    Fstructure_calc_HKL0(f_Al, DBW_list[2], L, Unitary, Al2_T, h2d, k2d, F_Al2_T)
    Fstructure_calc_HKL0(f_Ga, DBW_list[1], L, Unitary, Ga1, h2d, k2d, F_Ga1)
    Fstructure_calc_HKL0(f_Ga, DBW_list[1], L, Unitary, Ga1_T, h2d, k2d, F_Ga1_T)
    Fstructure_calc_HKL0(f_Ga, DBW_list[1], L, Unitary, Ga2, h2d, k2d, F_Ga2)
    Fstructure_calc_HKL0(f_Ga, DBW_list[1], L, Unitary, Ga2_T, h2d, k2d, F_Ga2_T)   
    F = F_Eu + F_Eu_T + F_Al1 + F_Al1_T + F_Al2 + F_Al2_T + F_Ga1 + F_Ga1_T + \
        F_Ga2 + F_Ga2_T
# =============================================================================
#     # Exclude unwanted kspace-points
# =============================================================================
    F, h_indices = excludekspacepoints_HKmap(h2d, k2d, deltak, F, hmax, kmax)
# =============================================================================
#     #  Compute Intensity 
# =============================================================================
    I = np.abs(F) ** 2  # I \propto F(Q)^2, F complex
# =============================================================================
#     # Normalize Intensity
# =============================================================================
    if normalization == True:
        I = I/np.max(I)
    else:
        I = I
# =============================================================================
#     # PLOTTING
# =============================================================================
    fig = plt.figure(figsize=(8, 3))
    if EuAl4 == True:
        plt.suptitle("EuAl4 I4/mmm, q={}rlu, [HK{}]".format(q_cdw, L))
    else:
        plt.suptitle("EuGa2Al2 I4/mmm, q={}rlu, [HK{}]".format(q_cdw, L))
    """LINECUTS"""
    h_indices = np.arange(0, int(2*hmax + 1), 1) * int(1/deltak)
    #  Only plot 1/4 of k-space, only plot for multiples of q_cdw in L
    for i in h_indices[len(h_indices) // 2:-int(len(h_indices)/4)]:
        plt.plot(k, I[:, i], ls='-', lw=0.5 , marker='x', 
                  ms=0.5, label='K={}'.format(np.round(k[i], 1)))
    plt.legend()
    plt.xlim(h0-hmax, h0+hmax)
    plt.ylabel("Intensity $I\propto F(\mathbf{Q})^2$")
    plt.xlabel("L (r.l.u.)") 
    # if savefig == True:
    #     if EuAl4 == True:
    #         plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/XRD_Simulation/Eu(Ga,Al)4_modulation1/L-cuts_EuAl4_q={}_H={}_center{}{}{}_optimized.jpg".format(q_cdw, H, H, k0, l0), dpi=300, bbox_inches='tight')
    #     else:
    #         plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/XRD_Simulation/Eu(Ga,Al)4_modulation1/L-cuts_EuGa2Al2_q={}_H={}_center{}{}{}_optimized.jpg".format(q_cdw, H, H, k0, l0), dpi=300, bbox_inches='tight')
    # else:
    #     pass
    
    fig = plt.figure()
    if EuAl4 == True:
        plt.suptitle("EuAl4 I4/mmm, q={}rlu, [HK{}]".format(q_cdw, L))
    else:
        if EuGa4 == True:
            plt.suptitle("EuGa4 I4/mmm, q={}rlu, [HK{}]".format(q_cdw, L))
        else:
            plt.suptitle("EuGa2Al2 I4/mmm, q={}rlu, [HK{}]".format(q_cdw, L))
    """CONVOLUTION"""
    plt.title(r"$|F(Q)|^2$-[HK{}]-map".format(L))
    x, y = np.linspace(-1, 1, kernelsize), np.linspace(-1, 1, kernelsize)
    X, Y = np.meshgrid(x, y)     
    kernel = 1/(2*np.pi*sigma**2)*np.exp(-(X**2+Y**2)/(2*sigma**2)) 
    Iconv = ndimage.convolve(I, kernel, mode='constant', cval=0.0)
    # Add noise
    Iconv = Iconv + np.abs(np.random.randn(len(h), len(h))) * noiseamplitude 
    if lognorm == True:
        plt.imshow(Iconv, cmap=cmap, 
                    extent=(-8.7722, 8.7722, -22.33, 22.33), #h0-hmax,h0+hmax,k0-kmax,k0+kmax
                    origin='lower', aspect=0.5/1.0,
                    norm=LogNorm(vmin = 1, vmax = np.max(Iconv))
                )
    else:
        plt.imshow(Iconv, cmap=cmap, 
                    extent=(h0-hmax,h0+hmax,k0-kmax,k0+kmax), 
                    origin='lower', 
                    vmin=0, vmax = np.max(Iconv)
                )          
    plt.colorbar()
    # plt.xlim(-5, 5)
    plt.ylabel("K (r.l.u.)")
    plt.xlabel("H (r.l.u.)")   
    if savefig == True:
        if EuAl4 == True:
            plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/XRD_Simulation/Eu(Ga,Al)4_modulation1/HK{}_EuAl4_q={}_optimized.jpg".format(L, q_cdw), dpi=300)
        else:
            if EuGa4 == True:
                plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/XRD_Simulation/Eu(Ga,Al)4_modulation1/HK{}_EuAl4_q={}_optimized.jpg".format(L, q_cdw), dpi=300, bbox_inches='tight')
            else:
                plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/XRD_Simulation/Eu(Ga,Al)4_modulation1/HK{}_EuGa2Al2_q={}_optimized.jpg".format(L, q_cdw), dpi=300, bbox_inches='tight')
    else:
        pass
    # """3D ATOM PLOT"""
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # # ax.title("Supercell with Wyckoff 4e z-parameter z0={}".format(z0))
    # # ax.scatter(Eu[0], Eu[1], Eu[2], label='2a (000)', c='yellow')
    # # ax.scatter(Eu_T[0], Eu_T[1], Eu_T[2], label='2a T(000)', facecolors='none', edgecolors='yellow')
    # ax.scatter(Al1[0], Al1[1], Al1[2], label='4d (1/2 0 1/4), (0 1/2 1/4)', c='blue')
    # ax.scatter(Al1_T[0], Al1_T[1], Al1_T[2], facecolors='none', edgecolors='blue')
    # ax.scatter(Al2[0], Al2[1], Al2[2], c='blue')
    # ax.scatter(Al2_T[0], Al2_T[1], Al2_T[2], facecolors='none', edgecolors='blue')
    # ax.scatter(Ga1[0], Ga1[1], Ga1[2], label='4e (0 0 z0), (0 0 -z0)', c='green')
    # ax.scatter(Ga1_T[0], Ga1_T[1], Ga1_T[2], facecolors='none', edgecolors='green')
    # ax.scatter(Ga2[0], Ga2[1], Ga2[2], c='green')
    # ax.scatter(Ga2_T[0], Ga2_T[1], Ga2_T[2], facecolors='none', edgecolors='green')
    # # ax.set_xlim(-0.5, 1.5)
    # # ax.set_ylim(-0.5, 1.5)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # plt.legend()
    # plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/XRD_Simulation/Eu(Ga,Al)4_modulation1/Structure_q={}.jpg".format(q_cdw), dpi=300)
    plt.show(block=False)
    
    return Iconv

