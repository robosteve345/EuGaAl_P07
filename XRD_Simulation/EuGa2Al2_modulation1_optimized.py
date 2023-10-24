#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Created on Tue Oct 24 12:30:02 2023

#@author: stevengebel

"""XRD Structure Factor F(Q) simulation for Eu(Ga,Al)4"""

import numpy as np
import matplotlib
import time
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.colors import LogNorm
from XRD_simulation_functions import kspacecreator, debyewallerfactor, excludekspacepoints
from XRD_simulation_functions import atomicformfactorEuGaAl


def calculate_F(f, DBW, h, Unitary, coords, k2d, l2d, result):
    for i in range(coords.shape[1]):
        result += f * DBW * np.exp(-2 * np.pi * 1j * (h * Unitary * coords[0, i] + k2d * coords[1, i] + l2d * coords[2, i]))


def structurefactorandplotting(a, c, k0, l0, k2d, k, l, kmax, lmax, l2d, h, 
                               deltak, Unitary, u_list, kernelsize, A, 
                               q_cdw, Nsc, kspacefactors, z0,
                               noiseamplitude, sigma, lognorm=False, 
                               normalization=False, DBW=False, savefig=False,
                               properatomicformfactor=False, EuAl4=False,
                               Lexclude=False):
    print("CENTERED PEAK: [HKL] = [{}{}{}]".format(h, k0, l0))
    print("K-space point density ∆k = {}".format(deltak))
    print("H = {}, K, L in {} to {}".format(h, -kmax, kmax))
    """Nominal atomic positions"""
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
    
    """Atomic positions"""
    ##########################################################################
    Nxy_neg, Nxy_pos = 0, 1#int(np.sqrt(Nsc/(q_cdw*0.3)))
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
    for i in range(0, len(posAl1[2])):  # begins with 1 because it evaluates every unit cell, 0 to n-1, BLEIBT SO
        for j in range(1,2):
            B = 0
            """Model 1: Al-intralayer distortion"""
            # posAl1[0][i] = posAl1[0][i] + \
            #           A * np.sin(q_cdw * 2 * np.pi * j * posAl1[2][i])  + \
            #           B * np.cos(q_cdw * 2 * np.pi * j * posAl1[2][i])
            # posAl1_T[1][i] = posAl1_T[1][i] + \
            #         A * np.sin(q_cdw * 2 * np.pi * j * posAl1_T[2][i]) + \
            #             B * np.cos(q_cdw * 2 * np.pi * j * posAl1_T[2][i])
            # posAl2[1][i] = posAl2[1][i] + \
            #           -A * np.sin(q_cdw * 2 * np.pi * j * posAl2[2][i]) + \
            #               -B * np.cos(q_cdw * 2 * np.pi * j * posAl2[2][i])
            # posAl2_T[0][i] = posAl2_T[0][i] + \
            #           -A * np.sin(q_cdw * 2 * np.pi * j * posAl2_T[2][i]) +\
            #               -B * np.cos(q_cdw * 2 * np.pi * j * posAl2_T[2][i])
            """Model 2: Ga-Al layer distortion"""
            posGa1[2][i] = posGa1[2][i] + \
                  A * np.sin(q_cdw * 2 * np.pi * j * posGa1[2][i]) 
            posGa1_T[2][i] = posGa1_T[2][i] + \
                    A * np.sin(q_cdw * 2 * np.pi * j * posGa1_T[2][i]) 
            posGa2[2][i] = posGa2[2][i] + \
                    A * np.sin(q_cdw * 2 * np.pi * j * posGa2[2][i]) 
            posGa2_T[2][i] = posGa2_T[2][i] + \
                    A * np.sin(q_cdw * 2 * np.pi * j * posGa2_T[2][i])  

    ##########################################################################
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
    print("# of Unit Cells Nz*Nxy = {}".format(Nz_pos*Nxy_pos))
    
    #  Compute Debye-Waller-Factor
    if DBW == True:
        DBW_list = debyewallerfactor(k2d, l2d, Unitary, h, a, c, u_list)
    else:
        DBW_list = np.ones(3)
    
    """Scattering amplitudes F"""
    #  Crystal structure factor for each atom at each Wyckoff position with 
    #  respect to the tI symmtry +1/2 transl.
    f_Eu, f_Ga, f_Al = atomicformfactorEuGaAl(h, k2d, l2d, Unitary, 
                                              properatomicformfactor)
    
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
    
    calculate_F(f_Eu, DBW_list[0], h, Unitary, Eu, k2d, l2d, F_Eu)
    calculate_F(f_Eu, DBW_list[0], h, Unitary, Eu_T, k2d, l2d, F_Eu_T)
    calculate_F(f_Al, DBW_list[2], h, Unitary, Al1, k2d, l2d, F_Al1)
    calculate_F(f_Al, DBW_list[2], h, Unitary, Al1_T, k2d, l2d, F_Al1_T)
    calculate_F(f_Al, DBW_list[2], h, Unitary, Al2, k2d, l2d, F_Al2)
    calculate_F(f_Al, DBW_list[2], h, Unitary, Al2_T, k2d, l2d, F_Al2_T)
    calculate_F(f_Ga, DBW_list[1], h, Unitary, Ga1, k2d, l2d, F_Ga1)
    calculate_F(f_Ga, DBW_list[1], h, Unitary, Ga1_T, k2d, l2d, F_Ga1_T)
    calculate_F(f_Ga, DBW_list[1], h, Unitary, Ga2, k2d, l2d, F_Ga2)
    calculate_F(f_Ga, DBW_list[1], h, Unitary, Ga2_T, k2d, l2d, F_Ga2_T)
    
    F = F_Eu + F_Eu_T + F_Al1 + F_Al1_T + F_Al2 + F_Al2_T + F_Ga1 + F_Ga1_T + \
        F_Ga2 + F_Ga2_T
    
    #  Compute Intensity 
    I = np.abs(F) ** 2  # I \propto F(Q)^2, F complex
    
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
    fig = plt.figure(figsize=(15, 5))
    plt.suptitle(r"EuGa2Al2, Centered around [{}{}{}], q={}rlu, {} Supercell(s)".format(h, k0, l0, q_cdw, Nsc))
    """MODULATED ATOMIC POSITIONS"""
    plt.subplot(2, 1, 1)
    plt.scatter(init_Ga1_z, np.ones(len(Ga1[0])),
                label=r'Ga1=$(0,0,{})$ equilibrium'.format(z0), facecolors='none',
                edgecolors='orange', s=100)
    plt.xlabel('z')
    plt.ylabel('')
    plt.scatter(Ga1[2, :], np.ones(len(init_Ga1_z)),
                label='Ga=$(0.5,0.5,0.5 + {} \sin({} \cdot 2\pi L))$ distorted'.format(A, q_cdw),
                marker='o')
    plt.legend()
    """LINECUTS"""
    plt.subplot(2, 1, 2)
    plt.plot(l2d[:, 0], I[:, len(k) // 2], ls='-', lw=1, marker='.', 
             ms=1, label='K={}'.format(np.round(k[len(k) // 2], 2)))
    plt.plot(l2d[:, 0], I[:, 0], ls='-', marker='.', lw=1, 
             ms=1, label='K={}'.format(np.round(k[0], 2)))
    plt.legend()
    plt.xlim(l0-kmax, l0+kmax)
    plt.ylabel(r"Intensity $I\propto F(\mathbf{Q})^2$")
    plt.xlabel("L (r.l.u.)") 
    if savefig == True:
        plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/XRD_Simulation/EuGa2Al2_modulation1/L-cuts_N={}{}{}_q={}_H={}_center{}{}{}_updated.jpg".format(Nxy_pos, Nxy_pos, Nz_pos, q_cdw, h, h, k0, l0), dpi=300)
    else:
        pass
    plt.subplots_adjust(wspace=0.3)
    
    fig = plt.figure(figsize=(15, 5))
    plt.suptitle("Eu(Ga,Al)4, Centered around [{}{}{}], q={}rlu, {} Supercell(s)".format(h, k0, l0, q_cdw, Nsc))
    """CONVOLUTION"""
    plt.subplot(1, 2, 1)
    plt.title("Gaussian conv., [{}KL]-map".format(h))
    x, y = np.linspace(-1,1,kernelsize), np.linspace(-1,1,kernelsize)
    X, Y = np.meshgrid(x, y)
    kernel = 1/(2*np.pi*sigma**2)*np.exp(-(X**2+Y**2)/(2*sigma**2))
    Iconv = ndimage.convolve(I, kernel, mode='constant', cval=0.0)
    if lognorm == True:
        plt.imshow(Iconv, cmap='viridis', extent=(k0-kmax, k0+kmax, l0-lmax, l0+lmax), 
                origin='lower',
                vmin=0, vmax=np.max(Iconv),
                norm=LogNorm(vmin = 0, vmax = 0.15*np.max(Iconv))
                )
    else:
        plt.imshow(Iconv, cmap='viridis', extent=(k0-kmax, k0+kmax, l0-lmax, l0+lmax), 
                origin='lower', vmin= 0, vmax= 0.15*np.max(Iconv)
                )          
    plt.colorbar()
    plt.ylabel("L (r.l.u.)")
    plt.xlabel("K (r.l.u.)")   
    """3D ATOM PLOT"""
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    # ax.title(r"(Modulated) unit cell(s) with Wyckoff 4e z-parameter z0={} and $T=(1/2 1/2 1/2)$".format(z0))
    # ax.scatter(Eu[0], Eu[1], Eu[2], label='Eu, Wyckoff 2a (000)', c='yellow')
    # ax.scatter(Eu_T[0], Eu_T[1], Eu_T[2], label='Eu_T, Wyckoff 2a T(000)', facecolors='none', edgecolors='yellow')
    ax.scatter(Al1[0], Al1[1], Al1[2], label='Al2, 4d (1/2 0 1/4), Al1, 4d (0 1/2 1/4)', c='blue')
    ax.scatter(Al1_T[0], Al1_T[1], Al1_T[2], facecolors='none', edgecolors='blue')
    ax.scatter(Al2[0], Al2[1], Al2[2], c='blue')
    ax.scatter(Al2_T[0], Al2_T[1], Al2_T[2], facecolors='none', edgecolors='blue')
    ax.scatter(Ga1[0], Ga1[1], Ga1[2], label='Ga1, Wyckoff 4e (0 0 z0), (0 0 -z0)', c='green')
    ax.scatter(Ga1_T[0], Ga1_T[1], Ga1_T[2], facecolors='none', edgecolors='green')
    ax.scatter(Ga2[0], Ga2[1], Ga2[2], c='green')
    ax.scatter(Ga2_T[0], Ga2_T[1], Ga2_T[2], facecolors='none', edgecolors='green')
    # ax.set_xlim(-0.5, 1.5)
    # ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    if savefig == True:
        plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/XRD_Simulation/EuGa2Al2_modulation1/Map_N={}{}{}_q={}_H={}_center{}{}{}_updated.jpg".format(Nxy_pos, Nxy_pos, Nz_pos, q_cdw, h, h, k0, l0), dpi=300)
    else:
        pass
    plt.show(block=False)


def main():
    ##########################################################################
    ########################## GENERAL INPUT #################################
    ##########################################################################
    print(__doc__)
    k0, l0, kmax = -5, 10, 3 # boundaries of K and L 
    lmax=kmax
    deltak = 0.01  # k-space point distance
    h = 0 # H value in k space
    a, c, z0 = 4, 11, 0.38  # Unit Cell Parameters in Angstrom
    ######################################
    #  INPUT FOR CDW
    A, q_cdw = 0.05, 0.1 
    Nsc = 1  #  Number of supercells
    kspacefactors = [1, 99, 10, 9]
    ######################################
    #  INPUT FOR DBW
    u_list = [1e-3, 1e-3, 1e-3]  # Isotropic displacements <u^2> in Å^2, ~1/m.
    ######################################
    # OTHER
    noiseamplitude = 1e-4  # Noise amplitude
    ######################################
    # GAUSSIAN KERNEL -- RESOLUTION FUNCTION
    sigma, kernelsize = 0.5, 10  #int(0.1/deltak) 
    
    
    #  PROGRAM
    st = time.time()
    k2d, l2d, k, l, Unitary = kspacecreator(k0, l0, kmax, lmax, deltak)
    print("(2) = {}".format(kmax*deltak*Nsc/q_cdw))  
    structurefactorandplotting(a, c, k0, l0, k2d, k, l,kmax, lmax, l2d, h, 
                               deltak=deltak, Unitary=Unitary, u_list=u_list, 
                               A=A, q_cdw=q_cdw, kernelsize = kernelsize, 
                               Nsc=Nsc, kspacefactors=kspacefactors, z0=z0,  
                               noiseamplitude=noiseamplitude, sigma=sigma,
                               normalization=False, DBW=False, 
                               lognorm=False, savefig=True,
                               properatomicformfactor=True, EuAl4=False,
                               Lexclude=True)
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')


if __name__ == '__main__':
        main()
    


