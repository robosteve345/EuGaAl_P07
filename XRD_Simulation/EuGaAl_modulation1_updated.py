#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Created on Thu Oct 19 15:41:29 2023

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

"""TO DO
-Implement possibility for HKL_0 & HK_0L maps
-Implement input-interface: kß, lß, deltak, Amplitude, q_cdw, kmax
"""


def structurefactorandplotting(a, c, k0, l0, k2d, k, l, kmax, lmax, l2d, h, 
                               deltak, Unitary, u_list, kernelsize, A, 
                               q_cdw, Nsc, kspacefactors, z0,
                               noiseamplitude, sigma, lognorm=False, 
                               normalization=False, DBW=False, savefig=False,
                               properatomicformfactor=False, EuAl4=False,
                               Lexclude=False):
    print("CENTERED PEAK: [HKL] = [{} {} {}]".format(h, k0, l0))
    print("∆k = {}".format(deltak))
    print("A = {}, q_cdw = {}".format(A, q_cdw))
    print("H = {}, kmax=lmax = {}".format(h, kmax))
    print("# of Supercells = {}".format(Nsc))
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
    """Atomic positions"""
    ##########################################################################
    Nxy_neg, Nxy_pos = 0, 1#int(np.sqrt(Nsc/(q_cdw*0.3)))
    Nz_neg, Nz_pos = 0, int(q_cdw**(-1)) * Nsc
    # print("Nz/Nxy = {}".format(np.round(q_cdw**(-1) * Nsc/Nxy_pos**2, 2)))
    # print("Nxy_neg, Nxy_pos = {}, {}".format(Nxy_neg, Nxy_pos))
    # print("Nz_neg, Nz_pos = {}, {}".format(Nz_neg, Nz_pos))
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
    
    # Basepositions
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
            posAl1[0][i] = posAl1[0][i] + \
                      A * np.sin(q_cdw * 2 * np.pi * j * posAl1[2][i])  + \
                      B * np.cos(q_cdw * 2 * np.pi * j * posAl1[2][i])
            posAl1_T[1][i] = posAl1_T[1][i] + \
                    A * np.sin(q_cdw * 2 * np.pi * j * posAl1_T[2][i]) + \
                        B * np.cos(q_cdw * 2 * np.pi * j * posAl1_T[2][i])
            posAl2[1][i] = posAl2[1][i] + \
                      -A * np.sin(q_cdw * 2 * np.pi * j * posAl2[2][i]) + \
                          -B * np.cos(q_cdw * 2 * np.pi * j * posAl2[2][i])
            posAl2_T[0][i] = posAl2_T[0][i] + \
                      -A * np.sin(q_cdw * 2 * np.pi * j * posAl2_T[2][i]) +\
                          -B * np.cos(q_cdw * 2 * np.pi * j * posAl2_T[2][i])
            # """Model 2: Ga-Al layer distortion"""
            # posGa1[2][i] = posGa1[2][i] + \
            #       A * np.sin(q_cdw * 2 * np.pi * j * posGa1[2][i]) 
            # posGa1_T[2][i] = posGa1_T[2][i] + \
            #         A * np.sin(q_cdw * 2 * np.pi * j * posGa1_T[2][i]) 
            # posGa2[2][i] = posGa2[2][i] + \
            #         -A * np.sin(q_cdw * 2 * np.pi * j * posGa2[2][i]) 
            # posGa2_T[2][i] = posGa2_T[2][i] + \
            #         -A * np.sin(q_cdw * 2 * np.pi * j * posGa2_T[2][i])  

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
    print("# of Unit Cells = {}".format(len(Eu[0])))
    
    #  Compute Debye-Waller-Factor
    if DBW == True:
        DBW_list = debyewallerfactor(k2d, l2d, Unitary, h, a, c, u_list)
    else:
        DBW_list = np.ones(3)
    
    """Scattering amplitudes F"""
    #  Crystal structure factor for each atom at each Wyckoff pos with respect to the tI symmtry +1/2 transl.
    F_Eu, F_Eu_T, F_Al1, F_Al1_T, F_Al2, F_Al2_T, F_Ga1, F_Ga1_T, \
    F_Ga2, F_Ga2_T = [], [], [], [], [], [], [], [], [], []
    # Calculate atomic form factors
    f_Eu, f_Ga, f_Al = atomicformfactorEuGaAl(h, k2d, l2d, Unitary, 
                                                properatomicformfactor, EuAl4)
    for i in range(0, len(Eu[0])):
        F_Eu.append(f_Eu * DBW_list[0] * np.exp(-2 * np.pi * 1j * 
                            (h * Unitary * Eu[0, i] + k2d * Eu[1, i] + 
                                                  l2d * Eu[2, i]))
        )
        F_Eu_T.append(f_Eu * DBW_list[0] * np.exp(-2 * np.pi * 1j * 
                            (h * Unitary * Eu_T[0, i] + k2d * Eu_T[1, i] + 
                                                  l2d * Eu_T[2, i]))
        )
        F_Al1.append(f_Al * DBW_list[1] * np.exp(-2 * np.pi * 1j * 
                            (h * Unitary * Al1[0, i] + k2d * Al1[1, i] + 
                                                  l2d * Al1[2, i]))
        )
        F_Al1_T.append(f_Al * DBW_list[1] *np.exp(-2 * np.pi * 1j * 
                            (h * Unitary * Al1_T[0, i] + k2d * Al1_T[1, i] + 
                                                  l2d * Al1_T[2, i]))
        )
        F_Al2.append(f_Al * DBW_list[1] *np.exp(-2 * np.pi * 1j * 
                            (h * Unitary * Al2[0, i] + k2d * Al2[1, i] + 
                                                  l2d * Al2[2, i]))
        )
        F_Al2_T.append(f_Al * DBW_list[1] *np.exp(-2 * np.pi * 1j * 
                            (h * Unitary * Al2_T[0, i] + k2d * Al2_T[1, i] + 
                                                  l2d * Al2_T[2, i]))
        )
        F_Ga1.append(f_Ga * DBW_list[2] * np.exp(-2 * np.pi * 1j * 
                            (h * Unitary * Ga1[0, i] + k2d * Ga1[1, i] + 
                                                  l2d * Ga1[2, i]))
        )
        F_Ga1_T.append(f_Ga * DBW_list[2] * np.exp(-2 * np.pi * 1j * 
                            (h * Unitary * Ga1_T[0, i] + k2d * Ga1_T[1, i] + 
                                                  l2d * Ga1_T[2, i]))
        )
        F_Ga2.append(f_Ga * DBW_list[2] * np.exp(-2 * np.pi * 1j * 
                            (h * Unitary * Ga2[0, i] + k2d * Ga2[1, i] + 
                                                  l2d * Ga2[2, i]))
        )
        F_Ga2_T.append(f_Ga * DBW_list[2] * np.exp(-2 * np.pi * 1j * 
                            (h * Unitary * Ga2_T[0, i] + 
                                                  k2d * Ga2_T[1, i] + 
                                                  l2d * Ga2_T[2, i]))
        )
    F_init = np.zeros((len(k2d), len(k2d)), dtype=complex)  # quadratic 0-matrix with dimensions of k-space
    F_list = [F_Eu, F_Eu_T, F_Al1, F_Al1_T, F_Al2, F_Al2_T, F_Ga1, F_Ga1_T, F_Ga2, F_Ga2_T]
    pre_F_init = [np.zeros((len(k2d), len(k2d)))]
    #  Zusammenfügen der Formfaktoren für die Atomsorten
    for i in F_list:  # put together the lists in a ndarray for each atom with each N poss, to get rid of 3rd dimension (better workaround probably possible...)
        pre_F_init = np.add(pre_F_init, i)
    for i in range(len(pre_F_init)):
        F_init = F_init + pre_F_init[i]
        
    #  Compute Intensity 
    I = np.abs(np.round(F_init, 3)) ** 2  # I \propto F(Q)^2, F complex
    
    # Exclude k-space points
    I = excludekspacepoints(kspacefactors, k2d, l2d, deltak, I, noiseamplitude, kmax, lmax, Lexclude)
    
    # Normalize Intensity
    if normalization == True:
        I = I/np.max(I)
    else:
        I = I
    
    # # PLOTTING
    fig = plt.figure(figsize=(15, 3))
    plt.suptitle(r"EuGa2Al2, Centered around [{}{}{}], q={}rlu, {} Supercell(s)".format(h, k0, l0, q_cdw, Nsc))
    """MODULATED ATOMIC POSITIONS"""
    plt.subplot(1, 2, 1)
    plt.scatter(1 / 2 * np.ones(len(Eu[0])) + init_Al1_z, np.ones(len(Eu[0])),
                label=r'Al_x=$(\frac{1}{2},\frac{1}{2},\frac{1}{2})$ equilibrium', facecolors='none',
                edgecolors='orange', s=100)
    plt.xlabel('z')
    plt.ylabel('')
    plt.scatter(Al1[0, :], np.ones(len(init_Al1_z) * len(init_Al1_z)**2),
                #label='Al=$(0.5,0.5,0.5 + {} \sin({} \cdot 2\pi L))$ distorted'.format(A, q_cdw),
                marker='o')
    plt.legend()
    """LINECUTS"""
    # plt.subplot(1, 1, 2)
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
    plt.suptitle("EuGa2Al2, Centered around [{}{}{}], q={}rlu, {} Supercell(s)".format(h, k0, l0, q_cdw, Nsc))
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
    # plt.xlim(-0.5*kmax, 0.5*kmax)
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
    k0, l0, kmax = 0, 0, 10 # boundaries of K and L 
    lmax=kmax
    deltak = 0.01  # k-space point distance
    h = 6 # H value in k space
    a, c, z0 = 4, 11, 0.38  # Unit Cell Parameters in Angstrom
    ######################################
    #  INPUT FOR CDW
    A, q_cdw = 0.03, 0.1 
    # A, q_cdw = 0.0, 1.0  # Default
    Nsc = 1  #  Number of supercells
    kspacefactors = [1, 99, 10, 9]
    ######################################
    #  INPUT FOR DBW
    u_list = [1e-3, 1e-3, 1e-3]  # Isotropic displacements <u^2> in Å^2, ~1/m. For more detail take bonding angles, distances into account.
    ######################################
    # OTHER
    noiseamplitude = 1e-4  # Noise amplitude
    ######################################
    # GAUSSIAN KERNEL -- RESOLUTION FUNCTION
    sigma, kernelsize = 0.5, 6  #int(0.1/deltak) 
    
    
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
                               properatomicformfactor=False, EuAl4=False,
                               Lexclude=True)
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')


if __name__ == '__main__':
        main()
    

