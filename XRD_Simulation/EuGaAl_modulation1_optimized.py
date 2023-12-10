#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Created on Tue Oct 24 12:30:02 2023

#@author: stevengebel

"""|F(Q)| HKL-map simulation for a supercell with modulation vector q
for Eu(Ga,Al)4 in I4/mmm symmetry"""

"""For single unit cell set q_cdw = 1.0
"""

import numpy as np
import time
from XRD_simulation_functions_main import main_HK0L, main_H0KL, main_HKL0
from XRD_simulation_functions import kspacecreator

def main():
# =============================================================================
#     ########################## GENERAL INPUT ################################
# =============================================================================
    print(__doc__)
    deltak = 0.02  # k-space point distance
    a, c, z0 = 4.3861, 11.165, 0.3849	# 4.3949, 11.1607, 0.3849  # Unit Cell Parameters in Angstrom
    ######################################
    #  INPUT FOR CDW
    q_cdw = 0.1
    Nsc = 1  #  Number of supercells
# =============================================================================
#     Model: Immm(00sigma)s00 from Ramakrishnan et al.
# =============================================================================
    #  From Table S5, Ramakrishnan et al
    #  Modulation Amplitudes for CDW (in Angstrom)         Ax/Bx  Ay/By  Az/Bz
    A = [1/a*np.array([0.1230, 0, 0]),                 #4d           ...
          1/a*np.array([0.1292, 0, 0]),                #4e           ...
          1/c*np.array([0.1282, 0, 0])]                #2a           ...
    B = [1/a*np.array([-0.0571, 0, 0]), 
          1/a*np.array([0.0562, 0, 0]), 
          1/c*np.array([0, 0, 0])]
    # A = [1/a*np.array([0, 0.1230, 0]),                 #4d           ...
    #       1/a*np.array([0, 0.1292, 0]),                #4e           ...
    #       1/c*np.array([0, 0.1282, 0])]                #2a           ...
    # B = [1/a*np.array([0, -0.0571, 0]), 
    #       1/a*np.array([0, 0.0562, 0]), 
    #       1/c*np.array([0, 0, 0])]
    # A = [1/a*np.array([0, 0, 0.1230]),                 #4d           ...
    #       1/a*np.array([0, 0, 0.1292]),                #4e           ...
    #       1/c*np.array([0, 0, 0.1282])]                #2a           ...
    # B = [1/a*np.array([0, 0, -0.0571]), 
    #       1/a*np.array([0, 0, 0.0562]), 
    #       1/c*np.array([0, 0, 0])]
    ######################################
    #  INPUT FOR DBW
    u_list = [0.057, 0.60, 0.53]  # Isotropic displacements <u^2> in Å^2.
    ######################################
    # OTHER
    noiseamplitude = 1  # Noise amplitude
# =============================================================================
#     # GAUSSIAN KERNEL -- DEPENDENT ON RESOLUTION FUNCTION
#     # ID28: qpixelx= |
#     # P07:  qpixelx= |
# =============================================================================
    sigma, kernelsize = 0.1, 10
    
    
# =============================================================================
#     #  PROGRAM: 
# =============================================================================    
    st = time.time()
    # # KL MAP
    k0, l0, kmax, lmax = 0, 0, 10, 10  # boundaries of K and L 
    H = 1 # H value in k space
    k2d, l2d, k, l, Unitary = kspacecreator(k0, l0, kmax, lmax, deltak) 
    main_H0KL(a, c, k0, l0, k2d, k, l, lmax, lmax, l2d, H, deltak, 
                        Unitary, u_list, kernelsize, q_cdw, Nsc, z0, A, B, 
                        noiseamplitude, sigma,
                        normalization=False, 
                        DBW=True, 
                        lognorm=True, 
                        savefig=True,
                        fatom=True, 
                        EuAl4=False, 
                        EuGa4=True
                        )
    main_H0KL(a, c, k0, l0, k2d, k, l, lmax, lmax, l2d, H, deltak, 
                        Unitary, u_list, kernelsize, q_cdw, Nsc, z0, A, B, 
                        noiseamplitude, sigma,
                        normalization=False, 
                        DBW=True, 
                        lognorm=True, 
                        savefig=True,
                        fatom=True, 
                        EuAl4=False, 
                        EuGa4=False
                        )
    main_H0KL(a, c, k0, l0, k2d, k, l, lmax, lmax, l2d, H, deltak, 
                        Unitary, u_list, kernelsize, q_cdw, Nsc, z0, A, B, 
                        noiseamplitude, sigma,
                        normalization=False, 
                        DBW=True, 
                        lognorm=True, 
                        savefig=True,
                        fatom=True, 
                        EuAl4=True, 
                        EuGa4=False
                        )
    # # # # HK MAP
    # h0, k0, hmax, kmax = 0, 0, 5, 5  # boundaries of K and L 
    # L = 1  # L value in k space
    # h2d, k2d, h, k, Unitary = kspacecreator(h0, k0, hmax, kmax, deltak)
    # main_HKL0(a, c, h0, k0, h2d, h, k, hmax, kmax, k2d, L, deltak, 
    #                     Unitary, u_list, kernelsize, q_cdw, Nsc, z0, A, B, 
    #                     noiseamplitude, sigma,
    #                     normalization=False, 
    #                     DBW=True, 
    #                     lognorm=True, 
    #                     savefig=True,
    #                     fatom=True, 
    #                     EuAl4=False, 
    #                     EuGa4=True
    #                     )   
    # # # HL MAP
    # h0, l0, hmax, lmax = 0, 0, 5, 5 # boundaries of K and L 
    # K = 0  # K value in k space
    # h2d, l2d, h, l, Unitary = kspacecreator(h0, l0, hmax, lmax, deltak)
    # main_HK0L(a, c, h0, l0, h2d, h, l, hmax, lmax, l2d, K, deltak, 
    #                     Unitary, u_list, kernelsize, q_cdw, Nsc, z0, A, B, 
    #                     noiseamplitude, sigma,
    #                     normalization=False, 
    #                     DBW=True, 
    #                     lognorm=True, 
    #                     savefig=True,
    #                     fatom=True, 
    #                     EuAl4=False, 
    #                     EuGa4=True
    #                     )
    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')


if __name__ == '__main__':
        main()
    


