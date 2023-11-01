#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 15:46:45 2023

@author: stevengebel
"""

"""Bessel-Playground"""

from scipy.special import jv
import matplotlib.pyplot as plt
import numpy as np
from XRD_simulation_functions import kspacecreator

# k = np.linspace(-20, 20, 10000) # in 2pi/a, a=1 Angstrom
# U = 0.1 # in Angstrom
# x = k*U
# jv1 = jv(1, x)
# jv0 = jv(0, x)
# plt.plot(k , jv1)
# plt.plot(k, jv0)
# m = 1
# z = 0.5
# q_cdw = 0.2
# Besselfactor = jv(x, m) * np.exp(-1j * 2 * np.pi * k * m * z * q_cdw)
# plt.plot(x, np.absolute(Besselfactor)**2)
# plt.show()




##############################################################################
# # Attempt 1
# # Define the mesh grid parameters
# k0 = 0
# l0 = 0
# kmax = 2
# lmax = 2
# deltak = 0.01
# # Create the mesh grid for k and l
# k = np.arange(k0 - kmax, k0 + kmax + deltak, deltak)
# l = np.arange(l0 - lmax, l0 + lmax + deltak, deltak)
# k2d, l2d = np.meshgrid(k, l)
# q_cdw = 0.1  # Adjust q_cdw to your desired periodicity

# y = np.abs(np.sin(k2d+l2d))



# # Set columns of y to zero where k-values are not integers
# column_indices = np.arange(k2d.shape[1])  # Create an array of column indices
# non_integer_columns = ~np.isclose(k[column_indices] % 1, 0)  # Check if k-values are not integers
# y[:, non_integer_columns] = 0  # Set columns to zero



# # Set rows of y to zero where l-values are not multiples of q_cdw
# non_multiple_rows = ~np.isclose(l % q_cdw, 0)  # Check if l-values are not multiples of q_cdw
# y[non_multiple_rows, :] = 0  # Set rows to zero


# print(y)
# plt.scatter(k2d, l2d, s=y)
# # plt.imshow(y, cmap='viridis', extent=(k0-kmax, k0+kmax, l0-lmax, l0+lmax))
# plt.show()
##############################################################################

########################################################################
#  Benchmarked table for extracting unwanted K,L points
#  q_cdw     0.1   0.1    0.2   0.2    0.5   0.5   0.125  0.125
#  ∆k        0.1   0.01   0.1   0.01   0.1   0.01
#  Kfactor1  1     1      1
#  Kfactor2  9     99     9
#  Lfactor1  1     10     1
#  Lfactor2  9     9      2
########################################################################
Kfactor1, Kfactor2, Lfactor1, Lfactor2 = 1, 99, 10, 9
k0, l0, kmax, lmax, deltak = 0, 0, 5, 5, 0.01
k2d, l2d, k, l, Unitary = kspacecreator(k0, l0, kmax, lmax, deltak)

I = np.exp(-np.abs(k2d**2 + l2d**2)*0.0001)

# #  Excluding unallowed K-points 
k_intlist = np.arange(0, len(k2d), 1)  # erstelle indices aller k-Werte
# print("k_integer={}".format(k_intlist))
for i in range(0, (2 * kmax*Kfactor1 + 1)):  
    # print(range(0,2*kmax+1))
    k_intlist = np.delete(k_intlist, i * Kfactor2)  #  n*9, since the list gets one less each time
    # print("k_intlist={}".format(k_intlist))
for i in k_intlist:  # Set unallowed K-values for intensities to 0
    I[:, i] = 0

# #  Exluding unallowed L-points
l_intlist = np.arange(0, len(l2d), 1)  # erstelle indices aller l-Werte
for i in range(0, 2 * kmax * Lfactor1 + 1):
    l_intlist = np.delete(l_intlist, i * Lfactor2)  # Lösche jeden zehnten index
    print("l_intlist={}".format(l_intlist))
for i in l_intlist:  # Set unallowed L-values for intensities to 0
    I[i, :] = 0

plt.imshow(I, extent=(k0-kmax, k0+kmax, l0-lmax, l0+lmax))
plt.show()



import numba as nb

@nb.jit(nopython=True)
def calculate_F(f, DBW, h, Unitary, coords, k2d, l2d, result):
    for i in range(len(coords[0])):
        result += f * DBW * np.exp(-2 * np.pi * 1j * (h * Unitary * coords[0, i] + k2d * coords[1, i] + l2d * coords[2, i]))

def structurefactorandplotting_optimized(a, c, k0, l0, k2d, k, l, kmax, lmax, l2d, h, deltak, Unitary, u_list, kernelsize, Nz, kspacefactors, z0, noiseamplitude, sigma, lognorm=False, normalization=False, DBW=False, savefig=False, Lexclude=False, properatomicformfactor=False, EuAl4=False):
    # ...

    F_Eu = np.empty((len(Eu[0]), len(k2d), len(l2d)), dtype=complex)
    F_Al1 = np.empty((len(Al1[0]), len(k2d), len(l2d)), dtype=complex)
    # Repeat for other atoms

    # Calculate DBW once
    if DBW:
        DBW_list = debyewallerfactor(k2d, l2d, Unitary, h, a, c, u_list)
    else:
        DBW_list = np.ones(3)

    # Calculate F for each atom
    calculate_F(f_Eu, DBW_list[0], h, Unitary, Eu, k2d, l2d, F_Eu)
    calculate_F(f_Al, DBW_list[1], h, Unitary, Al1, k2d, l2d, F_Al1)
    # Repeat for other atoms

    # Sum F for all atoms
    F_init = np.sum([F_Eu, F_Al1, ...], axis=0)

    # ...

# Call the optimized function
structurefactorandplotting_optimized(a, c, k0, l0, k2d, k, l, kmax, lmax, l2d, h, deltak=deltak, Unitary=Unitary, u_list=u_list, kernelsize=kernelsize, Nz=Nz, kspacefactors=kspacefactors, z0=z0, noiseamplitude=noiseamplitude, sigma=sigma, normalization=False, DBW=False, lognorm=False, savefig=True, Lexclude=True, properatomicformfactor=True, EuAl4=True)





















