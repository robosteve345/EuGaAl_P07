#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 17:52:02 2023

@author: stevengebel
"""
"""XRD EuGa2Al2 simulation"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.pyplot import figure
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rc('text', usetex=True)

# Include tool: HKL_0 or HK_0L maps?, if HKL0: else: K-space creator
"""K-space creator"""
k0, l0, kmax, lmax = 0, 0, 15, 15  #  boundaries of K and L for the intensity maps
deltak = 0.01  #  or 0.1, k-space point distance
h = 1  #  H value in k space
z_Ga = 0.35
k = np.arange(k0-kmax, k0+kmax + deltak, deltak)
l = np.arange(l0-lmax, l0+lmax + deltak, deltak)
k2d, l2d = np.meshgrid(k, l)
Unitary = np.ones((len(k2d), len(l2d)))  # Unitary matrix
"""EuGa2Al2 atomic positions"""
Eu1, Eu1T, Al1, Al1T, Al2, Al2T, Ga1, Ga1T, Ga2, Ga2T = np.array([[0],[0],[0]]), np.array([[1/2],[1/2],[1/2]]), \
                                                        np.array([[0],[1/2],[1/4]]), np.array([[1/2],[1],[3/4]]),\
                                                        np.array([[1/2],[0],[1/4]]), np.array([[1],[1/2],[3/4]]), \
                                                        np.array([[0],[0],[z_Ga]]), np.array([[1/2],[1/2],[1/2+z_Ga]]), \
                                                        np.array([[0],[0],[-z_Ga]]), np.array([[1/2],[1/2],[1/2-z_Ga]])
"""Scattering amplitudes F"""
# Form factors
f_Eu, f_Al, f_Ga = 2, 1, 1
# Scattering Amplitudes
F_Eu = f_Eu * (np.exp(-2*np.pi*1j*(h*Unitary*Eu1[0] + k2d*Eu1[1] + l2d*Eu1[2])) + np.exp(-2*np.pi*1j*(h*Unitary*Eu1T[0] + k2d*Eu1T[1] + l2d*Eu1T[2]))
               )
F_Al = f_Al * (np.exp(-2*np.pi*1j*(h*Unitary*Al1[0] + k2d*Al1[1] + l2d*Al1[2])) + np.exp(-2*np.pi*1j*(h*Unitary*Al2[0] + k2d*Al2[1] + l2d*Al2[2])) + np.exp(-2*np.pi*1j*(h*Unitary*Al1T[0] + k2d*Al1T[1] + l2d*Al1T[2])) + np.exp(-2*np.pi*1j*(h*Unitary*Al2T[0] + k2d*Al2T[1] + l2d*Al2T[2]))
               )
F_Ga = f_Ga * (np.exp(-2*np.pi*1j*(h*Unitary*Ga1[0] + k2d*Ga1[1] + l2d*Ga1[2])) + np.exp(-2*np.pi*1j*(h*Unitary*Ga2[0] + k2d*Ga2[1] + l2d*Ga2[2])) + np.exp(-2*np.pi*1j*(h*Unitary*Ga1T[0] + k2d*Ga1T[1] + l2d*Ga1T[2])) + np.exp(-2*np.pi*1j*(h*Unitary*Ga2T[0] + k2d*Ga2T[1] + l2d*Ga2T[2]))
               )
F = F_Eu + F_Al + F_Ga # + 0.1*np.random.rand(len(k2d), len(k2d)) # + F_Ga + F_Al
"""Intensity I"""
I = np.absolute(F)**2  # I \propto F(Q)^2, F complex
##############################################################################
# Excluding unallowed K-points (ONLY FOR deltak=/1)
k_intlist = np.arange(0,len(k2d), 1) # erstelle indices aller k-Werte
for i in range(0, 2*kmax + 1): # for kmax=lmax=2 and ∆k=0.1, up to 2*kmax=lmax + 1
    k_intlist = np.delete(k_intlist, i*9) # 9 for ∆k=0.1 and kmax=lmax=2, 99 for ∆k=0.01 and kmax=lmax=2
for i in k_intlist: # Set unallowed K-values for intensities to 0
    I[:, i] = 0
# Exluding unallowed L-points (ONLY FOR deltak=0.01 or deltak=0.001)
l_intlist = np.arange(0,len(l2d), 1) # erstelle indices aller l-Werte
if deltak == 0.1:
    for i in range(0, 2*kmax + 1):
        l_intlist = np.delete(l_intlist, i*9)  # Lösche jeden zehnten index
    for i in l_intlist: # Set unallowed L-values for intensities to 0
        I[i,:] = 0
else:
    for i in range(0, 2*kmax*10 + 1):
        l_intlist = np.delete(l_intlist, i*9)  # Lösche jeden zehnten index
    for i in l_intlist: # Set unallowed L-values for intensities to 0
        I[i,:] = 0
##############################################################################
# Noise
noisefactor = 0.0 # Amplitude of the noise for the intensity
I = I + noisefactor*np.random.rand(len(k2d), len(k2d)) # Add random noise with maximum 1
# Plotting
figure(figsize=(9,7), dpi=100)
plt.suptitle("EuGa2Al2, H={}".format(h)#, F($\mathbf{Q}$)=$f(1 + e^{-i\pi(h+k+l)})$ \n $I=f^2(2+2\cos(\pi(h+k+l))), f=1$
             )
plt.subplot(2, 2, 1)
plt.title('Countourplot')
plt.contourf(l2d, k2d, I, cmap='viridis', extent=(k0-kmax, k0+kmax, l0-lmax, l0+lmax))
plt.colorbar()
plt.xlabel("K(rlu)")
plt.ylabel("L(rlu)")

plt.subplot(2, 2, 2)
plt.title("Gaussian interpolation")
plt.imshow(I, cmap='viridis',
            interpolation='gaussian',
            extent=(k0-kmax, k0+kmax, l0-lmax, l0+lmax),
            origin='lower',
            #norm=LogNorm(vmin=0.1, vmax=np.max(I))
            )
plt.colorbar()
plt.xlabel("K(rlu)")
plt.ylabel("L(rlu)")

plt.subplot(2, 2, 3)
plt.scatter(k2d, l2d, c=I, s=I, cmap='viridis', label=r'$I \propto F(\mathbf{Q})^2$')
plt.colorbar()
plt.legend(loc='upper right')
plt.ylabel("L(rlu)")
plt.xlabel("K(rlu)")
plt.tight_layout()

plt.subplot(2, 2, 4)
# plt.title(r'$I(L)=f^2(4 + 2*(\cos(\pi(h+l)) + \cos(\pi(h-l)) + \cos(\pi(k+h)) + \cos(\pi(k-h))+\cos(\pi(k+l)) +\cos(\pi(k-l))), f=1$')
plt.plot(l2d[:,0], I[:,0], ls='--', marker='.', label='K={}'.format(np.round(k[0], 2)))
plt.plot(l2d[:,0], I[:,-1], ls='--', marker='.', label='K={}'.format(np.round(k[-1], 2)))
plt.plot(l2d[:,0], I[:,-10], ls='--', marker='.', label='K={}'.format(np.round(k[-10], 2)))
plt.plot(l2d[:,0], I[:,10], ls='--', marker='.', label='K={}'.format(np.round(k[10], 2)))
# plt.plot(l2d[:,0], I[:,int(2/deltak)], ls='--', marker='.', label='K={}'.format(int(k0-kmax+2)))
# plt.plot(l2d[:,0], I[:,int(3/deltak)], ls='--', marker='.', label='K={}'.format(int(k0-kmax+3)))
plt.legend(loc='lower center')
plt.ylabel(r"Intensity $I\propto F(\mathbf{Q})^2$")
plt.xlabel("L(rlu)")
plt.savefig("EuGa2Al2_Example2_H={}.jpg".format(h), dpi=300)
plt.subplots_adjust(wspace=0.3)
plt.show()

