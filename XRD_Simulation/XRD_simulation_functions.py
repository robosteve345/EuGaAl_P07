#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 10:11:21 2023

@author: stevengebel
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import ndimage
from matplotlib.colors import LogNorm
mpl.rc('text', usetex=True)
mpl.rcParams.update(mpl.rcParamsDefault) 
mpl.rcParams['font.family'] = "sans-serif"

"""TO DO
- Make single function for DBW
- Make plotting function

"""
"""XRD Simulation functions package"""


def Fstructure_calc_H0KL(f, DBW, H, Unitary, coords, k2d, l2d, result):
    """"""
    for i in range(coords.shape[1]):
        result += f * DBW * np.exp(-2 * np.pi * 1j * (H * Unitary * coords[0, i] + k2d * coords[1, i] + l2d * coords[2, i]))


def Fstructure_calc_HK0L(f, DBW, K, Unitary, coords, h2d, l2d, result):
    for i in range(coords.shape[1]):
        result += f * DBW * np.exp(-2 * np.pi * 1j * (h2d * coords[0, i] + K * Unitary * coords[1, i] + l2d * coords[2, i]))


def Fstructure_calc_HKL0(f, DBW, L, Unitary, coords, h2d, k2d, result):
    for i in range(coords.shape[1]):
        result += f * DBW * np.exp(-2 * np.pi * 1j * (h2d * coords[0, i] + k2d * coords[1, i] + L * Unitary * coords[2, i]))

    
def excludekspacepoints(x2d, y2d, deltak, I, q_cdw, xmax, ymax):
    """For H0KL and HK0L maps."""
    # =============================================================================
    # #  Excluding unallowed K-points 
    # =============================================================================
    x_intlist = np.arange(0, len(x2d), 1)  # erstelle indices aller k-Werte
    x_indices = np.arange(0, int(2*xmax + 1), 1) * int(1/deltak)
    x_intlist_copy = x_intlist.copy()
    for i in x_indices[::-1]:  # Iterate in reverse order
        if i < len(x_intlist_copy):
            x_intlist_copy = np.delete(x_intlist_copy, i)
    # print("Modified k_intlist={}".format(k_intlist_copy))
    # =============================================================================
    # # Excluding unallowed L-points 
    # =============================================================================
    y_intlist = np.arange(0, len(y2d), 1)  # erstelle indices aller l-Werte
    y_indices = np.arange(0, int(2*ymax/q_cdw + 1), 1) * int(q_cdw/deltak)
    y_intlist_copy = y_intlist.copy()
    for i in y_indices[::-1]:  # Iterate in reverse order
        if i < len(y_intlist_copy):
            y_intlist_copy = np.delete(y_intlist_copy, i)
    # l_intlist_copy now contains the modified list without the specified indices
    # l_intlist still contains the original indices
    # print("Modified l_intlist={}".format(l_intlist_copy))
    # Delete unwanted Intensities
    for i in x_intlist_copy:  
        I[:, i] = 0
    for i in y_intlist_copy: 
        I[i, :] = 0     
    return I, y_indices, x_indices


def excludekspacepoints_HKmap(x2d, y2d, deltak, I, xmax, ymax):
    # =============================================================================
    # #  Excluding unallowed H,K-points 
    # =============================================================================
    x_intlist = np.arange(0, len(x2d), 1)  # erstelle indices aller k-Werte
    x_indices = np.arange(0, int(2*xmax + 1), 1) * int(1/deltak)
    x_intlist_copy = x_intlist.copy()
    for i in x_indices[::-1]:  # Iterate in reverse order
        if i < len(x_intlist_copy):
            x_intlist_copy = np.delete(x_intlist_copy, i)
    # print("Modified k_intlist={}".format(k_intlist_copy))
    # Delete unwanted Intensities
    for i in x_intlist_copy:  
        I[:, i] = 0
    for i in x_intlist_copy: 
        I[i, :] = 0     
    return I, x_indices


def fatom_calc_H0KL(H, k2d, l2d, Unitary, fatom=False, EuAl4=False):
    if EuAl4 == True:
        ######################################################################
        # EMPIRIC FACTORS
        a_eu, a_ga, a_al  = [24.0063, 19.9504, 11.8034, 3.87243], \
            [15.2354, 6.7006, 4.3591, 2.9623], [4.17448, 3.3876, 1.20296, 0.528137]
        b_eu, b_ga, b_al = [2.27783, 0.17353, 11.6096, 26.5156], \
                [3.0669, 0.2412, 10.7805, 61.4135], [1.93816, 4.14553, 0.228753, 8.28524]
        c_eu, c_ga, c_al = 1.36389, 1.7189, 0.706786
        ######################################################################
        
        f_eu1 = a_eu[0] * np.exp(- b_eu[0] * (((H * Unitary) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))  
        f_eu2 = a_eu[1] * np.exp(- b_eu[1] * (((H * Unitary) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_eu3 = a_eu[2] * np.exp(- b_eu[2] * (((H * Unitary) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_eu4 = a_eu[3] * np.exp(- b_eu[3] * (((H * Unitary) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_Eu = f_eu1+f_eu2+f_eu3+f_eu4 + c_eu * Unitary
        f_al1 = a_al[0] * np.exp(- b_al[0] * (((H * Unitary) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_al2 = a_al[1] * np.exp(- b_al[1] * (((H * Unitary) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_al3 = a_al[2] * np.exp(- b_al[2] * (((H * Unitary) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_al4 = a_al[3] * np.exp(- b_al[3] * (((H * Unitary) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_Al = f_al1 + f_al2 + f_al3 + f_al4 + c_al * Unitary
        f_Ga = f_Al
    else:
        if fatom == True:
            a_eu, a_ga, a_al  = [24.0063, 19.9504, 11.8034, 3.87243], \
            [15.2354, 6.7006, 4.3591, 2.9623], [4.17448, 3.3876, 1.20296, 0.528137]
            b_eu, b_ga, b_al = [2.27783, 0.17353, 11.6096, 26.5156], \
            [3.0669, 0.2412, 10.7805, 61.4135], [1.93816, 4.14553, 0.228753, 8.28524]
            c_eu, c_ga, c_al = 1.36389, 1.7189, 0.706786
        else:
            # Make f_eu,f_ga,f_al trivial (1)
            a_eu, a_ga, a_al  = np.ones(4), np.ones(4), np.ones(4)
            b_eu, b_ga, b_al =  np.zeros(4), np.zeros(4), np.zeros(4)
            c_eu, c_ga, c_al = 0, 0, 0
        f_eu1 = a_eu[0] * np.exp(- b_eu[0] * (((H * Unitary) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_eu2 = a_eu[1] * np.exp(- b_eu[1] * (((H * Unitary) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_eu3 = a_eu[2] * np.exp(- b_eu[2] * (((H * Unitary) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_eu4 = a_eu[3] * np.exp(- b_eu[3] * (((H * Unitary) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_Eu = f_eu1 + f_eu2 + f_eu3 + f_eu4 + c_eu * Unitary
        f_ga1 = a_ga[0] * np.exp(- b_ga[0] * (((H * Unitary) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_ga2 = a_ga[1] * np.exp(- b_ga[1] * (((H * Unitary) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_ga3 = a_ga[2] * np.exp(- b_ga[2] * (((H * Unitary) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_ga4 = a_ga[3] * np.exp(- b_ga[3] * (((H * Unitary) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_Ga = f_ga1 + f_ga2 + f_ga3 + f_ga4 + c_ga * Unitary
        f_al1 = a_al[0] * np.exp(- b_al[0] * (((H * Unitary) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_al2 = a_al[1] * np.exp(- b_al[1] * (((H * Unitary) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_al3 = a_al[2] * np.exp(- b_al[2] * (((H * Unitary) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_al4 = a_al[3] * np.exp(- b_al[3] * (((H * Unitary) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_Al = f_al1 + f_al2 + f_al3 + f_al4 + c_al * Unitary      
    return f_Eu, f_Ga, f_Al


def fatom_calc_HK0L(K, h2d, l2d, Unitary, fatom=False, EuAl4=False):
    """Atomic form factors according to de Graed, structure of materials, 
    chapter 12: Eu2+. Ga1+, Al3+"""
    if EuAl4 == True:
        ######################################################################
        # EMPIRIC FACTORS
        a_eu, a_ga, a_al  = [24.0063, 19.9504, 11.8034, 3.87243], \
            [15.2354, 6.7006, 4.3591, 2.9623], [4.17448, 3.3876, 1.20296, 0.528137]
        b_eu, b_ga, b_al = [2.27783, 0.17353, 11.6096, 26.5156], \
                [3.0669, 0.2412, 10.7805, 61.4135], [1.93816, 4.14553, 0.228753, 8.28524]
        c_eu, c_ga, c_al = 1.36389, 1.7189, 0.706786
        ######################################################################
        
        f_eu1 = a_eu[0] * np.exp(- b_eu[0] * (((K* Unitary) ** 2 + h2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))  
        f_eu2 = a_eu[1] * np.exp(- b_eu[1] * (((K* Unitary) ** 2 + h2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_eu3 = a_eu[2] * np.exp(- b_eu[2] * (((K* Unitary) ** 2 + h2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_eu4 = a_eu[3] * np.exp(- b_eu[3] * (((K* Unitary) ** 2 + h2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_Eu = f_eu1+f_eu2+f_eu3+f_eu4 + c_eu * Unitary
        f_al1 = a_al[0] * np.exp(- b_al[0] * (((K* Unitary) ** 2 + h2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_al2 = a_al[1] * np.exp(- b_al[1] * (((K* Unitary) ** 2 + h2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_al3 = a_al[2] * np.exp(- b_al[2] * (((K* Unitary) ** 2 + h2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_al4 = a_al[3] * np.exp(- b_al[3] * (((K* Unitary) ** 2 + h2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_Al = f_al1 + f_al2 + f_al3 + f_al4 + c_al * Unitary
        f_Ga = f_Al
    else:
        if fatom == True:
            a_eu, a_ga, a_al  = [24.0063, 19.9504, 11.8034, 3.87243], \
            [15.2354, 6.7006, 4.3591, 2.9623], [4.17448, 3.3876, 1.20296, 0.528137]
            b_eu, b_ga, b_al = [2.27783, 0.17353, 11.6096, 26.5156], \
            [3.0669, 0.2412, 10.7805, 61.4135], [1.93816, 4.14553, 0.228753, 8.28524]
            c_eu, c_ga, c_al = 1.36389, 1.7189, 0.706786
        else:
            # Make f_eu,f_ga,f_al trivial (1)
            a_eu, a_ga, a_al  = np.ones(4), np.ones(4), np.ones(4)
            b_eu, b_ga, b_al =  np.zeros(4), np.zeros(4), np.zeros(4)
            c_eu, c_ga, c_al = 0, 0, 0
        f_eu1 = a_eu[0] * np.exp(- b_eu[0] * (((K* Unitary) ** 2 + h2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_eu2 = a_eu[1] * np.exp(- b_eu[1] * (((K* Unitary) ** 2 + h2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_eu3 = a_eu[2] * np.exp(- b_eu[2] * (((K* Unitary) ** 2 + h2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_eu4 = a_eu[3] * np.exp(- b_eu[3] * (((K* Unitary) ** 2 + h2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_Eu = f_eu1 + f_eu2 + f_eu3 + f_eu4 + c_eu * Unitary
        f_ga1 = a_ga[0] * np.exp(- b_ga[0] * (((K* Unitary) ** 2 + h2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_ga2 = a_ga[1] * np.exp(- b_ga[1] * (((K* Unitary) ** 2 + h2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_ga3 = a_ga[2] * np.exp(- b_ga[2] * (((K* Unitary) ** 2 + h2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_ga4 = a_ga[3] * np.exp(- b_ga[3] * (((K* Unitary) ** 2 + h2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_Ga = f_ga1 + f_ga2 + f_ga3 + f_ga4 + c_ga * Unitary
        f_al1 = a_al[0] * np.exp(- b_al[0] * (((K* Unitary) ** 2 + h2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_al2 = a_al[1] * np.exp(- b_al[1] * (((K* Unitary) ** 2 + h2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_al3 = a_al[2] * np.exp(- b_al[2] * (((K* Unitary) ** 2 + h2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_al4 = a_al[3] * np.exp(- b_al[3] * (((K* Unitary) ** 2 + h2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
        f_Al = f_al1 + f_al2 + f_al3 + f_al4 + c_al * Unitary       
    return f_Eu, f_Ga, f_Al


def fatom_calc_HKL0(L, h2d, k2d, Unitary, fatom=False, EuAl4=False):
    if EuAl4 == True:
        ######################################################################
        # EMPIRIC FACTORS
        a_eu, a_ga, a_al  = [24.0063, 19.9504, 11.8034, 3.87243], \
            [15.2354, 6.7006, 4.3591, 2.9623], [4.17448, 3.3876, 1.20296, 0.528137]
        b_eu, b_ga, b_al = [2.27783, 0.17353, 11.6096, 26.5156], \
                [3.0669, 0.2412, 10.7805, 61.4135], [1.93816, 4.14553, 0.228753, 8.28524]
        c_eu, c_ga, c_al = 1.36389, 1.7189, 0.706786
        ######################################################################
        
        f_eu1 = a_eu[0] * np.exp(- b_eu[0] * (((L* Unitary) ** 2 + h2d ** 2 + k2d ** 2) / (4 * np.pi) ** 2))  
        f_eu2 = a_eu[1] * np.exp(- b_eu[1] * (((L* Unitary) ** 2 + h2d ** 2 + k2d ** 2) / (4 * np.pi) ** 2))
        f_eu3 = a_eu[2] * np.exp(- b_eu[2] * (((L* Unitary) ** 2 + h2d ** 2 + k2d ** 2) / (4 * np.pi) ** 2))
        f_eu4 = a_eu[3] * np.exp(- b_eu[3] * (((L* Unitary) ** 2 + h2d ** 2 + k2d ** 2) / (4 * np.pi) ** 2))
        f_Eu = f_eu1+f_eu2+f_eu3+f_eu4 + c_eu * Unitary
        f_al1 = a_al[0] * np.exp(- b_al[0] * (((L* Unitary) ** 2 + h2d ** 2 + k2d ** 2) / (4 * np.pi) ** 2))
        f_al2 = a_al[1] * np.exp(- b_al[1] * (((L* Unitary) ** 2 + h2d ** 2 + k2d ** 2) / (4 * np.pi) ** 2))
        f_al3 = a_al[2] * np.exp(- b_al[2] * (((L* Unitary) ** 2 + h2d ** 2 + k2d ** 2) / (4 * np.pi) ** 2))
        f_al4 = a_al[3] * np.exp(- b_al[3] * (((L* Unitary) ** 2 + h2d ** 2 + k2d ** 2) / (4 * np.pi) ** 2))
        f_Al = f_al1 + f_al2 + f_al3 + f_al4 + c_al * Unitary
        f_Ga = f_Al
    else:
        if fatom == True:
            a_eu, a_ga, a_al  = [24.0063, 19.9504, 11.8034, 3.87243], \
            [15.2354, 6.7006, 4.3591, 2.9623], [4.17448, 3.3876, 1.20296, 0.528137]
            b_eu, b_ga, b_al = [2.27783, 0.17353, 11.6096, 26.5156], \
            [3.0669, 0.2412, 10.7805, 61.4135], [1.93816, 4.14553, 0.228753, 8.28524]
            c_eu, c_ga, c_al = 1.36389, 1.7189, 0.706786
        else:
            # Make f_eu,f_ga,f_al trivial (1)
            a_eu, a_ga, a_al  = np.ones(4), np.ones(4), np.ones(4)
            b_eu, b_ga, b_al =  np.zeros(4), np.zeros(4), np.zeros(4)
            c_eu, c_ga, c_al = 0, 0, 0
        f_eu1 = a_eu[0] * np.exp(- b_eu[0] * (((L* Unitary) ** 2 + h2d ** 2 + k2d ** 2) / (4 * np.pi) ** 2))
        f_eu2 = a_eu[1] * np.exp(- b_eu[1] * (((L* Unitary) ** 2 + h2d ** 2 + k2d ** 2) / (4 * np.pi) ** 2))
        f_eu3 = a_eu[2] * np.exp(- b_eu[2] * (((L* Unitary) ** 2 + h2d ** 2 + k2d ** 2) / (4 * np.pi) ** 2))
        f_eu4 = a_eu[3] * np.exp(- b_eu[3] * (((L* Unitary) ** 2 + h2d ** 2 + k2d ** 2) / (4 * np.pi) ** 2))
        f_Eu = f_eu1 + f_eu2 + f_eu3 + f_eu4 + c_eu * Unitary
        f_ga1 = a_ga[0] * np.exp(- b_ga[0] * (((L* Unitary) ** 2 + h2d ** 2 + k2d ** 2) / (4 * np.pi) ** 2))
        f_ga2 = a_ga[1] * np.exp(- b_ga[1] * (((L* Unitary) ** 2 + h2d ** 2 + k2d ** 2) / (4 * np.pi) ** 2))
        f_ga3 = a_ga[2] * np.exp(- b_ga[2] * (((L* Unitary) ** 2 + h2d ** 2 + k2d ** 2) / (4 * np.pi) ** 2))
        f_ga4 = a_ga[3] * np.exp(- b_ga[3] * (((L* Unitary) ** 2 + h2d ** 2 + k2d ** 2) / (4 * np.pi) ** 2))
        f_Ga = f_ga1 + f_ga2 + f_ga3 + f_ga4 + c_ga * Unitary
        f_al1 = a_al[0] * np.exp(- b_al[0] * (((L* Unitary) ** 2 + h2d ** 2 + k2d ** 2) / (4 * np.pi) ** 2))
        f_al2 = a_al[1] * np.exp(- b_al[1] * (((L* Unitary) ** 2 + h2d ** 2 + k2d ** 2) / (4 * np.pi) ** 2))
        f_al3 = a_al[2] * np.exp(- b_al[2] * (((L* Unitary) ** 2 + h2d ** 2 + k2d ** 2) / (4 * np.pi) ** 2))
        f_al4 = a_al[3] * np.exp(- b_al[3] * (((L* Unitary) ** 2 + h2d ** 2 + k2d ** 2) / (4 * np.pi) ** 2))
        f_Al = f_al1 + f_al2 + f_al3 + f_al4 + c_al * Unitary      
    return f_Eu, f_Ga, f_Al

    
def kspacecreator(x0, y0, xmax, ymax, deltak):
    """"""
    x = np.arange(x0-xmax, x0+xmax + deltak, deltak)
    y = np.arange(y0-ymax, y0+ymax + deltak, deltak)
    x2d, y2d = np.meshgrid(x, y)
    Unitary = np.ones((len(x2d), len(y2d)))  # Unitary matrix
    return x2d, y2d, x, y, Unitary


def dbw_H0KL(k2d, l2d, Unitary, H, a, c, u_list):
    lamb = 1e-10  # x-ray wavelength in m (Mo Ka)
    d_hkl = a / (np.sqrt((H * Unitary) ** 2 + k2d ** 2 + (a / c) ** 2 * l2d ** 2))
    theta = np.arcsin(lamb / (2 * d_hkl))
    B_iso_list = 8 * np.pi ** 2 / 3 * np.array(u_list)
    DBW_list = []
    for i in range(0, len(B_iso_list)):     
        DBW_list.append(np.exp(-B_iso_list[i] / lamb **2 * (np.sin(theta)) ** 2))
    return DBW_list


def dbw_HK0L(h2d, l2d, Unitary, K, a, c, u_list):
    lamb = 1e-10  # x-ray wavelength in m (Mo Ka)
    d_hkl = a / (np.sqrt((K * Unitary) ** 2 + h2d ** 2 + (a / c) ** 2 * l2d ** 2))
    theta = np.arcsin(lamb / (2 * d_hkl))
    B_iso_list = 8 * np.pi ** 2 / 3 * np.array(u_list)
    DBW_list = []
    for i in range(0, len(B_iso_list)):     
        DBW_list.append(np.exp(-B_iso_list[i] / lamb **2 * (np.sin(theta)) ** 2))
    return DBW_list


def dbw_HKL0(h2d, k2d, Unitary, L, a, c, u_list):
    lamb = 1e-10  # x-ray wavelength in m (Mo Ka)
    d_hkl = a / (np.sqrt(k2d ** 2 + h2d ** 2 + (a / c) ** 2 * (L * Unitary) ** 2))
    theta = np.arcsin(lamb / (2 * d_hkl))
    B_iso_list = 8 * np.pi ** 2 / 3 * np.array(u_list)
    DBW_list = []
    for i in range(0, len(B_iso_list)):     
        DBW_list.append(np.exp(-B_iso_list[i] / lamb **2 * (np.sin(theta)) ** 2))
    return DBW_list


def atomicformfactorBCC(h, k2d, l2d, Unitary, fatom):
    """Atomic form factors according to de Graed, structure of materials, 
    chapter 12: Eu2+. Ga1+, Al3+"""

    if fatom == True:
        f_Atom1 = 1
        f_Atom2 = 1/f_Atom1
    
    else:
        f_Atom1, f_Atom2 = 1, 1
    return f_Atom1, f_Atom2


# def fatom_calc_H0KL(H, k2d, l2d, Unitary, fatom=False, EuAl4=False):
#     if fatom or EuAl4:
#         a = {
#             'eu': [24.0063, 19.9504, 11.8034, 3.87243],
#             'ga': [15.2354, 6.7006, 4.3591, 2.9623],
#             'al': [4.17448, 3.3876, 1.20296, 0.528137]
#         }
#         b = {
#             'eu': [2.27783, 0.17353, 11.6096, 26.5156],
#             'ga': [3.0669, 0.2412, 10.7805, 61.4135],
#             'al': [1.93816, 4.14553, 0.228753, 8.28524]
#         }
#         c = {
#             'eu': 1.36389,
#             'ga': 1.7189,
#             'al': 0.706786
#         }
#     else:
#         a = b = c = {elem: [1.0, 1.0, 1.0, 1.0] for elem in ['eu', 'ga', 'al']}

#     f = {elem: 0 for elem in ['eu', 'ga', 'al']}

#     for elem in ['eu', 'ga', 'al']:
#         for i in range(4):
#             f[elem] += a[elem][i] * np.exp(-b[elem][i] * ((H * Unitary)**2 + \
#                                         k2d**2 + l2d**2) / (4 * np.pi)**2) 

#         for i in elem:
#             f[elem] += c[elem] * Unitary

#     if EuAl4:
#         f['ga'] = f['al'] 
#     return f['eu'], f['ga'], f['al']



