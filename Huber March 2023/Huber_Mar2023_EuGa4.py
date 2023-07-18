"""Data analysis of Eu(Ga,Al)4 DAC XRD experiment at HUBER"""
import matplotlib.pyplot as plt
from Praktikum import lin_reg
import numpy as np
from numpy import random
import scipy as sp
from scipy.optimize import curve_fit
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
tkw = dict(size=4, width=1.5)
########################################
# EuGa4 PHASE SPACE DIAGRAM
########################################
"""Mixed data points from Nakamura et. al: Q || [100] [110] [001]"""
p_CDW = np.array([0.7404580152671756, 0.7518904648977581, 1.1155308962369683, 1.4053872656494697, 1.4961832061068703,
1.6412213740458015, 1.8558324757259834, 2.1583516040986175, 2.0687022900763354, 2.2900763358778624, 2.5877862595419847])
T_CDW = np.array([100.00000000000003, 104.50579444270437, 122.74822139693052, 141.60812564320554, 141.9847328244275,
                  122.13740458015272, 153.24175578325654, 160.776768535505, 138.9312977099237, 161.83206106870233,
                  173.28244274809165])
"""EuGa4 probe from BSc thesis"""
p_exp = np.array([0, 2.03, 2.00, 3.30, 3.48, 5.09, 4.99])
dp_exp = np.array([0, 0.01, 0.01, 0.05, 0.05, 0.1, 0.1])
p_exp293 = np.array([0, 2.03, 3.30, 5.09])
p_exp250 = np.array([2.00, 3.48, 4.99])
p_exp2 = np.array([2.03, 5.09, 4.99]) # z-refinements that succeeded
"""P07_Ga_BSc_1""" # RUN01 RUN02 (p,T) MESSWERTEPAARE FÜR PHASENRAUMDIAGRAMM
p_exp_P07_Ga_BSc_1 = np.array([6.97, 7, 0.7, 0.9, 2.9]) # N>24 measurement points
dp_exp_P07_Ga_BSc_1 = np.array([0.15, 0.1, 0.1, 0.1, 0.1])
T_exp_P07_Ga_BSc_1 = np.array([293, 45, 35, 296, 20.022])
dT_exp_P07_Ga_BSc_1 = np.array([1, 0.2, 10, 10, 1])

###############################################################################
# CALIBRATION OF TOZERDAC 12mm with p_init = 6.7 GPa
################################################################################
# ABKÜHLEN
"""Huber Almax EasyLab 12mm TozerDAC cell calibration curve: p(T_cryo), p_initial = 6.93 GPa, dT1?, dT2?"""
T1, T2 = np.loadtxt("Cooldown_temperature.txt", usecols=(0, 1), skiprows=1, unpack=True)
# Reference: Vos 1991 - Journal Applied Physics
T_vos, R1_vos = np.loadtxt('rubyreference_vos.txt', usecols=(0, 1), skiprows=1, unpack=True)

"""Plot: Ruby fluorescence line R1(T)"""
# fig, ax1 = plt.subplots()
# ax1.tick_params(axis='x', labelcolor='k', labelsize=15, direction='in', **tkw)
# ax1.tick_params(axis='y', labelcolor='k', labelsize=15, direction='in', **tkw)
# ax1.set_xlabel(r'T(K)', fontsize=15)
# ax1.set_title(r'Ruby fluorescence line R1 characteristic', fontsize=15)
# ax1.set_ylabel(r'R1(nm)', fontsize=15)
# # REFERENCE DATA
# ax1.plot(T_vos, R1_vos, marker='s', markersize=12, ls='',
#          label=r'Vos 1991 - Journal for Applied physics', c='k')
# # WARMUP DATA
# plt.legend(fontsize=15)
# # plt.savefig("Huber March 2023/R1_calibrationcurve_TozerDAC12mm.jpg", dpi=500)
# plt.show()

"""Plot: Calibration curve p(T_cryo)"""
# # Neglect T1 sensor
# # m1, sigma_m1 , dm1, t1, sigma_t1, dt1 = lin_reg(T2, p_cell_old, sigma_y=np.ones(len(p_cell_old)), dy=np.ones(len(p_cell_old)), plot=False)
# m2, sigma_m2 , dm2, t2, sigma_t2, dt2 = lin_reg(T1[0:25], p_cell_old[0:25], sigma_y=np.ones(len(p_cell_old[0:25])), dy=np.ones(len(p_cell_old[0:25])), plot=False)
# fig, ax1 = plt.subplots()
# ax1.set_xlabel(r'T(K)', fontsize=15)
# ax1.set_title(r'TozerDAC 12mm $p_{\text{init}=6.97 \text{GPa}$}', fontsize=15)
# ax1.set_ylabel(r'p(GPa)', fontsize=15)
# #ax1.plot(T2, p_cell_old, color='k', linestyle='', lw=0.8, marker='o', markersize=6,
# #         label=r'Calibration curve  T2')
# #ax1.plot(np.linspace(T2[0], T2[-1], 500), m1*np.linspace(T2[0], T2[-1], 500) + t1, lw=1.5, c='k', ls='-')
# #print("T_CDW linear fit parameters: m={}+-{}, t={}+-{}".format(m1, sigma_m1, t1, sigma_t1))
# ax1.plot(T1, p_cell_old, color='b', linestyle='', lw=0.8, marker='o', markersize=6,
#          label=r'Sensor T1')
# ax1.plot(np.linspace(150, 300, 500), m2*np.linspace(150, 300, 500) + t2, lw=1.5, c='b', ls='-')
# print("p(T_cryo) Fit Parameter: m={}+-{}, t={}+-{}".format(m2, sigma_m2, t2, sigma_t2))
# plt.legend(fontsize=18, loc='center left')
# plt.show()
# # plt.savefig("Huber March 2023/TozerDAC_calibrationcurve.jpg", dpi=500)

"""Plot: CDW_phasediag"""
fig, ax1 = plt.subplots()
ax1.set_xlabel(r'p(GPa)', fontsize=20)
ax1.axvline(x=0.7,  ymin=0, ymax=0.3, linestyle = '--')
ax1.axvline(x=2.9, ymin=0, ymax=0.55, linestyle = '--')
ax1.text(x=1.25, y=62, s="CDW", fontsize=20, weight='bold')
ax1.set_ylabel(r'T$_{\mathrm{CDW}}$(K)', color='k', fontsize=20)
ax1.plot(p_CDW, T_CDW, color='k', linestyle='', lw=0.8, marker='.', markersize=12,
         label=r'$\mathbf{J}$, $\mathbf{Q}$ $||$ [100] [110] [001]')
m, sigma_m , dm, t, sigma_t, dt = lin_reg(p_CDW, T_CDW, sigma_y=np.ones(11), dy=np.ones(11), plot=False)
ax1.fill_between(np.linspace(0.7, 2.9, 100), m*np.linspace(0.7, 2.9, 100) + t, 0, color='b', alpha=0.15)
# ax1.plot(np.linspace(0, 6, 500), m*np.linspace(0, 6, 500) + t, lw=1.5, c='k', ls='-')
print("T_CDW linear fit parameters: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
ax1.tick_params(axis='y', labelcolor='k', labelsize=20, **tkw)
# # ax1.set_xticks([100, 180, 200, 293])
ax1.locator_params(axis='y', nbins=4)
ax1.set_ylim(0, 320)
ax1.tick_params(axis='x', labelsize=20, direction='in', **tkw)
ax1.legend(fontsize=18 , loc='upper center'
           )
# Adding Twin Axes
ax2 = ax1.twinx()
# ax2.errorbar(p_exp, np.array([293, 293, 250, 293, 250, 293, 250]), xerr=dp_exp, marker='.', markersize=12,
#              capsize=6, ls='', label=r'Bruker, 08/2022', c='b')
ax2.errorbar(p_exp_P07_Ga_BSc_1, T_exp_P07_Ga_BSc_1, xerr=dp_exp_P07_Ga_BSc_1, yerr = dT_exp_P07_Ga_BSc_1,
             marker='.', markersize=12,
             capsize=6, ls='', label=r'Our data', c='b')
ax2.set_ylabel(r'T(K)', color='b', fontsize=20)
ax2.tick_params(axis='y', labelcolor='b', labelsize=20, **tkw)
#ax2.annotate("", xy=(4.99, 250), xytext=(5.3, 150), fontsize=50,
#  arrowprops=dict(arrowstyle="->"))
ax2.set_ylim(0, 320)
ax2.legend(fontsize=18, loc='center right')
plt.tight_layout()
plt.savefig("CDW_phasediag_EuGa4_Mar2023.jpg", dpi=500)
plt.show()

"""Plot: Structure parameters Eu(Ga,Al)4"""
# EuGa4
# REIHENFOLGE: 0GPa @ 293K, 2GPa @ 293K, 3GPa @ 293K, 5GPa @ 293K
# a293 = np.array([4.484, 4.313181, 4.270780, 4.271346])
# da293 = np.array([0.011, 0.004827, 0.002990, 0.003721])
# c293 = np.array([10.74, 10.432373, 10.308905, 10.327200])
# dc293 = np.array([0.02, 0.009222, 0.039447, 0.029639])
# z = np.array([0, 0.38573259, 0, 0, 0, 0.38476077, 0.38476077])
# dz = np.array([0, 0.37381700E-02, 0, 0, 0, 0.85456884E-02, 0.85456884E-02])
# V293 = a293**2*c293
# dV293 = V293 * np.sqrt((2 * da293 / a293) ** 2 + (dc293 / c293) ** 2)
#
# # REIHENFOLGE: 2GPa @ 250K, 3GPa @ 250K, 5GPa @ 250K
# a250 = np.array([4.298343, 4.311010, 4.263003])
# da250 = np.array([0.004425, 0.002848, 0.004028])
# c250 = np.array([10.352694, 10.386235, 10.329618])
# dc250 = np.array([0.030567, 0.047502, 0.032752])
# V250 = a250**2*c250
# dV250 = V250 * np.sqrt((2 * da250 / a250) ** 2 + (dc250 / c250) ** 2)
