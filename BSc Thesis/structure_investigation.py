"""Single Crystal characterisation refinements analysis from BRUKER, comparison with STAVINOAH
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from Praktikum import lin_reg

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def stavinoah(z0, dz0, a, c, da, dc, compound=None, sample=None):
    alpha = np.sqrt(a**2/4 + (0.25 - z0)**2*c**2)
    d12 = alpha
    dd12 = np.sqrt( (-2/alpha*c**2*(0.25-z0) * dz0)**2 + (0.25*a/alpha * da)**2 + ((0.25-z0)*c*dc/alpha)**2 )
    d22 = (1 - 2*z0)*c
    dd22 = 2 * np.sqrt((z0*dc)**2 + (c*dz0)**2)
    theta = np.rad2deg(np.arccos((-0.25*a**2 + (0.25 - z0)**2 * c**2)/(0.25*a**2 + (0.25 - z0)**2*c**2)))
    dtheta = np.rad2deg(dz0 * (-((z0 - 0.25)/(z0**2 - z0/2 + 5/16)**2))/np.sqrt(1 - ((-0.25 + (0.25 - z0)**2)/(0.25 + (0.25 - z0)**2))**2))
    print("Comparison with STAVINOAH for {} sample {}:".format(compound, sample))
    print(r'd12 = ({} $\pm$ {})'.format(d12, dd12))
    print(r'd22 = ({} $\pm$ {})'.format(d22, dd22))
    print(r'theta = ({} $\pm$ {})'.format(theta, dtheta))

    return d12, dd12, d22, dd22, theta, dtheta


########################################
"""EuGa2Al2: Lets say 2 refinements (temp one included)"""
########################################
# Sample 1a SAFECALL
sinlambda1a, obs1a, cal1a, sigma1a, DIFsigma1a = np.loadtxt("EuGa2Al2/euga2al21a.txt",
                                                            usecols=(0, 1, 2, 3, 4), unpack=True, skiprows=2)
# Sample 1c
sinlambda1c, obs1c, cal1c, sigma1c, DIFsigma1c = np.loadtxt("EuGa2Al2/euga2al21c.txt",
                                                            usecols=(0, 1, 2, 3, 4), unpack=True, skiprows=2)

# Sample 1a4
sinlambda1a4, obs1a4, cal1a4, sigma1a4, DIFsigma1a4 = np.loadtxt("EuGa2Al2/euga2al21a4.txt",
                                                                 usecols=(0, 1, 2, 3, 4), unpack=True, skiprows=2)

# Sample 3b2
sinlambda3b2, obs3b2, cal3b2, sigma3b2, DIFsigma3b2 = np.loadtxt("EuGa2Al2/euga2al23b2.txt",
                                                                 usecols=(0, 1, 2, 3, 4), unpack=True, skiprows=2)
# Sample 4b2
sinlambda4b2, obs4b2, cal4b2, sigma4b2, DIFsigma4b2 = np.loadtxt("EuGa2Al2/euga2al24b2.txt",
                                                                 usecols=(0, 1, 2, 3, 4), unpack=True, skiprows=2)


# #######################################
# # Temperature dependent measurements
# #######################################
# sinlambda1a_2_100K, obs1a_2_100K, cal1a_2_100K, sigma1a_2_100K, DIFsigma1a_2_100K = np.loadtxt("euga2al2100K.txt",
#                                                       usecols=(0, 1, 2, 3, 4), unpack=True, skiprows=2)
# sinlambda1a_2_150K, obs1a_2_150K, cal1a_2_150K, sigma1a_2_150K, DIFsigma1a_2_150K = np.loadtxt("euga2al2150K.txt",
#                                                       usecols=(0, 1, 2, 3, 4), unpack=True, skiprows=2)
# sinlambda1a_2_200K, obs1a_2_200K, cal1a_2_200K, sigma1a_2_200K, DIFsigma1a_2_200K = np.loadtxt("euga2al2200K.txt",
#                                                       usecols=(0, 1, 2, 3, 4), unpack=True, skiprows=2)
# sinlambda1a_2_303K, obs1a_2_303K, cal1a_2_303K, sigma1a_2_303K, DIFsigma1a_2_303K = np.loadtxt("euga2al2303K.txt",
#                                                       usecols=(0, 1, 2, 3, 4), unpack=True, skiprows=2)

# a,c,V,z
T = np.array([100, 150, 200, 293]) # temperature in K
a = np.array([4.326090, 4.329725, 4.340179, 4.353075]) # a in angstrom cell parameter
da = np.array([0.000354, 0.000300, 0.000358, 0.000397])
c = np.array([10.916447, 10.922541, 10.935386, 10.970318]) # c in angstrom cell parameter
dc = np.array([0.001149, 0.001033, 0.001208, 0.001332])
z = np.array([0.38527993, 0.38544991, 0.38559416, 0.38626856]) # np.array([0.38626856, 0.38572708, 0.38559416, 0.38490966])  # np.array([0.3860138, 0.38572708, 0.38559416, 0.38490966]) # Wyckoff z-position Gallium # 0.38716856
dz = np.array([0.10797979E-02, 0.78963005E-03, 0.79795939E-03, 0.71762921e-03])
V = a**2*c # Volume in angstrom^3 cell parameter
# print("V_euga2al2 = {}".format(V))
dV = V*np.sqrt((2*da/a)**2 + (dc/c)**2)


# ########################################
# """EuGa4: 1-2 refinements + 1 temp. measurement??"""
# ########################################
# #Sample 1a
# sinlambda1a, obs1a, cal1a, sigma1a, DIFsigma1a = np.loadtxt("sample1a4.txt",
#                                                             usecols=(0, 1, 2, 3, 4), unpack=True, skiprows=2)
#Sample 1f
sinlambda1f, obs1f, cal1f, sigma1f, DIFsigma1f = np.loadtxt("EuGa4/euga41f.txt",
                                                            usecols=(0, 1, 2, 3, 4), unpack=True, skiprows=2)
# #temp measurements
sinlambda200K, obs200K, cal200K, sigma200K, DIFsigma200K = np.loadtxt("EuGa4/euga4_200K.txt",
                                                                      usecols=(0, 1, 2, 3, 4), unpack=True, skiprows=2)
sinlambda293K, obs293K, cal293K, sigma293K, DIFsigma293K = np.loadtxt("EuGa4/euga4_293K.txt",
                                                                      usecols=(0, 1, 2, 3, 4), unpack=True, skiprows=2)
sinlambda250K, obs250K, cal250K, sigma250K, DIFsigma250K = np.loadtxt("EuGa4/euga4_250K.txt",

                                                                      usecols=(0, 1, 2, 3, 4), unpack=True, skiprows=2)
sinlambda150K, obs150K, cal150K, sigma150K, DIFsigma150K = np.loadtxt("EuGa4/euga4_150K.txt",
                                                                      usecols=(0, 1, 2, 3, 4), unpack=True, skiprows=2)
sinlambda100K, obs100K, cal100K, sigma100K, DIFsigma100K = np.loadtxt("EuGa4/euga4_100K.txt",
                                                                      usecols=(0, 1, 2, 3, 4), unpack=True, skiprows=2)


def main():
    print(__doc__)
    n = 4 # scalefactor for blue differnce plots

    """EuGa2Al2"""
    # Plot Refinements
    # 1a
    # plt.errorbar(sinlambda1a4, obs1a4, yerr=sigma1a4, ls='', marker='x', capsize=1.0, c='k', label='Observed')
    # plt.plot(sinlambda1a4, cal1a4, marker='.', c='tab:red', ls='', label='Calculated')
    # plt.plot(sinlambda1a4, abs(cal1a4 - obs1a4) - np.max(obs1a4) / n, marker='.', c='tab:blue', ls='', label='Difference')
    # plt.plot(np.linspace(0.05, np.max(sinlambda1a4) +0.05, 1000), - np.max(obs1a4) / n * np.ones(1000), c='tab:blue', linewidth=0.5)
    # plt.xlim(0.1, np.max(sinlambda1a4) + 0.025)
    # plt.ylabel(r'Intensity (arb. units)', fontsize=13)
    # plt.xlabel(r'$\frac{\sin(\theta)}{\lambda}$(Å$^{-1}$)', fontsize=13)
    # plt.yticks([0, 5000, 10000, 15000, 20000])
    # plt.legend(fontsize=12)
    # plt.savefig("euga2al2_refinement_crystal4", dpi=300)
    # plt.show()

    # # # # 1c
    # plt.subplots(figsize=(10, 8))
    # plt.errorbar(sinlambda1c, obs1c, yerr=sigma1c, ls='', marker='x', capsize=4.0, c='k', label='Observed')
    # plt.plot(sinlambda1c, cal1c, marker='.', c='r', ls='', label='Calculated', markersize=10)
    # plt.plot(sinlambda1c, abs(cal1c - obs1c) - np.max(obs1c) / n, marker='.', c='b', ls='', label='Difference', markersize=10)
    # plt.plot(np.linspace(0.05, np.max(sinlambda1c) +0.05, 1000), - np.max(obs1c) / n* np.ones(1000), c='b', linewidth=0.5, markersize=10)
    # plt.xlim(0.1, np.max(sinlambda1c)+0.025)
    # plt.text(x=0.85, y=400000, s=r'R$_f=4.367$', fontsize=25)
    # plt.text(x=0.5, y=700000, s=r'EuGa$_2$Al$_2$', fontsize=30, fontstyle='oblique')
    # plt.ylabel(r'Integrated intensities (arb. units)', fontsize=25)
    # plt.tick_params(axis='y', labelsize=20, direction='in')
    # plt.tick_params(axis='x', labelsize=22, direction='in')
    # plt.xlabel(r'$\frac{\sin(\theta)}{\lambda}$(Å$^{-1}$)', fontsize=25)
    # plt.yticks([0, 200000, 400000, 600000, 800000])
    # plt.legend(fontsize=25)
    # plt.savefig("euga2al2_refinement_crystal1", dpi=300)
    # plt.show()
    # # 3b2
    # plt.errorbar(sinlambda3b2, obs3b2, yerr=sigma3b2, ls='', marker='x', capsize=1.0, c='k', label='Observed')
    # plt.plot(sinlambda3b2, cal3b2, marker='.', c='tab:red', ls='', label='Calculated')
    # plt.plot(sinlambda3b2, abs(cal3b2 - obs3b2) - np.max(obs3b2) / n, marker='.', c='tab:blue', ls='', label='Difference')
    # plt.plot(np.linspace(0.05, np.max(sinlambda3b2) + 0.05, 1000), - np.max(obs3b2) / n* np.ones(1000), c='tab:blue', linewidth=0.5)
    # plt.xlim(0.1, np.max(sinlambda3b2) + 0.025)
    # plt.ylabel(r'Intensity (arb. units)')
    # plt.xlabel(r'$\frac{\sin(\theta)}{\lambda}$(Å$^{-1}$)')
    # plt.legend()
    # plt.savefig("euga2al2_refinement_crystal3", dpi=300)
    # plt.show()
    #
    # # 4b2
    # plt.errorbar(sinlambda4b2, obs4b2, yerr=sigma4b2, ls='', marker='x', capsize=1.0, c='k', label='Observed')
    # plt.plot(sinlambda4b2, cal4b2, marker='.', c='tab:red', ls='', label='Calculated')
    # plt.plot(sinlambda4b2, abs(cal4b2 - obs4b2) - np.max(obs4b2) / n, marker='.', c='tab:blue', ls='', label='Difference')
    # plt.plot(np.linspace(0.05, np.max(sinlambda4b2) + 0.05, 1000), - np.max(obs4b2) / n* np.ones(1000), c='tab:blue', linewidth=0.5)
    # plt.xlim(0, np.max(sinlambda4b2) + 0.1)
    # plt.ylabel(r'Intensity (arb. units)')
    # plt.xlabel(r'$\frac{\sin(\theta)}{\lambda}$')
    # plt.legend()
    # plt.show()
    # # plt.savefig('EuGa2Al2_refinements.svg')
    # # plt.savefig('EuGa2Al2_T_refinements.svg')


    D12, D22, Theta = [],[],[]
    dD12, dD22, dTheta = [],[],[]
    """STAVINOAH STUFF"""
    for i in range(0,4):
        D12.append(stavinoah(z[i], dz[i], c=c[i], a=a[i], da=da[i], dc=dc[i], compound='EuGa2Al2', sample='1a [i]')[0])
        dD12.append(stavinoah(z[i], dz[i], c=c[i], a=a[i], da=da[i], dc=dc[i], compound='EuGa2Al2', sample='1a [i]')[1])
        D22.append(stavinoah(z[i], dz[i], c=c[i], a=a[i], da=da[i], dc=dc[i], compound='EuGa2Al2', sample='1a [i]')[2])
        dD22.append(stavinoah(z[i], dz[i], c=c[i], a=a[i], da=da[i], dc=dc[i], compound='EuGa2Al2', sample='1a [i]')[3])
        Theta.append(stavinoah(z[i], dz[i], c=c[i], a=a[i], da=da[i], dc=dc[i], compound='EuGa2Al2', sample='1a [i]')[4])
        dTheta.append(stavinoah(z[i], dz[i], c=c[i], a=a[i], da=da[i], dc=dc[i], compound='EuGa2Al2', sample='1a [i]')[5])
    stavinoah(z[0], dz[0], a=a[0], da=da[0], c=c[0], dc=dc[0], sample="100K")
    stavinoah(z[1], dz[1], a=a[1], da=da[1], c=c[1], dc=dc[1], sample="150K")
    stavinoah(z[2], dz[2], a=a[2], da=da[2], c=c[2], dc=dc[2], sample="200K")
    stavinoah(z[3], dz[3], a=a[3], da=da[3], c=c[3], dc=dc[3], sample="293K")
    print("euga2al2 d12={}+-{}, d22={}+-{}, theta={}+-{}".format(D12,dD12, D22, dD22,Theta, dTheta))

    ###################### Plots
    fig, ax1 = plt.subplots(figsize=(7, 5))
    tkw = dict(size=4, width=1.5)
    ax1.set_xticks([100, 200, 300])
    ax1.set_yticks([4.3, 4.4])
    ax1.tick_params(bottom=True, top=True, left=True, right=True, labelbottom=False, labeltop=False, labelright=False,
                    labelleft=False, **tkw)
    # ax1.locator_params(axis='y', nbins=3)
    # ax1.set_xlabel(r'$T$(K)', fontsize=30)
    # ax1.set_ylabel(r'$a$(Å)', color='b', fontsize=30)
    ax1.errorbar(T, a, yerr=da, color='b', linestyle='', lw=0.8, marker='o', markersize=12, capsize=4, label=r'$a$')
    m, sigma_m , dm, t, sigma_t, dt = lin_reg(T, a, dy=np.zeros(4), sigma_y=da, plot=False)
    ax1.plot(np.linspace(80, 310, 1000), m*np.linspace(80, 310, 1000) + t, lw=1.5, c='b', ls='-')
    print("a: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    # ax1.tick_params(axis='y', labelcolor='b', labelsize=30)
    ax1.text(x=160, y=4.45, s=r'EuGa$_2$Al$_2$', fontsize=40, style='oblique')
    ax1.set_ylim(4.2, 4.5)
    ax1.tick_params(axis='y', labelsize=30, direction='in', **tkw)
    ax1.tick_params(axis='x', labelsize=30, direction='in', **tkw)
    # ax1.legend(fontsize=27, loc='lower left')
    plt.tight_layout()
    #plt.savefig("stavinoah1_euga2al2_1_Version3.jpg", dpi=300)
    plt.show()
    
    
    # Adding Twin Axes
    fig, ax1 = plt.subplots(figsize=(7, 5))
    tkw = dict(size=4, width=1.5)
    ax1.set_xticks([100, 200, 300])
    ax1.tick_params(bottom=True, top=True, left=True, right=True, labelbottom=False, labeltop=False, labelright=False,
                    labelleft=False, **tkw)
    # ax1.locator_params(axis='y', nbins=3)
    ax1.tick_params(axis='x', labelsize=30, direction='in', **tkw)
    ax1.set_ylim(10.27, 11)
    ax1.set_yticks([10.4, 10.6, 10.8])
    m, sigma_m, dm, t, sigma_t, dt = lin_reg(T, c, dy=np.zeros(4), sigma_y=dc, plot=False)
    ax1.plot(np.linspace(80, 310, 1000), m * np.linspace(80, 310, 1000) + t, lw=1.5, c='g', ls='-')
    print("c: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    ax1.errorbar(T, c, yerr=dc, color='g', linestyle='', lw=0.8, marker='s', markersize=12, capsize=4, label=r'$c$')
    # ax1.set_ylabel(r'$c$(Å)', color='g', fontsize=30)
    ax1.tick_params(axis='y', direction='in', labelsize=30, **tkw)
    # ax1.text(x=210, y=10.965, s=r'(a)', style='oblique', fontsize=27)
    # ax1.text(x=80, y=10.950, s=r'EuGa$_2$Al$_2$', style='italic', fontsize=30, fontweight='bold')
    # ax1.legend(fontsize=27, loc='lower right')
    plt.tight_layout()
    #plt.savefig("stavinoah1_euga2al2_2_Version3.jpg", dpi=300)
    plt.show()
    
    
    fig, ax1 = plt.subplots(figsize=(7, 5))
    tkw = dict(size=4, width=1.5)
    # ax1.set_xlabel(r'$T$(K)', fontsize=30)
    ax1.set_ylim(2.53, 2.66)
    ax1.set_xticks([100, 200, 300])
    ax1.tick_params(bottom=True, top=True, left=True, right=True, labelbottom=False, labeltop=False, labelright=False,
                    labelleft=False, **tkw)
    # ax1.locator_params(axis='y', nbins=3)
    # ax1.set_ylabel(r'$d_{12}$(Å)', color='m', fontsize=30)
    m, sigma_m, dm, t, sigma_t, dt = lin_reg(T, D12, dy=np.zeros(4), sigma_y=dD12, plot=False)
    ax1.plot(np.linspace(80, 310, 1000), m * np.linspace(80, 310, 1000) + t, lw=1.5, c='m', ls='-')
    print("d12: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    ax1.errorbar(T, D12, yerr=dD12, color='m', linestyle='', lw=0.8, marker='o', markersize=12, capsize=4, label=r'$d_{12}$')
    ax1.tick_params(axis='y', direction='in', labelsize=30)
    # ax1.set_xticks([100, 150, 200, 293])
    ax1.tick_params(axis='x', labelsize=30, direction='in', **tkw)
    # ax1.legend(fontsize=27, loc='lower left')
    plt.tight_layout()
    #plt.savefig("stavinoah1_euga2al2_3_Version3.jpg", dpi=300)
    plt.show()
    
    
    fig, ax1 = plt.subplots(figsize=(7, 5))
    tkw = dict(size=4, width=1.5)
    ax1.set_xticks([100, 200, 300])
    ax1.tick_params(bottom=True, top=True, left=True, right=True, labelbottom=False, labeltop=False, labelright=False,
                    labelleft=False, **tkw)
    # ax1.locator_params(axis='y', nbins=3)
    ax1.set_ylim(2.35, 2.53)
    ax1.set_yticks([2.40, 2.45, 2.50])
    # ax1.set_ylabel('Temperat', color='g')
    m, sigma_m, dm, t, sigma_t, dt = lin_reg(T, D22, dy=np.zeros(4), sigma_y=dD22, plot=False)
    ax1.plot(np.linspace(80, 310, 1000), m * np.linspace(80, 310, 1000) + t, lw=1.5, c='y', ls='-')
    print("d22: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    ax1.errorbar(T, D22, yerr=dD22, color='y', linestyle='', lw=0.8, marker='s', markersize=12, capsize=4, label=r'$d_{22}$')
    # ax1.set_ylabel(r'$d_{22}$(Å)', color='y', fontsize=30)
    ax1.tick_params(axis='y', labelsize=30, direction='in', **tkw)
    ax1.tick_params(axis='x', labelsize=30, direction='in', **tkw)
    # ax1.text(x=220, y=2.525, s=r'(b)', style='oblique', fontsize=27)
    # ax1.legend(fontsize=27, loc='lower right')
    plt.tight_layout()
    #plt.savefig("stavinoah1_euga2al2_4_Version3.jpg", dpi=300)
    plt.show()
    
    
    fig, ax1 = plt.subplots(figsize=(7, 5))
    tkw = dict(size=4, width=1.5)
    ax1.set_ylim(0.3825, 0.39)
    ax1.set_yticks([0.384, 0.386, 0.388])
    ax1.set_xticks([100, 200, 300])
    ax1.tick_params(bottom=True, top=True, left=True, right=True, labelbottom=False, labeltop=False, labelright=False,
                    labelleft=False, **tkw)
    ax1.locator_params(axis='y', nbins=3)
    # ax1.set_xlabel(r'$T$(K)', fontsize=30)
    # ax1.set_ylabel(r'$z$', color='r', fontsize=30)
    m, sigma_m, dm, t, sigma_t, dt = lin_reg(T, z, dy=np.zeros(4), sigma_y=dz, plot=False)
    ax1.plot(np.linspace(80, 310, 1000), m * np.linspace(80, 310, 1000) + t, lw=1.5, c='r', ls='-')
    print("z: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    ax1.errorbar(T, z, yerr=dz, color='r', linestyle='', lw=0.8, marker='o', markersize=12, capsize=4, label=r'$z$')
    ax1.tick_params(axis='y', direction='in', labelsize=30)
    # ax1.set_yticks([0.377, 0.382, 0.387])
    ax1.tick_params(axis='x', labelsize=30, direction='in', **tkw)
    # ax1.locator_params(axis='y', nbins=3)
    # ax1.legend(fontsize=27, loc='upper left')
    plt.tight_layout()
    #plt.savefig("stavinoah1_euga2al2_5_Version3.jpg", dpi=300)
    plt.show()
    
    
    fig, ax1 = plt.subplots(figsize=(7, 5))
    tkw = dict(size=4, width=1.5)
    ax1.set_xticks([100, 200 , 300])
    ax1.tick_params(bottom=True, top=True, left=True, right=True, labelbottom=True, labeltop=False, labelright=False,
                    labelleft=False, **tkw)
    ax1.locator_params(axis='y', nbins=3)
    # ax1.set_ylabel('Temperat', color='g')
    m, sigma_m, dm, t, sigma_t, dt = lin_reg(T, Theta, dy=np.zeros(4), sigma_y=dTheta, plot=False)
    ax1.plot(np.linspace(80, 310, 1000), m * np.linspace(80, 310, 1000) + t, lw=1.5, c='c', ls='-')
    print("theta: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    ax1.errorbar(T, Theta, yerr=dTheta, color='c', linestyle='', lw=0.8, marker='s', markersize=12, capsize=4, label=r'$\theta$')
    # ax1.set_ylabel(r'$\theta$(°)', color='c', fontsize=30)
    ax1.tick_params(axis='y', labelsize=30, direction='in', **tkw)
    # ax1.text(x=255, y=111.525, s=r'(c)', style='oblique', fontsize=27)
    # ax1.locator_params(axis='y', nbins=3)
    # ax1.legend(fontsize=27, loc='upper right')
    ax1.set_ylim(110.7, 114.4)
    ax1.tick_params(axis='x', labelsize=30, direction='in', **tkw)
    ax1.text(x=180, y=110.820, s=r'$T$(K)', style='oblique', fontsize=30)
    # #plt.savefig("stavinoah1_euga2al2_3_Version2.jpg", dpi=300)
    # m, sigma_m, dm, t, sigma_t, dt = lin_reg(T, V, dy=np.zeros(4), sigma_y=dV, plot=False)
    # print("V: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    plt.tight_layout()
    #plt.savefig("stavinoah1_euga2al2_6_Version3.jpg", dpi=300)
    plt.show()
    
    
    # """
    # ============
    """EuGa4"""
    # ============
    # """
    # # Plot Refinements
    # # # 1f
    # # fig, ax = plt.subplots(figsize=(11, 8))
    # # plt.errorbar(sinlambda1f, obs1f, yerr=sigma1f, ls='', marker='x', capsize=4.0, c='k', label='Observed')
    # # plt.plot(sinlambda1f, cal1f, marker='.', c='tab:red', ls='', label='Calculated', markersize=10)
    # # plt.plot(sinlambda1f, abs(cal1f - obs1f) - np.max(obs1f) / n, marker='.', c='tab:blue', ls='', label='Difference', markersize=10)
    # # plt.plot(np.linspace(0.05, np.max(sinlambda1f) +0.05, 1000), - np.max(obs1f) / n * np.ones(1000), c='tab:blue', linewidth=0.5)
    # # plt.text(x=0.75, y=50000, s=r'R$_f=5.196$', fontsize=25)
    # # plt.text(x=0.6, y=90000, s=r'EuGa$_4$', fontsize=30, fontstyle='oblique')
    # # plt.xlim(0.05, np.max(sinlambda1f) + 0.025)
    # # plt.ylabel(r'Integrated intensities (arb. units)', fontsize=25)
    # # plt.xlabel(r'$\frac{\sin(\theta)}{\lambda}$(Å$^{-1}$)', fontsize=25)
    # # plt.legend(fontsize=25)
    # # plt.tick_params(axis='y', labelsize=22, direction='in')
    # # plt.tick_params(axis='x', labelsize=22, direction='in')
    # # plt.yticks([0, 4e4, 8e4, 12e4])
    # # #plt.savefig("euga4_refinement_crystal1", dpi=300)
    # # plt.show()

    # a,c,V,z
    T2 = np.array([100, 200, 250, 293])  # temperature in K    150
    a2 = np.array([4.375400, 4.387214, 4.387425, 4.403015])  # a in angstrom cell pa2rameter # 4.387414, 4.387225
    da2 = np.array([0.000735, 0.000352, 0.000246, 0.000308]) # 0.000603
    c2 = np.array([10.640713, 10.662181, 10.658446, 10.680388])  # c in angstrom cell parameter    10.667181, 10.656446
    dc2 = np.array([0.002541, 0.001104, 0.000836, 0.000977])              #0.002070
    B_eu = np.array([0.48221183, 0.89008331, 0.90516782, 1.1800470 ])      #0.95252866
    dB_eu = np.array([0.48541449E-01 , 0.97667180E-01, 0.45610957E-01, 0.75068019E-01])    #0.18984486
    B_ga1 = np.array([ 0.58138323 , 0.84271991, 0.83874643, 0.90734732])  #1.3838449
    dB_ga1 = np.array([0.61789177E-01, 0.10146012, 0.46784237E-01, 0.71538962E-01])    #0.26306602
    B_ga2 = np.array([0.40126890, 1.0198165, 0.83857411, 0.98966235])   #0.83032626
    dB_ga2 = np.array([0.70558026E-01, 0.10149224, 0.58203440E-01, 0.85263073E-01]) #0.18114553
    z2 = np.array([0.38288972, 0.38317543, 0.38373634,  0.38377333])  # Wyckoff z-position Gallium # 0.38716856   0.38448524
    dz2 = np.array([0.28512880E-03, 0.31877682E-03, 0.17599505E-03, 0.26537123E-03])           #0.59160584E-03
    V = a2 ** 2 * c2  # Volume in angstrom^3 cell parameter
    # print("V_euga2al2 = {}".format(V))
    dV = V * np.sqrt((2 * da2 / a2) ** 2 + (dc2 / c2) ** 2)

    D12, D22, Theta = [],[],[]
    dD12, dD22, dTheta = [],[],[]
    """STAVINOAH STUFF"""
    for i in range(0,4):
        D12.append(stavinoah(z2[i], dz2[i], c=c2[i], a=a2[i], da=da2[i], dc=dc2[i], compound='EuGa2Al2', sample='1a [i]')[0])
        dD12.append(stavinoah(z2[i], dz2[i], c=c2[i], a=a2[i], da=da2[i], dc=dc2[i], compound='EuGa2Al2', sample='1a [i]')[1])
        D22.append(stavinoah(z2[i], dz2[i], c=c2[i], a=a2[i], da=da2[i], dc=dc2[i], compound='EuGa2Al2', sample='1a [i]')[2])
        dD22.append(stavinoah(z2[i], dz2[i], c=c2[i], a=a2[i], da=da2[i], dc=dc2[i], compound='EuGa2Al2', sample='1a [i]')[3])
        Theta.append(stavinoah(z2[i], dz2[i], c=c2[i], a=a2[i], da=da2[i], dc=dc2[i], compound='EuGa2Al2', sample='1a [i]')[4])
        dTheta.append(stavinoah(z2[i], dz2[i], c=c2[i], a=a2[i], da=da2[i], dc=dc2[i], compound='EuGa2Al2', sample='1a [i]')[5])
    stavinoah(z2[0], dz2[0], a=a2[0], da=da2[0], c=c2[0], dc=dc2[0], sample="100K")
    stavinoah(z2[1], dz2[1], a=a2[1], da=da2[1], c=c2[1], dc=dc2[1], sample="200K")
    stavinoah(z2[2], dz2[2], a=a2[2], da=da2[2], c=c2[2], dc=dc2[2], sample="250K")
    stavinoah(z2[3], dz2[3], a=a2[3], da=da2[3], c=c2[3], dc=dc2[3], sample="250K")
    print("EuGa4 Stavinoha parameters from 100K to 293K: d12={}+-{}, d22={}+-{}, theta={}+-{}".format(D12, dD12, D22, dD22, Theta, dTheta))


    #############################
    # Temperature measurement Plot
    #############################
    fig, ax1 = plt.subplots(figsize=(8,5))
    tkw = dict(size=4, width=1.5)
    ax1.set_xticks([100, 200, 300])
    ax1.tick_params(bottom=True, top=True, left=True, right=True, labelbottom=False, labeltop=False, labelright=False,
                    labelleft=True, **tkw)
    # ax1.set_xlabel(r'$T$(K)', fontsize=30)
    # ax1.set_title(r'EuGa$_4$', fontsize=40, style='oblique')
    ax1.text(x=160, y=4.45, s=r'EuGa$_4$', fontsize=40, style='oblique')
    ax1.set_ylim(4.2, 4.5)
    ax1.set_yticks([4.3, 4.4])
    ax1.set_ylabel(r'$a$(Å)', color='b', fontsize=30)
    ax1.errorbar(T2, a2, yerr=da2, color='b', linestyle='', lw=0.8, marker='o', markersize=12, capsize=4, label=r'$a$')
    m, sigma_m , dm, t, sigma_t, dt = lin_reg(T2, a2, dy=np.zeros(4), sigma_y=da2, plot=False)
    ax1.plot(np.linspace(80, 310, 1000), m*np.linspace(80, 310, 1000) + t, lw=1.5, c='b', ls='-')
    print("a: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    ax1.tick_params(axis='y', labelcolor='b', labelsize=30, direction='in', **tkw)
    # ax1.set_xticks([100, 180, 200, 293])
    ax1.tick_params(axis='x', labelsize=30, direction='in', **tkw)
    # ax1.locator_params(axis='y', nbins=3)
    # ax1.tick_params(axis='x', labelsize=30, direction='in', **tkw)
    ax1.legend(fontsize=27, loc='upper left')
    plt.tight_layout()
    #plt.savefig("stavinoah1_euga4_1_Version3.jpg", dpi=300)
    plt.show()


    fig, ax1 = plt.subplots(figsize=(8,5))
    tkw = dict(size=4, width=1.5)
    ax1.set_xticks([100, 200, 300])
    ax1.tick_params(bottom=True, top=True, left=True, right=True, labelbottom=False, labeltop=False, labelright=False,
                    labelleft=True, **tkw)
    ax1.set_ylim(10.27, 11)
    ax1.set_yticks([10.4, 10.6, 10.8])
    ax1.tick_params(axis='y', labelcolor='g', labelsize=30, direction='in', **tkw)
    # ax1.locator_params(axis='y', nbins=3)
    m, sigma_m, dm, t, sigma_t, dt = lin_reg(T2, c2, dy=np.zeros(4), sigma_y=dc2, plot=False)
    ax1.plot(np.linspace(80, 310, 1000), m * np.linspace(80, 310, 1000) + t, lw=1.5, c='g', ls='-')
    print("c: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    ax1.errorbar(T2, c2, yerr=dc2, color='g', linestyle='', lw=0.8, marker='s', markersize=12, capsize=4, label=r'$c$')
    ax1.set_ylabel(r'$c$(Å)', color='g', fontsize=30)
    ax1.tick_params(axis='x', labelsize=30, direction='in', **tkw)
    ax1.tick_params(axis='y', labelcolor='g', direction='in', labelsize=30, **tkw)
    # ax1.text(x=210, y=10.678, s=r'(a)', style='oblique', fontsize=27)
    # ax1.text(x=80, y=10.665, s=r'EuGa$_4$', style='italic', fontsize=30, fontweight='bold')
    ax1.legend(fontsize=27, loc='upper left')
    plt.tight_layout()
    #plt.savefig("stavinoah1_euga4_2_Version3.jpg", dpi=300)
    plt.show()


    fig, ax1 = plt.subplots(figsize=(8,5))
    tkw = dict(size=4, width=1.5)
    ax1.set_xticks([100, 200, 300])
    ax1.tick_params(bottom=True, top=True, left=True, right=True, labelbottom=False, labeltop=False, labelright=False,
                    labelleft=True, **tkw)
    ax1.set_ylim(2.53, 2.66)
    # ax1.set_xlabel(r'$T$(K)', fontsize=30)
    ax1.set_ylabel(r'$d_{12}$(Å)', color='m', fontsize=30)
    m, sigma_m, dm, t, sigma_t, dt = lin_reg(T2, D12, dy=np.zeros(4), sigma_y=dD12, plot=False)
    ax1.plot(np.linspace(80, 310, 1000), m * np.linspace(80, 310, 1000) + t, lw=1.5, c='m', ls='-')
    print("d12: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    ax1.errorbar(T2, D12, yerr=dD12, color='m', linestyle='', lw=0.8, marker='o', markersize=12, capsize=4, label=r'$d_{12}$')
    ax1.tick_params(axis='y', labelcolor='m', labelsize=30, direction='in', **tkw)
    # ax1.set_xticks([100, 150, 200, 293])
    ax1.tick_params(axis='x', labelsize=30, direction='in', **tkw)
    # ax1.locator_params(axis='y', nbins=3)
    # ax1.tick_params(axis='x', labelsize=30, direction='in', **tkw)
    ax1.legend(fontsize=27, loc='upper left')
    plt.tight_layout()
    #plt.savefig("stavinoah1_euga4_3_Version3.jpg", dpi=300)
    plt.show()


    fig, ax1 = plt.subplots(figsize=(8,5))
    tkw = dict(size=4, width=1.5)
    ax1.set_xticks([100, 200, 300])
    ax1.tick_params(bottom=True, top=True, left=True, right=True, labelbottom=False, labeltop=False, labelright=False,
                    labelleft=True, **tkw)
    # ax1.locator_params(axis='y', nbins=3)
    ax1.set_ylim(2.35, 2.53)
    ax1.set_yticks([2.40, 2.45, 2.50])
    # ax1.set_ylabel('Temperat', color='g')
    m, sigma_m, dm, t, sigma_t, dt = lin_reg(T2, D22, dy=np.zeros(4), sigma_y=dD22, plot=False)
    ax1.plot(np.linspace(80, 310, 1000), m * np.linspace(80, 310, 1000) + t, lw=1.5, c='y', ls='-')
    print("d22: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    ax1.errorbar(T2, D22, yerr=dD22, color='y', linestyle='', lw=0.8, marker='s', markersize=12, capsize=4, label=r'$d_{22}$')
    ax1.set_ylabel(r'$d_{22}$(Å)', color='y', fontsize=30)
    ax1.tick_params(axis='y', labelcolor='y', labelsize=30, direction='in', **tkw)
    ax1.tick_params(axis='x', labelsize=30, direction='in', **tkw)
    # ax1.text(x=240, y=2.4967, s=r'(b)', style='oblique', fontsize=27)
    ax1.legend(fontsize=27, loc='center left')
    plt.tight_layout()
    #plt.savefig("stavinoah1_euga4_4_Version3.jpg", dpi=300)
    plt.show()


    fig, ax1 = plt.subplots(figsize=(8,5))
    tkw = dict(size=4, width=1.5)
    ax1.set_xticks([100, 200, 300])
    ax1.tick_params(bottom=True, top=True, left=True, right=True, labelbottom=False, labeltop=False, labelright=False,
                    labelleft=True, **tkw)
    # ax1.locator_params(axis='y', nbins=3)
    # fig.subplots_adjust(right=0.75)
    tkw = dict(size=4, width=1.5)
    ax1.set_ylim(0.3825, 0.39)
    ax1.set_yticks([0.384, 0.386, 0.388])
    # ax1.set_xlabel(r'$T$(K)', fontsize=30)
    ax1.set_ylabel(r'$z$', color='r', fontsize=30)
    m, sigma_m, dm, t, sigma_t, dt = lin_reg(T2, z2, dy=np.zeros(4), sigma_y=dz2, plot=False)
    ax1.plot(np.linspace(80, 310, 1000), m * np.linspace(80, 310, 1000) + t, lw=1.5, c='r', ls='-')
    print("z: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    ax1.errorbar(T2, z2, yerr=dz2, color='r', linestyle='', lw=0.8, marker='o', markersize=12, capsize=4, label=r'$z$')
    ax1.tick_params(axis='y', labelcolor='r', labelsize=30, direction='in', **tkw)
    # ax1.set_yticks([0.383, 0.388, 0.393])
    # ax1.tick_params(axis='x', labelsize=30, direction='in', **tkw)
    ax1.locator_params(axis='y')
    ax1.tick_params(axis='x', labelsize=30, direction='in', **tkw)
    ax1.legend(fontsize=27, loc='upper left')
    # ax1.set_xlabel(r'$T$(K)', fontsize=30)
    plt.tight_layout()
    #plt.savefig("stavinoah1_euga4_5_Version3.jpg", dpi=300)
    plt.show()


    fig, ax1 = plt.subplots(figsize=(8,5))
    tkw = dict(size=4, width=1.5)
    m, sigma_m, dm, t, sigma_t, dt = lin_reg(T2, Theta, dy=np.zeros(4), sigma_y=dTheta, plot=False)
    ax1.plot(np.linspace(80, 310, 1000), m * np.linspace(80, 310, 1000) + t, lw=1.5, c='c', ls='-')
    print("theta: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    ax1.errorbar(T2, Theta, yerr=dTheta, color='c', linestyle='', lw=0.8, marker='s', markersize=12, capsize=4, label=r'$\theta$')
    ax1.set_ylabel(r'$\theta$(°)', color='c', fontsize=30)
    ax1.tick_params(axis='y', direction='in', labelcolor='c', labelsize=30, **tkw)
    # ax1.text(x=100, y=114.12, s=r'(c)', style='oblique', fontsize=27)
    # ax1.locator_params(axis='y', nbins=3)
    ax1.legend(fontsize=27, loc='center left')
    ax1.set_xticks([100, 200, 300])
    ax1.tick_params(bottom=True, top=True, left=True, right=True, labelbottom=True, labeltop=False, labelright=False,
                    labelleft=True, **tkw)
    ax1.set_ylim(110.7, 114.4)
    ax1.tick_params(axis='x', labelsize=30, direction='in', **tkw)
    # ax1.set_xlabel(r'$T$(K)', fontsize=30)
    ax1.text(x=180, y=110.84, s=r'$T$(K)', style='oblique', fontsize=30)
    plt.tight_layout()
    ##plt.savefig("stavinoah1_euga4_6_Version3.jpg", dpi=300)
    plt.show()
    m, sigma_m, dm, t, sigma_t, dt = lin_reg(T, V, dy=np.zeros(4), sigma_y=dV, plot=False)
    print("V: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    print("V for EuGa4 for T=100,200, 250, 293 K: ={}".format(V))
    print("dV for EuGa4 for T=100,200, 250, 293 K: ={}".format(dV))


    # """XXXXtra special plot: B_iso shit"""
    # y1 = np.array([0.48, 0.891, 0.91, 1.18])
    # y2 = np.array([0.58, 0.81, 0.84, 0.91])
    # y3 = np.array([0.40, 1.02, 0.84, 0.99])
    # dy1 = np.array([0.05, 0.021, 0.05, 0.08])
    # dy2 = np.array([0.06, 0.12, 0.05, 0.07])
    # dy3 = np.array([0.07, 0.13, 0.06, 0.09])
    # T_biso = np.array([100, 200, 250, 293])
    # fig, ax1 = plt.subplots()
    # tkw = dict(size=4, width=1.5)
    # # ax1.set_ylim(0.3825, 0.395)
    # ax1.set_xlabel(r'$T$(K)', fontsize=30)
    # ax1.set_ylabel(r'B$_{\mathrm{iso}}$', fontsize=30)
    # m, sigma_m, dm, t, sigma_t, dt = lin_reg(T_biso, y1, dy=np.zeros(4), sigma_y=dy1, plot=False)
    # ax1.plot(np.linspace(80, 310, 1000), m * np.linspace(80, 310, 1000) + t, lw=1.5, c='b', ls='-')
    # print("B_iso Eu: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    # m, sigma_m, dm, t, sigma_t, dt = lin_reg(T_biso, y2, dy=np.zeros(4), sigma_y=dy2, plot=False)
    # ax1.plot(np.linspace(80, 310, 1000), m * np.linspace(80, 310, 1000) + t, lw=1.5, c='r', ls='-')
    # print("B_iso Ga(1): m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    # m, sigma_m, dm, t, sigma_t, dt = lin_reg(T_biso, y3, dy=np.zeros(4), sigma_y=dy3, plot=False)
    # ax1.plot(np.linspace(80, 310, 1000), m * np.linspace(80, 310, 1000) + t, lw=1.5, c='g', ls='-')
    # print("B_iso Ga(2): m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    # ax1.errorbar(T_biso, y1, yerr=dy1, color='b', linestyle='', lw=0.8, marker='o', markersize=12, capsize=4, label=r'Eu')
    # ax1.errorbar(T_biso, y2, yerr=dy2, color='r', linestyle='', lw=0.8, marker='s', markersize=12, capsize=4, label=r'Ga(1)')
    # ax1.errorbar(T_biso, y3, yerr=dy3, color='g', linestyle='', lw=0.8, marker='^', markersize=12, capsize=4, label=r'Ga(2)')
    # ax1.tick_params(axis='y', labelsize=30, direction='in', **tkw)
    # # ax1.set_xticks([100, 180, 200, 293])
    # ax1.tick_params(axis='x', labelsize=30, direction='in', **tkw)
    # ax1.tick_params(bottom=True, top=True, left=True, right=True, labelbottom=True, labeltop=False, labelright=False,
    #                 labelleft=True, **tkw)
    # ax1.legend(fontsize=25)
    # plt.tight_layout()
    # plt.savefig("B_isoshitttt.jpg", dpi=300)
    # # m, sigma_m, dm, t, sigma_t, dt = lin_reg(T, V, dy=np.zeros(4), sigma_y=dV, plot=False)
    # # print("V: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    # # print("V for EuGa4 for T=100,200, 250, 293 K: ={}".format(V))
    # # print("dV for EuGa4 for T=100,200, 250, 293 K: ={}".format(dV))
    # plt.show()


if __name__ == '__main__':
    main()