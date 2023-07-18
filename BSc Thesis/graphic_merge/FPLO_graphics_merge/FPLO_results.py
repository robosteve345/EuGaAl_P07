"""Evaluating the Nesting vector evolution in EuGa4 and EuGa2Al2
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from Praktikum import lin_reg
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

#EuGa2Al2, T-dep
ratios1 = np.array([0.536, 0.542, 0.549])
T1 = np.array([200, 250, 300])
M1 = np.array([0.5743392, 0.573598242, 0.57285918])
q_1_T = M1 * ratios1
dq_1_T = 0.01 * q_1_T

#EuGa2Al2, p-dep
p2 = np.array([2, 5]) # in GPa
ratios2 = np.array([0.136, 0.178])
M2 = np.array([0.58715287, 0.60254297])
q_2_p = M2 * ratios2
dq_2_p = 0.01 * q_2_p

#EuGa4, T-dep
ratios3 = np.array([0.311, 0.308, 0.306])
T3 = np.array([200, 250, 300])
M3 = np.array([0.58959, 0.58907, 0.588556])
q_3_T = M3 * ratios3
dq_3_T = 0.01 * q_3_T

#EuGa4, p-dep
p4 = np.array([0, 2, 5]) # in GPa
ratios4 = np.array([0.268, 0.310, 0.380])
M4 = np.array([0.59005605, 0.6003289, 0.6164269])
q_4_p = M4 * ratios4
dq_4_p = 0.01 * q_4_p
"""Holy Grail: Nesting vectors"""
def main():
    fig, ax1 = plt.subplots()
    tkw = dict(size=4, width=1.5)
    ax1.set_xlabel(r'$p$(GPa)', fontsize=30)
    ax1.set_ylabel(r'$q_{\mathrm{Nest.}}(\frac{2\pi}{c})$', fontsize=30)
    m, sigma_m, dm, t, sigma_t, dt = lin_reg(p4, q_4_p, dy=np.zeros(3), sigma_y=dq_4_p, plot=False)
    ax1.plot(np.linspace(-0.2,6, 1000), m * np.linspace(-0.2,6, 1000) + t, lw=1.5, c='b', ls='-')
    print("q_1_p: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    m, sigma_m, dm, t, sigma_t, dt = lin_reg(p2, q_2_p, dy=np.zeros(2), sigma_y=dq_2_p, plot=False)
    ax1.plot(np.linspace(1.5,6, 1000), m * np.linspace(1.5,6, 1000) + t, lw=1.5, c='g', ls='-')
    print("q_2_p: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    ax1.errorbar(p4, q_4_p, yerr=dq_4_p, color='b', linestyle='', lw=0.8, marker='o', markersize=12, capsize=4, label=r'q$_1$')
    ax1.errorbar(p2, q_2_p, yerr=dq_2_p, color='g', linestyle='', lw=0.8, marker='s', markersize=12, capsize=4, label=r'q$_2$')
    ax1.tick_params(axis='y', labelsize=30, direction='in', **tkw)
    ax1.tick_params(axis='x', labelsize=30, direction='in', **tkw)
    ax1.tick_params(bottom=True, top=True, left=True, right=True, labelbottom=True, labeltop=False, labelright=False,
                    labelleft=True, **tkw)
    ax1.legend(fontsize=25)
    plt.tight_layout()
    plt.savefig("Nesting_p.jpg", dpi=300)
    plt.show()

    fig, ax1 = plt.subplots()
    tkw = dict(size=4, width=1.5)
    ax1.set_xlabel(r'$T$(K)', fontsize=30)
    ax1.set_ylabel(r'$q_{\mathrm{Nest.}}(\frac{2\pi}{c})$', fontsize=30)
    m, sigma_m, dm, t, sigma_t, dt = lin_reg(T3, q_3_T, dy=np.zeros(3), sigma_y=dq_3_T, plot=False)
    ax1.plot(np.linspace(180, 310, 1000), m * np.linspace(180, 310, 1000) + t, lw=1.5, c='b', ls='-')
    #print("q_1_T: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    #m, sigma_m, dm, t, sigma_t, dt = lin_reg(T1, q_1_T, dy=np.zeros(3), sigma_y=dq_1_T, plot=False)
    #ax1.plot(np.linspace(180, 310, 1000), m * np.linspace(180, 310, 1000) + t, lw=1.5, c='g', ls='-')
    print("q_3_T: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    ax1.errorbar(T3, q_3_T, yerr=dq_3_T, color='b', linestyle='', lw=0.8, marker='o', markersize=12, capsize=4,
                 label=r'q$_1$')
    #ax1.errorbar(T1, q_1_T, yerr=dq_1_T, color='g', linestyle='', lw=0.8, marker='s', markersize=12, capsize=4,
    #             label=r'q$_3$')
    ax1.tick_params(axis='y', labelsize=30, direction='in', **tkw)
    ax1.tick_params(axis='x', labelsize=30, direction='in', **tkw)
    ax1.tick_params(bottom=True, top=True, left=True, right=True, labelbottom=True, labeltop=False, labelright=False,
                    labelleft=True, **tkw)
    ax1.legend(fontsize=25)
    plt.tight_layout()
    plt.savefig("Nesting_T.jpg", dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
