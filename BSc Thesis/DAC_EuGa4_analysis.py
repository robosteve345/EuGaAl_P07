"""Evaluation of EuGa4 DAC-Data: Dependence of structural properties on temperature T and pressure p:
d12, d22, theta, a, c, z"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
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


"""CDW experiment"""
# # # mixed data points from Nakamura et. al: Q || [100] [110] [001]
# p_CDW = np.array([0.7404580152671756, 0.7518904648977581, 1.1155308962369683, 1.4053872656494697, 1.4961832061068703,
# 1.6412213740458015, 1.8558324757259834, 2.1583516040986175, 2.0687022900763354, 2.2900763358778624, 2.5877862595419847])
# T_CDW = np.array([100.00000000000003, 104.50579444270437, 122.74822139693052, 141.60812564320554, 141.9847328244275, 122.13740458015272,
# 153.24175578325654, 160.776768535505, 138.9312977099237, 161.83206106870233, 173.28244274809165])
# # Our experimental data
p_exp = np.array([0, 2.03, 2.00, 3.30, 3.48, 5.09, 4.99])
dp_exp = np.array([0, 0.01, 0.01, 0.05, 0.05, 0.1, 0.1, 0.2])
p_exp293 = np.array([0, 2.03, 3.30, 5.09])
p_exp250 = np.array([2.00, 3.48, 4.99])
p_exp2 = np.array([2.03, 5.09, 4.99]) # z-refinements that succeeded
# fig, ax1 = plt.subplots()
# fig.subplots_adjust(right=0.8)
# tkw = dict(size=4, width=1.5)
# ax1.set_xlabel(r'p(GPa)', fontsize=20)
# ax1.set_ylabel(r'T$_{\mathrm{CDW}}$(K)', color='k', fontsize=20)
# ax1.plot(p_CDW, T_CDW, color='k', linestyle='', lw=0.8, marker='o', markersize=12, label=r'$\mathbf{J}$, $\mathbf{Q}$ $||$ [100] [110] [001]')
# m, sigma_m , dm, t, sigma_t, dt = lin_reg(p_CDW, T_CDW, sigma_y=np.ones(11), dy=np.ones(11), plot=False)
# ax1.plot(np.linspace(0, 6, 500), m*np.linspace(0, 6, 500) + t, lw=1.5, c='k', ls='-')
# print("T_CDW linear fit parameters: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
# ax1.tick_params(axis='y', labelcolor='k', labelsize=20)
# # # ax1.set_xticks([100, 180, 200, 293])
# ax1.locator_params(axis='y', nbins=4)
# ax1.tick_params(axis='x', labelsize=20, direction='in', **tkw)
# ax1.legend(fontsize=18 , loc='lower right'
#            )
# # Adding Twin Axes
# ax2 = ax1.twinx()
# ax2.errorbar(p_exp, np.array([293, 293, 250, 293, 250, 293, 250]), xerr=dp_exp, marker='x', markersize=12, capsize=6, ls='', label=r'Exp. points', c='b')
# ax2.set_ylabel(r'T(K)', color='b', fontsize=20)
# ax2.tick_params(axis='y', labelcolor='b', labelsize=20, **tkw)
# ax2.annotate("", xy=(4.99, 250), xytext=(5.3, 150), fontsize=50,
#   arrowprops=dict(arrowstyle="->"))
# ax2.set_ylim(0, 320)
# ax2.legend(fontsize=18, loc='center left'
#                )
# plt.tight_layout()
# # plt.savefig("CDW_DAC_comparison_linfit.jpg", dpi=500)
# plt.show()

######################################
"""DAC Pressure and Temperature data
"""
######################################
# REIHENFOLGE: 0GPa @ 293K, 2GPa @ 293K, 3GPa @ 293K, 5GPa @ 293K
a293 = np.array([4.484, 4.313181, 4.270780, 4.271346])
da293 = np.array([0.011, 0.004827, 0.002990, 0.003721])
c293 = np.array([10.74, 10.432373, 10.308905, 10.327200])
dc293 = np.array([0.02, 0.009222, 0.039447, 0.029639])
z = np.array([0, 0.38573259, 0, 0, 0, 0.38476077, 0.38476077])
dz = np.array([0, 0.37381700E-02, 0, 0, 0, 0.85456884E-02, 0.85456884E-02])
V293 = a293**2*c293
dV293 = V293 * np.sqrt((2 * da293 / a293) ** 2 + (dc293 / c293) ** 2)

# REIHENFOLGE: 2GPa @ 250K, 3GPa @ 250K, 5GPa @ 250K
a250 = np.array([4.298343, 4.311010, 4.263003])
da250 = np.array([0.004425, 0.002848, 0.004028])
c250 = np.array([10.352694, 10.386235, 10.329618])
dc250 = np.array([0.030567, 0.047502, 0.032752])
V250 = a250**2*c250
dV250 = V250 * np.sqrt((2 * da250 / a250) ** 2 + (dc250 / c250) ** 2)

D12293, D22293, Theta293 = [2.5799177159492483, 2.5491041595699375],[2.38416048572786, 2.3801971521120002],[113.42235488002747, 113.8198192861541]
dD12293, dD22293, dTheta293 = [0.04315422695116597, 0.0977263348399128],[0.07831977004886498, 0.17797356399181286],[-0.797923784509957, -1.8258913673748525]

D12250, D22250, Theta250 = [2.5457884005154154 ],[2.3807544490282804],[113.70501731174919]
dD12250, dD22250, dTheta250 = [0.09819701498275951],[0.1783372982940546],[-1.8258913673748525]

"""STAVINOHA STUFF"""
stavinoah(z[1], dz[1], a=a293[1], da=da293[1], c=c293[1], dc=dc293[1], sample="2 GPa @ 293K")
stavinoah(z[5], dz[5], a=a293[3], da=da293[3], c=c293[3], dc=dc293[3], sample="5 GPa @ 293K")
stavinoah(z[6], dz[6], a=a250[2], da=da250[2], c=c250[2], dc=dc250[2], sample="5 GPa @ 250K")


def main():
    print(__doc__)
    # fig, ax1 = plt.subplots()
    # fig.subplots_adjust(right=0.75)
    # tkw = dict(size=4, width=1.5)
    # # ax1.suptitle(r'EuGa$_4$ DAC', size=20)
    # ax1.set_xlabel(r'$p$(GPa)', fontsize=30)
    # ax1.set_title(r'EuGa$_4$ DAC', fontsize=32, style='oblique')
    # ax1.set_yticks([])
    # # ax1.set_ylabel(r'$a$(Å)', color='b', fontsize=30)
    # ax1.errorbar(p_exp293, a293, yerr=da293, color='b', linestyle='', lw=0.8, marker='s', markersize=12, capsize=4)
    # ax1.errorbar(p_exp250, a250, yerr=da250, color='b', linestyle='', lw=0.8, marker='o', markersize=12, capsize=4)
    # m, sigma_m , dm, t, sigma_t, dt = lin_reg(p_exp293, a293, dy=np.zeros(4), sigma_y=da293, plot=False)
    # print("a293: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    # ax1.plot(np.linspace(0, 6, 1000), m * np.linspace(0, 6, 1000) + t, lw=1.5, c='b', ls='-.')
    # m, sigma_m, dm, t, sigma_t, dt = lin_reg(p_exp250, a250, dy=np.zeros(3), sigma_y=da250, plot=False)
    # print("a250: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    # ax1.plot(np.linspace(0, 6, 1000), m*np.linspace(0, 6, 1000) + t, lw=1.5, c='b', ls='-')
    # # ax1.tick_params(axis='y', labelcolor='b', labelsize=30)
    # ax1.set_xticks([1,3,5])
    # ax1.set_ylim(4.2, 4.5)
    # ax1.locator_params(axis='y', nbins=3)
    # ax1.tick_params(axis='x', labelsize=30, direction='in', **tkw)
    # legend_elements = [Line2D([0], [0], color='k', lw=2, label=r'$293$ K', ls='-.'),
    #                    Line2D([0], [0], color='k', lw=2, label=r'$250$ K', ls='-'),
    #                    Line2D([0], [0], marker='s', color='w', label='$293$ K',
    #                           markerfacecolor='k', markersize=15),
    #                    Line2D([0], [0], marker='o', color='w', label='$250$ K',
    #                           markerfacecolor='k', markersize=15)]
    # ax1.legend(fontsize=18, loc='upper right', handles=legend_elements)
    # # Adding Twin Axes
    # ax2 = ax1.twinx()
    # ax2.set_ylim(10.27, 11)
    # ax2.errorbar(p_exp293, c293, yerr=dc293, color='g', linestyle='', lw=0.8, marker='s', markersize=12, capsize=4)
    # ax2.errorbar(p_exp250, c250, yerr=dc250, color='g', linestyle='', lw=0.8, marker='o', markersize=12, capsize=4)
    # m, sigma_m, dm, t, sigma_t, dt = lin_reg(p_exp293, c293, dy=np.zeros(4), sigma_y=dc293, plot=False)
    # ax2.plot(np.linspace(0, 6, 1000), m * np.linspace(0, 6, 1000) + t, lw=1.5, c='g', ls='-.')
    # print("c293: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    # m, sigma_m, dm, t, sigma_t, dt = lin_reg(p_exp250, c250, dy=np.zeros(3), sigma_y=dc250, plot=False)
    # ax2.plot(np.linspace(0, 6, 1000), m * np.linspace(0, 6, 1000) + t, lw=1.5, c='g', ls='-')
    # print("c250: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    # ax2.set_ylabel(r'$c$(Å)', color='g', fontsize=30)
    # ax2.tick_params(axis='y', labelcolor='g', labelsize=30, **tkw)
    # # ax2.text(x=0.8, y=10.2, s=r'(a)', style='oblique', fontsize=27)
    # # ax2.text(x=1.2, y=10.62, s='EuGa$_4$ \n DAC', style='italic', fontsize=30, fontweight='bold')
    # ax2.legend(fontsize=27, loc='lower right')
    # plt.tight_layout()
    # # plt.savefig("DAC_EuGa4_structure_1_Version2.jpg", dpi=300)
    #
    # fig, ax1 = plt.subplots()
    # fig.subplots_adjust(right=0.75)
    # tkw = dict(size=4, width=1.5)
    #
    # ax1.set_xlabel(r'$p$(GPa)', fontsize=30)
    # # ax1.set_ylabel(r'$d_{12}$(Å)', color='m', fontsize=30)
    # ax1.errorbar(p_exp2[0:-1], D12293, yerr=dD12293, color='m', linestyle='', lw=0.8, marker='s', markersize=12, capsize=4,
    #              label=r'$293$ K')
    # ax1.errorbar(p_exp2[2], D12250, yerr=dD12250, color='m', linestyle='', lw=0.8, marker='o', markersize=12, capsize=4,
    #              label=r'$250$ K')
    # m, sigma_m, dm, t, sigma_t, dt = lin_reg(p_exp2[0:-1], D12293, dy=np.zeros(2), sigma_y=dD12293, plot=False)
    # print("d12293: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    # ax1.plot(np.linspace(0, 6, 1000), m * np.linspace(0, 6, 1000) + t, lw=1.5, c='m', ls='-.')
    # # m, sigma_m, dm, t, sigma_t, dt = lin_reg(p_exp2[2], D12250, dy=np.zeros(1), sigma_y=dD12250, plot=False)
    # # print("d12250: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    # # ax1.plot(np.linspace(0, 6, 1000), m * np.linspace(0, 6, 1000) + t, lw=1.5, c='m', ls='-')
    # # ax1.tick_params(axis='y', labelcolor='m', labelsize=30)
    # # ax2.legend(fontsize=27, loc='lower right')
    # ax1.set_xticks([1,3,5])
    # ax1.set_ylim(2.53, 2.66)
    # ax1.set_yticks([])
    # # ax1.locator_params(axis='y', nbins=3)
    # ax1.tick_params(axis='x', labelsize=30, direction='in', **tkw)
    # # legend_elements = [Line2D([0], [0], color='k', lw=2, label=r'$293$ K', ls='-.'),
    # #                   Line2D([0], [0], marker='s', color='w', label='$293$ K',
    # #                          markerfacecolor='k', markersize=15),
    # #                   Line2D([0], [0], marker='o', color='w', label='$250$ K',
    # #                          markerfacecolor='k', markersize=15)]
    # # ax1.legend(fontsize=18, loc='upper right'# , handles=legend_elements
    # #           )
    # # Adding Twin Axes
    # ax2 = ax1.twinx()
    # ax2.set_ylim(2.35, 2.53)
    # ax2.errorbar(p_exp2[0:-1], D22293, yerr=dD22293, color='y', linestyle='', lw=0.8, marker='s', markersize=12, capsize=4)
    # ax2.errorbar(p_exp2[2], D22250, yerr=dD22250, color='y', linestyle='', lw=0.8, marker='o', markersize=12, capsize=4)
    # m, sigma_m, dm, t, sigma_t, dt = lin_reg(p_exp2[0:-1], D22293, dy=np.zeros(2), sigma_y=dD22293, plot=False)
    # ax2.plot(np.linspace(0, 6, 1000), m * np.linspace(0, 6, 1000) + t, lw=1.5, c='y', ls='-.')
    # print("D22293: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    # # m, sigma_m, dm, t, sigma_t, dt = lin_reg(p_exp2[2], D22250, dy=np.zeros(1), sigma_y=dD22250, plot=False)
    # # ax2.plot(np.linspace(0, 6, 1000), m * np.linspace(0, 6, 1000) + t, lw=1.5, c='y', ls='-')
    # # print("D22250: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    # ax2.set_ylabel(r'$d_{22}$(Å)', color='y', fontsize=30)
    # legend_elements = [Line2D([0], [0], color='k', lw=2, label=r'$293$ K', ls='-.'),
    #                    Line2D([0], [0], color='k', lw=2, label=r'$250$ K', ls='-'),
    #                    Line2D([0], [0], marker='s', color='w', label='$293$ K',
    #                           markerfacecolor='k', markersize=15),
    #                    Line2D([0], [0], marker='o', color='w', label='$250$ K',
    #                           markerfacecolor='k', markersize=15)]
    # ax2.legend(fontsize=18, loc='upper left', handles=legend_elements)
    # ax2.tick_params(axis='y', labelcolor='y', labelsize=30, **tkw)
    # # ax2.text(x=3.2, y=2.51, s=r'(b)', style='oblique', fontsize=27)
    # plt.tight_layout()
    # # plt.savefig("DAC_EuGa4_structure_2_Version2.jpg", dpi=300)

    fig, ax1 = plt.subplots()
    fig.subplots_adjust(right=0.75)
    tkw = dict(size=4, width=1.5)

    ax1.set_xlabel(r'$p$(GPa)', fontsize=30)
    # ax1.set_ylabel(r'$z$', color='r', fontsize=30)
    # ax1.errorbar(p_exp2[0:-1], np.array([0.38573259, 0.38476077]), yerr=np.array([0.37381700E-02, 0.85456884E-02]), color='r', linestyle='', lw=0.8, marker='s', markersize=12,
    #              capsize=4)
    # ax1.errorbar(p_exp2[2], np.array([0.38476077]), yerr=np.array([0.85456884E-02]), color='r', linestyle='', lw=0.8, marker='o', markersize=12, capsize=4)
    # m, sigma_m, dm, t, sigma_t, dt = lin_reg(p_exp2[0:-1], np.array([0.38573259, 0.38476077]), dy=np.zeros(2), sigma_y=np.array([0.37381700E-02, 0.85456884E-02]), plot=False)
    # print("z293: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    # ax1.plot(np.linspace(0, 6, 1000), m * np.linspace(0, 6, 1000) + t, lw=1.5, c='r', ls='-.')
    # m, sigma_m, dm, t, sigma_t, dt = lin_reg(p_exp2[2], D12250, dy=np.zeros(1), sigma_y=dD12250, plot=False)
    # print("d12250: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    # ax1.plot(np.linspace(0, 6, 1000), m * np.linspace(0, 6, 1000) + t, lw=1.5, c='m', ls='-')
    # ax1.tick_params(axis='y', labelcolor='r', labelsize=30)
    # ax1.set_xticks([1, 3, 5])
    # ax1.set_yticks([])
    # ax1.set_ylim(0.3825, 0.395)
    # ax1.locator_params(axis='y', nbins=3)
    # ax1.tick_params(axis='x', labelsize=30, direction='in', **tkw)
    # legend_elements = [Line2D([0], [0], color='k', lw=2, label=r'$293$ K', ls='-.'),
    #                   Line2D([0], [0], marker='s', color='w', label='$293$ K',
    #                          markerfacecolor='k', markersize=15),
    #                   Line2D([0], [0], marker='o', color='w', label='$250$ K',
    #                          markerfacecolor='k', markersize=15)]
    # ax1.legend(fontsize=18, loc='upper left' , handles=legend_elements
    #           )
    # Adding Twin Axes
    # ax2 = ax1.twinx()
    # ax2.errorbar(p_exp2[0:-1], Theta293, yerr=dTheta293, color='c', linestyle='', lw=0.8, marker='s', markersize=12,
    #              capsize=4)
    # ax2.errorbar(p_exp2[2], Theta250, yerr=dTheta250, color='c', linestyle='', lw=0.8, marker='o', markersize=12, capsize=4)
    m, sigma_m, dm, t, sigma_t, dt = lin_reg(p_exp2[0:-1], Theta293, dy=np.zeros(2), sigma_y=dTheta293, plot=False)
    # ax2.plot(np.linspace(0, 6, 1000), m * np.linspace(0, 6, 1000) + t, lw=1.5, c='c', ls='-.')
    print("Theta293: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    # m, sigma_m, dm, t, sigma_t, dt = lin_reg(p_exp2[2], D22250, dy=np.zeros(1), sigma_y=dD22250, plot=False)
    # ax2.plot(np.linspace(0, 6, 1000), m * np.linspace(0, 6, 1000) + t, lw=1.5, c='y', ls='-')
    # print("D22250: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    # ax2.set_ylabel(r'$\theta$(°)', color='c', fontsize=30)
    # ax2.tick_params(axis='y', labelcolor='c', labelsize=30, **tkw)
    # ax2.set_ylim(110.7, 114.4)
    # ax2.text(x=3.2, y=115.2, s=r'(c)', style='oblique', fontsize=27)
    # plt.tight_layout()
    # # plt.savefig("DAC_EuGa4_structure_3_Version2.jpg", dpi=300)
    # plt.show()

    # m, sigma_m, dm, t, sigma_t, dt = lin_reg(p_exp293, V293, dy=np.zeros(4), sigma_y=dV293, plot=False)
    # print("V293: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    # print("V for EuGa4 for T=293 K and p = 0,2,3.48,5 GPa: ={}".format(V293))
    # print("dV for EuGa4 for T=293 K and p = 0,2,3.48,5 GPa: ={}".format(dV293))
    # m, sigma_m, dm, t, sigma_t, dt = lin_reg(p_exp250, V250, dy=np.zeros(3), sigma_y=dV250, plot=False)
    # print("V250: m={}+-{}, t={}+-{}".format(m, sigma_m, t, sigma_t))
    # print("V for EuGa4 for T=250 K and p = 2,3.48,5 GPa: ={}".format(V250))
    # print("dV for EuGa4 for T=250 K and p = 2,3.48,5 GPa: ={}".format(dV250))


    # T_v = np.array([250, 293])
    # fig, ax1 = plt.subplots()
    # fig.subplots_adjust(right=0.75)
    # tkw = dict(size=4, width=1.5)
    # ax1.set_xlabel(r'$T$(K)', fontsize=30)
    # ax1.set_ylabel(r'$V$(Å$^3)$', color='k', fontsize=30)
    # ax1.errorbar(T_v, np.array([205.16973635, 207.05578082]), yerr=np.array([0.02807694, 0.03461056]), color='y',
    #              linestyle='', marker='x', markersize=12, capsize=4, label=r'$0$ GPa temp. set')
    # ax1.errorbar(T_v[1], np.array([215.94118944]), yerr=np.array([1.13322613]), color='orange',
    #              linestyle='', marker='s', markersize=12, capsize=4, label=r'$0$ GPa')
    # ax1.errorbar(T_v, np.array([191.27381252, 194.07896761]), yerr=np.array([0.46704956, 0.68850187]), color='r',
    #              linestyle='', marker='s', markersize=12, capsize=4, label=r'$2$ GPa')
    # ax1.errorbar(T_v, np.array([193.02617522, 188.02990992]), yerr=np.array([0.91891691, 0.68850187]), color='m',
    #              linestyle='', marker='s', markersize=12, capsize=4, label=r'$3.48$ GPa')
    # ax1.errorbar(T_v, np.array([187.72215783, 188.4135331]), yerr=np.array([0.69290614, 0.63258988]), color='purple',
    #              linestyle='', marker='s', markersize=12, capsize=4, label=r'$5$ GPa')
    # ax1.tick_params(axis='y', labelcolor='k', labelsize=30)
    # ax1.set_xticks([250, 293])
    # ax1.tick_params(axis='x', labelsize=30, direction='in', **tkw)
    # ax1.locator_params(axis='y', nbins=3)
    # ax1.legend(fontsize=18, loc='upper center')
    # plt.tight_layout()
    # # plt.savefig("DAC_V_comparison.jpg", dpi=300)
    # plt.show()


if __name__ == '__main__':
    main()