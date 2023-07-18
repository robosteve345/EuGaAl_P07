import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
tkw = dict(size=4, width=1.5)
#shift = [0.19444444444444442, 0.8333333333333333, 1.3888888888888888]
#p = [0.5652173913043479, 2.217391304347826, 3.6956521739130435]
m = 2.76 # (p[-1] - p[0]) / (shift[-1] - shift[0])


def pressurecalc(R1_inside, R1_outside):
    # Reference Yamaoka et al. T=77K
    m = 2.76 # (p[-1] - p[0]) / (shift[-1] - shift[0])
    print(m)  # in GPa/nm
    measuredshift = R1_inside - R1_outside # in nm
    p = m * measuredshift
    plt.plot(np.linspace(0,8, 1000), m*np.linspace(0,8,1000), c='k', lw=0.6, label=r'Calibration by ref.2')
    plt.xlim(0,8.1)
    plt.xlabel(r"$\Delta$R$_1$ (nm)")
    plt.ylabel(r"p(GPa)")
    plt.ylim(np.min(y), np.max(y))
    plt.legend(title=r'R$_1$=({}$\pm${})nm'.format(np.round(R1, 3), np.round(dR1, 3)))
    plt.show()
    print("p={}GPa".format(p))
    return m*measuredshift

