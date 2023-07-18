
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
plt.rcParams["figure.autolayout"] = True
#plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1.50
A = 6  # Want figures to be A6
plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)])


def secondorderphasetransition(x, T_c, c, a):
    return (1-(x/T_c)**2)**(1/a) + c


def main():
    #################################
    """x=0.50"""
    #################################
    x0p50, y0p50 = np.loadtxt("ga2al2.txt", unpack=True, usecols=(0, 1), skiprows=0)
    opt0p50, cov0p50 = curve_fit(secondorderphasetransition, x0p50[40:130], y0p50[40:130]/np.max(y0p50), p0=[55, 1, 2])
    print("FIT FOR X=0.50:")
    print("T_c = ({}+-{})K".format(opt0p50[0], np.sqrt(np.diag(cov0p50))[0]))
    print("c = ({}+-{})K".format(opt0p50[1], np.sqrt(np.diag(cov0p50))[1]))
    print("a = ({}+-{})K".format(opt0p50[2], np.sqrt(np.diag(cov0p50))[2]))
    plt.plot(x0p50[40:130], secondorderphasetransition(x0p50[40:130], *opt0p50), ls='--', lw=2, c='k')
    plt.scatter(x0p50, y0p50 / np.max(y0p50), label='x=0.50', marker='.')
    #################################
    """x=0.58"""
    #################################
    x0p58, y0p58 = np.loadtxt("gaal_058.txt", unpack=True, usecols=(0, 1), skiprows=0)
    opt0p58, cov0p58 = curve_fit(secondorderphasetransition, x0p58[100:219], y0p58[100:219] / np.max(y0p58),
                                 p0=[60, 1, 2])
    print("FIT FOR X=0.58:")
    print("T_c = ({}+-{})K".format(opt0p58[0], np.sqrt(np.diag(cov0p58))[0]))
    print("c = ({}+-{})K".format(opt0p58[1], np.sqrt(np.diag(cov0p58))[1]))
    print("a = ({}+-{})K".format(opt0p58[2], np.sqrt(np.diag(cov0p58))[2]))
    plt.plot(x0p58[100:219], secondorderphasetransition(x0p58[100:219], *opt0p58), ls='--', lw=2, c='k')
    plt.scatter(x0p58, y0p58/np.max(y0p58), label='x=0.58', marker='.')
    #################################
    """x=0.9"""
    #################################
    x0p9, y0p9 = np.loadtxt("gaal_0p9.txt", unpack=True, usecols=(0, 1), skiprows=0)
    opt0p9, cov0p9 = curve_fit(secondorderphasetransition, x0p9[30:170], y0p9[30:170] / np.max(y0p9),
                                 p0=[70, 1, 2])
    print("FIT FOR X=0.9:")
    print("T_c = ({}+-{})K".format(opt0p9[0], np.sqrt(np.diag(cov0p9))[0]))
    print("c = ({}+-{})K".format(opt0p9[1], np.sqrt(np.diag(cov0p9))[1]))
    print("a = ({}+-{})K".format(opt0p9[2], np.sqrt(np.diag(cov0p9))[2]))
    plt.plot(x0p9[30:170], secondorderphasetransition(x0p9[30:170], *opt0p9), ls='--', lw=2, c='k')
    plt.scatter(x0p9, y0p9/np.max(y0p9), label='x=0.9', marker='.')
    plt.tick_params(bottom=True, top=True, left=True, right=True)
    plt.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    plt.tick_params(axis='both', labelcolor='k', labelsize=10,
                    size=5, direction='in', length=15, width=1.5)
    plt.ylabel(r"\textbf{Intensity (a.u.)}")
    plt.xlabel(r"\textbf{T(K)}")
    plt.legend()
    plt.show()
if __name__ == '__main__':
    main()