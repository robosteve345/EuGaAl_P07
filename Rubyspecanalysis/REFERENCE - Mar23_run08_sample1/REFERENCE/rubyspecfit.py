import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')
tkw = dict(size=4, width=1.5)

def pressurecalc(R1_inside, R1_outside, dR1_inside):
    # Reference Yamaoka et al. T=77K
    m = 2.76 # (p[-1] - p[0]) / (shift[-1] - shift[0])
    print(m)  # in GPa/nm
    measuredshift = R1_inside - R1_outside # in nm
    p = m * measuredshift
    plt.plot(np.linspace(0,8, 1000), m*np.linspace(0,8,1000), c='k', lw=0.6, label=r'Calibration by ref.2')
    plt.xlim(0,8.1)
    plt.xlabel(r"$\Delta$R$_1$ (nm)")
    plt.ylabel(r"p(GPa)")
    plt.ylim(np.min(p), np.max(p))
    plt.legend(title=r'R$_1$=({}$\pm${})nm'.format(np.round(R1_inside, 3), np.round(dR1_inside, 3)))
    plt.show()
    print("p={}GPa".format(p))
    return m*measuredshift


def weinstein_temp(I1, I2):
    """Reference: Weinstein, boltzman distribution"""
    kB = 1.380649*1e-23 #m^2 kg s^-2 K^-1
    h = 6.62607015*1e-34 # m^2kg s^-1
    eta = 0.65
    Delta = 3.55 # meV
    T_weinstein = -(Delta*1e-3*1.6e-19)/(kB*np.log(I2/(eta*I1)))
    return T_weinstein


def doublegaussianfit(x, A1, A2, sigma1, sigma2, B, x01, x02):
    return A1 * np.exp(-(x - x01) ** 2 / (2 * sigma1 ** 2)) + A2 * np.exp(-(x - x02) ** 2 / (2 * sigma2 ** 2)) + B


def fittingtool(x, y, initconditions, filename):
    """Fit Ruby-fluorescence spectra to double-gaussian, determine fit-data and plot"""
    opt, cov = curve_fit(doublegaussianfit, x, y, p0=initconditions)
    A1, dA1, A2, dA2, FWHM1, dFWHM1, FWHM2, dFWHM2, B, dB, R2, dR2, R1, dR1 =  opt[0], np.sqrt(np.diag(cov))[0], \
    opt[1], np.sqrt(np.diag(cov))[1], opt[2], np.sqrt(np.diag(cov))[2], 2*np.sqrt(np.log(2))*opt[3], np.sqrt(np.diag(cov))[3], \
    opt[4], np.sqrt(np.diag(cov))[4], opt[5], np.sqrt(np.diag(cov))[5], opt[6], np.sqrt(np.diag(cov))[6]
    print("FIT TO FILE: {}".format(filename))
    print("A1 = ({} +- {})a.u.".format(A1, dA1))
    print("A2 = ({} +- {})a.u.".format(A2, dA2))
    print("FWHM1_R2 = ({} +- {})nm".format(np.round(FWHM1, 3), np.round(dFWHM1, 3)))
    print("R1 = ({} +- {})nm".format(np.round(R1, 3), np.round(dR1, 3)))
    print("FWHM_R1 = ({} +- {})nm".format(np.round(FWHM2, 3), np.round(dFWHM2, 3)))
    print("R2 = ({} +- {})nm".format(np.round(R2, 3), np.round(dR2, 3)))
    print("Mean background = ({} +- {})a.u.".format(B, dB))
    plt.plot(x, y, c='r', lw=0.6, label='{}'.format(filename))
    plt.xlim(685, 700)
    plt.xlabel(r"$\lambda$ (nm)")
    plt.ylabel(r"Intensity (a.u.)")
    plt.ylim(np.min(y), np.max(y))
    plt.plot(np.linspace(opt[5] - 10, opt[6] + 10, 1000), doublegaussianfit(np.linspace(opt[5] - 10, opt[6] + 10, 1000), *opt),
             c='b', lw=1.5, ls='--')
    plt.legend(title=r'R$_1$=({}$\pm${})nm'.format(np.round(R1, 3), np.round(dR1, 3)))
    plt.show()
    T_weinstein = weinstein_temp(A2, A1)
    print("T_weinstein={}".format(T_weinstein))
    return opt, cov, T_weinstein


def main():
    A1, dA1, A2, dA2, FWHM1, dFWHM1, FWHM2, dFWHM2, R2, dR2, R1, dR1, B, dB = [], [], [], [], [], [], [], [], [], [], [], [], [], []
    spectra = [spectrum for spectrum in os.listdir() if spectrum.endswith(".txt")]
    sorted_spectra = sorted(spectra)
    # T_sensor = []
    T_weinstein = []
    for spectrum in sorted_spectra:
        x, y = np.loadtxt(spectrum, skiprows=3, delimiter=",", usecols=(0, 1), unpack=True)
        initconditions = [np.max(y) / 3, np.max(y), 0.5, 0.5, 0.01 * np.mean(y), x[np.argmax(y)] - 1.5, x[np.argmax(y)]]
        fittingtool(x, y, initconditions, spectrum)
        fitparametereingabe = input("Initial guesses für Fit ändern? (y/n): ")
        if fitparametereingabe == "y":
            initconditions[5] = float(input("R2 = "))
            initconditions[6] = float(input("R1 = "))
            opt, cov, t_weinstein = fittingtool(x, y, initconditions, spectrum)
            FWHM2.append(2*np.sqrt(2*np.log(2))*opt[3]), dFWHM2.append(np.sqrt(np.diag(cov))[3]), R1.append(opt[6]), \
            dR1.append(np.sqrt(np.diag(cov))[6]), FWHM1.append(2*np.sqrt(2*np.log(2))*opt[2]), dFWHM1.append(np.sqrt(np.diag(cov))[2]), \
            R2.append(opt[5]), dR2.append(np.sqrt(np.diag(cov))[5]), A1.append(opt[0]), dA1.append(np.sqrt(np.diag(cov))[0]), \
            A2.append(opt[1]), dA2.append(np.sqrt(np.diag(cov))[1]), \
            B.append(opt[4]), dB.append(np.sqrt(np.diag(cov))[4]),
            T_weinstein.append(t_weinstein)
        else:
            opt, cov, t_weinstein = fittingtool(x, y, initconditions, spectrum)
            FWHM2.append(2 * np.sqrt(2 * np.log(2)) * opt[3]), dFWHM2.append(np.sqrt(np.diag(cov))[3]), R1.append(
                opt[6]), \
            dR1.append(np.sqrt(np.diag(cov))[6]), FWHM1.append(2 * np.sqrt(2 * np.log(2)) * opt[2]), dFWHM1.append(
                np.sqrt(np.diag(cov))[2]), \
            R2.append(opt[5]), dR2.append(np.sqrt(np.diag(cov))[5]), A1.append(opt[0]), dA1.append(
                np.sqrt(np.diag(cov))[0]), \
            A2.append(opt[1]), dA2.append(np.sqrt(np.diag(cov))[1]),
            B.append(opt[4]), dB.append(np.sqrt(np.diag(cov))[4]),
            T_weinstein.append(t_weinstein)
if __name__ == '__main__':
    main()