import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
tkw = dict(size=4, width=1.5)

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
    return opt, cov


def main():
    T_sensor = [293, 293, 293, 293, 292.549, 292.594, 276.886, 276.886, 269.895, 269.895, 261.280, 261.280, 254.412, 254.412,
                248.629, 248.629, 239.749, 239.729, 228.795, 228.795, 228.795, 204.019, 204.019, 204.019, 196.376, 196.376, 181.817, 181.817, 174.102, 174.102,
                159.787, 159.787, 148.976, 148.976, 140.085, 140.085, 121.159, 121.159, 110.627, 110.627,
                89.725, 89.725, 72.537, 73.537, 54.727, 54.727, 26.171, 26.171, 15.072, 15.072, 13.206, 13.206, 13.106,
                13.106]
    R1, dR1, R2, dR2 = np.loadtxt("Fulldata.txt", usecols=(0, 1, 4, 5), unpack=True)
    # A1, dA1, A2, dA2, FWHM1, dFWHM1, FWHM2, dFWHM2, R2, dR2, R1, dR1, B, dB = [], [], [], [], [], [], [], [], [], [], [], [], [], []
    # spectra = [spectrum for spectrum in os.listdir() if spectrum.endswith("1aqq.txt")]
    # sorted_spectra = sorted(spectra)
    # T_sensor = []
    # T_weinstein = []
    # for spectrum in sorted_spectra:
    #     x, y = np.loadtxt(spectrum, skiprows=3, delimiter=",", usecols=(0, 1), unpack=True)
    #     initconditions = [np.max(y) / 3, np.max(y), 0.5, 0.5, 0.01 * np.mean(y), x[np.argmax(y)] - 1.5, x[np.argmax(y)]]
    #     fittingtool(x, y, initconditions, spectrum)
    #     fitparametereingabe = input("Initial guesses für Fit ändern? (y/n): ")
    #     if fitparametereingabe == "y":
    #         initconditions[5] = float(input("R2 = "))
    #         initconditions[6] = float(input("R1 = "))
    #         opt, cov = fittingtool(x, y, initconditions, spectrum)
    #         FWHM2.append(2*np.sqrt(2*np.log(2))*opt[3]), dFWHM2.append(np.sqrt(np.diag(cov))[3]), R1.append(opt[6]), \
    #         dR1.append(np.sqrt(np.diag(cov))[6]), FWHM1.append(2*np.sqrt(2*np.log(2))*opt[2]), dFWHM1.append(np.sqrt(np.diag(cov))[2]), \
    #         R2.append(opt[5]), dR2.append(np.sqrt(np.diag(cov))[5]), A1.append(opt[0]), dA1.append(np.sqrt(np.diag(cov))[0]), \
    #         A2.append(opt[1]), dA2.append(np.sqrt(np.diag(cov))[1]), \
    #         B.append(opt[4]), dB.append(np.sqrt(np.diag(cov))[4]), T_weinstein.append(weinstein_temp(opt[0], opt[1],
    #                                                                                                 opt[6], opt[5]))
    #     else:
    #         opt, cov = fittingtool(x, y, initconditions, spectrum)
    #         FWHM2.append(2 * np.sqrt(2 * np.log(2)) * opt[3]), dFWHM2.append(np.sqrt(np.diag(cov))[3]), R1.append(
    #             opt[6]), \
    #         dR1.append(np.sqrt(np.diag(cov))[6]), FWHM1.append(2 * np.sqrt(2 * np.log(2)) * opt[2]), dFWHM1.append(
    #             np.sqrt(np.diag(cov))[2]), \
    #         R2.append(opt[5]), dR2.append(np.sqrt(np.diag(cov))[5]), A1.append(opt[0]), dA1.append(
    #             np.sqrt(np.diag(cov))[0]), \
    #         A2.append(opt[1]), dA2.append(np.sqrt(np.diag(cov))[1]),
    #         B.append(opt[4]), dB.append(np.sqrt(np.diag(cov))[4]), T_weinstein.append(weinstein_temp(opt[0], opt[1],
    #                                                                                                 opt[6], opt[5]))
    #     temperatureingabe = input("Temperaturpunkt vorhanden? (y/n)")
    #     if temperatureingabe == "y":
    #         t = float(input("T = ")) # in Kelvin
    #         T_sensor.append(t)
    #     else:
    #         pass
    # print("R2 = {}".format(R2))
    # print(T_sensor, T_weinstein)
    # speicherueberpruefung = input("Datensatz speichern? (y/n): ")
    # if speicherueberpruefung == "y":
    #     np.savetxt(r"/Users/stevengebel/PycharmProjects/Eu(Ga,Al)4 @ BSc&DESY P07/Rubyspecanalysis/Fulldata.txt",
    #                np.transpose([R1, dR1, FWHM2, dFWHM2, R2, dR2, FWHM1, dFWHM1, A1, dA1, A2, dA2, B, dB]),
    #                header='T_sensor(K) T_weinstein  R1  dR1 FWHM_R1  dFWHM_R1 R2 dR2 FWHM_R2  dFWHM_R2 A1  dA1 A2  dA2  B  dB')
    # else:
    #     pass
    # print(T_sensor, R1, dR1, FWHM1, dFWHM2, T_weinstein, T_sensor)
    print(len(T_sensor), len(R1))
    T_vos, R1_vos = np.loadtxt("rubyreference_vos.txt", usecols = (0, 1), unpack = True)
    plt.errorbar(T_sensor, R1, yerr=dR1, c='b', label='R1', marker='.', ls='', capsize=3)
    # plt.errorbar(T_sensor, R2, yerr=dR2, c='g', label='R2', marker='.', ls='', capsize=3)
    plt.plot(T_vos, R1_vos, c='k', label='Reference: Vos et al.', marker='.', ls='')
    plt.xlabel("T$_{sensor}$(K)")
    plt.ylabel(r"R$_{1}$(nm)")
    plt.legend()
    plt.savefig("cooldown2.jpg", dpi=500)
    plt.show()
if __name__ == '__main__':
    main()