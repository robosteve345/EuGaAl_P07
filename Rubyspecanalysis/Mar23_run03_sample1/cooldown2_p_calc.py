import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
tkw = dict(size=4, width=1.5)
R1, dR1, R2, dR2 = np.loadtxt("../Warming/Fulldata.txt", usecols=(0, 1, 4, 5), unpack=True)
inside = R1[4:-2:2]
outside = R1[5:-1:2]
shift = inside-outside
# print(shift, len(shift))

def weinstein_temp(I1, I2):
    """Reference: Weinstein, boltzman distribution"""
    kB = 1.380649*1e-23 #m^2 kg s^-2 K^-1
    h = 6.62607015*1e-34 # m^2kg s^-1
    eta = 0.65
    Delta = 3.55 # meV
    T_weinstein = -(Delta*1e-3*1.6e-19)/(kB*np.log(I2/(eta*I1)))
    return T_weinstein

I2, I1, dI2, dI1 = np.loadtxt("../Warming/Fulldata.txt", skiprows=3, usecols=(10, 8, 9, 11), unpack=True)
print(weinstein_temp(I2, I1))
# # handpickedshift = [6.945284910105727931e+02 - 6.942060530393470117e+02, 6.945251100874767189e+02 - 6.941935450338256715e+02,
# #                    0.31688884184211474, 0.28838359268399927, 0.2802946937700881, 0.28383254913558176, 0.30492215041113013,
# #                    0.29692343909584906, 0.2828970472567107, 0.2667512118067634, 0.21516184767085633]
# # shift: disregard first 4 sets (misordered: inside day1, inside day2, outside day1, outside day2...=
# plt.errorbar(np.abs(np.asarray(shift)), 2.76*np.asarray(shift), c='g', label='Data with fit to Nakano 77K', marker='s', ls='')
# plt.errorbar(np.abs(np.asarray(shift)), 2.74*np.asarray(shift), c='k', label='Data with fit to Nakano 10K', marker='s', ls='')
# plt.errorbar(np.linspace(0, max(shift), 1000), 2.76*np.linspace(0, max(shift), 1000), c='g', ls='-.')
# plt.errorbar(np.linspace(0, max(shift), 1000), 2.74*np.linspace(0, max(shift), 1000), c='k', ls='--')
# plt.xlabel(r"$\Delta$R$_1$(nm)")
# plt.ylabel(r"p(GPa)")
# plt.ylim(0.3)
# plt.xlim(0.15)
# plt.legend()
# # plt.savefig("p_cooldown2.jpg", dpi=500)
# plt.show()
# ########################
# #Second plot
# ########################
# T_sensor = [293, 293, 293, 293, 292.549, 292.594, 276.886, 276.886, 269.895, 269.895, 261.280, 261.280, 254.412, 254.412,
#                 248.629, 248.629, 239.749, 239.729, 228.795, 228.795, 228.795, 204.019, 204.019, 204.019, 196.376, 196.376, 181.817, 181.817, 174.102, 174.102,
#                 159.787, 159.787, 148.976, 148.976, 140.085, 140.085, 121.159, 121.159, 110.627, 110.627,
#                 89.725, 89.725, 72.537, 73.537, 54.727, 54.727, 26.171, 26.171, 15.072, 15.072, 13.206, 13.206, 13.106,
#                 13.106]
T_weinstein = [  440.77481296, -3248.95288409,   353.26156603,   614.60145951,
  -459.54150478,   933.97520218, -1441.71791393 ,  588.53068148,
 16084.22580268,   490.53925317 , -787.89728214 ,  549.61512186,
 -3525.00207789,   426.64724797 ,-2837.45120482 ,  442.33299122,
   734.9033345 ,   338.10972182 ,  789.36651783 ,  496.10454694,
   217.84875066,   289.65931203 ,  482.09733606 ,  218.06608007,
   225.50776115,   208.84634065 ,  182.64949638 ,  199.76967788,
   161.87816024,   197.89611111 ,  126.31050308 ,  193.98679932,
   121.20715961,   182.16626713 ,  109.80836875 ,  176.36701088,
   116.42970377,   157.14802419 ,  103.50370674 ,  145.56693774,
    97.81417698,   134.10027836 ,   91.95672486 ,  128.42342938,
    83.66571039,   108.83168122 ,   85.6965076  ,   97.89147945,
    89.06967464,   93.95627692  ,  87.45038939   , 91.28676167,
    57.54590304,    65.53269803]
plt.plot(T_weinstein[4:-2:2], 2.76*np.asarray(shift), ls='', c='k', marker='s')
plt.xlabel(r"T$_{weinstein}$")
plt.ylabel(r"p(GPa)")
plt.ylim(0.4)
plt.xlim(0,300)
plt.savefig("pvsT_cooldown2.jpg", dpi=500)
plt.show()
