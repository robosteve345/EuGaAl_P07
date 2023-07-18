import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.pyplot import figure
import matplotlib
matplotlib.rcParams['font.family'] = "sans-serif"


"""Modulation of monoatomic chain
x_n = n*a + cos(q*a*n*Q)
F(Q) = sum(exp(-i*Q*x_n))_n = sum(exp(-i*Q*n*a) * exp(-i*Q*u*cos(q*n*a)))_n"""
# For N->+inf, Intensities at Q-values with Q_0+-q should appear, in this case Q_0=1, Q values in 2*pi/a
u = 0.2 # Modulation amplitude
q = 0.1 # Modulation vector
a = 1 # Lattice spacing
deltaQ = 0.1 # Q-value spacing
Qmax = 1 # maximum Q-value
Q = np.arange(0, Qmax + deltaQ, deltaQ)
for N in [1, 10, 100, 1000, 2000, 5000]:
    print("F(Q)-values for N={}".format(N))
    f0, f01, f02, f03, f04, f10, f101, f102, f103, f104, f05, f06, f07, f08, f09, f105, f106, f107, f108, f109, f20\
        = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    f = [f0, f01, f02, f03, f04, f05, f06, f07, f08, f09, f10, f101, f102, f103, f104, f105, f106, f107, f108, f109, f20]  # Structure factor
    f_sum = [[] for i in range(21)]  # summarized structure factor for Q-value over N unit cells
    I = []  # Intensity
    # evaluate for N atoms
    print("f = {}".format(f))
    for i in range(0, N + 1):
        f10.append(
            np.exp(-1j * 2 * np.pi * i * a * 1.0) * np.exp(-1j * u * 2 * np.pi * 1.0 * np.cos(q * i * 2 * np.pi * a)))
        f101.append(
            np.exp(-1j * 2 * np.pi * i * a * 1.1) * np.exp(-1j * u * 2 * np.pi * 1.1 * np.cos(q * i * 2 * np.pi * a)))
        f102.append(
            np.exp(-1j * 2 * np.pi * i * a * 1.2) * np.exp(-1j * u * 2 * np.pi * 1.2 * np.cos(q * i * 2 * np.pi * a)))
        f103.append(
            np.exp(-1j * 2 * np.pi * i * a * 1.3) * np.exp(-1j * u * 2 * np.pi * 1.3 * np.cos(q * i * 2 * np.pi * a)))
        f104.append(
            np.exp(-1j * 2 * np.pi * i * a * 1.4) * np.exp(-1j * u * 2 * np.pi * 1.4 * np.cos(q * i * 2 * np.pi * a)))
        f05.append(
            np.exp(-1j * 2 * np.pi * i * a * 0.5) * np.exp(-1j * u * 2 * np.pi * 0.5 * np.cos(q * i * 2 * np.pi * a)))
        f06.append(
            np.exp(-1j * 2 * np.pi * i * a * 0.6) * np.exp(-1j * u * 2 * np.pi * 0.6 * np.cos(q * i * 2 * np.pi * a)))
        f07.append(
            np.exp(-1j * 2 * np.pi * i * a * 0.7) * np.exp(-1j * u * 2 * np.pi * 0.7 * np.cos(q * i * 2 * np.pi * a)))
        f08.append(
            np.exp(-1j * 2 * np.pi * i * a * 0.8) * np.exp(-1j * u * 2 * np.pi * 0.8 * np.cos(q * i * 2 * np.pi * a)))
        f09.append(
            np.exp(-1j * 2 * np.pi * i * a * 0.9) * np.exp(-1j * u * 2 * np.pi * 0.9 * np.cos(q * i * 2 * np.pi * a)))
        f0.append(
            np.exp(-1j * 2 * np.pi * i * a * 0) * np.exp(-1j * u * 2 * np.pi * 0 * np.cos(q * i * 2 * np.pi * a)))
        f01.append(
            np.exp(-1j * 2 * np.pi * i * a * 0.1) * np.exp(-1j * u * 2 * np.pi * 0.1 * np.cos(q * i * 2 * np.pi * a)))
        f02.append(
            np.exp(-1j * 2 * np.pi * i * a * 0.2) * np.exp(-1j * u * 2 * np.pi * 0.2 * np.cos(q * i * 2 * np.pi * a)))
        f03.append(
            np.exp(-1j * 2 * np.pi * i * a * 0.3) * np.exp(-1j * u * 2 * np.pi * 0.3 * np.cos(q * i * 2 * np.pi * a)))
        f04.append(
            np.exp(-1j * 2 * np.pi * i * a * 0.4) * np.exp(-1j * u * 2 * np.pi * 0.4 * np.cos(q * i * 2 * np.pi * a)))
        f105.append(
            np.exp(-1j * 2 * np.pi * i * a * 1.5) * np.exp(-1j * u * 2 * np.pi * 1.5 * np.cos(q * i * 2 * np.pi * a)))
        f106.append(
            np.exp(-1j * 2 * np.pi * i * a * 1.6) * np.exp(-1j * u * 2 * np.pi * 1.6 * np.cos(q * i * 2 * np.pi * a)))
        f107.append(
            np.exp(-1j * 2 * np.pi * i * a * 1.7) * np.exp(-1j * u * 2 * np.pi * 1.7 * np.cos(q * i * 2 * np.pi * a)))
        f108.append(
            np.exp(-1j * 2 * np.pi * i * a * 1.8) * np.exp(-1j * u * 2 * np.pi * 1.8 * np.cos(q * i * 2 * np.pi * a)))
        f109.append(
            np.exp(-1j * 2 * np.pi * i * a * 1.9) * np.exp(-1j * u * 2 * np.pi * 1.9 * np.cos(q * i * 2 * np.pi * a)))
        f20.append(
            np.exp(-1j * 2 * np.pi * i * a * 2.0) * np.exp(-1j * u * 2 * np.pi * 2.0 * np.cos(q * i * 2 * np.pi * a)))
    for i in range(len(f)):
        print("sum f = {}".format(np.sum(f[i])))
        f_sum.append(np.sum(f[i]))
        print("I = {}".format(np.abs(np.sum(f[i])) ** 2))
        I.append(np.abs(np.sum(f[i])) ** 2)
        print("I={}".format(I))
    print("I0={}".format(np.abs(np.sum(f0)) ** 2))
    print("I01={}".format(np.abs(np.sum(f01)) ** 2))
    print("I02={}".format(np.abs(np.sum(f02)) ** 2))
    print("I03={}".format(np.abs(np.sum(f03)) ** 2))
    print("I04={}".format(np.abs(np.sum(f04)) ** 2))
    print("I05={}".format(np.abs(np.sum(f05)) ** 2))
    print("I06={}".format(np.abs(np.sum(f06)) ** 2))
    print("I07={}".format(np.abs(np.sum(f07)) ** 2))
    print("I08={}".format(np.abs(np.sum(f08)) ** 2))
    print("I09={}".format(np.abs(np.sum(f09)) ** 2))
    print("I10={}".format(np.abs(np.sum(f10)) ** 2))
    print("I101={}".format(np.abs(np.sum(f101)) ** 2))
    print("I102={}".format(np.abs(np.sum(f102)) ** 2))
    print("I103={}".format(np.abs(np.sum(f103)) ** 2))
    print("I104={}".format(np.abs(np.sum(f104)) ** 2))
    print("I105={}".format(np.abs(np.sum(f105)) ** 2))
    print("I106={}".format(np.abs(np.sum(f106)) ** 2))
    print("I107={}".format(np.abs(np.sum(f107)) ** 2))
    print("I108={}".format(np.abs(np.sum(f108)) ** 2))
    print("I109={}".format(np.abs(np.sum(f109)) ** 2))
    print("I20={}".format(np.abs(np.sum(f20)) ** 2))
    plt.plot(np.arange(0.0, 2.0 + 0.1, 0.1), I, ls='', marker='x', label='N={}, q={}'.format(N, q))
    plt.xlabel("Q(rlu)")
    plt.ylabel("Intensity I")
    plt.legend()
    plt.savefig('1D_monoatomic_modulated_chain_N={}_q={}.jpg'.format(N, q), dpi=300)
    plt.show()


def main():
    print(__doc__)
    u = 0.1 # Modulation amplitude
    q = 0.2 # Modulation vector
    a = 1 # Lattice spacing
    Q = np.linspace(0, 1, 10) # k-space interval
    # N_list = [1, 2, 10, 100, 1000]
    # for i in N_list:
    #     plt.subplot(1, 2, 1)
    #     plt.plot(Q, oneDstructurefactor(Q, a, i, u, q)[0] / np.max(oneDstructurefactor(Q, a, i, u, q)[0]), label='N={}'.format(i))
    #     plt.xlabel('Q')
    #     plt.ylabel(r'$F(Q)/F_{max}$')
    #     plt.title(r'$F(Q)=\frac{\sin(NQa/2)}{\sin(NQa/2)}$')
    #     plt.subplot(1, 2, 2)
    #     plt.plot(Q, oneDstructurefactor(Q, a, i, u, q)[1] / np.max(oneDstructurefactor(Q, a, i, u, q)[1]), label='N={}'.format(i))
    #     plt.xlabel(r'$Q(2\np.pi/2)')
    #     plt.ylabel(r'$I(Q)/I_{max}$')
    #     plt.title(r'$I(Q)=\left(\frac{\sin(NQa/2)}{\sin(NQa/2)}\right)^2$')
    #     plt.subplots_adjust(wspace=0.3)
    # plt.legend()
    # plt.savefig('1D_monoatomic_modulated_chain.jpg', dpi=300)
    # plt.show()

if __name__ == '__main__':
    main()
