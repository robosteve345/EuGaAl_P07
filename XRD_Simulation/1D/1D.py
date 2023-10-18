import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.pyplot import figure
import matplotlib
matplotlib.rcParams['font.family'] = "sans-serif"
# matplotlib.rc('text', usetex=True)

"""Simulate 1D monoatomic chain structure factor """
def oneDstructurefactor(Q, a, N):
    F = np.sin(N*Q*a/2) / np.sin(Q*a/2)
    I = np.absolute(F)**2
    return F, I


Q = np.linspace(-5*np.pi, 5*np.pi,10000)
a=1


for i in [1, 2, 10, 100, 1000]:
    # figure(figsize=(13, 7))
    plt.subplot(1,2,1)
    plt.plot(Q, oneDstructurefactor(Q, a, i)[0] / np.max(oneDstructurefactor(Q, a, i)[0]), label='N={}'.format(i))
    plt.xlabel('Q')
    plt.ylabel(r'$F(Q)/F_{max}$')
    plt.title(r'$F(Q)=\frac{\sin(NQa/2)}{\sin(NQa/2)}$')
    plt.subplot(1, 2, 2)
    plt.plot(Q, oneDstructurefactor(Q, a, i)[1] / np.max(oneDstructurefactor(Q, a, i)[1]), label='N={}'.format(i))
    plt.xlabel('Q')
    plt.ylabel(r'$I(Q)/I_{max}$')
    plt.title(r'$I(Q)=\left(\frac{\sin(NQa/2)}{\sin(NQa/2)}\right)^2$')
    plt.subplots_adjust(wspace=0.3)

plt.legend()
plt.savefig('1D_monoatomic_chain.jpg', dpi=300)
plt.show()