import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.pyplot import figure
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rc('text', usetex=True)

"""TO DO
-Implement deltak possibilities for 0.1&0.01 in L for q_cdw=0.2, 0.1 & 0.5
-Implement gaussian distribution on every desired (K,L)-value
-Implement possibility for HKL_0 & HK_0L maps
-Implement input-interface: kß, lß, deltak, Amplitude, q_cdw, kmax
-IDEAL: fourier series for the modulations
-Implement DBW & Q-dependent atomic form factors (copy from main.py)"""
def ztranslation(x, storage_x, storage_y, storage_z, i):
    """Translate atomic positions one given direction x,y,z
    """
    x_transl = x + np.array([0, 0, i])
    storage_x.append(x_transl[0])
    storage_y.append(x_transl[1])
    storage_z.append(x_transl[2])


def xtranslation(x, storage_x, storage_y, storage_z, i):
    """Translate atomic positions one given direction x,y,z
    """
    x_transl = x + np.array([i, 0, 0])
    storage_x.append(x_transl[0])
    storage_y.append(x_transl[1])
    storage_z.append(x_transl[2])


def ytranslation(x, storage_x, storage_y, storage_z, i):
    """Translate atomic positions one given direction x,y,z
    """
    x_transl = x + np.array([0, i, 0])
    storage_x.append(x_transl[0])
    storage_y.append(x_transl[1])
    storage_z.append(x_transl[2])


def kspacecreator(k0, l0, kmax, lmax, deltak):
    """K-space creator"""
    k = np.arange(k0-kmax, k0+kmax + deltak, deltak)
    l = np.arange(l0-lmax, l0+lmax + deltak, deltak)
    k2d, l2d = np.meshgrid(k, l)
    Unitary = np.ones((len(k2d), len(l2d)))  # Unitary matrix
    return k2d, l2d, k, Unitary


def structurefactorandplotting(k0, l0, k2d, k, kmax, lmax, l2d, h, deltak, Unitary, Amplitude, q_cdw, z0, N, noisefactor, f_list, savefig=False):
    for n in N:
        print("SIMULATION FOR N={} unmodulated unit cells / {} supercells".format(n, int(n/int(q_cdw**(-1)))))
        print("INVESTIGATED PEAK: [HKL] = [{}{}{}]".format(h, k0, l0))
        print("CDW Modulation A*sin(q_cdw*z*2*pi):")
        print("q_cdw={}, A={}".format(q_cdw, Amplitude))
        """EuGa2Al2 unit cell Wyckoff positions"""
        #  Europium:
        x_Eu, y_Eu, z_Eu = [0], [0], [0]
        x_Eu_T, y_Eu_T, z_Eu_T = [0.5], [0.5], [0.5]
        #  Aluminium:
        x_Al1, y_Al1, z_Al1 = [0], [0.5], [0.25]
        x_Al1_T, y_Al1_T, z_Al1_T = [0.5], [1], [0.75]
        x_Al2, y_Al2, z_Al2 = [0.5], [0], [0.25]
        x_Al2_T, y_Al2_T, z_Al2_T = [1], [0.5], [0.75]
        #  Gallium:
        x_Ga1, y_Ga1, z_Ga1 = [0], [0], [z0]
        x_Ga1_T, y_Ga1_T, z_Ga1_T = [0.5], [0.5], [0.5 + z0]
        x_Ga2, y_Ga2, z_Ga2 = [0], [0], [-z0 + 1]
        x_Ga2_T, y_Ga2_T, z_Ga2_T = [0.5], [0.5], [0.5 - z0]

        """Atomic positions modulation: commensurate with q_CDW = q*2*pi/a of atom 2 (1/2, 1/2, 1/2)"""
        # Full translation of Wyckoff 2a(Eu), 4d(Al) & 4e(Ga) for N unit cells
        for i in range(1, n):
            ztranslation(np.array([x_Eu[0], y_Eu[0], z_Eu[0]]), x_Eu, y_Eu, z_Eu, i)
            ztranslation(np.array([x_Eu_T[0], y_Eu_T[0], z_Eu_T[0]]), x_Eu_T, y_Eu_T, z_Eu_T, i)
            ztranslation(np.array([x_Al1[0], y_Al1[0], z_Al1[0]]), x_Al1, y_Al1, z_Al1, i)
            ztranslation(np.array([x_Al1_T[0], y_Al1_T[0], z_Al1_T[0]]), x_Al1_T, y_Al1_T, z_Al1_T, i)
            ztranslation(np.array([x_Al2[0], y_Al2[0], z_Al2[0]]), x_Al2, y_Al2, z_Al2, i)
            ztranslation(np.array([x_Al2_T[0], y_Al2_T[0], z_Al2_T[0]]), x_Al2_T, y_Al2_T, z_Al2_T, i)
            ztranslation(np.array([x_Ga1[0], y_Ga1[0], z_Ga1[0]]), x_Ga1, y_Ga1, z_Ga1, i)
            ztranslation(np.array([x_Ga1_T[0], y_Ga1_T[0], z_Ga1_T[0]]), x_Ga1_T, y_Ga1_T, z_Ga1_T, i)
            ztranslation(np.array([x_Ga2[0], y_Ga2[0], z_Ga2[0]]), x_Ga2, y_Ga2, z_Ga2, i)
            ztranslation(np.array([x_Ga2_T[0], y_Ga2_T[0], z_Ga2_T[0]]), x_Ga2_T, y_Ga2_T, z_Ga2_T, i)
        #  Modulation of Al
        for i in range(0, n):  # begins with 1 because it evaluates every unit cell
            x_Al1[i] = x_Al1[i] + Amplitude * np.cos(q_cdw * 2 * np.pi * i)
            # print("n={}".format(i))
        print("x_Modulation = {}".format(Amplitude * np.cos(q_cdw * 2 * np.pi * np.arange(0, n+1, 1))))
        #  Final positions
        Eu, Eu_T, Al1, Al1_T, Ga1, Ga1_T, Al2, Al2_T, Ga2, Ga2_T = np.array([x_Eu, y_Eu, z_Eu]), np.array([x_Eu_T, y_Eu_T, z_Eu_T]),\
                                       np.array([x_Al1, y_Al1, z_Al1]), np.array([x_Al1_T, y_Al1_T, z_Al1_T]),\
                                       np.array([x_Al2, y_Al2, z_Al2]), np.array([x_Al2_T, y_Al2_T, z_Al2_T]), \
                                       np.array([x_Ga1, y_Ga1, z_Ga1]), np.array([x_Ga1_T, y_Ga1_T, z_Ga1_T]), \
                                       np.array([x_Ga2, y_Ga2, z_Ga2]), np.array([x_Ga2_T, y_Ga2_T, z_Ga2_T]), \
            # print("Eu={}".format(Eu))
        # print("Al={}".format(Al))
        # print("Al={}".format(Al))
        """Scattering amplitudes F"""
        #  Crystal structure factor for each atom at each Wyckoff position with respect to the tI symmtry +1/2 transl.
        F_Eu, F_Eu_T, F_Al1, F_Al1_T, F_Al2, F_Al2_T, F_Ga1, F_Ga1_T, F_Ga2, F_Ga2_T = [], [], [], [], [], [], [], [], [], []
        for i in range(0, n):
            F_Eu.append(
                f_list[0] * np.exp(-2 * np.pi * 1j * (h * Unitary * Eu[0, i] + k2d * Eu[1, i] + l2d * Eu[2, i]))
            )
            F_Eu_T.append(
                f_list[0] * np.exp(-2 * np.pi * 1j * (h * Unitary * Eu_T[0, i] + k2d * Eu_T[1, i] + l2d * Eu_T[2, i]))
            )
            F_Al1.append(
                f_list[1] * np.exp(-2 * np.pi * 1j * (h * Unitary * Al1[0, i] + k2d * Al1[1, i] + l2d * Al1[2, i]))
            )
            F_Al1_T.append(
                f_list[1] * np.exp(-2 * np.pi * 1j * (h * Unitary * Al1_T[0, i] + k2d * Al1_T[1, i] + l2d * Al1_T[2, i]))
            )
            F_Al2.append(
                f_list[1] * np.exp(-2 * np.pi * 1j * (h * Unitary * Al2[0, i] + k2d * Al2[1, i] + l2d * Al2[2, i]))
            )
            F_Al2_T.append(
                f_list[1] * np.exp(
                    -2 * np.pi * 1j * (h * Unitary * Al2_T[0, i] + k2d * Al2_T[1, i] + l2d * Al2_T[2, i]))
            )
            F_Ga1.append(
                f_list[2] * np.exp(-2 * np.pi * 1j * (h * Unitary * Ga1[0, i] + k2d * Ga1[1, i] + l2d * Ga1[2, i]))
            )
            F_Ga1_T.append(
                f_list[2] * np.exp(-2 * np.pi * 1j * (h * Unitary * Ga1_T[0, i] + k2d * Ga1_T[1, i] + l2d * Ga1_T[2, i]))
            )
            F_Ga2.append(
                f_list[2] * np.exp(-2 * np.pi * 1j * (h * Unitary * Ga2[0, i] + k2d * Ga2[1, i] + l2d * Ga2[2, i]))
            )
            F_Ga2_T.append(
                f_list[2] * np.exp(-2 * np.pi * 1j * (h * Unitary * Ga2_T[0, i] + k2d * Ga2_T[1, i] + l2d * Ga2_T[2, i]))
            )
        F_init = np.zeros((len(k2d), len(k2d)), dtype=complex)  # quadratic 0-matrix with dimensions of k-space
        F_list = [F_Eu, F_Eu_T, F_Al1, F_Al1_T, F_Al2, F_Al2_T, F_Ga1, F_Ga1_T, F_Ga2, F_Ga2_T]
        pre_F_init = [np.zeros((len(k2d), len(k2d)))]
        #  Zusammenfügen der Formfaktoren für die Atomsorten
        for i in F_list:  # put together the lists in a ndarray for each atom with each N positions, to get rid of 3rd dimension (better workaround probably possible...)
            pre_F_init = np.add(pre_F_init, i)
        for i in range(len(pre_F_init)):
            F_init = F_init + pre_F_init[i]
        #  Intensity I
        I = np.abs(np.round(F_init, 3)) ** 2  # I \propto F(Q)^2, F complex
        ########################################################################
        #  Excluding unwanted kspace-points
        #  Table for extracting unwanted K,L points
        #  q_cdw     0.1   0.1    0.2   0.2    0.5   0.5
        #  ∆k        0.1   0.01   0.1   0.01   0.1   0.01
        #  Kfactor1  1     1      1
        #  Kfactor2  9     99     9
        #  Lfactor1  1     10     1
        #  Lfactor2  9     9      2
        Kfactor1, Kfactor2, Lfactor1, Lfactor2 = 1, 9, 1, 1
        # #  Excluding unallowed K-points (ONLY FOR deltak=/1)
        k_intlist = np.arange(0, len(k2d), 1)  # erstelle indices aller k-Werte
        # print("k_integer={}".format(k_intlist))
        for i in range(0, (2 * kmax*Kfactor1 + 1)):  # LEAVES ONLY INTEGER K-values
            # print(range(0,2*kmax+1))
            k_intlist = np.delete(k_intlist, i * Kfactor2)  #  n*9, since the list gets one less each time
            print("k_intlist={}".format(k_intlist))
        for i in k_intlist:  # Set unallowed K-values for intensities to 0
            I[:, i] = 0
        # #  Exluding unallowed L-points
        # l_intlist = np.arange(0, len(l2d), 1)  # erstelle indices aller l-Werte
        # for i in range(0, 2 * kmax * Lfactor1 + 1):
        #     l_intlist = np.delete(l_intlist, i * Lfactor2)  # Lösche jeden zehnten index
        #     print("l_intlist={}".format(l_intlist))
        # for i in l_intlist:  # Set unallowed L-values for intensities to 0
        #     I[i, :] = 0
        # # if deltak == 0.1:
        #     for i in range(0, 2 * kmax + 1):
        #         l_intlist = np.delete(l_intlist, i * Lfactor)  # Lösche jeden zehnten index
        #     for i in l_intlist:  # Set unallowed L-values for intensities to 0
        #         I[i, :] = 0
        # else:
        #     for i in range(0, 2 * kmax * 10 + 1):
        #         l_intlist = np.delete(l_intlist, i * Lfactor)  # Lösche jeden zehnten index
        #     for i in l_intlist:  # Set unallowed L-values for intensities to 0
        #         I[i, :] = 0
        ########################################################################
        I = I + noisefactor * np.random.rand(len(k2d), len(k2d))  # Add random noise with maximum 1
        #  Plotabschnitt
        fig = plt.figure(figsize=(12, 7), dpi=100)
        plt.suptitle("EuGa2Al2, {} unit cells / {} supercells, modulation q={}rlu".format(n, int(n/int(q_cdw**(-1))), q_cdw))
        #  Modulated atomic positions
        plt.subplot(2, 2, 1)
        # print(np.ones(n), np.arange(0, n, 1), Al[2, :], np.ones(n))
        plt.scatter(0 * np.ones(n) + np.arange(0, n, 1), np.ones(n),
                    label=r'Al1=$(0,\frac{1}{2},\frac{1}{4})$ equilibrium', facecolors='none',
                    edgecolors='orange', s=100)
        plt.xlabel('z')
        plt.ylabel('')
        plt.scatter(Al1[0, :], np.ones(n),
                    label='Al1=$(0+{} \cos({} \cdot 2\pi L),1/2,1/4)$ distorted'.format(Amplitude, q_cdw),
                    marker='o')
        plt.legend()
        #  Imshow convolution plot
        plt.subplot(2, 2, 2)
        plt.title("Gaussian interpolation")
        plt.imshow(I, cmap='plasma',
                   interpolation='gaussian',
                   extent=(k0 - kmax, k0 + kmax, l0 - lmax, l0 + lmax),
                   origin='lower'
                   # ,norm=LogNorm(vmin=0.01 , vmax=np.max(I))
                   )
        plt.colorbar()
        plt.xlabel("K(rlu)")
        plt.ylabel("L(rlu)")
        #  2D Scatter plot
        plt.subplot(2, 2, 3)
        plt.scatter(k2d, l2d, cmap='viridis', label=r'$I \propto F(\mathbf{Q})^2$'
                    , s=I / np.max(I),
                    # , c = I / np.max(I),
                    )
        plt.colorbar()
        plt.legend()
        plt.ylabel("L(rlu)")
        plt.xlabel("K(rlu)")
        plt.tight_layout()
        #  Linecuts
        plt.subplot(2, 2, 4)
        plt.title(r'Centered around [{}{}{}]'.format(h, k0, l0))
        plt.plot(l2d[:, 0], I[:, len(k) // 2], ls='--', lw=0.5, marker='.', ms=1.5, label='K={}'.format(np.round(k[len(k) // 2], 2)))
        plt.plot(l2d[:, 0], I[:, 0], ls='--', marker='.', lw=0.5, ms=1.5, label='K={}'.format(np.round(k[0], 2)))
        plt.legend()
        plt.ylabel(r"Intensity $I\propto F(\mathbf{Q})^2$")
        plt.xlabel("L(rlu)")
        if savefig == True:
            plt.savefig("EuGa2Al2_{}UC_{}SC_A={}_q={}_H={}_center{}{}{}.jpg".format(n, int(n/int(q_cdw**(-1))), Amplitude, q_cdw, h, h, k0, l0), dpi=300)
        else:
            pass
        #  3D Atomic Plot
        fig = figure()
        ax = fig.add_subplot(projection='3d')
        # print("Eu={}, Al={}".format(Eu, Al))
        # ax.title(r"(Modulated) unit cell(s) with Wyckoff 4e z-parameter z0={} and $T=(1/2 1/2 1/2)$".format(z0))
        ax.scatter(Eu[0], Eu[1], Eu[2], label='Eu, Wyckoff 2a (000)', c='yellow')
        ax.scatter(Eu_T[0], Eu_T[1], Eu_T[2], label='Eu_T, Wyckoff 2a T(000)', facecolors='none', edgecolors='yellow')
        ax.scatter(Al1[0], Al1[1], Al1[2], label='Al1, Wyckoff 4d (1/2 0 1/4), (0 1/2 1/4)', c='blue')
        ax.scatter(Al1_T[0], Al1_T[1], Al1_T[2], facecolors='none', edgecolors='blue')
        ax.scatter(Al2[0], Al2[1], Al2[2], c='blue')
        ax.scatter(Al2_T[0], Al2_T[1], Al2_T[2], facecolors='none', edgecolors='blue')
        ax.scatter(Ga1[0], Ga1[1], Ga1[2], label='Ga1, Wyckoff 4e (0 0 z0), (0 0 -z0)', c='green')
        ax.scatter(Ga1_T[0], Ga1_T[1], Ga1_T[2], facecolors='none', edgecolors='green')
        ax.scatter(Ga2[0], Ga2[1], Ga2[2], c='green')
        ax.scatter(Ga2_T[0], Ga2_T[1], Ga2_T[2], facecolors='none', edgecolors='green')
        ax.set_xlim(-1.5 * kmax, 1.5 * kmax)
        ax.set_ylim(-1.5 * kmax, 1.5 * kmax)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.legend()
        plt.show()


def euga2al2modulation1():
    ########################################################################
    #  General input
    #  --> Need to define atoms at unique wyckoff positions in Structurefactor
    k0, l0, kmax, lmax = 0, 0, 15, 15  # boundaries of K and L for the intensity maps
    z0 = 0.35  #Wyckoff 4e z-parameter
    deltak = 0.1  #or 0.01, k-space point distance
    h = 1  #H value in k space
    f_Eu, f_Ga, f_Al = 1, 1, 1
    f_list = [f_Eu, f_Ga, f_Al]  # Atomic form factor list --> later as Q-dependent quantity
    #####################################
    #  Modulation
    Amplitude = 0.0  # Modulation amplitude
    q_cdw = 0.2  # Modulation vector in rlu
    #####################################
    #  Other
    N = [1, 5, 10
        # int(q_cdw**(-1))*1
        # ,int(q_cdw**(-1))*2
        # ,int(q_cdw**(-1))*3
        #,int(q_cdw**(-1))*5
        # ,int(q_cdw**(-1))*8
         ]  ## of unit unmodulated cells list
    noisefactor = 0.01  #Noise amplitude
    #  PROGRAM
    k2d, l2d, k, Unitary = kspacecreator(k0, l0, kmax, lmax, deltak)
    structurefactorandplotting(k0, l0, k2d, k, kmax, lmax, l2d, h, deltak, Unitary,
                               Amplitude, q_cdw, z0, N, noisefactor, f_list, savefig=True)

if __name__ == '__main__':
    euga2al2modulation1()

