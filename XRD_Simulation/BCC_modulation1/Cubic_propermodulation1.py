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


def pseudoconvolution(I, sigma, Unitary):
    for i in range(2):
        I + np.exp(-((k2d**2 - Unitary*k0)+ (l**2 - Unitary*l0))/(2*sigma**2))
    return 1


def structurefactorandplotting(k0, l0, k2d, k, kmax, lmax, l2d, h, deltak, Unitary, Amplitude, q_cdw, N, noisefactor, f_list, savefig=False):
    for n in N:
        print("SIMULATION FOR N={} unmodulated unit cells / {} supercells".format(n, int(n/int(q_cdw**(-1)))))
        print("INVESTIGATED PEAK: [HKL] = [{}{}{}]".format(h, k0, l0))
        print("CDW Modulation A*sin(q_cdw*z*2*pi):")
        print("q_cdw={}, A={}".format(q_cdw, Amplitude))
        """BCC unit cell Wyckoff positions"""
        x_Atom1, y_Atom1, z_Atom1 = [0], [0], [0]
        x_Atom2, y_Atom2, z_Atom2 = [0.5], [0.5], [0.5]
        """Atomic positions modulation: commensurate with q_CDW = q*2*pi/a of atom 2 (1/2, 1/2, 1/2)"""
        # Full translation of Atom1 & Atom2 for N unit cells
        for i in range(1, n):  # begins at 1 because unit cell 0 is already given
            ztranslation(np.array([x_Atom1[0], y_Atom1[0], z_Atom1[0]]), x_Atom1, y_Atom1, z_Atom1, i)
            ztranslation(np.array([x_Atom2[0], y_Atom2[0], z_Atom2[0]]), x_Atom2, y_Atom2, z_Atom2, i)
        #  Modulation of Atom2
        for i in range(0, n):  # begins with 1 because it evaluates every unit cell
            z_Atom2[i] = z_Atom2[i] + Amplitude * np.cos(q_cdw * 2 * np.pi * i)
            # print("n={}".format(i))
        print("z_Modulation = {}".format(Amplitude * np.cos(q_cdw * 2 * np.pi * np.arange(0, n+1, 1))))
            #  Final positions
        Atom1, Atom2 = np.array([x_Atom1, y_Atom1, z_Atom1]), np.array([x_Atom2, y_Atom2, z_Atom2])
        # print("Atom1={}".format(Atom1))
        # print("Atom2={}".format(Atom2))
        """Scattering amplitudes F"""
        #  Atomic form factors
        f_Atom1, f_Atom2 = f_list[0], f_list[1]
        #  Crystal structure factor for each atom
        F_Atom1, F_Atom2 = [], []
        for i in range(0, n):
            F_Atom1.append(
                f_Atom1 * np.exp(-2 * np.pi * 1j * (h * Unitary * Atom1[0, i] + k2d * Atom1[1, i] + l2d * Atom1[2, i])))
            F_Atom2.append(
                f_Atom2 * np.exp(-2 * np.pi * 1j * (h * Unitary * Atom2[0, i] + k2d * Atom2[1, i] + l2d * Atom2[2, i])))
        F_init = np.zeros((len(k2d), len(k2d)), dtype=complex)  # quadratic 0-matrix with dimensions of k-space
        F_list = [F_Atom1, F_Atom2]
        pre_F_init = [np.zeros((len(k2d), len(k2d)))]
        #  Zusammenfügen der Formfaktoren für die Atomsorten
        for i in F_list:  # put together the lists in a ndarray for each atom with each N positions, to get rid of 3rd dimension (better workaround probably possible...)
            pre_F_init = np.add(pre_F_init, i)
        for i in range(len(pre_F_init)):
            F_init = F_init + pre_F_init[i]
        #  Intensity I
        I = np.abs(np.round(F_init, 3)) ** 2  # I \propto F(Q)^2, F complex
        #  Table for extracting unwanted K,L points
        #  q_cdw     0.1   0.1    0.2   0.2    0.5   0.5
        #  ∆k        0.1   0.01   0.1   0.01   0.1   0.01
        #  Kfactor1  1     1      1
        #  Kfactor2  9     99     9
        #  Lfactor1  1     10     1
        #  Lfactor2  9     9      2
        ########################################################################
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
        plt.suptitle("Body centered cubic (bcc), {} unit cells / {} supercells, modulation q={}rlu".format(n, int(n/int(q_cdw**(-1))), q_cdw))
        #  Modulated atomic positions
        plt.subplot(2, 3, 1)
        # print(np.ones(n), np.arange(0, n, 1), Atom2[2, :], np.ones(n))
        plt.scatter(1 / 2 * np.ones(n) + np.arange(0, n, 1), np.ones(n),
                    label=r'Atom2=$(\frac{1}{2},\frac{1}{2},\frac{1}{2})$ equilibrium', facecolors='none',
                    edgecolors='orange', s=100)
        plt.xlabel('z')
        plt.ylabel('')
        plt.scatter(Atom2[2, :], np.ones(n),
                    label='Atom2=$(0.5,0.5,0.5+{} \cos({} \cdot 2\pi L))$ distorted'.format(Amplitude, q_cdw),
                    marker='o')
        plt.legend()
        #  Imshow convolution plot
        plt.subplot(2, 3, 2)
        plt.title("Gaussian interpolation")
        plt.imshow(I, cmap='viridis',
                   interpolation='gaussian',
                   extent=(k0 - kmax, k0 + kmax, l0 - lmax, l0 + lmax),
                   origin='lower'
                   # ,norm=LogNorm(vmin=0.01 , vmax=np.max(I))
                   )
        plt.colorbar()
        plt.xlabel("K(rlu)")
        plt.ylabel("L(rlu)")
        #  2D Scatter plot
        plt.subplot(2, 3, 3)
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
        plt.subplot(2, 3, 5)
        plt.title(r'Centered around [{}{}{}]'.format(h, k0, l0))
        plt.plot(l2d[:, 0], I[:, len(k) // 2], ls='--', lw=0.5, marker='.', ms=1.5, label='K={}'.format(np.round(k[len(k) // 2], 2)))
        plt.plot(l2d[:, 0], I[:, 0], ls='--', marker='.', lw=0.5, ms=1.5, label='K={}'.format(np.round(k[0], 2)))
        plt.legend()
        plt.ylabel(r"Intensity $I\propto F(\mathbf{Q})^2$")
        plt.xlabel("L(rlu)")
        plt.subplots_adjust(wspace=0.3)
        #  3D Atom Plot
        ax = fig.add_subplot(2, 3, 4, projection='3d')
        # Atomlist = np.concatenate((Atom1, Atom2), axis=1)
        print("Atom1={}, Atom2={}".format(Atom1, Atom2))
        # print("Atomlist={}".format(Atomlist))
        ax.scatter(Atom1[0], Atom1[1], Atom1[2], label='Atom1')
        ax.scatter(Atom2[0], Atom2[1], Atom2[2], label='Atom2')
        ax.set_xlim(-2 * kmax, 2 * kmax)
        ax.set_ylim(-2 * kmax, 2 * kmax)
        # ax.set_zlim(-2 * kmax, 2 * kmax)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.legend()
        if savefig == True:
            plt.savefig("BCC_{}UC_{}SC_A={}_q={}_H={}_center{}{}{}.jpg".format(n, int(n/int(q_cdw**(-1))), Amplitude, q_cdw, h, h, k0, l0), dpi=300)
        else:
            pass
        plt.subplots_adjust(wspace=0.3)
        plt.show(block=True)


def bccmodulation1():
    ########################################################################
    #  General input
    #  --> Need to define atoms at unique wyckoff positions in Structurefactor
    k0, l0, kmax, lmax = 0, 0, 1, 1  # boundaries of K and L for the intensity maps
    deltak = 0.1  # or 0.1, k-space point distance
    h = 1  # H value in k space
    f_list = [1, 1]  # Atomic form factor list --> later as Q-dependent quantity
    #####################################
    #  Modulation
    Amplitude = 0.3  # Modulation amplitude
    q_cdw = 0.2  # Modulation vector in rlu
    #####################################
    #  Other
    N = [int(q_cdw**(-1))*1
        ,int(q_cdw**(-1))*2
        #,int(q_cdw**(-1))*3
        #,int(q_cdw**(-1))*5
        # ,int(q_cdw**(-1))*8
         ]  ## of unit unmodulated cells list
    noisefactor = 0.01  #Noise amplitude
    #  PROGRAM

    k2d, l2d, k, Unitary = kspacecreator(k0, l0, kmax, lmax, deltak)
    structurefactorandplotting(k0, l0, k2d, k, kmax, lmax, l2d, h, deltak, Unitary,
                               Amplitude, q_cdw, N, noisefactor, f_list, savefig=False)


if __name__ == '__main__':
    bccmodulation1()

