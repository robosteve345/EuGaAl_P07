import numpy as np
import matplotlib
import time
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.colors import LogNorm
from matplotlib.pyplot import figure
matplotlib.rcParams['font.family'] = "sans-serif"
matplotlib.rc('text', usetex=True)
import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)
from XRD_simulation_functions import ztranslationpos, kspacecreator, excludekspacepoints, debyewallerfactor
from XRD_simulation_functions import atomicformfactorBCC

"""-----> OUT-DATED: BBX_unmodulated_updated has all features"""
"""TO DO
-Implement deltak possibilities for 0.1&0.01 in L for q_cdw=0.2, 0.1 & 0.5
-Implement possibility for HKL_0 & HK_0L maps
-Implement gaussian distribution on every desired (K,L)-value (done)
-Implement input-interface: kß, lß, deltak, Amplitude, q_cdw, kmax
"""


def structurefactorandplotting(a, c, k0, l0, k2d, k, kmax, lmax, l2d, h, deltak, Unitary, 
                               Amplitude, u_list, lamb, q_cdw, N, kernelsize, 
                               noiseamplitude, sigma, kspacefactors, Lexclude=False, lognorm=True, 
                               normalization=False, DBW=False, savefig=False,
                               properatomicformfactor=True):
    for n in N:
        print("SIMULATION FOR N={} unmodulated unit cells / {} supercells".format(n, int(n/int(q_cdw**(-1)))))
        print("CENTERED PEAK: [HKL] = [{}{}{}]".format(h, k0, l0))
        print("CDW periodic modulation {}cos({}*z*2*pi)".format(Amplitude, q_cdw))
        """BCC unit cell Wyckoff positions"""
        x_Atom1, y_Atom1, z_Atom1 = [0], [0], [0]
        x_Atom2, y_Atom2, z_Atom2 = [0.5], [0.5], [0.5]
        """Atomic positions modulation: commensurate with q_CDW = q*2*pi/a of atom 2 (1/2, 1/2, 1/2)"""
        #  Full translation of Atom1 & Atom2 for N unit cells
        for i in range(1, n):  # begins at 1 because unit cell 0 is already given 1, to n-1
            ztranslationpos(np.array([x_Atom1[0], y_Atom1[0], z_Atom1[0]]), x_Atom1, y_Atom1, z_Atom1, i)
            ztranslationpos(np.array([x_Atom2[0], y_Atom2[0], z_Atom2[0]]), x_Atom2, y_Atom2, z_Atom2, i)

        #  Modulation of Atom2
        for i in range(0, n):  # begins with 1 because it evaluates every unit cell, 0 to n-1, BLEIBT SO
            z_Atom2[i] = z_Atom2[i] + Amplitude * np.cos(q_cdw * 2 * np.pi * z_Atom2[i]) # damit i: 1...n und nicht 0...n-1
        
        #  Final positions
        Atom1, Atom2 = np.array([x_Atom1, y_Atom1, z_Atom1]), np.array([x_Atom2, y_Atom2, z_Atom2])
        # print("Atom2={}".format(Atom2))
        # print("Atom1={}".format(Atom1))
        
        #  Compute Debye-Waller-Factor
        if DBW == True:
            DBW_list = debyewallerfactor(lamb, k2d, l2d, Unitary, h, a, c, u_list)
        else:
            DBW_list = np.ones(3)
            
        """Scattering amplitudes F"""
        #  Atomic form factors
        f_Atom1, f_Atom2 = atomicformfactorBCC(h, k2d, l2d, Unitary, properatomicformfactor)
        
        #  Crystal structure factor for each atom
        F_Atom1, F_Atom2 = [], []
        for i in range(0, n):
            F_Atom1.append(
                f_Atom1 * DBW_list[0] * np.exp(-2 * np.pi * 1j * 
            (h * Unitary * Atom1[0, i] + k2d * Atom1[1, i] + l2d * Atom1[2, i])))
            F_Atom2.append(
                f_Atom2 * DBW_list[1] * np.exp(-2 * np.pi * 1j * 
            (h * Unitary * Atom2[0, i] + k2d * Atom2[1, i] + l2d * Atom2[2, i])))
        F_init = np.zeros((len(k2d), len(k2d)), dtype=complex)  # quadratic 0-matrix with dimensions of k-space
        F_list = [F_Atom1, F_Atom2]
        pre_F_init = [np.zeros((len(k2d), len(k2d)))]
        
        #  Zusammenfügen der Formfaktoren für die Atomsorten
        for i in F_list:  # put together the lists in a ndarray for each atom with each N positions, to get rid of 3rd dimension (better workaround probably possible...)
            pre_F_init = np.add(pre_F_init, i)
        for i in range(len(pre_F_init)):
            F_init = F_init + pre_F_init[i]
        # print(F_init)
        #  Compute Intensity
        I = np.abs(np.round(F_init, 3)) ** 2  # I \propto F(Q)^2, F complex
        # I = I + noiseamplitude * np.random.rand(len(k2d), len(k2d))  # Add random noise with maximum 1
        
        # Compute normalization
        if normalization == True:
            I = I/np.max(I)
        else:
            I = I
            
        # # Exclude k-space points 
        # I = excludekspacepoints(kspacefactors, k2d, l2d, deltak, I, noiseamplitude, 
        #                         kmax, lmax, Lexclude)

        
        # # PLOTTING
        fig = plt.figure(figsize=(15, 3))
        plt.suptitle("Body centered cubic (bcc), {} unit cells / {} supercells, modulation q={}rlu".format(n, int(n/int(q_cdw**(-1))), q_cdw))
        """MODULATED ATOMIC POSITIONS"""
        plt.subplot(2, 1, 1)
        plt.scatter(1 / 2 * np.ones(n) + np.arange(0, n, 1), np.ones(n),
                    label=r'Atom2=$(\frac{1}{2},\frac{1}{2},\frac{1}{2})$ equilibrium', facecolors='none',
                    edgecolors='orange', s=100)
        plt.xlabel('z')
        plt.ylabel('')
        plt.scatter(Atom2[2, :], np.ones(n),
                    label='Atom2=$(0.5,0.5,0.5 + {} \cos({} \cdot 2\pi L))$ distorted'.format(Amplitude, q_cdw),
                    marker='o')
        plt.legend()
        """LINECUTS"""
        plt.subplot(2, 1, 2)
        plt.title(r'Centered around [{}{}{}]'.format(h, k0, l0))
        plt.plot(l2d[:, 0], I[:, len(k) // 2], ls='-', lw=0.5, marker='.', 
                 ms=1, label='K={}'.format(np.round(k[len(k) // 2], 2)))
        plt.plot(l2d[:, 0], I[:, 0], ls='-', marker='.', lw=0.5, 
                 ms=1, label='K={}'.format(np.round(k[0], 2)))
        plt.legend()
        plt.xlim(l0-kmax, l0+kmax)
        plt.ylabel(r"Intensity $I\propto F(\mathbf{Q})^2$")
        plt.xlabel("L (r.l.u.)")
        plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/XRD_Simulation/BCC_modulated/Linecuts_BCC_{}UC_{}SC_A={}_q={}_H={}_center{}{}{}.jpg".format(n, int(n/int(q_cdw**(-1))), Amplitude, q_cdw, h, h, k0, l0), dpi=300)
        if savefig == True:
            plt.savefig("/Users/stevengebel/PycharmProjects/EuGaAl_P07/ +\
                        XRD_Simulation/BCC_modulated/ +\
                        L-cuts_N={}_q={}_H={}_center{}{}{}.jpg +\
                            ".format(N, q_cdw, h, h, k0, l0), dpi=300)
        else:
            pass
        plt.subplots_adjust(hspace=0.8)
        
        fig = plt.figure(figsize=(15,2.5))
        plt.suptitle("BCC, {} unit cells / {} supercells, modulation $\mathbf{q}$={}rlu".format(n, int(n/int(q_cdw**(-1))), q_cdw))
        """INTERPOLATION"""
        plt.subplot(1, 3, 1)
        plt.title("Gaussian interpolation, H={}".format(h))
        if lognorm == True:
            plt.imshow(I, cmap='inferno',
                       interpolation='gaussian',
                       extent=(k0 - kmax, k0 + kmax, l0 - lmax, l0 + lmax),
                       origin='lower',
                       norm=LogNorm(vmin = 10, vmax = np.max(I))
                       )
        else:
            plt.imshow(I, cmap='inferno',
                       interpolation='gaussian',
                       extent=(k0 - kmax, k0 + kmax, l0 - lmax, l0 + lmax),
                       origin='lower',
                       )
            
        plt.colorbar()
        plt.xlabel("K(rlu)")
        plt.ylabel("L(rlu)")
        """CONVOLUTION"""
        plt.subplot(1, 3, 3)
        plt.title("Gaussian conv., H={}".format(h))
        x, y = np.linspace(-1,1,kernelsize), np.linspace(-1,1,kernelsize)
        X, Y = np.meshgrid(x, y)
        kernel = 1/(2*np.pi*sigma**2)*np.exp(-(X**2+Y**2)/(2*sigma**2))
        Iconv = ndimage.convolve(I, kernel, mode='constant', cval=0.0)
        if lognorm == True:
            plt.imshow(Iconv, cmap='inferno', extent=(k0-kmax, k0+kmax, l0-lmax, l0+lmax), 
                   origin='lower',
                   #vmin=0, vmax=0.1*np.max(Iconv)
                   norm=LogNorm(vmin = 1, vmax = np.max(Iconv))
                   )
        else:
            plt.imshow(Iconv, cmap='inferno', extent=(k0-kmax, k0+kmax, l0-lmax, l0+lmax), 
                   origin='lower',
                   )
        plt.colorbar()
        plt.ylabel("L (r.l.u.)")
        plt.xlabel("K (r.l.u.)")       
        
        """3D Atom Plot"""
        ax = fig.add_subplot(1,3,2, projection='3d')
        # Atomlist = np.concatenate((Atom1, Atom2), axis=1)
        # print("Atom1={}, Atom2={}".format(Atom1, Atom2))
        # print("Atomlist={}".format(Atomlist))
        ax.scatter(Atom1[0], Atom1[1], Atom1[2], label='Atom1')
        ax.scatter(Atom2[0], Atom2[1], Atom2[2], label='Atom2')
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        # ax.set_zlim(-2 * kmax, 2 * kmax)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.legend()
        
        
        plt.show(block=False)


def bccmodulation1():
    ##########################################################################
    ########################## GENERAL INPUT #################################
    ##########################################################################
    k0, l0, kmax, lmax = 10, 5, 1, 1  # boundaries of K and L 
    deltak = 0.01  # k-space point distance
    h = 5  # H value in k space
    a, c = 1, 1  # Unit Cell Parameters
    ##########################################################################
    #  CDW MODULATION
    Amplitude = 0.2  # Modulation amplitude
    q_cdw = 0.2  # Modulation vector in rlu
    ##########################################################################
    #  INPUT FOR DBW
    lamb = 7e-10  # x-ray wavelength in m (Mo Ka)
    u_list = [1, 1]  # Isotropic displacements <u^2> in Å^2, ~1/m. For more detail take bonding angles, distances into account.
    noiseamplitude = 1e-4  # Noise amplitude
    sigma, kernelsize = 0.5, int(0.1/deltak) # Gaussian Kernel parameters
    #  Other
    kspacefactors = [1, 9, 1, 9]
    N = [#int(q_cdw**(-1)),
        100
         ]  # # of unit unmodulated cells list
    #  PROGRAM
    st = time.time()
    k2d, l2d, k, Unitary = kspacecreator(k0, l0, kmax, lmax, deltak)
    structurefactorandplotting(a, c, k0, l0, k2d, k, kmax, lmax, l2d, h, deltak=deltak, 
                               Unitary=Unitary, Amplitude=Amplitude, u_list=u_list, lamb=lamb,
                               q_cdw=q_cdw, N=N, kernelsize = kernelsize, 
                               noiseamplitude=noiseamplitude, sigma=sigma,
                               kspacefactors=kspacefactors, Lexclude=False,
                               properatomicformfactor=False,
                               normalization=False, DBW=False, 
                               lognorm=False, savefig=False)

    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

if __name__ == '__main__':
    bccmodulation1()
