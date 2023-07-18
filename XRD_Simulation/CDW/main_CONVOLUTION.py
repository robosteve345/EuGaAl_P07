"""-- EuGa2Al2 single-crystal XRD intensity map simulation --
DBW with isotropic B considered. Atomic form factor in series expansion considered.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# Times, Palatino, New Century Schoolbook, Bookman, Computer Modern Roman


def translation(x, storage_x, storage_y, storage_z, i):
    """
    input: 3-dim x
    output:
    Translate atomic positions of X_3 position
    """
    x_transl = x + np.array([0, 0, i])
    storage_x.append(x_transl[0])
    storage_y.append(x_transl[1])
    storage_z.append(x_transl[2])


def sin_modulation(GA_Z, z_ampl, q_cdw):
    """
    Simuliere Modulierung der Gallium-Atome in z-Richtung
    :param GA_Z: Wyckoff z-position
    :param z_ampl: amplitude of modulation...
    :param q_cdw: amplitude of CDW modulation vector
    :param n: factor
    :return: z_mod, array like
    """
    z_mod = z_ampl * np.sin(2 * np.pi * q_cdw * np.asarray(GA_Z))
    return z_mod


def anharmonic_modulation(GA_Z, z_ampl, q_cdw, n, phi):
    """
    :param GA_Z:
    :param z_ampl:
    :param q_cdw:
    :param n: # of fourier components
    :param phi:
    :return:
    """
    z_mod = [0] * n
    for i in (range(n)):
        z_mod.append( z_ampl * np.cos(i * q_cdw * np.asarray(GA_Z) + phi))
    return z_mod


def kspacecreator(boundary, diff):
    """Creates kspace with points evenly distributed with periodic boundaries in K, L
    :returns: n: integer, 2 * boundary / diff + 1
              k2d, l2d: meshgrid of k, l = np.linspace(-boundary, boundary, n)
    """
    n = np.intc(boundary / diff * 2 + 1)
    print("n = {}".format(n))
    k = np.linspace(-boundary, boundary, n)
    l = np.linspace(-boundary, boundary, n)
    k2d, l2d = np.meshgrid(k, l)
    # print("k2d[0][i] = {}".format(k2d[0,:]))
    # print("l2d[i][0] = {}".format(l2d[:,0]))
    # print("k2d={}".format(k2d), "l2d={}".format(l2d))
    return n, k2d, l2d


def thetacalc(lamb, k2d, l2d, n, h, a, c):
    """theta-input for DBW
    DBW = exp(- (B_iso * sin(theta)**2) / lambda )
    :param lamb: x-ray wavelength
    :param k2d, l2d: kspace 2D parametrization
    :param a, c: crystal parameters
    :return: B_iso parameters, theta: size as l2d, k2d
    """
    # Compute all distances corresponding to lattice diffractions with hkl
    d_hkl = a / (np.sqrt((h * np.ones((n, n))) ** 2 + k2d ** 2 + (a / c) ** 2 * l2d ** 2))
    # 1. compute all angles corresponding to the k points according to braggs law
    theta = np.arcsin(lamb / (2 * d_hkl))

    return theta


def F_Q(k2d, l2d, h, n, a_eu, a_ga, a_al, b_eu, b_ga, b_al, c_eu, c_ga, c_al, u_list, theta, z0, lamb, DBW=False):
    """ Crystal symmetry input"""
    # Positions of atoms in 10 unit cells
    # Gebe ersten Vektor der verschobenen und nicht verschobenen Positionen der Atome an:
    # Translated positions about \vec{r} = (0.5,0.5,0.5)
    x_Eu, y_Eu, z_Eu = [0], [0], [0]
    x_Eu_T, y_Eu_T, z_Eu_T = [0.5], [0.5], [0.5]

    # Aluminium:
    x_Al1, y_Al1, z_Al1 = [0], [0.5], [0.25]
    x_Al1_T, y_Al1_T, z_Al1_T = [0.5], [1], [0.75]
    x_Al2, y_Al2, z_Al2 = [0.5], [0], [0.25]
    x_Al2_T, y_Al2_T, z_Al2_T = [1], [0.5], [0.75]

    # Gallium:
    x_Ga1, y_Ga1, z_Ga1 = [0], [0], [z0]
    x_Ga1_T, y_Ga1_T, z_Ga1_T = [0.5], [0.5], [0.5 + z0]
    x_Ga2, y_Ga2, z_Ga2 = [0], [0], [-z0 + 1]
    x_Ga2_T, y_Ga2_T, z_Ga2_T = [0.5], [0.5], [0.5 - z0]

    # Full translation of Eu, Al, first translation without modulation of Ga:
    for i in range(1, 10):
        translation(np.array([x_Eu[0], y_Eu[0], z_Eu[0]]), x_Eu, y_Eu, z_Eu, i)  # Eu
        translation(np.array([x_Eu_T[0], y_Eu_T[0], z_Eu_T[0]]), x_Eu_T, y_Eu_T, z_Eu_T, i)
        translation(np.array([x_Al1[0], y_Al1[0], z_Al1[0]]), x_Al1, y_Al1, z_Al1, i)  # Al1
        translation(np.array([x_Al1_T[0], y_Al1_T[0], z_Al1_T[0]]), x_Al1_T, y_Al1_T, z_Al1_T, i)
        translation(np.array([x_Al2[0], y_Al2[0], z_Al2[0]]), x_Al2, y_Al2, z_Al2, i)  # Al2
        translation(np.array([x_Al2_T[0], y_Al2_T[0], z_Al2_T[0]]), x_Al2_T, y_Al2_T, z_Al2_T, i)
        translation(np.array([x_Ga1[0], y_Ga1[0], z_Ga1[0]]), x_Ga1, y_Ga1, z_Ga1, i)  # Ga1
        translation(np.array([x_Ga1_T[0], y_Ga1_T[0], z_Ga1_T[0]]), x_Ga1_T, y_Ga1_T, z_Ga1_T, i)
        translation(np.array([x_Ga2[0], y_Ga2[0], z_Ga2[0]]), x_Ga2, y_Ga2, z_Ga2, i)  # Ga2
        translation(np.array([x_Ga2_T[0], y_Ga2_T[0], z_Ga2_T[0]]), x_Ga2_T, y_Ga2_T, z_Ga2_T, i)

    """Ga_z modulation:"""
    q_cdw = 1 / 10  # in r.l.u.
    z_ampl = 0
    print("CDW Modulation A*sin(q_cdw*z*2*pi):")
    print("q_cdw={}, A={}".format(q_cdw, z_ampl))
    zp = 0.13  # relative position Ga-Al in z-direction, FESTER WERT
    z = []
    dz, weird = [], []
    for i in range(1, 41):
        z.append((i / 4 - 1 / 8) + 1 / 8 * (-1) ** (i + 1) + zp * (-1) ** i)
        #Modulation 1: (-1)**i * z_ampl * np.sin(2*np.pi* q_cdw * z)
        dz.append((-1)**i * z_ampl * (np.sin(2*np.pi*q_cdw *  (i/4 - 1/8) + 1/8*(-1)**(i+1)+zp*(-1)**i)))
        weird.append((i/4 + 1/8 + 1/8 * (-1)**(i+1)))
    # print("input for sin from marein: {}".format(weird))
    print("Gallium z-positions and deviations dz:")
    print("z={}".format(np.round(z, 2)), "# z={}".format(len(z)),
          "dz={}".format(np.round(dz, 2)), "# dz={}".format(len(dz)),
          "z+dz={}".format(np.asarray(z) + np.asarray(dz))
          )
    mod = np.asarray(z) + np.asarray(dz)  # modulated ga_z positions
    fig, axs = plt.subplots(2, 2, figsize=(10, 5))

    # fig.suptitle(r'CDW modulation simulation for $EuGa_2Al_2$')
    axs[0, 0].scatter(np.asarray(z) + np.asarray(dz), np.ones(40) * 0.5, c='r', marker='.', label='Modulated')
    axs[0, 0].scatter(np.asarray(z), np.ones(40) * 0.5, c='k', marker='.', label='Unmodulated')

    # plt.scatter(z_Ga1_T_mod, np.ones(10) * 0.5, c='r', marker='.')
    # plt.scatter(z_Ga2_mod, np.ones(10) * 0.5, c='k', marker='.', label='Wyckoff 4e(2)')
    # plt.tick_params(direction='in', length=200, width=1, zorder=1, colors='k', grid_linestyle='--',
    #                grid_color='tab:orange', grid_alpha=0.25, axis='x', top=True)
    every_1st, every_2nd, every_3rd, every_4th = mod[0::4], mod[1::4], mod[2::4], mod[
                                                                                  3::4]  # transform 40 array into 4*10 arrays
    # print("every_1st={}".format(every_1st), "every_2nd={}".format(every_2nd), "every_3rd={}".format(every_3rd),
    #      "every_4th={}".format(every_4th))
    """ Final Atomic vector positions """
    Eu, Eu_T = np.array([x_Eu, y_Eu, z_Eu]), np.array([x_Eu_T, y_Eu_T, z_Eu_T])
    Ga1, Ga1_T, Ga2, Ga2_T = np.array([x_Ga1, y_Ga1, every_2nd]), np.array([x_Ga1_T, y_Ga1_T, every_4th]), np.array(
        [x_Ga2, y_Ga2, every_3rd]), np.array([x_Ga2_T, y_Ga2_T, every_1st])
    Al1, Al1_T, Al2, Al2_T = np.array([x_Al1, y_Al1, z_Al1]), np.array([x_Al1_T, y_Al1_T, z_Al1_T]), np.array(
        [x_Al2, y_Al2, z_Al2]), np.array([x_Al2_T, y_Al2_T, z_Al2_T])
    """Form factor calculation"""
    f_eu1 = a_eu[0] * np.exp(- b_eu[0] * (((h * np.ones((n, n))) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    f_eu2 = a_eu[1] * np.exp(- b_eu[1] * (((h * np.ones((n, n))) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    f_eu3 = a_eu[2] * np.exp(- b_eu[2] * (((h * np.ones((n, n))) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    f_eu4 = a_eu[3] * np.exp(- b_eu[3] * (((h * np.ones((n, n))) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    f_Eu = f_eu1 + f_eu2 + f_eu3 + f_eu4 + c_eu * np.ones((n, n))
    f_ga1 = a_ga[0] * np.exp(- b_ga[0] * (((h * np.ones((n, n))) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    f_ga2 = a_ga[1] * np.exp(- b_ga[1] * (((h * np.ones((n, n))) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    f_ga3 = a_ga[2] * np.exp(- b_ga[2] * (((h * np.ones((n, n))) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    f_ga4 = a_ga[3] * np.exp(- b_ga[3] * (((h * np.ones((n, n))) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    f_Ga = f_ga1 + f_ga2 + f_ga3 + f_ga4 + c_ga * np.ones((n, n))
    f_al1 = a_al[0] * np.exp(- b_al[0] * (((h * np.ones((n, n))) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    f_al2 = a_al[1] * np.exp(- b_al[1] * (((h * np.ones((n, n))) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    f_al3 = a_al[2] * np.exp(- b_al[2] * (((h * np.ones((n, n))) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    f_al4 = a_al[3] * np.exp(- b_al[3] * (((h * np.ones((n, n))) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    f_Al = f_al1 + f_al2 + f_al3 + f_al4 + c_al * np.ones((n, n))

    F_Eu_list, F_Eu_T_list = [], []
    F_Ga1_list, F_Ga1_T_list, F_Ga2_list, F_Ga2_T_list = [], [], [], []
    F_Al1_list, F_Al1_T_list, F_Al2_list, F_Al2_T_list = [], [], [], []

    # Compute DBW factors
    B_iso_Eu, B_iso_Ga, B_iso_Al = 8 * np.pi ** 2 / 3 * u_list[0], 8 * np.pi ** 2 / 3 * u_list[1], \
                                   8 * np.pi ** 2 / 3 * u_list[2] # ISOTROPIC
    B_iso_list = [B_iso_Eu, B_iso_Ga, B_iso_Al]
    if DBW == True:
        DBW_Eu = np.exp(-B_iso_list[0] / lamb **2 * (np.sin(theta)) ** 2)
        DBW_Ga = np.exp(-B_iso_list[1] / lamb **2 *  (np.sin(theta)) ** 2)
        DBW_Al = np.exp(-B_iso_list[2] / lamb **2 * (np.sin(theta)) ** 2)
    else:
        DBW_Eu = np.ones(n, n)
        DBW_Ga = np.ones(n, n)
        DBW_Al = np.ones(n, n)
    for i in range(10):  # 2 atoms for each sort, for Eu with 1 position and 2 for each al and ga with translation
        F_Eu_list.append(
            f_Eu * DBW_Eu * np.exp(
                1j * 2 * np.pi * (h * np.ones((n, n)) * Eu[0, i] + k2d * Eu[1, i] + l2d * Eu[2, i]))
        )
        F_Eu_T_list.append(
            f_Eu * DBW_Eu * np.exp(
                1j * 2 * np.pi * (h * np.ones((n, n)) * Eu_T[0, i] + k2d * Eu_T[1, i] + l2d * Eu_T[2, i]))
        )
        F_Ga1_list.append(
            f_Ga * DBW_Ga * np.exp(
                1j * 2 * np.pi * (h * np.ones((n, n)) * Ga1[0, i] + k2d * Ga1[1, i] + l2d * Ga1[2, i]))
        )
        F_Ga1_T_list.append(
            f_Ga * DBW_Ga * np.exp(
                1j * 2 * np.pi * (h * np.ones((n, n)) * Ga1_T[0, i] + k2d * Ga1_T[1, i] + l2d * Ga1_T[2, i]))
        )
        F_Ga2_list.append(
            f_Ga * DBW_Ga * np.exp(
                1j * 2 * np.pi * (h * np.ones((n, n)) * Ga2[0, i] + k2d * Ga2[1, i] + l2d * Ga2[2, i]))
        )
        F_Ga2_T_list.append(
            f_Ga * DBW_Ga * np.exp(
                1j * 2 * np.pi * (h * np.ones((n, n)) * Ga2_T[0, i] + k2d * Ga2_T[1, i] + l2d * Ga2_T[2, i]))
        )
        F_Al1_list.append(
            f_Al * DBW_Al * np.exp(
                1j * 2 * np.pi * (h * np.ones((n, n)) * Al1[0, i] + k2d * Al1[1, i] + l2d * Al1[2, i]))
        )
        F_Al1_T_list.append(
            f_Al * DBW_Al * np.exp(
                1j * 2 * np.pi * (h * np.ones((n, n)) * Al1_T[0, i] + k2d * Al1_T[1, i] + l2d * Al1_T[2, i]))
        )
        F_Al2_list.append(
            f_Al * DBW_Al * np.exp(
                1j * 2 * np.pi * (h * np.ones((n, n)) * Al2[0, i] + k2d * Al2[1, i] + l2d * Al2[2, i]))
        )
        F_Al2_T_list.append(
            f_Al * DBW_Al * np.exp(
                1j * 2 * np.pi * (h * np.ones((n, n)) * Al2_T[0, i] + k2d * Al2_T[1, i] + l2d * Al2_T[2, i]))
        )
    """The glorious structure factor"""
    F_str = np.zeros((n, n), dtype=complex)  # Structure factor with dimensions nxn
    atoms_list = [ F_Eu_list, F_Eu_T_list, F_Ga1_list, F_Ga1_T_list, F_Ga2_list, F_Ga2_T_list, F_Al1_list, F_Al1_T_list,
                   F_Al2_list, F_Al2_T_list ]  # create list with components corresponding to F evaluated for all atomic positions of element X with all k-space points
    pre_F_str = [np.zeros((n, n))]  # Hilfsgröße, in nächstem loop beschrieben
    # Zusammenfügen der Formfaktoren für die Atomsorten
    for i in atoms_list:  # put together the lists in a ndarray for each atom with each N positions, to get rid of 3rd dimension (better workaround probably possible...)
        pre_F_str = np.add(pre_F_str, i)
    for i in range(len(pre_F_str)):
        F_str = F_str + pre_F_str[i]

    I = np.absolute(F_str) ** 2

    return I


def plotfunction(k2d, l2d, I, boundary, diff, h, k0, l0, vmax, savefig=False, norm=False):
    """Plots intensity I on k, l meshgrid via imshow. Initially deletes forbidden intensities for
    certain kspace points governed by the LAUE condition."""
    cmap = "binary"
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    print("INVESTIGATED PEAK: [HKL] = [{}{}{}]".format(h, k0, l0))
    ax.set_title('XRD intensity map simulation', size=15)
    ax.set_xlabel('K (r.l.u.)')
    ax.set_ylabel('L (r.l.u.)')
    integers = np.arange(-boundary, boundary + 1, 1)
    print("integers={}".format(integers))
    indices = 1 / diff * np.arange(0, 2 * boundary + 1, 1) + 0
    print("newindices={}".format(np.intc(indices)))
    # print("I_before={}".format(I)) # Intensity evaluated at every kpoint.
    # ADD NOISE
    # I = I  # + np.max(I) / 20 * np.abs(np.random.randn(n, n))
    # Manually set F(Q) elements to zero, that are forbidden by crystal symmetry.
    for i in range(len(l2d)):
        if i in np.intc(indices):
            I[i, :] = I[i, :]
        else:
            I[i, :] = 0
    # print("I_after={}".format(I))
    print("I dim: {}x{}".format(I.shape[0], I.shape[1]))
    if norm == True:
        pos = ax.imshow(I / np.max(I), cmap=cmap, interpolation='gaussian',
                        extent=[-boundary, boundary, -boundary, boundary],
                        vmax = vmax
                        )
    else:
        pos = ax.imshow(I, cmap=cmap, interpolation='gaussian',
                        extent=[-boundary, boundary, -boundary, boundary],
                        vmax = vmax
                        )
    fig.colorbar(pos, ax=ax, label='Intensity (r.l.u.)')
    if savefig == True:
        plt.savefig('{}kl_CDW_simulation_CONVOLUTION'.format(h), dpi=300)
        plt.show()
    else:
        plt.show()

    axs[0, 0].set_xlim(0, 10)
    axs[0, 0].set_ylim(0.48, 0.52)
    axs[0, 0].legend(fontsize=15)
    axs[0, 0].set_xticks(np.arange(0, 11, 1))
    axs[0, 0].tick_params(axis='x', labelsize=15, direction='in')
    axs[0, 0].tick_params(direction='in')
    axs[0, 0].set_yticks([])
    axs[0, 0].set_xlabel(r"$z$", fontsize=15)
    # plot whole space:
    axs[1,0].scatter(k2d, l2d, s=I*scatterfactor, linewidth=0.05, c='k')
    axs[1,0].set_ylabel(r"L (r.l.u.)", fontsize=15)
    axs[1,0].set_xlabel(r"K (r.l.u.)", fontsize=15)
    axs[1,0].tick_params(axis='x', labelsize=15, direction='in')
    axs[1,0].tick_params(axis='y', labelsize=15, direction='in')
    readout = l_boundary * 10 # readout: l_k_boundary * 10, centers around desired [HKL] peak
    # print("I_min/I_max = {}".format(np.min(I)/np.max(I)))
    # Centered peak
    axs[0,1].set_xlabel(r'L (r.l.u.)', fontsize=15)
    axs[0,1].set_ylabel(r'K (r.l.u.)', fontsize=15)
    axs[0,1].tick_params(axis='x', labelsize=15, direction='in')
    axs[0,1].tick_params(axis='y', labelsize=15, direction='in')
    # axs[0,1].set_xlim(k0-k_boundary-0.5, k0+k_boundary+0.5)
    axs[0,1].set_ylim(k0 - k_boundary, k0 + k_boundary)
    axs[0,1].set_xlim(l0 - l_boundary, l0 + l_boundary)
    axs[0,1].scatter(l2d[:, readout], k2d[:, readout], s=I[:, readout]*scatterfactor, linewidth=0.05, c='k',
                   label='[{}KL]'.format(h))
    axs[0,1].legend(fontsize=15)
    # print("single peak: k={}, l={}".format(k2d[:, readout], l2d[:, readout]))
    # Projected intensity of desired [HKL]-peak
    axs[1,1].plot(l2d[:, readout], I[:, readout] / np.max(I[:, readout]),
                  marker='x', c='k', ls='--', lw=0.5, ms=3, label='[{}KL]'.format(h))
    axs[1,1].set_xlim(l0 - l_boundary, l0 + l_boundary)
    axs[1,1].set_xlabel(r"L (r.l.u.)", fontsize=15)
    axs[1,1].set_ylabel(r"Intensity (Rel.)", fontsize=15)
    axs[1,1].tick_params(axis='x', labelsize=15, direction='in')
    axs[1,1].tick_params(axis='y', labelsize=15, direction='in')
    axs[1,1].legend(fontsize=15)
    print("I_max = {}".format(np.sort(I[:, readout])[-1]))
    print("(I_max - I_[max-1])/I_max = {}".format((np.sort(I[:, readout])[-1] - np.sort(I[:, readout])[-2]) / np.sort(I[:, readout])[-1]))
    plt.show()


def main():
    print(__doc__)
    l_boundary, k_boundary = 2, 2
    h = -3
    # n, k2d, l2d, h = kspacecreator(boundary, h) # for normal plot
    ############################################ # For CDW Plot
    k0, l0 = 5, -6
    k = np.arange(k0 - k_boundary, k0 + k_boundary + 0.1, 0.1)
    l = np.arange(l0 - l_boundary, l0 + l_boundary + 0.1, 0.1)
    n = len(k)
    k2d, l2d = np.meshgrid(k, l)
    ####################################################################################################################
    ####### INPUT KSPACE: boundary / diff ≈ 100 sufficient resolution
    boundary = 5  # Symmetric boundaries for K, L
    diff = 1  # Distance between two kspace points
    h = 1  # H plane in kspace
    n, k2d, l2d = kspacecreator(boundary, diff)
    ####### INPUT CRYSTAL SYMMETRY
    a = 4.40696  # in Å
    c = 10.68626  # in Å
    z0 = 0.3854 # 0.3854  # 4e Wyckoff z parameter
    ####### INPUT FOR DBW
    lamb = 7e-10  # x-ray wavelength in m (Mo Ka)
    u_Eu, u_Ga, u_Al = 1e-2, 1e-2, 1e-2  # Isotropic displacements <u^2> in Å^2, ~1/m. For more detail take bonding angles, distances into account.
    u_list = [u_Eu, u_Ga, u_Al]
    theta = thetacalc(lamb, k2d, l2d, n, h, a, c)
    ####### INPUT FOR ATOMIC FORM FACTORS
    # Atomic form factors according to de Graed, structure of materials, chapter 12: Eu2+. Ga1+, Al3+
    # http://lampx.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php
    a_eu, a_ga, a_al = [24.0063, 19.9504, 11.8034, 3.87243], \
                       [15.2354, 6.7006, 4.3591, 2.9623], [4.17448, 3.3876, 1.20296, 0.528137]
    b_eu, b_ga, b_al = [2.27783, 0.17353, 11.6096, 26.5156], \
                       [3.0669, 0.2412, 10.7805, 61.4135], [1.93816, 4.14553, 0.228753, 8.28524]
    c_eu, c_ga, c_al = 1.36389, 1.7189, 0.706786
    ####################################################################################################################

    ####################################################################################################################
    # FINAL CALCULATION & PLOTTING
    I = F_Q(k2d, l2d, h, n, a_eu, a_ga, a_al, b_eu, b_ga, b_al, c_eu, c_ga, c_al, u_list, theta, z0, lamb, DBW=True)
    plotfunction(k2d, l2d, I, boundary, diff, h=h, k0=k0, l0=l0, vmax=1/10, savefig=False, norm=True)
    # print("max(I)/min(I)= {}".format(np.max(I) / np.min(I)))
    ####################################################################################################################

    ####################################################################################################################
    # PRINT STATEMENTS
    print("k_max = {}".format(boundary))
    print("l_max = {}".format(boundary))
    print("delta_k = {}".format(diff))
    print("H (r.l.u) = {}".format(h))
    print("# of kspace points n = {}".format(n))
    print("Crystallographic input in angstroem: a = {}, c = {}, z0 = {}".format(a, c, z0))
    print("Isotropic displacement <u^2> for Eu = {} and Ga = {} and Al = {}".format(u_list[0], u_list[1], u_list[2]))
    ####################################################################################################################


if __name__ == '__main__':
    main()