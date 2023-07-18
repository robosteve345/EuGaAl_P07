"""Single-crystal XRD intensity map simulation for CDW phase in EuGa2Al2
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def translation(x, storage_x, storage_y, storage_z, i):
    """
    Translate atomic positions of X_x position
    """
    x_transl = x + np.array([0, 0, i])
    storage_x.append(x_transl[0])
    storage_y.append(x_transl[1])
    storage_z.append(x_transl[2])


def modulation(q_cdw):
    """Ga z modulation."""
    # # Modify Gallium atoms positions (if wished)
    # print("z_Ga1_modulation={}".format(np.round(sin_modulation(z_Ga1, z_ampl, q_cdw), 2))), print("z_Ga1_T_modulation={}".format(np.round(sin_modulation(z_Ga1_T, z_ampl, q_cdw), 2))), print("z_Ga2_modulation={}".format(np.round(sin_modulation(z_Ga2, z_ampl, q_cdw), 2))), print("z_Ga2_T_modulation={}".format(np.round(sin_modulation(z_Ga2_T, z_ampl, q_cdw), 2)))
    # z_Ga1_mod = z_Ga1 #+ sin_modulation(z_Ga1, z_ampl, q_cdw)
    # z_Ga1_T_mod = z_Ga1_T #+ sin_modulation(z_Ga1_T, z_ampl, q_cdw)
    # z_Ga2_mod = z_Ga2# + sin_modulation(z_Ga2, z_ampl, q_cdw)
    # z_Ga2_T_mod = z_Ga2_T #+ sin_modulation(z_Ga2_T, z_ampl, q_cdw)
    # Ga_mod = z_Ga1_mod, z_Ga1_T_mod, z_Ga2_mod, z_Ga1_T_mod
    # print("z_Ga1={}".format(z_Ga1_mod)), print("z_Ga1_T={}".format(z_Ga1_T_mod)), print("z_Ga2={}".format(z_Ga2_mod)), print("z_Ga2_T={}".format(z_Ga2_T_mod))
    # print("# Ga = {}".format(len(z_Ga1_mod) + len(z_Ga1_T_mod) + len(z_Ga2_mod) + len(z_Ga2_T_mod)))
    # """Plot Ga_z positions"""
    # for i in Ga_mod:
    #     plt.scatter(i, np.ones(10)*1, marker='.')
    #     plt.tick_params(direction='in', length=200, width=1, zorder=1, colors='k',
    #                     grid_color='tab:orange', grid_alpha=0.25, axis='x', top=True)
    #     plt.xticks(np.arange(0, 11, 1))
    #     plt.xlim(0)
    #     plt.yticks([])
    #     plt.xlabel(r"$Ga_z$")
    # plt.show()
    # plt.scatter(z_Ga1_T, np.ones(11) * 0.5, c='g', marker='.')
    # plt.scatter(z_Ga2_T, np.ones(11) * 0.5, c='b', marker='.')
    # #################################################################
    z_ampl = -0.08
    print("CDW Ga-modulation z' = A*sin(q_cdw*z*2*pi):")
    print("q_cdw={}, A={}".format(q_cdw, z_ampl))
    zp = 0.1  # relative position Ga-Al in z-direction, FESTER WERT
    z, dz, weird = [], [], []
    for i in range(1, 41):
        z.append((i / 4 - 1 / 8) + 1 / 8 * (-1) ** (i + 1) + zp * (-1) ** i)
        # Modulation 1: (-1)**i * z_ampl * np.sin(2*np.pi* q_cdw * z)
        dz.append((-1) ** i * z_ampl * (
            np.sin(2 * np.pi * q_cdw * (i / 4 - 1 / 8) + 1 / 8 * (-1) ** (i + 1) + zp * (-1) ** i)))
        weird.append((i / 4 + 1 / 8 + 1 / 8 * (-1) ** (i + 1)))
    # print("input for sin from marein: {}".format(weird))
    print("Gallium z-positions and deviations dz:")
    print("z={}".format(np.round(z, 2)), "# z={}".format(len(z)),
          "dz={}".format(np.round(dz, 2)), "# dz={}".format(len(dz)),
          "z+dz={}".format(np.asarray(z) + np.asarray(dz)))
    mod = np.asarray(z) + np.asarray(dz)  # modulated ga_z positions

    return mod


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


def kspacecreator(k_boundary, l_boundary, diff, k0, l0):
    """Create kspace with integer points evenly distributed with periodic boundaries in KL
    :returns: n: integer, dimension of k-space
              k2d, l2d: ndarrays that resemble K and L in k-space
    """
    n = np.intc(k_boundary / diff * 2 + 1)
    k = np.linspace(k0 - k_boundary, k0 + k_boundary, n)
    l = np.linspace(l0 - l_boundary, l0 + l_boundary, n)
    k2d, l2d = np.meshgrid(k, l)
    print("#n={}".format(n))
    return n, k2d, l2d


def plotfunction(k2d, l2d, I, h, vmax, scatterfactor, savefig=False, norm=False):
    """Plots intensity I on k, l meshgrid via imshow. Initially deletes forbidden intensities for
    certain kspace points governed by the LAUE condition.
    """
    cmap = "viridis"
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(('XRD intensity map simulation, H = {}').format(h), size=15)
    print("I dim: {}x{}".format(I.shape[0], I.shape[1]))
    # 2d scatter plot
    # ax.scatter(k2d, l2d, s=I * scatterfactor, linewidth=0.05, c='k')
    ax.set_ylabel(r"L (r.l.u.)", fontsize=15)
    ax.set_xlabel(r"K (r.l.u.)", fontsize=15)
    ax.tick_params(axis='x', labelsize=15, direction='in')
    ax.tick_params(axis='y', labelsize=15, direction='in')
    if norm == True:
        pos = ax.imshow(I / np.max(I), cmap=cmap, interpolation='gaussian',
                        extent=[np.min(k2d), np.max(k2d), np.min(l2d), np.max(l2d)],
                        vmax=vmax
                        )
    else:
        pos = ax.imshow(I, cmap=cmap, interpolation='gaussian',
                        extent=[np.min(k2d), np.max(k2d), np.min(l2d), np.max(l2d)],
                        vmax=vmax
                        )
    fig.colorbar(pos, ax=ax, label='Intensity (r.l.u.)')
    if savefig == True:
        plt.savefig('{}kl_CDW_EuGa2Al2'.format(h), dpi=300)
        plt.show()
    else:
        plt.show()

    # fig, axs = plt.subplots(2, 2, figsize=(10, 5))
    # fig.suptitle(r'CDW modulation simulation for $EuGa_2Al_2$')
    # axs[0, 0].scatter(np.asarray(z) + np.asarray(dz), np.ones(40) * 0.5, c='r', marker='.', label='Modulated')
    # axs[0, 0].scatter(np.asarray(z), np.ones(40) * 0.5, c='k', marker='.', label='Unmodulated')

    # plt.scatter(z_Ga1_T_mod, np.ones(10) * 0.5, c='r', marker='.')
    # plt.scatter(z_Ga2_mod, np.ones(10) * 0.5, c='k', marker='.', label='Wyckoff 4e(2)')
    # plt.tick_params(direction='in', length=200, width=1, zorder=1, colors='k', grid_linestyle='--',
    #                grid_color='tab:orange', grid_alpha=0.25, axis='x', top=True)
    # print("every_1st={}".format(every_1st), "every_2nd={}".format(every_2nd), "every_3rd={}".format(every_3rd),
    #      "every_4th={}".format(every_4th))
    # axs[0, 0].set_xlim(0, 10)
    # axs[0, 0].set_ylim(0.48, 0.52)
    # axs[0, 0].legend(fontsize=15)
    # axs[0, 0].set_xticks(np.arange(0, 11, 1))
    # axs[0, 0].tick_params(axis='x', labelsize=15, direction='in')
    # axs[0, 0].tick_params(direction='in')
    # axs[0, 0].set_yticks([])
    # axs[0, 0].set_xlabel(r"$z$", fontsize=15)


def F_Q(diff, k2d, l2d, k_boundary, l_boundary, mod, q_cdw, h, n, a_eu, a_ga, a_al, b_eu, b_ga, b_al, c_eu, c_ga, c_al, u_list, theta, z0, lamb, DBW=False):
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

    every_1st, every_2nd, every_3rd, every_4th = mod[0::4], \
                                                 mod[1::4], mod[2::4], mod[3::4]  # transform 40 array into 4*10 arrays

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

    # Final Atomic vector positions
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

    # Compute DBW factors
    B_iso_Eu, B_iso_Ga, B_iso_Al = 8 * np.pi ** 2 / 3 * u_list[0], 8 * np.pi ** 2 / 3 * u_list[1], \
                                   8 * np.pi ** 2 / 3 * u_list[2]  # ISOTROPIC
    B_iso_list = [B_iso_Eu, B_iso_Ga, B_iso_Al]
    if DBW == True:
        DBW_Eu = np.exp(-B_iso_list[0] / lamb ** 2 * (np.sin(theta)) ** 2)
        DBW_Ga = np.exp(-B_iso_list[1] / lamb ** 2 * (np.sin(theta)) ** 2)
        DBW_Al = np.exp(-B_iso_list[2] / lamb ** 2 * (np.sin(theta)) ** 2)
    else:
        DBW_Eu = np.ones(n, n)
        DBW_Ga = np.ones(n, n)
        DBW_Al = np.ones(n, n)

    """Structure Factor F_hkl"""
    F_Eu_list, F_Eu_T_list = [], []  # Europium Eu
    F_Ga1_list, F_Ga1_T_list, F_Ga2_list, F_Ga2_T_list = [], [], [], []  # Gallium Ga
    F_Al1_list, F_Al1_T_list, F_Al2_list, F_Al2_T_list = [], [], [], []  # Alluminium Al
    # fill exponential components in each structure factor list with size nxn
    for i in range(10):
        F_Eu_list.append(f_Eu * DBW_Eu * np.exp(
            1j * 2 * np.pi * (h * np.ones((n, n)) * Eu[0, i] + k2d * Eu[1, i] + l2d * Eu[2, i]))
                         )  # 2 Europium atoms
        F_Eu_T_list.append(f_Eu * DBW_Eu * np.exp(
            1j * 2 * np.pi * (h * np.ones((n, n)) * Eu_T[0, i] + k2d * Eu_T[1, i] + l2d * Eu_T[2, i]))
                           )
        F_Ga1_list.append(f_Ga * DBW_Ga * np.exp(
            1j * 2 * np.pi * (h * np.ones((n, n)) * Ga1[0, i] + k2d * Ga1[1, i] + l2d * Ga1[2, i]))
                          )  # 4 modulated Gallium atoms
        F_Ga1_T_list.append(f_Ga * DBW_Ga * np.exp(
            1j * 2 * np.pi * (h * np.ones((n, n)) * Ga1_T[0, i] + k2d * Ga1_T[1, i] + l2d * Ga1_T[2, i]))
                            )
        F_Ga2_list.append(f_Ga * DBW_Ga * np.exp(
            1j * 2 * np.pi * (h * np.ones((n, n)) * Ga2[0, i] + k2d * Ga2[1, i] + l2d * Ga2[2, i]))
                          )
        F_Ga2_T_list.append(f_Ga * DBW_Ga * np.exp(
            1j * 2 * np.pi * (h * np.ones((n, n)) * Ga2_T[0, i] + k2d * Ga2_T[1, i] + l2d * Ga2_T[2, i]))
                            )
        F_Al1_list.append(f_Al * DBW_Al * np.exp(
            1j * 2 * np.pi * (h * np.ones((n, n)) * Al1[0, i] + k2d * Al1[1, i] + l2d * Al1[2, i]))
                          )
        F_Al1_T_list.append(f_Al * DBW_Al * np.exp(
            1j * 2 * np.pi * (h * np.ones((n, n)) * Al1_T[0, i] + k2d * Al1_T[1, i] + l2d * Al1_T[2, i]))
                            )
        F_Al2_list.append(f_Al * DBW_Al * np.exp(
            1j * 2 * np.pi * (h * np.ones((n, n)) * Al2[0, i] + k2d * Al2[1, i] + l2d * Al2[2, i]))
                          )
        F_Al2_T_list.append(f_Al * DBW_Al * np.exp(
            1j * 2 * np.pi * (h * np.ones((n, n)) * Al2_T[0, i] + k2d * Al2_T[1, i] + l2d * Al2_T[2, i]))
                            )
    F_str = np.zeros((n, n), dtype=complex)  # n dimension of k-space
    atoms_list = [F_Eu_list, F_Eu_T_list, F_Ga1_list, F_Ga1_T_list, F_Ga2_list, F_Ga2_T_list, F_Al1_list, F_Al1_T_list,
                  F_Al2_list, F_Al2_T_list]
    pre_F_str = [np.zeros((n, n))]
    # Zusammenfügen der Formfaktoren für die Atomsorten
    for i in atoms_list:  # put together the lists in a ndarray for each atom with each N positions, to get rid of 3rd dimension (better workaround probably possible...)
        pre_F_str = np.add(pre_F_str, i)
    for i in range(len(pre_F_str)):
        F_str = F_str + pre_F_str[i]

    """Intensity I"""
    I = np.absolute(F_str) ** 2  # un
    #### MANIPULATE I so only allowed Q-transfers allowed (governed by CDW)
    c = np.intc(q_cdw / diff)  # every #th desired kpoint, so that evaluation occurs at F(Q) with CDW symmetry
    desiredkpoints = k2d[0::c]
    print("every #={}".format(c))
    print("desiredkpoints={}".format(desiredkpoints))
    indices = np.intc(c * np.arange(0, 2 * l_boundary / (diff * c) + 1, 1))
    indices_k = np.intc(1 / diff * np.arange(0, 2 * k_boundary + 1, 1) + 0)
    print("foundindices={}".format(indices))
    print("l2d[indices]={}".format(l2d[indices, 0]))
    print("indices_k={}".format(indices_k))
    print("k2d[indices_k]={}".format(k2d[0, indices_k]))
    for i in range(n):
        if i in np.intc(indices):
            I[i, :] = I[i, :]
        else:
            I[i, :] = 0
            for j in range(n):
                if j in np.intc(indices_k):
                    I[:, j] = I[:, j]
                else:
                    I[:, j] = 0
    print("Intensity I_after = {}".format(I))

    return I


def main():
    print(__doc__)
    scatterfactor = 1e-5 # For 2d scatter-plot
    # CDW turn OFF/ON: z_ampl
    ########################################################################################
    ### BENCHMARKED: ((boundary, boundary): [diff]): (5, 5): [0.05, 0.02, 0.01], (2,2): [0.01, 0.02, 0.05], (1,1): [0.01, 0.005]
    #### INPUT KSPACE: boundary / diff ≈ 100 sufficient resolution
    k_boundary, l_boundary = 10, 10 # Boundary in kspace
    h = 0 # H plane in kspace (r.l.u)
    diff = 0.05 # Distance between kspace-points
    k0, l0 = 0, 0  # centering of kspace in (r.l.u.)
    n, k2d, l2d = kspacecreator(k_boundary, l_boundary, diff, k0, l0)
    ### For CDW Plot
    a = 4.254 # in Å
    c = 10.397  # in Å
    z0 = 0.390 # 4e Wyckoff degree of freedom
    ### INPUT FOR DBW
    lamb = 7e-10  # x-ray wavelength in m (Mo Ka)
    u_Eu, u_Ga, u_Al = 1e-2, 1e-2, 1e-2  # Isotropic displacements <u^2> in Å^2, ~1/m. For more detail take bonding angles,
    # distances into account.
    u_list = [u_Eu, u_Ga, u_Al]
    theta = thetacalc(lamb, k2d, l2d, n, h, a, c)
    ### INPUT FOR GA-modulation (CDW)
    q_cdw = 0.10  # in r.l.u.
    mod = modulation(q_cdw) # Compute modulated Ga-z-positions
    ### FORM FACTOR INPUT
    # Atomic form factors according to de Graed, structure of materials, chapter 12: Eu2+. Ga1+, Al3+
    # http://lampx.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php
    a_eu, a_ga, a_al = [24.0063, 19.9504, 11.8034, 3.87243], [15.2354, 6.7006, 4.3591, 2.9623], \
                       [4.17448, 3.3876, 1.20296, 0.528137]
    b_eu, b_ga, b_al = [2.27783, 0.17353, 11.6096, 26.5156], [3.0669, 0.2412, 10.7805, 61.4135], \
                       [1.93816, 4.14553, 0.228753, 8.28524]
    c_eu, c_ga, c_al = 1.36389, 1.7189, 0.706786
    ########################################################################################

    ####################################################################################################################
    # FINAL CALCULATION & PLOTTING
    I = F_Q(diff, k2d, l2d, k_boundary, l_boundary, mod, q_cdw,
            h, n, a_eu, a_ga, a_al, b_eu, b_ga, b_al, c_eu,
            c_ga, c_al, u_list, theta, z0, lamb, DBW=True)
    plotfunction(k2d, l2d, I, h, vmax=0.01, scatterfactor=scatterfactor,
                 savefig=True, norm=True)
    ####################################################################################################################

    ####################################################################################################################
    # PRINT STATEMENTS
    print("Centered peak: [HKL] = [{}{}{}]".format(h, k0, l0))
    print("k_max = {}".format(k_boundary))
    print("l_max = {}".format(l_boundary))
    print("delta_k = {}".format(diff))
    print("H (r.l.u) = {}".format(h))
    print("Kspace dimension = {} x {}".format(n, n))
    print("Crystallographic input in angstroem: a = {}, c = {}, z0 = {}".format(a, c, z0))
    print("Scaling factor for Plot = {}".format(scatterfactor))
    print("Isotropic displacement <u^2> for Eu = {} and Ga = {} and Al = {}".format(u_list[0], u_list[1], u_list[2]))
    ####################################################################################################################


if __name__ == '__main__':
    main()