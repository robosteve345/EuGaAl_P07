"""-- EuGa4 single-crystal XRD intensity map simulation --
DBW with isotropic B considered. Atomic form factor in series expansion considered.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rc('text', usetex=True)
plt.rc('font', family='sans-serif')

def gaussian2d(A, X, Y, s):
    return A/s * np.exp(-(X**2 + Y**2)/(2*s))


def plotfunction(k2d, l2d, I, boundary, diff, h, vmax, savefig=False, norm=False):
    """Plots intensity I on k, l meshgrid via imshow. Initially deletes forbidden intensities for
    certain kspace points governed by the LAUE condition.
    """
    cmap = "viridis"
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # ax.legend(loc='upper right')
    ax.set_title(r'XRD intensity map simulation, H={}'.format(h))
    ax.set_xlabel('K (r.l.u.)')
    ax.set_ylabel('L (r.l.u.)')
    ##############################################################################
    # For manual convolution (not needed)
    # Kernel = gaussian2d(A, k2d, l2d, s)
    # print("Kernel dim: {}x{}".format(Kernel.shape[0], Kernel.shape[1]))
    ##############################################################################
    # integers = np.arange(-boundary, boundary + 1, 1)
    # print("integers={}".format(integers))
    indices = 1 / diff * np.arange(0, 2 * boundary + 1, 1) + 0
    # print("newindices={}".format(np.intc(indices)))
    # print("I_before={}".format(I)) # Intensity evaluated at every kpoint.
    # ADD NOISE
    # I = I  # + np.max(I) / 20 * np.abs(np.random.randn(n, n))
    # Manually set F(Q) elements to zero, that are forbidden by crystal symmetry.
    for i in range(len(k2d)):
        if i in np.intc(indices):
            I[i, :] = I[i, :]
        else:
            I[i, :] = 0
            for j in range(len(l2d)):
                if j in np.intc(indices):
                    I[:, j] = I[:, j]
                else:
                    I[:, j] = 0
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
    fig.colorbar(pos, ax=ax, label='Intensity')
    if savefig == True:
        plt.savefig('{}kl_EuGa4'.format(h), dpi=300)
        plt.show()
    else:
        plt.show()


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


def F_Q(k2d, l2d, h, n, a_eu, a_ga, b_eu, b_ga, c_eu, c_ga, u_list, theta, z0, lamb, DBW=False):
    """ Crystal symmetry input"""
    Eu = np.array([[0, 0, 0], [0, 0, 1]]).T  # For now: two cells.
    Eu_T = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 1.5]]).T
    Ga1 = np.array([[0, 0, z0], [0, 0, z0 + 1]]).T
    Ga1_T = np.array([[0.5, 0.5, z0 + 0.5], [0.5, 0.5, 1.5 + z0]]).T
    Ga2 = np.array([[0, 0, -z0], [0, 0, 1.0 - z0]]).T
    Ga2_T = np.array([[0.5, 0.5, 0.5 - z0], [0.5, 0.5, 1.5 - z0]]).T
    Ga3 = np.array([[0, 0.5, 0.25], [0, 0.5, 0.25 + 1]]).T
    Ga3_T = np.array([[0.5, 1, 0.75], [0.5, 1, 0.75 + 1]]).T
    Ga4 = np.array([[0.5, 0, 0.25], [0.5, 0, 0.25 + 1]]).T
    Ga4_T = np.array([[1, 0.5, 0.75], [1, 0.5, 0.75 + 1]]).T
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

    F_Eu_list, F_Eu_T_list = [], []
    F_Ga1_list, F_Ga1_T_list, F_Ga2_list, F_Ga2_T_list = [], [], [], []
    F_Ga3_list, F_Ga3_T_list, F_Ga4_list, F_Ga4_T_list = [], [], [], []

    # Compute DBW factors
    B_iso_Eu, B_iso_Ga = 8 * np.pi ** 2 / 3 * u_list[0], 8 * np.pi ** 2 / 3 * u_list[1] # ISOTROPIC
    B_iso_list = [B_iso_Eu, B_iso_Ga]
    if DBW == True:
        DBW_Eu = np.exp(-B_iso_list[0] / lamb **2 * (np.sin(theta)) ** 2)
        DBW_Ga = np.exp(-B_iso_list[1] / lamb **2 *  (np.sin(theta)) ** 2)
    else:
        DBW_Eu = np.ones(n, n)
        DBW_Ga = np.ones(n, n)

    for i in range(2):  # 2 atoms for each sort, for Eu with 1 position and 2 for each al and ga with translation
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
        F_Ga3_list.append(
            f_Ga * DBW_Ga * np.exp(
                1j * 2 * np.pi * (h * np.ones((n, n)) * Ga3[0, i] + k2d * Ga3[1, i] + l2d * Ga3[2, i]))
        )
        F_Ga3_T_list.append(
            f_Ga * DBW_Ga * np.exp(
                1j * 2 * np.pi * (h * np.ones((n, n)) * Ga3_T[0, i] + k2d * Ga3_T[1, i] + l2d * Ga3_T[2, i]))
        )
        F_Ga4_list.append(
            f_Ga * DBW_Ga* np.exp(
                1j * 2 * np.pi * (h * np.ones((n, n)) * Ga4[0, i] + k2d * Ga4[1, i] + l2d * Ga4[2, i]))
        )
        F_Ga4_T_list.append(
            f_Ga * DBW_Ga * np.exp(
                1j * 2 * np.pi * (h * np.ones((n, n)) * Ga4_T[0, i] + k2d * Ga4_T[1, i] + l2d * Ga4_T[2, i]))
        )

    """The glorious structure factor"""
    F_str = np.zeros((n, n), dtype=complex)  # Structure factor with dimensions nxn
    atoms_list = [F_Eu_list, F_Eu_T_list, F_Ga1_list, F_Ga1_T_list, F_Ga2_list, F_Ga2_T_list, F_Ga3_list, F_Ga3_T_list,
                  F_Ga4_list,
                  F_Ga4_T_list]  # create list with components corresponding to F evaluated for all atomic positions of element X with all k-space points
    pre_F_str = [np.zeros((n, n))]  # Hilfsgröße, in nächstem loop beschrieben
    # Zusammenfügen der Formfaktoren für die Atomsorten
    for i in atoms_list:  # put together the lists in a ndarray for each atom with each N positions, to get rid of 3rd dimension (better workaround probably possible...)
        pre_F_str = np.add(pre_F_str, i)
    for i in range(len(pre_F_str)):
        F_str = F_str + pre_F_str[i]

    I = np.absolute(F_str) ** 2

    return I


def main():
    print(__doc__)
    ####################################################################################################################
    ####### INPUT KSPACE: boundary / diff ≈ 100 sufficient resolution
    boundary = 15  # Symmetric boundaries for K, L
    diff = 0.2  # Distance between two kspace points
    h = 1 # H plane in kspace
    n, k2d, l2d = kspacecreator(boundary, diff)
    vmax=0.5
    ####### INPUT CRYSTAL SYMMETRY
    a = 4.40696  # in Å
    c = 10.68626  # in Å
    z0 = 0.3854  # 4e Wyckoff z parameter
    ####### INPUT FOR DBW
    lamb = 7e-10 # x-ray wavelength in m (Mo Ka)
    u_Eu, u_Ga = 1e-2, 1e-2 # Isotropic displacements <u^2> in Å^2, ~1/m. For more detail take bonding angles, distances into account.
    u_list = [u_Eu, u_Ga]
    theta = thetacalc(lamb, k2d, l2d, n, h, a, c)
    ####### INPUT FOR ATOMIC FORM FACTORS
    # Atomic form factors according to de Graed, structure of materials, chapter 12: Eu2+. Ga1+, Al3+
    # http://lampx.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php
    a_eu, a_ga = [24.0063, 19.9504, 11.8034, 3.87243], [15.2354, 6.7006, 4.3591, 2.9623]
    b_eu, b_ga = [2.27783, 0.17353, 11.6096, 26.5156], [3.0669, 0.2412, 10.7805, 61.4135]
    c_eu, c_ga = 1.36389, 1.7189
    ####################################################################################################################

    ####################################################################################################################
    # PRINT STATEMENTS
    print("k_max = {}".format(boundary))
    print("l_max = {}".format(boundary))
    print("delta_k = {}".format(diff))
    print("H (r.l.u) = {}".format(h))
    print("# of kspace points n = {}".format(n))
    print("Crystallographic input in angstroem: a = {}, c = {}, z0 = {}".format(a, c, z0))
    print("Isotropic displacement <u^2> for Eu = {} and Ga = {}".format(u_list[0], u_list[1]))
    ####################################################################################################################

    ####################################################################################################################
    # FINAL CALCULATION & PLOTTING
    I = F_Q(k2d, l2d, h, n, a_eu, a_ga, b_eu, b_ga, c_eu, c_ga, u_list, theta, z0, lamb, DBW=True)
    plotfunction(k2d, l2d, I, boundary, diff, h=h, vmax = vmax, savefig=True, norm=True)
    # print("max(I)/min(I)= {}".format(np.max(I) / np.min(I)))
    ####################################################################################################################


if __name__ == '__main__':
    main()