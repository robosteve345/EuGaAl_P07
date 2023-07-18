"""XRD_Simulation of CDW pattern UNIT CELL in k-space for one unit cell of EuGa4 with symmetry I4/mmm
The only difference to EuGa2Al2 is the different contributions to the reflexes by the substitution from 4d: Ga->Al in
form of the different atomic form factors.
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def kspacecreator(boundary, h):
    """Create kspace with integer points evenly distributed with periodic boundaries in KL
    :returns: n: integer, dimension of k-space
              k2d, l2d: ndarrays that resemble K and L in k-space
    """
    k = np.arange(-boundary, boundary + 1, 1)  # np.linspace(-boundary, boundary, n)
    l = np.arange(-boundary, boundary + 1, 1)  # np.linspace(-boundary, boundary, n)
    k2d, l2d = np.meshgrid(k, l)
    n = len(k)
    h=h
    return n, k2d, l2d, h


def plotallthisshit(k2d, l2d, I, boundary, h, scatterfactor):
    """Plot all this shit
    """
    # # imshow
    # color = 'binary'
    fig = plt.figure(figsize=(6,7))
    # #fig.suptitle('XRD of EuGa2Al2')
    # ax = fig.add_subplot(1, 2, 1)
    # im = ax.imshow(I, cmap=color, vmin=abs(I).min(), vmax=abs(I).max(),
    #                extent=[-boundary, boundary, -boundary, boundary], 
    #                interpolation='catrom')
    # print("I_min ={}".format(abs(I).min()))
    # print("I_min ={}".format(abs(I).max()))
    # #im.set_interpolation('gaussian')
    # # interpolation methods: 'nearest', 'bilinear', 'bicubic', 'spline16',
    # #            'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
    # #            'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
    # # HOW DOES IT CONVOLVE HOW DOES INTERPOLATION WORK???
    # cb = fig.colorbar(im, ax=ax) #shrink=1, aspect=1
    # # Set up a figure twice as tall as it is wide
    # #ax.set_title('HKL-plot')
    # ax.set_xlabel('k')
    # ax.set_ylabel('l')

    # # 2d contourplot
    # ax = fig.add_subplot(1, 2, 1)
    # levels = np.linspace(np.min(I), np.max(I), 1000)
    # contourplot = ax.contourf(k2d, l2d, I, levels=levels, cmap=color)
    # fig.colorbar(contourplot)  # Add a colorbar to a plot
    # ax.set_title('HKL-plot')
    # ax.set_xlabel('k')
    # ax.set_ylabel('l')

    # 2d scatter plot
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(-boundary + 10, boundary - 10)
    ax.set_ylim(-boundary, boundary)
    ax.scatter(k2d, l2d, s=scatterfactor * I, linewidth=1.0, c='k')
    #ax.set_yticks(np.arange(-boundary+2, boundary, 2))
    #ax.set_xticks(np.arange(-boundary+2, boundary, 2))
    ax.set_xlabel(r'K (r.l.u.)', fontsize=20)
    ax.set_ylabel(r'L (r.l.u.)', fontsize=20)
    plt.tick_params(axis='x', labelsize=17, direction='in')
    plt.tick_params(axis='y', labelsize=15, direction='in')

    # 3d surface plot
    # ax = plt.axes(projection='3d')
    # surface = ax.plot_surface(k2d, l2d, I, rstride=1, cstride=1,
    #                         linewidth=0, antialiased=False, cmap='viridis')
    # fig.colorbar(surface, shrink=1, aspect=5)
    
    # # 3d scatter plot
    # ax = fig.add_subplot(4, 1, 4, projection='3d')
    # ax.scatter(k2d, l2d, I, c=I, cmap='plasma', linewidth=0.5)

    plt.savefig('{}kl_euga4_oneunitcell.jpg'.format(h), dpi=500)
    plt.show()


def main():
    print(__doc__)
    boundary = 16
    h = 1
    n, k2d, l2d, h = kspacecreator(boundary, h)
    scatterfactor = 0.00005
    # Strcture parameters
    a = 4.40696  # in angstrom
    c = 10.68626  # in angstrom
    z0 = 0.3854  # Equilibrium 4e Wyckoff z parameter. experimentally for T = 293 K
    print("K-space dimension n={}".format(n))
    print("H={}".format(h))
    print("Crystallographic input in angstroem: a={}, c={}, z0={}".format(a, c, z0))
    print("Scaling factor for Plot={}".format(scatterfactor))

    Eu = np.array([[0, 0, 0], [0, 0, 1]]).T  # For now: two cells.
    Eu_T = np.array([[0.5, 0.5, 0.5], [0.5, 0.5,1.5]]).T
    Ga1 = np.array([[0, 0, z0], [0, 0, z0 + 1]]).T  
    Ga1_T = np.array([[0.5, 0.5, z0+0.5], [0.5, 0.5, 1.5+z0]]).T
    Ga2 = np.array([[0, 0, -z0], [0, 0, 1.0 - z0]]).T
    Ga2_T = np.array([[0.5, 0.5, 0.5-z0], [0.5, 0.5, 1.5-z0]]).T
    Ga3 = np.array([[0, 0.5, 0.25], [0, 0.5, 0.25 + 1]]).T
    Ga3_T = np.array([[0.5, 1, 0.75], [0.5, 1, 0.75 + 1]]).T
    Ga4 = np.array([[0.5, 0, 0.25], [0.5, 0, 0.25 + 1]]).T
    Ga4_T = np.array([[1, 0.5, 0.75], [1, 0.5, 0.75 + 1]]).T


    """Form factor calculation"""
    # Form factors according to de Graed, structure of materials, chapter 12: Eu2+. Ga1+, Al3+
    a_eu, a_ga, a_al  = [24.0063, 19.9504, 11.8034, 3.87243], [15.2354, 6.7006, 4.3591, 2.9623], [4.17448, 3.3876, 1.20296, 0.528137]
    b_eu, b_ga, b_al = [2.27783, 0.17353, 11.6096, 26.5156], [3.0669, 0.2412, 10.7805, 61.4135], [1.93816, 4.14553, 0.228753, 8.28524]
    # c_eu, c_ga, c_al = 1.36389, 1.7189, 0.706786
    f_eu1 = a_eu[0] * np.exp(- b_eu[0] * (((h * np.ones((n, n)) )**2 + k2d**2 + l2d**2) / (4 * np.pi)**2  ))
    f_eu2 = a_eu[1] * np.exp(- b_eu[1] * (((h * np.ones((n, n))) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    f_eu3 = a_eu[2] * np.exp(- b_eu[2] * (((h * np.ones((n, n))) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    f_eu4 = a_eu[3] * np.exp(- b_eu[3] * (((h * np.ones((n, n))) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    f_Eu = f_eu1+f_eu2+f_eu3+f_eu4 
    f_ga1 = a_ga[0] * np.exp(- b_ga[0] * (((h * np.ones((n, n)) )**2 + k2d**2 + l2d**2) / (4 * np.pi)**2  ))
    f_ga2 = a_ga[1] * np.exp(- b_ga[1] * (((h * np.ones((n, n))) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    f_ga3 = a_ga[2] * np.exp(- b_ga[2] * (((h * np.ones((n, n))) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    f_ga4 = a_ga[3] * np.exp(- b_ga[3] * (((h * np.ones((n, n))) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    f_Ga = f_ga1+f_ga2+f_ga3+f_ga4 
    # f_al1 = a_al[0] * np.exp(- b_al[0] * (((h * np.ones((n, n)) )**2 + k2d**2 + l2d**2) / (4 * np.pi)**2  ))
    # f_al2 = a_al[1] * np.exp(- b_al[1] * (((h * np.ones((n, n))) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    # f_al3 = a_al[2] * np.exp(- b_al[2] * (((h * np.ones((n, n))) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    # f_al4 = a_al[3] * np.exp(- b_al[3] * (((h * np.ones((n, n))) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    # f_Al = f_ga3+f_al2+f_al3+f_al4

    # ########################################################################
    # """ Debye-Waller factor"""
    lamb = 0.0000000001 # Wellenlänge
    # Compute all distances corresponding to lattice diffractions with hkl
    d_hkl = a/(np.sqrt( (h * np.ones((n, n))) **2 + k2d**2 + (a/c)**2 * l2d**2))
    # 1. compute all angles corresponding to the k points according to braggs law
    theta = np.arcsin(lamb/(2 * d_hkl))
    B_iso_Eu, B_iso_Ga = 0, 0 # isotropic movements around equilibrium, inverse proportional to the mass of the atom
    B_iso_list = [B_iso_Eu, B_iso_Ga]
    # ########################################################################
    
    F_Eu_list, F_Eu_T_list = [], []
    F_Ga1_list, F_Ga1_T_list, F_Ga2_list, F_Ga2_T_list = [],[],[],[]
    F_Ga3_list, F_Ga3_T_list, F_Ga4_list, F_Ga4_T_list = [],[],[],[]

    for i in range(2):  # 2 atoms for each sort, for Eu with 1 position and 2 for each al and ga with translation
        F_Eu_list.append(
            f_Eu * np.exp(-B_iso_list[0] / lamb * (np.sin(theta))**2) * np.exp(1j * 2 * np.pi * (h * np.ones((n, n)) * Eu[0, i] + k2d * Eu[1, i] + l2d * Eu[2, i]))
            )
        F_Eu_T_list.append(
            f_Eu * np.exp(-B_iso_list[0] / lamb * (np.sin(theta))**2) * np.exp(1j * 2 * np.pi * (h * np.ones((n, n)) * Eu_T[0, i] + k2d * Eu_T[1, i] + l2d * Eu_T[2, i]))
            )
        F_Ga1_list.append(
            f_Ga * np.exp(-B_iso_list[1] / lamb * (np.sin(theta))**2) * np.exp(1j * 2 * np.pi * (h * np.ones((n, n)) *Ga1[0, i] + k2d *Ga1[1, i] + l2d *Ga1[2, i]))
            )
        F_Ga1_T_list.append(
            f_Ga * np.exp(-B_iso_list[1] / lamb * (np.sin(theta))**2) * np.exp(1j * 2 * np.pi * (h * np.ones((n, n)) *Ga1_T[0,i] + k2d *Ga1_T[1, i] + l2d *Ga1_T[2, i]))
            )
        F_Ga2_list.append(
            f_Ga * np.exp(-B_iso_list[1] / lamb * (np.sin(theta))**2) * np.exp(1j * 2 * np.pi * (h * np.ones((n, n)) *Ga2[0, i] + k2d *Ga2[1, i] + l2d *Ga2[2, i]))
            )
        F_Ga2_T_list.append(
            f_Ga * np.exp(-B_iso_list[1] / lamb * (np.sin(theta))**2) * np.exp(1j * 2 * np.pi * (h * np.ones((n, n)) *Ga2_T[0, i] + k2d *Ga2_T[1, i] + l2d *Ga2_T[2, i]))
            )
        F_Ga3_list.append(
            f_Ga * np.exp(-B_iso_list[1] / lamb * (np.sin(theta))**2) * np.exp(1j * 2 * np.pi * (h * np.ones((n, n)) *Ga3[0, i] + k2d *Ga3[1, i] + l2d *Ga3[2, i]))
            )
        F_Ga3_T_list.append(
            f_Ga * np.exp(-B_iso_list[1] / lamb * (np.sin(theta))**2) * np.exp(1j * 2 * np.pi * (h * np.ones((n, n)) *Ga3_T[0, i] + k2d *Ga3_T[1, i] + l2d *Ga3_T[2, i]))
            )
        F_Ga4_list.append(
           f_Ga * np.exp(-B_iso_list[1] / lamb * (np.sin(theta))**2) * np.exp(1j * 2 * np.pi * (h * np.ones((n, n)) *Ga4[0, i] + k2d *Ga4[1, i] + l2d *Ga4[2, i]))
           )
        F_Ga4_T_list.append(
           f_Ga * np.exp(-B_iso_list[1] / lamb * (np.sin(theta))**2) * np.exp(1j * 2 * np.pi * (h * np.ones((n, n)) *Ga4_T[0, i] + k2d *Ga4_T[1, i] + l2d *Ga4_T[2, i]))
           )
        
    """The glorious structure factor"""
    F_str = np.zeros((n, n), dtype=complex)  # Structure factor with dimensions nxn
    atoms_list = [ F_Eu_list, F_Eu_T_list, F_Ga1_list, F_Ga1_T_list, F_Ga2_list, F_Ga2_T_list, F_Ga3_list, F_Ga3_T_list, F_Ga4_list, F_Ga4_T_list ]  #create list with components corresponding to F evaluated for all atomic positions of element X with all k-space points
    pre_F_str = [np.zeros((n, n))]  # Hilfsgröße, in nächstem loop beschrieben
    # Zusammenfügen der Formfaktoren für die Atomsorten
    for i in atoms_list:  # put together the lists in a ndarray for each atom with each N positions, to get rid of 3rd dimension (better workaround probably possible...)
        pre_F_str = np.add(pre_F_str, i)
    for i in range(len(pre_F_str)):
        F_str = F_str + pre_F_str[i]

    I = np.absolute(F_str) ** 2

    plotallthisshit(k2d, l2d, I, boundary, h, scatterfactor)


if __name__ == '__main__':
    main()
# def gaussian(theta, lamb, sigma):
#     gauss = (1 / (2 * np.pi * sigma)) * np.exp(-(theta ** 2 + lamb ** 2) / (2 * sigma))
#     return gauss


# def convolute2d(matrix, sigma, len_k2d):
#     """Faltung eines 2d-Datensatzes mit Gauß-peaks"""
#     boundary = len_k2d # muss gleiche dimension, wie k space haben, dehsalb len(k2d)=len(l2d)
#     x_conv, y_conv = np.meshgrid(np.linspace(-boundary,  boundary, 2*boundary+1), np.linspace(-boundary,  boundary, 2*boundary+1)) # np.meshgrid(np.arange(-boundary,  boundary, 0.1), np.arange(- boundary,  boundary, 0.1))
#     g = gaussian(x_conv, y_conv, sigma)
#     convolved = scipy.signal.convolve2d(matrix, g,  boundary='wrap', mode='same') #boundary='symm', mode='same')
#     plt.imshow(convolved, cmap='viridis', interpolation='gaussian', vmin=0, vmax=10)
#     plt.show()
