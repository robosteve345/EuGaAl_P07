"""XRD_Simulation of CDW pattern in k-space for 10 unit cells of EuGa2Al2 with symmetry I4/mmm
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# Times, Palatino, New Century Schoolbook, Bookman, Computer Modern Roman


def translation(x, storage_x, storage_y, storage_z, i):
    """
    Translate atomic positions of X_x position
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


def kspacecreator(boundary, h):
    """Create kspace with integer points evenly distributed with periodic boundaries in KL
    :returns: n: integer, dimension of k-space
              k2d, l2d: ndarrays that resemble K and L in k-space
    """
    k = np.arange(-boundary, boundary + 1, 1)  # np.linspace(-boundary, boundary, n)
    l = np.arange(-boundary, boundary + 1, 1)  # np.linspace(-boundary, boundary, n)
    
    k2d, l2d = np.meshgrid(k, l)
    # print(k2d, l2d)
    n = len(k)
    h=h
    return n, k2d, l2d, h


def plotallthisshit(k2d, l2d, I, h, scatterfactor):
    """Plot all this shit
    """
    # # imshow
    # color = 'binary'
    # fig = plt.figure(figsize = (20,20))
    # #fig.suptitle('XRD of EuGa2Al2')
    # ax = fig.add_subplot(1, 2, 1)
    # im = ax.imshow(I, cmap=color, vmin=abs(I).min(), vmax=abs(I).max(),
    #                extent=[-boundary, boundary, -boundary, boundary],
	# 			   interpolation='catrom')
    #
    # # interpolation methods: 'nearest', 'bilinear', 'bicubic', 'spline16',
    # #            'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
    # #            'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'
    # # HOW DOES IT CONVOLVE HOW DOES INTERPOLATION WORK???
    # cb = fig.colorbar(im, ax=ax, shrink=0.3, aspect=1)
    # print("I_min ={}".format(abs(I).min()))
    # print("I_min ={}".format(abs(I).max()))
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
    # ax = fig.add_subplot(1, 1, 1)
    # ax.scatter(k2d, l2d, s=I*scatterfactor, linewidth=1.0, c='k')
    # #ax.set_title('HKL-plot')
    # ax.set_xlabel('k')
    # ax.set_ylabel('l')

    # 3d surface plot
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # surface = ax.plot_surface(k2d, l2d, I, rstride=1, cstride=1,
    #                        linewidth=0, antialiased=False, cmap='viridis')
    # fig.colorbar(surface, shrink=1, aspect=5)
    # # 3d scatter plot
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # ax.scatter(k2d, l2d, I, c=I, cmap='viridis', linewidth=0.5)

    # plt.savefig('{}kl_10unitcells'.format(h), dpi=300)
    # plt.show()


def main():
    print(__doc__)
    ####### INPUT KSPACE: boundary / diff ≈ 100 sufficient resolution
    diff = 0.01  # Distance between two kspace points
    l_boundary, k_boundary = 2, 2
    h = -3
    ############################################ # For CDW Plot
    k0, l0 = 5, -6
    k = np.arange(k0 - k_boundary, k0 + k_boundary + 0.1, 0.1)
    l = np.arange(l0 - l_boundary, l0 + l_boundary + 0.1, 0.1)
    n = len(k)
    k2d, l2d = np.meshgrid(k, l)
    # print("#n={}".format(len(k)))
    # print("k2d={}".format(k2d))
    # print("l2d={}".format(l2d))
    ############################################
    scatterfactor = 0.00008
    a, c = 4.35, 10.9  # in angstrom
    z0 = 0.385
    # print("K-space dimension n={}".format(n))
    print("K-space dimension n={}".format(n))
    print("H={}".format(h))
    print("Crystallographic input in angstroem: a={}, c={}, z0={}".format(a, c, z0))
    print("Scaling factor for Plot={}".format(scatterfactor))
    
    ########################################################################################################
    ### FORM FACTOR CALCULATION
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
    f_al1 = a_al[0] * np.exp(- b_al[0] * (((h * np.ones((n, n)) )**2 + k2d**2 + l2d**2) / (4 * np.pi)**2  ))
    f_al2 = a_al[1] * np.exp(- b_al[1] * (((h * np.ones((n, n))) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    f_al3 = a_al[2] * np.exp(- b_al[2] * (((h * np.ones((n, n))) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    f_al4 = a_al[3] * np.exp(- b_al[3] * (((h * np.ones((n, n))) ** 2 + k2d ** 2 + l2d ** 2) / (4 * np.pi) ** 2))
    f_Al = f_al1+f_al2+f_al3+f_al4 
    #########################################################################################################
    ### INPUT CRYSTALLOGAPHY
    # Positions of atoms in 10 unit cells
    # Gebe ersten Vektor der verschobenen und nicht verschobenen Positionen der Atome an:
    # Translated positions about \vec{r} = (0.5,0.5,0.5)
    e
    # Full translation of Eu, Al, first translation without modulation of Ga:
    for i in range(1,10):
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
    z_ampl = np.array([-0.12, -0.01, -0.01, -0.01, -0.01, -0.01]) # -0.08
    q_cdw = 1/10  # in r.l.u.
    print("CDW Modulation A*sin(q_cdw*z*2*pi):")
    print("q_cdw={}, A={}".format(q_cdw, z_ampl))
    zp = 0.13 #relative position Ga-Al in z-direction, FESTER WERT
    z=[]
    dz, weird = [], []
    for i in range(1, 41):
        z.append((i/4 - 1/8) + 1/8*(-1)**(i+1)+zp*(-1)**i)
        # Modulation 1: (-1)**i * z_ampl * np.sin(2*np.pi* q_cdw * z)
        #dz.append((-1)**i * z_ampl * (np.sin(2*np.pi*q_cdw *  (i/4 - 1/8) + 1/8*(-1)**(i+1)+zp*(-1)**i)))
        # weird.append((i/4 + 1/8 + 1/8 * (-1)**(i+1)))

        # Modulation 2: fourier series with q_cdw=0.10
        dz.append((-1)**i * z_ampl[0] * ( np.cos(2*np.pi*q_cdw *  (i/4 - 1/8) + 1/8*(-1)**(i+1)+zp*(-1)**i) +  np.sin(2*np.pi*q_cdw *  (i/4 - 1/8) + 1/8*(-1)**(i+1)+zp*(-1)**i)) +
                  (-1)**i * z_ampl[1] * ( np.cos(2*2*np.pi*q_cdw *  (i/4 - 1/8) + 1/8*(-1)**(i+1)+zp*(-1)**i) +  np.sin(2*2*np.pi*q_cdw *  (i/4 - 1/8) + 1/8*(-1)**(i+1)+zp*(-1)**i)) +
                  (-1)**i * z_ampl[2] * ( np.cos(2*3*np.pi*q_cdw *  (i/4 - 1/8) + 1/8*(-1)**(i+1)+zp*(-1)**i) +  np.sin(2*3*np.pi*q_cdw *  (i/4 - 1/8) + 1/8*(-1)**(i+1)+zp*(-1)**i)) +
                  (-1) ** i * z_ampl[3] * (np.cos(4*
            2 * np.pi * q_cdw * (i / 4 - 1 / 8) + 1 / 8 * (-1) ** (i + 1) + zp * (-1) ** i) + np.sin(4*
            2 * np.pi * q_cdw * (i / 4 - 1 / 8) + 1 / 8 * (-1) ** (i + 1) + zp * (-1) ** i)) +
                  (-1) ** i * z_ampl[4] * (np.cos(
            5 * 2 * np.pi * q_cdw * (i / 4 - 1 / 8) + 1 / 8 * (-1) ** (i + 1) + zp * (-1) ** i) + np.sin(
            5 * 2 * np.pi * q_cdw * (i / 4 - 1 / 8) + 1 / 8 * (-1) ** (i + 1) + zp * (-1) ** i)) +
                  (-1) ** i * z_ampl[5] * (np.cos(
            6 * 3 * np.pi * q_cdw * (i / 4 - 1 / 8) + 1 / 8 * (-1) ** (i + 1) + zp * (-1) ** i) + np.sin(
            6 * 3 * np.pi * q_cdw * (i / 4 - 1 / 8) + 1 / 8 * (-1) ** (i + 1) + zp * (-1) ** i))
                  )
    # print("input for sin from marein: {}".format(weird))
    print("Gallium z-positions and deviations dz:")
    print("z={}".format(np.round(z, 2)), "# z={}".format(len(z)),
          "dz={}".format(np.round(dz, 2)), "# dz={}".format(len(dz)),
          "z+dz={}".format(np.asarray(z) + np.asarray(dz))
          )
    mod = np.asarray(z) + np.asarray(dz) # modulated ga_z positions
    fig, axs = plt.subplots(2, 2, figsize=(10,5))

    # fig.suptitle(r'CDW modulation simulation for $EuGa_2Al_2$')
    axs[0, 0].scatter(np.asarray(z) + np.asarray(dz), np.ones(40) * 0.5, c='r', marker='.', label='Modulated')
    axs[0, 0].scatter(np.asarray(z), np.ones(40) * 0.5, c='k', marker='.', label='Unmodulated')

    # plt.scatter(z_Ga1_T_mod, np.ones(10) * 0.5, c='r', marker='.')
    # plt.scatter(z_Ga2_mod, np.ones(10) * 0.5, c='k', marker='.', label='Wyckoff 4e(2)')
    # plt.tick_params(direction='in', length=200, width=1, zorder=1, colors='k', grid_linestyle='--',
    #                grid_color='tab:orange', grid_alpha=0.25, axis='x', top=True)
    every_1st, every_2nd, every_3rd, every_4th = mod[0::4], mod[1::4], mod[2::4], mod[3::4] # transform 40 array into 4*10 arrays
    #print("every_1st={}".format(every_1st), "every_2nd={}".format(every_2nd), "every_3rd={}".format(every_3rd),
    #      "every_4th={}".format(every_4th))
    axs[0, 0].set_xlim(0, 10)
    axs[0, 0].set_ylim(0.48, 0.52)
    axs[0, 0].legend(fontsize=15)
    axs[0, 0].set_xticks(np.arange(0, 11, 1))
    axs[0, 0].tick_params(axis='x', labelsize=15, direction='in')
    axs[0, 0].tick_params(direction='in')
    axs[0, 0].set_yticks([])
    axs[0, 0].set_xlabel(r"$z$", fontsize=15)
    
    """ Final Atomic vector positions """
    Eu, Eu_T = np.array([x_Eu, y_Eu, z_Eu]), np.array([x_Eu_T, y_Eu_T, z_Eu_T])
    Ga1, Ga1_T, Ga2, Ga2_T = np.array([x_Ga1, y_Ga1, every_2nd]), np.array([x_Ga1_T, y_Ga1_T, every_4th]), np.array([x_Ga2, y_Ga2, every_3rd]), np.array([x_Ga2_T, y_Ga2_T, every_1st])
    Al1, Al1_T, Al2, Al2_T = np.array([x_Al1, y_Al1, z_Al1]), np.array([x_Al1_T, y_Al1_T, z_Al1_T]), np.array([x_Al2, y_Al2, z_Al2]), np.array([x_Al2_T, y_Al2_T, z_Al2_T])


    """Debye-Waller factor"""
    lamb = 1.1 # Wellenlänge in nm
    # Compute all distances corresponding to lattice diffractions with hkl
    d_hkl = a/(np.sqrt( (h * np.ones((n, n)))**2 + k2d**2 + (a/c)**2 * l2d**2)) # in angstrom
    # 1. compute all angles corresponding to the k points according to braggs law
    theta = np.arcsin(lamb/(2 * d_hkl))
    B_iso_Eu, B_iso_Ga, B_iso_Al = 0.1, 0.1, 0.1 # isotropic movements around equilibrium, inverse proportional to the mass of the atom
    B_iso_list = [B_iso_Eu, B_iso_Ga, B_iso_Al]

    
    """Structure Factor F_hkl"""
    F_Eu_list, F_Eu_T_list = [], []  # Europium Eu
    F_Ga1_list, F_Ga1_T_list, F_Ga2_list, F_Ga2_T_list = [], [], [], []  # Gallium Ga
    F_Al1_list, F_Al1_T_list, F_Al2_list, F_Al2_T_list = [], [], [], []  # Alluminium Al
    # fill exponential components in each structure factor list with size nxn
    for i in range(10):
        F_Eu_list.append( f_Eu * np.exp(-B_iso_list[0] * (  np.sin(theta) / lamb  )**2 ) * np.exp(
            1j * 2 * np.pi * (h * np.ones((n, n)) * Eu[0, i] + k2d * Eu[1, i] + l2d * Eu[2, i]))
            )  # 2 Europium atoms
        F_Eu_T_list.append( f_Eu * np.exp(-B_iso_list[0] * (  np.sin(theta) / lamb  )**2 ) * np.exp(
            1j * 2 * np.pi * (h * np.ones((n, n)) * Eu_T[0, i] + k2d * Eu_T[1, i] + l2d * Eu_T[2, i]))
            )
        F_Ga1_list.append( f_Ga * np.exp(-B_iso_list[1] * (  np.sin(theta) / lamb  )**2 ) * np.exp(
            1j * 2 * np.pi * (h * np.ones((n, n)) * Ga1[0, i] + k2d * Ga1[1, i] + l2d * Ga1[2, i]))
            )  # 4 modulated Gallium atoms
        F_Ga1_T_list.append( f_Ga * np.exp(-B_iso_list[1] * (  np.sin(theta) / lamb  )**2 ) * np.exp(
            1j * 2 * np.pi * (h * np.ones((n, n)) * Ga1_T[0, i] + k2d * Ga1_T[1, i] + l2d *  Ga1_T[2, i]))
            )
        F_Ga2_list.append(  f_Ga * np.exp(-B_iso_list[1] * (  np.sin(theta) / lamb  )**2 ) * np.exp(
            1j * 2 * np.pi * (h * np.ones((n, n)) * Ga2[0, i] + k2d * Ga2[1, i] + l2d *  Ga2[2, i]))
            )
        F_Ga2_T_list.append( f_Ga * np.exp(-B_iso_list[1] * (  np.sin(theta) / lamb  )**2 ) * np.exp(
            1j * 2 * np.pi * (h * np.ones((n, n)) * Ga2_T[0, i] + k2d * Ga2_T[1, i] + l2d *  Ga2_T[2, i]))
            )
        F_Al1_list.append(f_Al * np.exp(-B_iso_list[2] * (  np.sin(theta) / lamb  )**2 ) * np.exp(
            1j * 2 * np.pi * (h * np.ones((n, n)) * Al1[0, i] + k2d * Al1[1, i] + l2d * Al1[2, i]))
            )  
        F_Al1_T_list.append(f_Al * np.exp(-B_iso_list[2] * (  np.sin(theta) / lamb  )**2 ) * np.exp(
            1j * 2 * np.pi * (h * np.ones((n, n)) * Al1_T[0, i] + k2d * Al1_T[1, i] + l2d * Al1_T[2, i]))
            )
        F_Al2_list.append(f_Al * np.exp(-B_iso_list[2] * (  np.sin(theta) / lamb  )**2 ) * np.exp(
            1j * 2 * np.pi * (h * np.ones((n, n)) * Al2[0, i] + k2d * Al2[1, i] + l2d * Al2[2, i]))
            )
        F_Al2_T_list.append( f_Al * np.exp(-B_iso_list[2] * (  np.sin(theta) / lamb  )**2 ) * np.exp(
            1j * 2 * np.pi * (h * np.ones((n, n)) * Al2_T[0, i] + k2d * Al2_T[1, i] + l2d * Al2_T[2, i]))
            )
    F_str = np.zeros((n, n), dtype=complex)  # n dimension of k-space
    atoms_list = [F_Eu_list, F_Eu_T_list, F_Ga1_list, F_Ga1_T_list, F_Ga2_list,
                  F_Ga2_T_list, F_Al1_list, F_Al1_T_list, F_Al2_list, F_Al2_T_list]
    pre_F_str = [np.zeros((n, n))]
    # Zusammenfügen der Formfaktoren für die Atomsorten
    for i in atoms_list:  # put together the lists in a ndarray for each atom with each N positions, to get rid of 3rd dimension (better workaround probably possible...)
        pre_F_str = np.add(pre_F_str, i)
    for i in range(len(pre_F_str)):
        F_str = F_str + pre_F_str[i]
        
        
    """Intensity I"""
    print("INVESTIGATED PEAK: [HKL] = [{}{}{}]".format(h, k0, l0))
    I = np.absolute(F_str) ** 2
    print("Intensity as n+1 x n+1 array: {}".format(I))
    # 2d scatter plot
    # print("k2d={}, l2d={}".format(k2d, l2d))
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
    axs[1,1].plot(l2d[:, readout], I[:, readout] / np.max(I[:, readout]), marker='x', c='k', ls='--', lw=0.5, ms=3, label='[{}KL]'.format(h))
    axs[1,1].set_xlim(l0 - l_boundary, l0 + l_boundary)
    axs[1,1].set_xlabel(r"L (r.l.u.)", fontsize=15)
    axs[1,1].set_ylabel(r"Intensity (Rel.)", fontsize=15)
    axs[1,1].tick_params(axis='x', labelsize=15, direction='in')
    axs[1,1].tick_params(axis='y', labelsize=15, direction='in')
    axs[1,1].legend(fontsize=15)
    print("I_max = {}".format(np.sort(I[:, readout])[-1]))
    print("(I_max - I_[max-1])/I_max = {}".format((np.sort(I[:, readout])[-1] - np.sort(I[:, readout])[-2]) / np.sort(I[:, readout])[-1]))
    plt.subplots_adjust(wspace=0.2, hspace=0.4)
    # plt.savefig('CDW_sim_HK0L0_FOURIER={}{}{}'.format(h, k0, l0), dpi=300)

    plt.show()


if __name__ == '__main__':
    main()