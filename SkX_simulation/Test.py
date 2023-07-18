import numpy as np
import matplotlib.pyplot as plt

"""Simulation of Skyrmion magnetizations m(r), depending on polarity, vorticity and helicity."""
# Grid dimensions
x, y, z = np.meshgrid(np.arange(-10, 10, 0.5),
                      np.arange(-10, 10, 0.5),
                   np.arange(0, 0.5, 0.5))
# Make the direction data for the arrows
gamma = 3 # Helicity
m = -1 # Vorticity
r0 = 2 # simulation radius
p = 1 # Polarity
#mx = 1 / (np.sqrt(x**2 + y**2)) * ( (x * np.cos(gamma) - m * y * np.sin(gamma)) * np.sin(np.pi/r0 * np.sqrt(x**2 + y**2)) )
#my = 1 / (np.sqrt(x**2 + y**2)) * ( (x * np.sin(gamma) + m * y * np.cos(gamma)) * np.sin(np.pi/r0 * np.sqrt(x**2 + y**2)) )
#mz = p * np.cos(np.pi/r0 * np.sqrt(x**2 + y**2))
n = -4
# fig, ax  = plt.subplots(figsize=(7, 7))
# ax = plt.figure().add_subplot(projection='3d')
#c_map = np.sqrt(((mx-my)/2)*2 + ((my-mx)/2)*2)
# ax.quiver(x, y, z, mx, my, mz, normalize=True)

m, M, kappa, a = 1, 1.1, 1, 1

def omega1(k):
    return kappa * (1/m + 1/M) + kappa * np.sqrt((1/m-1/M)**2 + 4/(m*M)*np.cos(k*a)**2)


def omega2(k):
    return kappa * (1/m + 1/M) - kappa * np.sqrt((1/m-1/M)**2 + 4/(m*M)*np.cos(k*a)**2)

# plt.plot(np.linspace(-1.5, 1.5, 1000), np.sqrt(omega1(np.linspace(-1.5,1.5,1000))))
# plt.plot(np.linspace(-1.5, 0, 1000), np.sqrt(omega(np.linspace(-1.5,0,1000))))
# plt.plot(np.linspace(-1.5, 1.5, 1000), np.sqrt(omega2(np.linspace(-1.5, 1.5, 1000))))
# plt.plot(np.linspace(-1.5, 1.5, 1000),  2*np.sqrt(kappa/m)*np.abs(np.cos(np.linspace(-1.5, 1.5, 1000)*a/2)), label='omega1, m=M')
# plt.plot(np.linspace(-1.5, 1.5, 1000),  2*np.sqrt(kappa/m)*np.abs(np.sin(np.linspace(-1.5, 1.5, 1000)*a/2)), label='omega2, m=M')

x = np.linspace(-10, 10, 1000)

plt.plot()
plt.legend()
plt.show()