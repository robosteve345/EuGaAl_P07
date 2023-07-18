import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.signal

ax = plt.axes(projection='3d')
Z = np.linspace(-10,10,100)
X = np.cos(Z)
Y = np.sin(Z)
def f(X,Y):
    return np.sin(5*np.sqrt(X**2+Y**2))/np.sqrt(X**2+Y**2)

x = np.linspace(-5,5,1000)
y = np.linspace(-5,5,1000)
X2d, Y2d = np.meshgrid(x, y)
#ax.plot_surface(X2d, Y2d, f(X2d, Y2d), cmap='viridis')
Xrand = X + 0.1 * np.random.randn(100)
Yrand = Y + 0.1 * np.random.randn(100)
#ax.scatter3D(Xrand, Yrand, Z, c=Z, cmap='seismic')
#ax.plot3D(X, Y, Z, 'gray')
# plt.show()
n = 100
a = np.linspace(-5, 5, n)
b = np.linspace(-5, 5, n)
A, B = np.meshgrid(a, b)
def g(X, Y, s):
 return 0.05/s**2 * np.exp(-(X**2 + Y**2)/2*s)

def h(X, Y):
    return np.sin(2*X) + np.cos(2*Y)
Signal  = np.zeros((A.shape[0], B.shape[0]))
#Signal[5,5] = 10
Signal[9,9] = 10
Signal[0,0] = 10
Signal[4,3] = 5
print(Signal)
print("Signal dim: {}x{}".format(Signal.shape[0], Signal.shape[1]))
Kernel = g(A, B, 100)
print("Kernel dim: {}x{}".format(Kernel.shape[0], Kernel.shape[1]))
#ax.contourf(A, B, Kernel)
#ax.scatter3D(A, B, Kernel, marker='x')
conv = sp.signal.convolve2d(Signal, Kernel, mode='same', boundary='fill', fillvalue=0)
# print(conv)
print("Conv dim: {}x{}".format(conv.shape[0], conv.shape[1]))
#ax.contourf(A, B, conv, marker='v', cmap='binary')
fig = plt.figure()
ax = fig.add_subplot(111)
#ax.imshow(conv, cmap='binary', interpolation='gaussian')
#plt.show()
boundary = 2
diff=0.5
n = boundary/diff * 2 + 1
Arrayx = np.zeros(round(n)) # np.arange(-boundary, boundary + diff, diff)
print("len(Arrayx) = {}".format(len(Arrayx)))   #len(arrayx) = boundary/diff * 2 + 1
print("zeros_array={}".format(Arrayx))
print(np.arange(-boundary, boundary + diff, diff))
integers = np.arange(-boundary, boundary+1, 1)
print("integers={}".format(integers))
newindices = 1/diff * np.arange(0, 2*boundary + 1, 1) + 0
print("newindices={}".format(np.intc(newindices)))
X2d, Y2d = np.meshgrid(Arrayx, Arrayx)
X = np.ones((11, 11))
I = h(X2d, Y2d)
for i in range(len(X2d)):
    if i in newindices:
        I[i, :] = I[i, :]
    else:
        I[i, :] = 0
        for j in range(len(X2d)):
            if j in newindices:
                I[:, j] = I[:, j]
            else:
                I[:, j] = 0

# print(I)

x = np.linspace(-3, 3, 1000)
y = np.linspace(-3, 3, 1000)
kx, ky = np.meshgrid(x, y)
a = 1
E_graphen1 = np.sqrt(3 + 2*np.cos(np.sqrt(3)*ky*a) + 4*np.cos(3/2*kx*a)*np.cos(np.sqrt(3)/2*ky*a))
E_graphen2 = - E_graphen1
ax = plt.axes(projection='3d')
ax.plot_surface(kx, ky, E_graphen1, cmap='viridis')
ax.plot_surface(kx, ky, E_graphen2, cmap='viridis')
plt.show()










