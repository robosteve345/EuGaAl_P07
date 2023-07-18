import numpy as np
from scipy.interpolate import Rbf
from matplotlib import cm
import matplotlib.pyplot as plt

q_cdw = 0.1
diff = 0.01
boundary = 1
k0 = 0
n = np.intc(boundary/ diff * 2 + 1)
#print("n = {}".format(n))
x = np.linspace(k0-boundary, k0+boundary, n)
#print("x={}".format(x))
c = np.intc(q_cdw / diff) # every #th desired kpoint, so that evaluation occurs at F(Q) with CDW symmetry
desiredkpoints = x[0::c]
#print("every #={}".format(c))
#print("desiredkpoints={}".format(desiredkpoints))
slice = x[0::c]
indices = c * np.arange(0, 2 * (boundary)/(diff*c) + 1, 1)
#print("foundindices={}".format(np.intc(indices)))
#print("x[indices] = {}".format(x[np.intc(indices)]))
y0 = 1
y = np.linspace(y0-boundary, y0+boundary, n)
#print("len(y)={}".format(len(y)))
X, Y = np.meshgrid(x, y)
I = np.sin(X + Y)
indices_k = 1 / diff * np.arange(0, 2 * (boundary) + 1, 1) + 0
#print("y[indices_k]={}".format(y[np.intc(indices_k)]))
#print("indices_k={}".format(np.intc(indices_k)))
#print(I)
for i in range(len(Y)):
    if i in np.intc(indices):
        I[i, :] = I[i, :]
    else:
        I[i, :] = 0
        for j in range(len(X)):
            if j in np.intc(indices_k):
                I[:, j] = I[:, j]
            else:
                I[:, j] = 0
#print(I)
#print(I[:,0])
#print(I[:,20])