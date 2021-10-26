#%matplotlib widget

import ipywidgets as widgets
from ipywidgets import interact, fixed
import numpy as np
from numpy import pi
from numpy.linalg import solve, norm
import matplotlib.pyplot as plt


class ExplicitRungeKuttaAlt:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def solve(self, y0, t0, T, f, Nmax):
        # Extract Butcher table
        a, b, c = self.a, self.b, self.c

        # Initiate stages
        s = len(b)
        ks = [np.zeros_like(y0, dtype=np.double) for i in range(s)]

        # Start time-stepping
        ys = [y0]
        ts = [t0]
        dt = (T-t0)/Nmax

        while ts[-1] < T:
            t, y = ts[-1], ys[-1]

            # Compute stages derivatives k_j
            for j in range(s):
                t_j = t + c[j]*dt
                dY_j = np.zeros_like(y, dtype=np.double)
                for l in range(j):
                    dY_j += a[j, l]*ks[l]

                ks[j] = f(t_j, y + dt*dY_j)

                # Compute next time-step by computing the incremement dy
            dy = np.zeros_like(y, dtype=np.double)
            for j in range(s):
                dy += b[j]*ks[j]

            ys.append(y + dt*dy)
            ts.append(t + dt)

        return np.array(ts), np.array(ys)


def f(t, y):
    return -y


def __main__():
    newparams = {'figure.figsize': (6.0, 6.0),
                 'axes.grid': True,
                 'lines.markersize': 8,
                 'lines.linewidth': 2,
                 'font.size': 14}
    plt.rcParams.update(newparams)
    arr1 = np.array([1.0, 2.0])
    s = 3
    ks = [ np.zeros_like(arr1, dtype=np.double) for i in range(s) ]
    print(ks)

    #Definerer Butcher-tabellen for midtpunkt
    am = np.array([[0, 0], [1/2, 0]])
    bm = np.array([0, 1])
    cm = np.array([0, 1/2])

    #Definerer Butcher-tabellen for Gottlieb & Gottlieb
    ag = np.array([[0, 0, 0], [1, 0, 0], [1/4, 1/4, 0]])
    bg = np.array([1/6, 1/6, 2/3])
    cg = np.array([0, 1, 1/2])

    midtpunkt = ExplicitRungeKuttaAlt(am, bm, cm)
    gottlieb = ExplicitRungeKuttaAlt(ag, bg, cg)

    t0 = 0
    T = 10
    y0 = 1
    Nmax = 10

    tsMp, ysMp = midtpunkt.solve(y0, t0, T, f, Nmax)
    plt.plot(tsMp, ysMp, color="green", label="Midtpunkt")

    tsG, ysG = gottlieb.solve(y0, t0, T, f, Nmax)
    plt.plot(tsG, ysG, color="blue", label="Gottlieb")

    plt.show()

if __name__ == '__main__':
    __main__()