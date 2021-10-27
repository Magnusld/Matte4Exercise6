# import ipywidgets as widgets
# from ipywidgets import interact, fixed
import numpy as np
from numpy import pi
from numpy.linalg import solve, norm
import matplotlib.pyplot as plt

class RungeKutta:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def solve(self, y0, t0, T, f, N_max):
        dt = (T-t0)/N_max
        yk = [y0]
        tk = [t0]

        a, b, c = self.a, self.b, self.c

        n = len(b)
        kj = [np.zeros_like(y0, dtype=np.double) for i in range(n)]

        while tk[-1] < T:
            y = yk[-1]
            t = tk[-1]

            for j in range(n):
                dY_j = np.zeros_like(y, dtype=np.double)
                for l in range(j):
                    dY_j += a[j, l]*kj[l]
                kj[j] = f(t + c[j] * dt, y + dt * dY_j)

            dy = np.zeros_like(y, dtype=np.double)
            for j in range(n):
                dy += b[j]*kj[j]

            yk.append(y + dt*dy)
            tk.append(t + dt)

        return np.array(tk), np.array(yk)

def f(tk, yk):
    return -yk

def explicit_mid_point(y0, t0, T, f, N_max):
    a = np.array([[0, 0], [1/2, 0]])
    b = np.array([0, 1])
    c = np.array([0, 1/2])
    midpoint = RungeKutta(a, b, c)
    return midpoint.solve(y0, t0, T, f, N_max)

def ssprk3(y0, t0, T, f, N_max):
    a = np.array([[0, 0, 0], [1, 0, 0], [1/4, 1/4, 0]])
    b = np.array([1/6, 1/6, 2/3])
    c = np.array([0, 1, 1/2])
    gottlieb = RungeKutta(a, b, c)
    return gottlieb.solve(y0, t0, T, f, N_max)

class a():
    t0 = 0
    T = 10
    y0 = 1
    N_max = 10

    tkMp, ykMp = explicit_mid_point(y0, t0, T, f, N_max)
    plt.plot(tkMp, ykMp, color="green", label="Midtpunkt")

    tkG, ykG = ssprk3(y0, t0, T, f, N_max)
    plt.plot(tkG, ykG, color="blue", label="Gottlieb")

    plt.show()

def b():
    m = []
    for i in range(10):
        m.append(2**-i)

    N = []
    for s in m:
        N.append(int(10/s))
    
    y0 = 1
    t0 = 0
    T = 10

    y_actual = np.e**(-10)

    y10 = []
    for n in N:
        y10.append(ssprk3(y0, t0, T, f, n)[-1][-1])
    e = []
    for y in y10:
        e.append(abs(y_actual - y))
    p = []
    for i in range(len(N)-1):
        p.append(np.log(e[i]/e[i+1])/np.log(N[i]/N[i+1]))
    print(p)

def c():

    y0 = 1
    t0 = 0
    T = 0.5
    
    y_actual = np.e**(-(T)**2)

    y_m = explicit_mid_point_rule(y0, t0, T, f, 3)[1]
    y_s = ssprk3(y0, t0, T, f, 2)[1]

    e_midpoint = abs(y_real - y_m[-1])
    e_gottlieb = abs(y_real - y_s[-1])

    print(e_m, e_s)
    

def __main__():
    parameters = {'figure.figsize': (7.0, 7.0), 'axes.grid': True, 'lines.markersize': 6, 'lines.linewidth': 2, 'font.size': 12}
    plt.rcParams.update(parameters)


if __name__ == '__main__':
    __main__()