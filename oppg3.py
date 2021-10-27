import matplotlib.pyplot as plt
# 3a

# g(t) = s(t) + I(t) + R(t)
# g'(t) = s'(t) + I'(t) + R'(t) = (-βSI) + (βSI -γI) + (γI) = 0
# g(t) is therefore constant, and the system is conservative

# 3b

beta = 0.002
gamma = 0.15

def f(t, y):
    if len(y) != 3:
        raise Error(y)
    return np.asarray([-beta*y[0]*y[1], beta*y[0]*y[1] - gamma*y[1], gamma*y[1]])

N_max = 1000
u0 = np.asarray([50, 10, 0])**T
T = 2



t, l = ssprk3(u0, 0, T, f, N_max)
#t, l = kk(u0, 0, T, f, N_max)
h, i, r = zip(*l)
total = [h[j]+i[j]+r[j] for j in range(len(h))]

plt.plot(t, h, color="green")
plt.plot(t, i, color="red")
plt.plot(t, r, color='blue')
plt.plot(t, total, color="black")