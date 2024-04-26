import numpy as np

def lorenz96(t, x, F=8):
    """Lorenz 96 model with constant forcing"""
    N = len(x)
    dxdt = np.zeros(N)
    for i in range(N):
        dxdt[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
    return dxdt

def lorenz96_twoscale(t, u, N=40, n=5, F=8):
    dx = np.zeros(N)
    dy = np.zeros((n, N))

    u = u.reshape(n + 1, N)
    x = u[0, :]
    y = u[1:, :]

    for i in range(N):
        dx[i] = (x[(i+1) % N] - x[(i-2) % N])*x[(i-1) % N] - x[i] + F - p["h"]*p["c"]/p["b"]*np.sum(y[:, i])

        for j in range(n):
            if j == n - 1:
                jp1 = n
                jp2 = 1
                jm1 = n - 2
                ip1 = i
                ip2 = (i + 1) % N
                im1 = i
            elif j == 0:
                jp1 = 1
                jp2 = 2
                jm1 = n
                ip1 = i
                ip2 = i
                im1 = (i - 1) % N
            else:
                jp1 = j + 1
                jp2 = j + 2
                jm1 = j - 1
                ip1 = ip2 = im1 = i

            dy[j, i] = p["c"]*p["b"]*y[jp1, ip1]*(y[jm1, im1] - y[jp2, ip2]) - p["c"]*y[j, i] + p["h"]*p["c"]/p["b"]*x[i]

    du = np.concatenate((dx, dy.flatten()))

    return du