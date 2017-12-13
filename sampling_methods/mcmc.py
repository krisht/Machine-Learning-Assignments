


import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy
import os
import scipy.stats
import random
import shutil


fig = plt.figure()
# plt.ylabel("Normalized frequency")
# plt.xlabel("x")
# plt.title("EM Algorithm in 1D With 3 Clusters (Iteration %d)" % 0)
# plt.xlim([-10, 16])
# plt.ylim([0, 0.7])


# All values from book

N = 500
sigma = 0.2
beta = 1/np.power(sigma, 2.0)
alpha = 2
w_0 = -0.3
w_1 = 0.5

x_n = np.linspace (-1, 1, N)
t_n = w_0 + w_1*x_n

plt.plot(x_n, t_n)

q = mlab.normpdf(x_n, 0, 1)

k_scale = 0
for zz in range (N):
    if k_scale < x_n[zz] / q[zz]:
        k_scale = x_n[zz] / q[zz]
q = q * k_scale

plt.plot(x_n, q)

z_0_accepted = np.zeros(500)
num_accepted = 0
p_z_T = 0

while num_accepted < 500:
    z_star = np.random.normal(0, 1)
    p_z_star = w_0 + w_1*z_star

    if p_z_T == 0:
        A_z_star = 1
    else:
        A_z_star = min(1, float(p_z_star)/p_z_T)

    u = np.random.uniform(0, 1)
    if A_z_star > u:
        z_0_accepted[num_accepted] = z_star
        p_z_T = p_z_star
        num_accepted += 1

plt.hist(z_0_accepted, bins = 100, normed = True)

plt.show()

