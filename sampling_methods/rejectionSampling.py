


import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy
import os
import scipy.stats
import random
import shutil

n_iter = 100

fig = plt.figure()
# plt.ylabel("Normalized frequency")
# plt.xlabel("x")
# plt.title("EM Algorithm in 1D With 3 Clusters (Iteration %d)" % 0)
plt.xlim([-10, 16])
plt.ylim([0, 0.7])

muk = np.asarray([-7, 3, 13], np.float32)
sigk = np.asarray([1.0, 1.0, 1.0], np.float32)
pik = np.asarray([0.25, 0.5, 0.25], np.float32)

X = np.linspace(-10, 16, 500)
sample_pdf_gm = np.zeros(500)

for kk in range(3):
    sample_pdf_gm += mlab.normpdf(X, muk[kk], sigk[kk])

plt.plot(X, sample_pdf_gm)


q = mlab.normpdf(X, 3, 10)

k_scale = 0
for zz in range (500):
    if k_scale < sample_pdf_gm[zz] / q[zz]:
        k_scale = sample_pdf_gm[zz] / q[zz]
q = q * k_scale

plt.plot(X, q)

z_0_accepted = np.zeros(500)
num_accepted = 0

while num_accepted < 500:
    z_0 = np.random.normal(3, 10)
    u_0 = np.random.uniform(0, k_scale * mlab.normpdf(z_0, 3, 10))

    p_z0 = 0
    for kk in range (3):
        p_z0 += mlab.normpdf(z_0, muk[kk], sigk[kk])

    if u_0 <= p_z0:
        z_0_accepted[num_accepted] = z_0
        num_accepted += 1

plt.hist(z_0_accepted, bins = 100, normed = True)

plt.show()

