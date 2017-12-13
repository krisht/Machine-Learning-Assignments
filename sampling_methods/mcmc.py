import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal, norm
import scipy.stats

fig = plt.figure()
# plt.ylabel("Normalized frequency")
# plt.xlabel("x")
# plt.title("EM Algorithm in 1D With 3 Clusters (Iteration %d)" % 0)
# plt.xlim([-10, 16])
# plt.ylim([0, 0.7])


# All values from book

N = 25
sigma = 0.2
alpha = 2
w_0 = -0.3
w_1 = 0.5

x_n = np.linspace(-1, 1, N)
t_n = w_0 + w_1 * x_n
noise = np.random.normal(0, sigma, N)
t_n = t_n + noise

mu_n = np.zeros((N, 2))
S_n = np.zeros((N, 2, 2))

mu_0 = [0, 0]
S_0 = (1.0 / alpha) * (np.identity(2))

phi = np.array([[1, x_n[0]]])
for i in range(N):
    if i != 0:
        phi = np.concatenate((phi, np.array([[1, x_n[i]]])), axis=0)
phiT = phi.T

plt.scatter(x_n, t_n)

q = mlab.normpdf(x_n, 0, 1)

k_scale = 0
for zz in range(N):
    if k_scale < x_n[zz] / q[zz]:
        k_scale = x_n[zz] / q[zz]
q = q * k_scale

plt.plot(x_n, q)

z_t = np.random.multivariate_normal(mu_0, S_0)

z_0_accepted = np.zeros(1500)
num_accepted = 0
p_z_T = 0

num_burn = 0

w_test = np.asarray([0, 0])

def likelihood(t, x, w, noise_sigma, noise_mean):
    return 1 / (np.sqrt(2 * np.pi) * noise_sigma) * np.exp(
        -(w[0] + x * w[1] - t - noise_mean) ** 2 / (2 * noise_sigma ** 2))


def prior(w, weight_sigma):
    return 1 / (np.sqrt(2 * np.pi ** 2) * weight_sigma ** 2) * np.exp(
        -(w[0] ** 2 + w[1] ** 2) / (2 * weight_sigma ** 2))


def posterior(t, x, w, noise_sigma, noise_mean, weight_sigma):
    return np.sum(np.log(likelihood(t, x, w, noise_sigma, noise_mean))) + np.log(prior(w, weight_sigma))


while num_accepted < 1500:
    z_star = multivariate_normal(0, 1)

    p_z_T = posterior(t_n, x_n, w_test, sigma, 0, S_0)

    p_z_star = posterior(t_n, x_n, w_test, sigma, 0, S_0)

    A_z_star = min(0, p_z_star - p_z_T)
    u = np.random.uniform(0, 1)

    if A_z_star > u:
        num_burn += 1
        z_t = z_star
        if (num_burn > 100):
            z_0_accepted[num_accepted] = z_star
            num_accepted += 1
plt.hist(z_0_accepted, bins=100, normed=True)

plt.show()
