import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal, norm
import scipy.stats

fig = plt.figure()
plt.ylabel("w_1")
plt.xlabel("w_0")
plt.title("Estimation with MCMC")
plt.xlim([-0.5, 0.5])
plt.ylim([-1, 1])


# All values from book

N = 25
sigma = 0.2
alpha = 2
w_0 = -0.2
w_1 = 0.5

x_n = np.random.uniform(-1, 1, N)
t_n = w_0 + w_1 * x_n
noise = np.random.normal(0, sigma, N)
t_n = t_n + noise

mu_n = np.zeros((N, 2))
S_n = np.zeros((N, 2, 2))

mu_0 = [0, 0]
S_0 = sigma * (np.identity(2))

phi = np.array([[1, x_n[0]]])
for i in range(N):
    if i != 0:
        phi = np.concatenate((phi, np.array([[1, x_n[i]]])), axis=0)
phiT = phi.T


q = mlab.normpdf(x_n, 0, 1)

k_scale = 0
for zz in range(N):
    if k_scale < x_n[zz] / q[zz]:
        k_scale = x_n[zz] / q[zz]
q = q * k_scale

z_t = np.random.multivariate_normal(mu_0, S_0)
p_z_T = 0

z_0_accepted = np.zeros((1500, 2))
num_accepted = 0

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
    z_star = np.random.multivariate_normal(z_t, S_0)

    p_z_T = np.sum(np.log(scipy.stats.multivariate_normal.pdf(t_n, phi[:, 0] * z_t[0] + phi[:, 1] * z_t[1], sigma)))
    p_z_star = np.sum(np.log(scipy.stats.norm.pdf(t_n, phi[:, 0]*z_star[0] + phi[:, 1]*z_star[1], sigma)))

    p_z_T += np.log(scipy.stats.multivariate_normal.pdf(z_t, mu_0, S_0))
    p_z_star += np.log(scipy.stats.multivariate_normal.pdf(z_star, mu_0, S_0))

    A_z_star = min(0, p_z_star - p_z_T)
    u = np.log(np.random.uniform(0, 1))

    if A_z_star > u:
        num_burn += 1
        z_t = z_star
        if (num_burn > 100):
            z_0_accepted[num_accepted] = z_star
            num_accepted += 1
w_0_mean = np.sum(z_0_accepted[:, 0])/num_accepted
w_1_mean = np.sum(z_0_accepted[:, 1])/num_accepted
print("Actual values: w_0 = ", w_0, ", w_1 = ", w_1)
print("Estimated values: w_0 = ", w_0_mean, ", w_1 = ", w_1_mean)

plt.scatter(z_0_accepted[:, 0], z_0_accepted[:, 1], c="blue", marker="x", alpha=0.2, label="Sampled values")
plt.scatter(w_0, w_1, c="brown", label="Actual weights")
plt.scatter(w_0_mean, w_1_mean, c="red", label="Estimated weights")
plt.legend(loc="upper right")


plt.show()
