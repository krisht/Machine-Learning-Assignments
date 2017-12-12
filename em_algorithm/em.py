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

# 1D Case
k = 3
mu = np.asarray([-6, 0, 6])
sigma = np.ones(len(mu))
N = 20
X = np.asarray([])
for m, s in zip(mu, sigma):
	X = np.concatenate((X, np.random.normal(m, s, N)))

fig = plt.figure()
plt.ylabel("Normalized frequency")
plt.xlabel("x")
plt.title("EM Algorithm in 1D With 3 Clusters (Iteration %d)" % 0)
plt.xlim([-10, 10])
plt.ylim([0, 0.7])
hist = plt.hist(X, bins=100, color='k', normed=True)

muk = np.asarray([-7, 3, 13], np.float32)
sigk = np.asarray([1.0, 1.0, 1.0], np.float32)
pik = np.asarray([0.25, 0.5, 0.25], np.float32)

for kk in range(3):
	x = np.linspace(muk[kk] - 3 * sigk[kk], muk[kk] + 3 * sigk[kk], 100)
	plt.plot(x, mlab.normpdf(x, muk[kk], sigk[kk]))


def update_hist(num):
	fig.clf()
	global muk, sigk, pik

	plt.ylabel("Normalized frequency")
	plt.xlabel("x")
	plt.title("EM Algorithm in 1D With 3 Clusters (Iteration %d)" % num)
	plt.xlim([-10, 10])
	plt.ylim([0, 0.7])
	plt.hist(X, bins=100, color='k', normed=True)

	# Create frozen Norm distributions for current muks and sigks
	norm_pdfs = [None, None, None]
	for kk in range(3):
		norm_pdfs[kk] = scipy.stats.multivariate_normal(muk[kk], sigk[kk])

	# Calculate the denominator for the gamma function
	temp_sum = np.zeros(3 * N)
	for kk in range(3):
		temp_sum += (pik[kk] * norm_pdfs[kk].pdf(X))

	# Calculate new gamma matrix
	gamma_matrix = np.zeros((3, 3 * N))

	for kk in range(3):
		for nn in range(len(X)):
			gamma_matrix[kk, nn] = pik[kk] * norm_pdfs[kk].pdf(X[nn]) / temp_sum[nn]

	# Calculate new Nks
	Nk = np.sum(gamma_matrix, 1)

	# Calculate new muks
	muk = np.sum(gamma_matrix * X, 1) / Nk

	# Calculate new sigks
	for kk in range(3):
		sigk[kk] = np.sum(gamma_matrix[kk, :] * (X - mu[kk]) * (X - mu[kk]).T) / Nk[kk]

	# Calculate new piks
	pik = Nk / (3 * N)

	# Plot it out
	for kk in range(3):
		x = np.linspace(-10, 10, 300)
		plt.plot(x, mlab.normpdf(x, muk[kk], sigk[kk]))


# Animate it
anim = animation.FuncAnimation(fig, update_hist, np.arange(1, 50), interval=200)

anim.save('em_algorithm_1d.mp4')

# 2D Case

# Generate three clusters of data
mu = np.asarray([[3, 3], [3, -3], [-3, -3]])
sigma = np.asarray([np.eye(2) / 2, np.eye(2) / 2, np.eye(2) / 2])
N = 10

X = None
y = np.asarray([])
i = 0
eps = 1e-5
for m, s in zip(mu, sigma):
	if X is not None:
		X = np.concatenate((X, np.random.multivariate_normal(m, s, N)))
		y = np.concatenate((y, np.repeat(i, N)))
		i += 1
	else:
		X = np.random.multivariate_normal(m, s, N)
		y = np.repeat(i, N)
		i += 1

# Create initial figure
fig = plt.figure()
plt.ylabel("Y")
plt.xlabel("X")
plt.title("EM Algorithm in 2D with 3 Clusters (Iteration %d)" % 0)
plt.scatter(X[:, 0], X[:, 1], c='k')
plt.ylim([-10, 10])
plt.xlim([-10, 10])

# Initialize values
muk = np.asarray([[0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]], np.float32)
covk = np.asarray([3 * np.eye(2), 3 * np.eye(2), 3 * np.eye(2)], np.float32)
pik = np.asarray([0.4, 0.2, 0.4], np.float32)

# Set up mesh for contour
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)
x, y = np.meshgrid(x, y)
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y


# Plot initial contours
for kk in range(3):
	z = scipy.stats.multivariate_normal(muk[kk], covk[kk])
	z = z.pdf(pos)
	plt.contour(x, y, z, 3)


def update_scatter(num):
	fig.clf() # Clear figure

	global muk, sigk, pik, X, y
	# Set up plot
	plt.ylabel("Y")
	plt.xlabel("X")
	plt.title("EM Algorithm in 2D with 3 Clusters (Iteration %d)" % num)
	plt.scatter(X[:, 0], X[:, 1], c='k')
	plt.ylim([-10, 10])
	plt.xlim([-10, 10])
	x = np.linspace(-10, 10, 100)
	y = np.linspace(-10, 10, 100)
	x, y = np.meshgrid(x, y)
	pos = np.empty(x.shape + (2,))
	pos[:, :, 0] = x
	pos[:, :, 1] = y

	# Create frozen pdfs
	norm_pdfs = [None, None, None]

	for kk in range(3):
		norm_pdfs[kk] = scipy.stats.multivariate_normal(np.ravel(muk[kk]), covk[kk])

	# Calculate denominator for gamma s
	temp_sum = 0
	for kk in range(3):
		temp_sum += (pik[kk] * norm_pdfs[kk].pdf(X))

	# Calculate gamma matrix
	gamma_matrix = np.zeros((3, 3 * N))
	for kk in range(3):
		for nn in range(len(X)):
			gamma_matrix[kk, nn] = pik[kk] * norm_pdfs[kk].pdf(X[nn]) / temp_sum[nn]


	# Define Nk used in muk, covk and pik update equations
	Nk = np.zeros(3, np.float32)

	for k in range(3):
		Nk[k] = 0
		for n in range(len(X)):
			Nk[k] += gamma_matrix[k][n]

	# Update muk with given equations
	for kk in range(3):
		muk[kk] = np.zeros((2))
		for nn in range(len(X)):
			muk[kk] += gamma_matrix[kk][nn] * X[nn, :] / Nk[kk]


	# Update covk with update equations
	for kk in range(3):
		covk[kk] = np.zeros((2, 2))
		for nn in range(len(X)):
			covk[kk] += gamma_matrix[kk][nn] * (X[nn, :] - muk[kk]) * (X[nn, :] - muk[kk]).T / Nk[kk]
		# Set indexes [0, 1] ad [1, 0] to zeros so that python doesn't complain that matrix is singular
		covk[kk, 0, 1] = covk[kk, 1, 0] = 0.0

	# Update pik with update equations
	for kk in range(3):
		pik[kk] = Nk[kk] / (3 * N)

	# Plot out new contours
	for kk in range(3):
		z = scipy.stats.multivariate_normal(muk[kk], covk[kk])
		z = z.pdf(pos)
		plt.contour(x, y, z, 3)


# Animate it
anim = animation.FuncAnimation(fig, update_scatter, 50, interval=200)

anim.save('em_algorithm_2d.mp4')
