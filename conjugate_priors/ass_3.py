# Krishna Thiyagarajan & Abhinav Jain
# Prof. Keene
# Machine Learning - Assignment 1
# Gaussian variance value estimation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import math
from scipy.stats import norm

file_name = 'known_mu2'
var = .5 # unknown
mu = -2.5 # known
num_samples = 100
num_trials = 100
num_exp = 100

def gaussian(x, mu, var):
	return norm(mu, math.sqrt(var)).pdf(x)

min_diff = -100000
gif_vars = None

alphas = np.random.uniform(0.0, 10, 8)
betas = np.random.uniform(0.0, 10, 8)

# alphas = [100, 200, 300, 400, 500, 600]
# betas  = [600, 500, 400, 300, 200, 100]

for ii, (a, b) in enumerate(zip(alphas, betas)):
	samples = np.random.normal(mu, math.sqrt(var), size=(num_exp, num_samples))
	beta = b + np.cumsum(np.square(samples - mu), axis=1)/2
	alpha = a + np.cumsum(np.ones((num_exp, num_samples)), axis=1)/2
	temp_var = np.average(beta/alpha, axis=0)
	mse_cp = np.square(temp_var - var)

	if mse_cp[-1] > min_diff:
		gif_vars = temp_var
		min_diff = mse_cp[-1]

 	plt.plot(mse_cp, label=r'$\alpha$=%.4f, $\beta$=%.4f' % (a, b))

plt.legend()
plt.ylim([-0.1,1])

samples = np.random.normal(mu, math.sqrt(var), size=(num_exp, num_samples))

totals = np.average(np.cumsum(np.square(samples - mu), axis = 1), axis = 0)
num_count = np.cumsum(np.ones(num_samples))
var_mle = totals/num_count

mse_mle = np.square(var_mle - var)

plt.plot(mse_mle, label='MLE Error')
plt.title(r"MSE for CP & ML for Gaussian with known $\mu$=%.4f & unknown $\sigma^2_{true}$=%.4f" % (mu, var))
plt.legend()
plt.savefig(file_name + '_error.pdf')

plt.clf()

fig, ax = plt.subplots()
fig.set_size_inches(10, 7.5)
ax.set_ylim([0,2])
ax.set_title(r'Bayesian Inference of a Gaussian $\sigma^2$ with $\mu$=%.4f & unknown $\sigma^2_{true}$=%.4f' % (mu, var))

ax.set_ylabel('Posterior')
fig.set_tight_layout(True)

x = np.linspace(-11, 11, 10000)
distrib,  = ax.plot(x, gaussian(x, mu, gif_vars[0]), 'k-', lw=2, alpha = 0.6, label='Predicted Distribution')

def init():
	initial, = ax.plot(x, gaussian(x, mu, var), 'b-', lw=2, alpha=0.6, label='True Distribution')
	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles, labels)
	return initial

def update(i):
	pred_var = gif_vars[i]
	pred_mu = mu
	label = r"Timestep: %.4d, $\mu_{est}$ = %0.4f, $\sigma_{est}^2$ = %0.4f"  % (i, pred_mu, pred_var)
	distrib.set_ydata(gaussian(x, pred_mu, pred_var))
	ax.set_xlabel(label)
	return distrib, ax


anim = FuncAnimation(fig, update, init_func = init, frames=np.arange(0, num_samples), interval=100)
anim.save(file_name + '.gif', dpi=100, writer='imagemagick')
