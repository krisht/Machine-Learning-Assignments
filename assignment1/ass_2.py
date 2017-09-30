# Krishna Thiyagarajan & Abhinav Jain
# Prof. Keene
# Machine Learning - Assignment 1
# Gaussian mean value estimation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import math
from scipy.stats import norm

file_name = 'known_var2'
var = .5 # Known
mu = -2.5 # Unknown
num_samples = 100
num_trials = 100
num_exp = 100

def gaussian(x, mu, var):
	return norm(mu, math.sqrt(var)).pdf(x)

def get_mus(summed, m, v, var, n):
	a = m/v + summed/var 
	b = (1/ v + n / var)
	return a/b

def get_vars(summed, m, v, var, n):
	return 1/(1/ v + n / var)

min_diff = -100000
gif_vars = None
gif_mus = None

mu_0 = np.random.uniform(-10.0, 10.0, 8)
var_0 = np.random.uniform(0.001, 10, 8)

# alphas = [100, 200, 300, 400, 500, 600]
# betas  = [600, 500, 400, 300, 200, 100]

for ii, (m, v) in enumerate(zip(mu_0, var_0)):
	samples = np.random.normal(mu, math.sqrt(var), size=(num_exp, num_samples))
	samples = np.average(samples, axis=0)
	n = np.cumsum(np.ones(num_samples))
	totals = np.cumsum(samples)
	temp_mu = get_mus(totals, m, v, var, n)
	temp_var = get_vars(totals, m, v, var, n)
	mse_cp = np.square(temp_mu - mu)

	if(mse_cp[-1] > min_diff):
		gif_vars = temp_var
		gif_mus = temp_mu
		min_diff = mse_cp[-1]


	plt.plot(mse_cp, label=r'$\sigma_0$=%.4f, $\mu_0$=%.4f' % (m, v))


samples = np.random.normal(mu, math.sqrt(var), size=(num_exp, num_samples))
samples = np.average(samples, axis = 0)

totals = np.cumsum(samples)
num_count = np.cumsum(np.ones(num_samples))
mu_mle = totals/num_count

mse_mle = np.square(mu_mle - mu)	

plt.plot(mse_mle, label='MLE Error')
plt.title(r"MSE for CP & ML for Gaussian with known $\sigma^2$=%.4f & unknown $\mu_{true}$=%.4f" % (var, mu))
plt.legend()
plt.savefig(file_name + '_error.pdf')

plt.clf()

fig, ax = plt.subplots()
fig.set_size_inches(10, 7.5)
ax.set_ylim([0,2])
ax.set_title(r'Bayesian Inference of a Gaussian $\mu$ with $\sigma^2$=%.4f & unknown $\mu_{true}$=%.4f' % (var, mu))

# ax.set_xlim([0, 1])
# ax.set_ylim([0, 100])

ax.set_ylabel('Posterior')
fig.set_tight_layout(True)

x = np.linspace(-11, 11, 10000)
distrib,  = ax.plot(x, gaussian(x, gif_mus[0], gif_vars[0]), 'k-', lw=2, alpha = 0.6, label='Predicted Distribution')

def init():
	initial, = ax.plot(x, gaussian(x, mu, var), 'b-', lw=2, alpha=0.6, label='True Distribution')
	handles, labels = ax.get_legend_handles_labels()
	ax.legend(handles, labels)
	return initial

def update(i):
	pred_var = var + gif_vars[i]
	pred_mu = gif_mus[i]
	label = r"Timestep: %.4d, $\mu_{est}$ = %0.4f, $\sigma_{est}^2$ = %0.4f"  % (i, pred_mu, pred_var)
	distrib.set_ydata(gaussian(x, pred_mu, pred_var))
	ax.set_xlabel(label)
	return distrib, ax


anim = FuncAnimation(fig, update, init_func = init, frames=np.arange(0, num_samples), interval=100)
anim.save(file_name + '.gif', dpi=100, writer='imagemagick')
