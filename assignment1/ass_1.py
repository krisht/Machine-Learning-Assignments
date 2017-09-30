# Krishna Thiyagarajan & Abhinav Jain
# Prof. Keene
# Machine Learning - Assignment 1
# Binomial p Value estimation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy.stats import beta

file_name = 'binomial2'
p_true = 0.25
num_samples = 1000
num_trials = 100
num_exp = 100

min_diff = 100000
gif_alphas = None
gif_betas = None
gif_p_ests = None

alphas = np.random.randint(1, 600, 8)
betas = np.random.randint(1, 600, 8)

# alphas = [100, 200, 300, 400, 500, 600]
# betas  = [600, 500, 400, 300, 200, 100]

for ii, (a, b) in enumerate(zip(alphas, betas)):
	samples = np.random.binomial(num_trials, p_true, size=(num_exp, num_samples))
	samples = np.average(samples, axis=0)
	ran_betas = b + num_trials*np.cumsum(np.ones(num_samples)) - np.cumsum(samples)
	ran_alphas = a + np.cumsum(samples)
	p_est = ran_alphas/(ran_betas+ran_alphas)

	mse_cp = np.square(p_est - p_true)

	if(mse_cp[-1] < min_diff):
		gif_alphas = ran_alphas
		gif_betas = ran_betas
		gif_p_ests = p_est
		min_diff = mse_cp[-1]


	plt.plot(mse_cp, label=r'$\alpha$=%d, $\beta$=%d' % (a, b))

samples = np.random.binomial(num_trials, p_true, size=(num_exp, num_samples))
samples = np.average(samples, axis=0)
ran_alphas = np.cumsum(samples)
num_count = np.cumsum(np.ones(num_samples))
ran_total = ran_alphas/(num_trials*num_count)
p_mle = np.square(ran_total - p_true)

plt.plot(p_mle, label='MLE Error')
plt.title("MSE for CP & ML for Binomial Trials with n=%d, p=%.4f" % (num_trials, p_true))
plt.legend()
plt.savefig(file_name + '_error.pdf')

plt.clf()

fig, ax = plt.subplots()
fig.set_size_inches(10, 7.5)
ax.set_title(r'Bayesian Inference of Bernoulli $p$ Value with $p_{true} = $ %0.4f' % p_true)
ax.set_xlim([0, 1])
ax.set_ylim([0, 100])

ax.set_ylabel('Posterior')
fig.set_tight_layout(True)

x = np.linspace(-0.1, 1.1, 10000)
distrib, = ax.plot(x, beta.pdf(x, gif_alphas[0], gif_betas[0]), 'k-', lw=2, alpha=0.6)

# Draw p = p_true
#initline, = ax.plot(np.repeat(p_true, 102), np.asarray(range(-1, 101)), 'r-')


def update(i):
	est_p = gif_p_ests[i]
	label = r'Timestep: %.4d, $p_{est}$ = %0.4f, $p_{true}$ = %0.4f' % (i, est_p, p_true)
	distrib.set_ydata(beta.pdf(x, gif_alphas[i], gif_betas[i]))
	ax.set_xlabel(label)
	return distrib, ax


anim = FuncAnimation(fig, update, frames=np.arange(0, num_samples), interval=100)
anim.save(file_name + '.gif', dpi=100, writer='imagemagick')
