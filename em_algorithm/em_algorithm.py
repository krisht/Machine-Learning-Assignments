import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
import scipy.stats
import shutil

n_iter = 100


def em_algorithm(x, k=2):
    n = x.shape[0]
    m = x.shape[1]

    mu = np.random.rand(k, m, 1)
    cov = np.repeat(np.expand_dims(np.identity(m, dtype=np.float64), 0), k, axis=0)
    pi = np.repeat([1.0 / k], k)

    likelihood = 0  # Technically log likelihood

    gamma = np.zeros((k, n, 1))
    n_k = np.zeros(k)
    if os.path.exists('./temp/'):
        shutil.rmtree('./temp/')
    # Make plots for initial parameters, also creates frame 0 of gif
    plot(x, mu, cov, 0, scale=20, gamma=gamma)

    for ii in range(n_iter):
        normpdf = list(np.zeros(k, dtype=np.int))
        for kk in range(k):
            normpdf[kk] = scipy.stats.multivariate_normal(mean=np.ravel(mu[kk]), cov=cov[kk])

        # E Step
        for kk in range(k):
            for nn in range(n):
                temp = 0

                for jj in range(k):
                    temp += (pi[jj] * normpdf[jj].pdf(np.ravel(x[nn, :])))
                gamma[kk][nn, 0] = pi[kk] * normpdf[kk].pdf(np.ravel(x[nn, :])) / temp

        for kk in range(k):
            n_k[kk] = 0
            for nn in range(n):
                n_k[kk] += gamma[kk][nn, 0]

        # M Step

        for kk in range(k):
            mu[kk] = np.zeros((m, 1))
            for nn in range(n):
                mu[kk] += gamma[kk][nn, 0] * x[nn, :].T / n_k[kk]

            cov[kk] = np.zeros((m, m))
            for nn in range(n):

                cov[kk] += gamma[kk][nn, 0] * (x[nn,:].T - mu[kk]) * (x[nn,:].T-mu[kk]).T/n_k[kk]
            pi[kk] = n_k[kk] / float(n)

        old_likelihood = likelihood
        likelihood = 0

        for nn in range(n):
            temp = 0
            for kk in range(k):
                temp += pi[kk] * normpdf[kk].pdf(np.ravel(x[nn, :]))
            likelihood += np.log(temp)

        plot(x, mu, cov,  ii+1, scale=20, gamma=gamma)

        print("Iteration %3d out of %d: %f" % (ii + 1, n_iter, likelihood))

        if abs(old_likelihood - likelihood) < 0.001:
            print("Converged at iteration %d" % (ii + 1))
            break

    return gamma, mu, cov, pi, likelihood, ii + 1


def plot(x, mu, cov, frame, scale=None, gamma=None):
    k = len(mu)
    n = x.shape[0]
    m = x.shape[1]
    plt.figure(figsize=(8, 6))
    plt.title('Expectation-Maximization Algorithm after %3d iterations' % frame)
    if not os.path.exists('./temp'):
        os.makedirs('./temp')
    dir_name = './temp'
    fig_name = "pic_%03d.png" % frame

    if m == 2 and k > 3:
        plt.scatter(x[:, 0], x[:, 1], color='r')
        plt.scatter(mu[:, 0], mu[:, 1], color='k')

        num = 100
        a = np.linspace(-10, 10, num)
        b = np.linspace(-10, 10, num)
        x1, x2 = np.meshgrid(a, b)
        for kk in range(k):
            normpdf = scipy.stats.multivariate_normal(mean=np.ravel(mu[kk]), cov=cov[kk])
            temp = np.array([np.ravel(x1), np.ravel(x2)])
            z = []
            for ii in range(num**2):
                z.append(normpdf.pdf(np.ravel(temp[:, ii])))
            z = np.reshape(z, (num, num))
            plt.contour(a, b, z)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
    elif m == 2 and k > 1:
        # Plot color-coded data
        for nn in range(n):
            r = gamma[0][nn, 0]  # red is k = 1
            g = gamma[1][nn, 0]  # green is k = 2
            if k == 2:
                b = 0  # fix one of the RGB values if only 2 clusters
            elif k == 3:
                b = gamma[2][nn, 0]  # blue is k = 3
            plt.scatter(x[nn, 0], x[nn, 1], c=(r, g, b))

        # One contour plot for each mean
        num = 100
        a = np.linspace(-10, 10, num)
        b = np.linspace(-10, 10, num)
        x1, x2 = np.meshgrid(a, b)
        for kk in range(k):
            normpdf = scipy.stats.multivariate_normal(mean=np.ravel(mu[kk]), cov=cov[kk])
            temp = np.array([np.ravel(x1), np.ravel(x2)])
            z = []
            for ii in range(num**2):
                z.append(normpdf.pdf(np.ravel(temp[:, ii])))
            z = np.reshape(z, (num, num))
            rgb = 'r' if (kk == 0) else ('g' if (kk == 1) else 'b')
            plt.contour(a, b, z, 1, colors=rgb)

        # Plot each mean
        for kk in range(k):
            plt.scatter([mu[kk][0]], [mu[kk][1]], color='k')

        # Plot labels
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
    elif m == 1:
        plt.hist(x, bins=scale * 5, color='y')

        for kk in range(k):
            n = 100
            s = 3
            sigma_k = np.sqrt(cov[kk])
            x = np.linspace(np.asscalar(mu[kk] - s * sigma_k), np.asscalar(mu[kk] + s * sigma_k), n)
            if m == 3:
                rgb = 'r' if (kk == 0) else ('g' if (kk == 1) else 'b')
                plt.plot(x, scale * mlab.normpdf(x, np.asscalar(mu[kk]), np.asscalar(sigma_k)), color=rgb)
            else:
                plt.plot(x, scale * mlab.normpdf(x, np.asscalar(mu[kk]), np.asscalar(sigma_k)))
        plt.xlabel('x')
        plt.xlim([-60, 60])
        plt.ylim([0, scale * 0.5])

    plt.savefig(dir_name + '/' + fig_name)


# Data set that Keene asked us to generate in class
def getKeeneData(N1, N2):
    # Total number of observations
    N = N1 + N2;

    # Parameters for Gaussian (these are unknown and we're to estimate them)
    mu1 = np.array([2, 2])
    mu2 = np.array([-2, -2])
    cov = np.eye(2)/2

    # Generate data samples from Gaussian
    # Samples are nObservations-by-nFeatures
    X1 = np.random.multivariate_normal(mu1, cov, N1)
    X2 = np.random.multivariate_normal(mu2, cov, N2)
    X = np.concatenate((X1, X2), axis=0)
    # concatenate data into one matrix
    Y = np.concatenate((np.ones((X1.shape[0], 1)), np.zeros((X2.shape[0], 1))), axis=0)
    # generate column vector for corresponding values (we want to predict these)

    # Convert data to matrices for simpler Python operations
    X = np.asmatrix(X)
    Y = np.asmatrix(Y)

    return X, Y

def animate():
    files = [f for f in os.listdir('./temp/') if os.path.isfile(os.path.join('./temp/', f))]
    files.sort()
    with open('image_list.txt', 'w') as file:
        for item in files:
            file.write("./temp/%s\n" % item)

    os.system('convert @image_list.txt -delay {} result.gif'.format(100))
    shutil.rmtree('./temp/')


def generate1DClusters(K, sizes):
    if K > len(sizes):
        print('Specified K %d exceeds size of list %d' % (K, len(sizes)))
        return

    # Parameters to generate data with using for Gaussian
    mu = []
    for k in range(K):
        # Distribute clusters so that means are evenly separated from -x_range to x_range
        # Draw it visually and you'll see that this is the formula to do this
        x_range = 50
        x = -x_range + x_range / float(K) + 2 * k * x_range / float(K)
        mu.append(x)
    sigma = 1  # just use 1 for variance

    # Generate data samples from Gaussian
    # Samples are nObservations-by-nFeatures
    X = []
    for k in range(K):
        X_k = np.random.normal(mu[k], sigma, sizes[k])
        X.append(X_k)

    # Convert data to matrices for simpler Python operations
    # First unravel list to make all data in one row
    # Then convert to matrix and transpose to have N rows
    X = np.asmatrix(np.ravel(X)).T

    return X

K=5
print( '----------------------------------------------------------')
print ('K = %d' % (K))
print ('----------------------------------------------------------\n')
size = 20
X = generate1DClusters(K,[size,size,size,size,size])


em_algorithm(X, K)
animate()