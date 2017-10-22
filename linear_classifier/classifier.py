import numpy as np
import scipy
import matplotlib.pyplot as plt
import pylab

N = 1000 # number of samples
mean1 = [1, 1]
mean2 = [-1, -1]
cov = np.eye(2)

#Sigmoid

def sigmoid(x):
	return scipy.special.expit(x)

#Generate first set
x1, y1 = np.random.multivariate_normal(mean1, cov, N).T
labels = np.zeros(len(x1))
samples1 = np.asarray([x1, y1, labels])

#Generate second set
x2, y2 = np.random.multivariate_normal(mean2, cov, N).T
labels = np.ones(len(x2))
samples2 = np.asarray([x2, y2, labels])


total = np.concatenate((samples1, samples2), axis = 1).T

np.random.shuffle(total)

train = total[0:800]
test = total[800:1000]

print(train.shape)
print(test.shape)


pi = float(np.count_nonzero(train.T[2]))/train.T.shape[1]
print(pi)

plt.scatter(train.T[0], train.T[1], c=train.T[2], cmap=pylab.cm.bwr)
plt.show()
