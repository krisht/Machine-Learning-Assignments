# Krishna Thiyagarjan
# Abhinav Jain
# Linear Classification

import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import pylab
import random

N1 = 1000
N2 = 1000
N = N1 + N2
test_set_ratio = 0.4
mu1 = np.asarray([1, 1])
mu2 = np.asarray([-1, -1])
cov = np.eye(2)

n_iter = 50 # Number of iterations for IRLW

def sigmoid(x):
	return scipy.special.expit(x)

# Generate Training and Testing Data
x1_train = np.random.multivariate_normal(mu1, cov, N1).T
x1_train_labels = np.expand_dims(np.ones(len(x1_train.T)), 0)
x1_train = np.append(x1_train, x1_train_labels, axis = 0)

x1_test = np.random.multivariate_normal(mu1, cov, int(test_set_ratio*N1)).T
x1_test_labels = np.expand_dims(np.ones(len(x1_test.T)), 0)
x1_test = np.aeppend(x1_test, x1_test_labels, axis = 0)

x2_train = np.random.multivariate_normal(mu2, cov, N2).T
x2_train_labels = np.expand_dims(np.zeros(len(x2_train.T)), 0)
x2_train = np.append(x2_train, x2_train_labels, axis = 0)

x2_test = np.random.multivariate_normal(mu2, cov, int(test_set_ratio*N2)).T
x2_test_labels = np.expand_dims(np.zeros(len(x2_test.T)), 0)
x2_test = np.append(x2_test, x2_test_labels, axis = 0)

all_data = np.concatenate((x1_train, x2_train), axis = 1)
all_test = np.concatenate((x1_test, x2_test), axis = 1)

np.random.shuffle(all_data.T)
np.random.shuffle(all_test.T)

################ Gaussian Generative Model ################

######### pi(4.73), mu1_est(4.75), mu2_est (4.76) #########
pi = N1/float(N1 + N2) #pi according to 4.73
mu1_est = np.sum(all_data[0:2]*all_data[2], axis = 1)/float(N1) #4.75
mu2_est = np.sum(all_data[0:2]*(1 - all_data[2]), axis = 1)/float(N2) #4.76


S1 = (1/float(N1)) * (np.multiply((all_data[:2].T - mu1_est).T, all_data[2]).dot(np.multiply((all_data[:2].T - mu1_est).T, all_data[2]).T))
S2 = (1/float(N2)) * (np.multiply((all_data[:2].T - mu2_est).T, 1 - all_data[2]).dot(np.multiply((all_data[:2].T - mu2_est).T, 1 - all_data[2]).T))
S = (float(N1)/N) * S1 + (float(N2)/N) * S2

######### Model parameters with estimated parameters #########
w = np.linalg.inv(S).dot(mu1_est - mu2_est)
w_0 = (-0.5 * mu1_est.dot(np.linalg.inv(S).dot(mu1_est))) + (0.5 * mu2_est.dot(np.linalg.inv(S).dot(mu2_est))) + (np.log(pi/(1-pi)))

print("Weight vector for Gaussian Generative Model: %s" % [w_0, w[0], w[1]])


######### Predict class based on 0.5 probability boundary #########

np.set_printoptions(suppress=True)
predictions = np.round(sigmoid(w.dot(all_test[:2]) + w_0))
accuracy = np.sum(predictions == all_test[2])/float(len(all_test[2]))*100
x_line = np.linspace(-4, 4, 100)
y_line = ((-w[0]*x_line) - w_0)/w[1]

plt.figure()

plt.title("Decision Boundary for Gaussian Generative Model: %.4f%%" % accuracy)
plt.xlabel("X")
plt.ylabel("Y")
plt.scatter(x1_test[0], x1_test[1], color='blue', s=2, label='Class 1')
plt.scatter(x2_test[0], x2_test[1], c='red', s = 2, label='Class 2')
plt.plot(x_line, y_line, color='green', lw=1, label='Boundary')
plt.legend()

################ Logistic Regression Model ################

iota = np.concatenate((np.ones((1,N)), all_data[:2]), axis = 0)
w_old = np.array([[0, 0, 0]])

for _ in range(n_iter):
	t = all_data[2]
	y = sigmoid(w_old.dot(iota))
	R = (y.T).dot(y)
	R = R * np.eye(N)

	w = w_old - np.linalg.inv(iota.dot(R).dot(iota.T)).dot(iota).dot((y - t).T).T
	w_old = w 

w = np.ndarray.flatten(w)
print("Weight vector for Logistic Regression: %s" % w)
np.set_printoptions(suppress=True)
predictions = np.round(sigmoid(w[0] + w[1:3].dot(all_test[:2])))
accuracy = np.sum(predictions == all_test[2])/float(len(all_test[2]))*100
x_line = np.linspace(-4, 4, 100)
y_line = ((-w[1]*x_line- w[0]))/w[2]

plt.figure()

plt.title("Decision Boundary for Logistic Regression: %.4f%%" % accuracy)
plt.xlabel("X")
plt.ylabel("Y")
plt.scatter(x1_test[0], x1_test[1], color='blue', s=2, label='Class 1')
plt.scatter(x2_test[0], x2_test[1], c='red', s = 2, label='Class 2')
plt.plot(x_line, y_line, color='green', lw=1, label='Boundary')
plt.legend()
plt.show()