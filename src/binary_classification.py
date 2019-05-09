import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from feedforward.neural_network import NeuralNetwork

X, y = make_moons(n_samples=1000, noise=0.2, random_state=50)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=50)
mean = np.mean(Xtrain, axis=0, keepdims=True)
std = np.std(Xtrain, axis=0, keepdims=True)
Xtrain = (Xtrain - mean) / std
Xtest = (Xtest - mean) / std

hiddenLayers = {"0": [25, "relu"], "1": [50, "relu"], "2": [25, "relu"], "3": [1, "logistic"]}
nn = NeuralNetwork(Xtrain, ytrain, hiddenLayers)
nn.fit(10000, 0.02)

x = np.arange(-3, 3, 0.01)
y = np.arange(-3, 3, 0.01)
xx, yy = np.meshgrid(x, y)
positions = np.vstack([xx.ravel(), yy.ravel()]).T

ypred = nn.predict(positions)
zz = np.reshape(ypred, xx.shape)
plt.figure(figsize=(7, 5))
plt.contourf(xx, yy, zz, cmap=plt.get_cmap("plasma"))
plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest, cmap=plt.get_cmap("plasma"), s=5)
plt.show()
