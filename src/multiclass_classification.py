import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from feedforward.neural_network import NeuralNetwork


def make_spiral(n_samples, n_classes):
    # http://cs231n.github.io/neural-networks-case-study/#together
    X = np.zeros((n_samples * n_classes, 2))
    y = np.zeros(n_samples * n_classes, dtype='int')
    for j in range(n_classes):
        i = range(n_samples * j, n_samples * (j + 1))
        radius = np.linspace(0.0, 1, n_samples)
        theta = np.linspace(j * 4, (j + 1) * 4, n_samples) + np.random.randn(n_samples) * 0.2
        X[i] = np.c_[radius * np.sin(theta), radius * np.cos(theta)]
        y[i] = j
    return X, y


# X, y = make_classification(n_samples = 1000, n_features=2, n_informative=2, n_redundant=0, n_classes=4, n_clusters_per_class=1)
# Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=50)

# plt.figure()
# plt.scatter(Xtrain[:,0], Xtrain[:,1], c=ytrain, cmap=plt.get_cmap("plasma"), s=5)
# plt.show()

X, y = make_spiral(1000, 3)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=50)
plt.scatter(Xtrain[:, 0], Xtrain[:, 1], c=ytrain, s=5, cmap=plt.get_cmap("plasma"))
plt.show()

mean = np.mean(Xtrain, axis=0, keepdims=True)
std = np.std(Xtrain, axis=0, keepdims=True)
Xtrain = (Xtrain - mean) / std
Xtest = (Xtest - mean) / std

hiddenLayers = {"0": [25, "relu"], "1": [50, "relu"], "2": [25, "relu"], "3": [3, "softmax"]}
nn = NeuralNetwork(Xtrain, ytrain, hiddenLayers)
nn.fit(10000, 0.03)

x = np.arange(-3, 3, 0.01)
y = np.arange(-3, 3, 0.01)
xx, yy = np.meshgrid(x, y)
positions = np.vstack([xx.ravel(), yy.ravel()]).T

ypred = nn.predict(positions)
zz = np.reshape(ypred, xx.shape)
plt.figure(figsize=(7, 5))
plt.contourf(xx, yy, zz, cmap=plt.get_cmap("plasma"))
plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest, s=5, cmap=plt.get_cmap("plasma"))
plt.show()
