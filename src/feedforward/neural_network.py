import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:

    def __init__(self, X, y, architecture, regularization="Frob", regularization_parameter=0.001):
        """"
        X = (N,D)
        y = (N,)
        architecture = { i : [ nb neurons, activation function ] }
        
        labels must follow 0, 1, 2, ...
        """

        self.Xtrain = X.T
        self.ytrain = y.reshape(1, len(y))
        self.D = self.Xtrain.shape[0]
        self.N = self.Xtrain.shape[1]

        self.reg = regularization
        self.reg_parameter = regularization_parameter

        self.memory = {}
        for i in range(len(architecture)):
            idx = str(i)

            if i == 0:
                self.memory[idx] = {"activation": architecture[idx][1],
                                    "W": np.random.randn(architecture[idx][0], self.D) * np.sqrt(2.0 / self.N),
                                    # output x input
                                    "b": np.zeros((architecture[idx][0], 1))}
            else:
                self.memory[idx] = {"activation": architecture[idx][1],
                                    "W": np.random.randn(architecture[idx][0], architecture[str(i - 1)][0]) * np.sqrt(
                                        2.0 / self.N),
                                    "b": np.zeros((architecture[idx][0], 1))}

    def activation(self, activation):
        if activation == "relu":
            return self.relu
        elif activation == "logistic":
            return self.logistic
        elif activation == "softmax":
            return self.softmax
        elif activation == "tanh":
            return self.tanh
        else:
            raise ValueError("Activation function is not supported")

    def dactivation(self, activation):
        if activation == "relu":
            return self.drelu
        elif activation == "logistic":
            return self.dlogistic
        elif activation == "tanh":
            return self.dtanh
        else:
            raise ValueError("Derivative of activation function is not supported")

    def logistic(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def tanh(self, x):
        return np.tanh(x)

    def dlogistic(self, dA, x):
        tmp = self.logistic(x)
        return dA * tmp * (1 - tmp)

    def drelu(self, dA, x):
        return dA * (x > 0);

    def dtanh(self, dA, x):
        return dA * (1 - self.tanh(x) ** 2)

    def softmax(self, x):
        exps = np.exp(x)
        return exps / np.sum(exps, axis=0, keepdims=True)

    def dbinaryCrossEntropy(self):
        return ((1 - self.ytrain) / (1 - self.output) - (self.ytrain / self.output)) / self.N

    def doutput(self, activation, z):

        if activation == "softmax":  # multiclass cross entropy
            dZ = np.array(self.output, copy=True)
            dZ[np.squeeze(self.ytrain), range(self.N)] -= 1
            return dZ / self.N

        elif activation == "logistic":
            dA = self.dbinaryCrossEntropy()
            dZ = self.dlogistic(dA, z)
            return dZ

        elif activation == "relu":
            dA = self.dbinaryCrossEntropy()
            dZ = self.drelu(dA, z)
            return dZ

        elif activation == "tanh":
            dA = self.dbinaryCrossEntropy()
            dZ = self.dtanh(dA, z)
            return dZ

        else:
            raise ValueError("Activation function is not supported in output layer")

    def feedforward(self, X):

        """"
        X = (D,N)
        """

        for i in range(len(self.memory)):
            idx = str(i)
            if i == 0:
                self.memory[idx]["Z"] = self.memory[idx]["W"] @ X + self.memory[idx]["b"]
                self.memory[idx]["A"] = self.activation(self.memory[idx]["activation"])(
                    self.memory[idx]["Z"])  # output of layer i
            else:
                self.memory[idx]["Z"] = self.memory[idx]["W"] @ self.memory[str(i - 1)]["A"] + self.memory[idx]["b"]
                self.memory[idx]["A"] = self.activation(self.memory[idx]["activation"])(self.memory[idx]["Z"])
        self.output = self.memory[str(len(self.memory) - 1)]["A"]

        # self.A0 = X
        # self.Z1 = self.W1 @ self.A0 + self.b1
        # self.A1 = activationFunction(self.Z1)
        # self.Z2 = self.W2 @ self.A1 + self.b2
        # self.A2 = activationFunction(self.Z2)
        # A2 is the output

    def backpropagate(self, learning_rate):

        for i in reversed(range(len(self.memory))):
            idx = str(i)

            if i == len(self.memory) - 1:  # output layer
                # dA = self.dcostFunc(self.output, self.ytrain)
                # self.memory[idx]["dZ"] = self.memory[idx]["dactivation"](dA, self.memory[idx]["Z"])
                self.memory[idx]["dZ"] = self.doutput(self.memory[idx]["activation"], self.memory[idx]["Z"])
                self.memory[idx]["dW"] = self.memory[idx]["dZ"] @ self.memory[str(i - 1)]["A"].T + self.dregularization(
                    self.memory[idx]["W"])
                self.memory[idx]["db"] = np.sum(self.memory[idx]["dZ"], axis=1, keepdims=True)
                self.memory[idx]["dA"] = self.memory[idx]["W"].T @ self.memory[idx]["dZ"]

            elif i == 0:
                self.memory[idx]["dZ"] = self.dactivation(self.memory[idx]["activation"])(self.memory[str(i + 1)]["dA"],
                                                                                          self.memory[idx]["Z"])
                self.memory[idx]["dW"] = self.memory[idx]["dZ"] @ self.Xtrain.T + self.dregularization(
                    self.memory[idx]["W"])
                self.memory[idx]["db"] = np.sum(self.memory[idx]["dZ"], axis=1, keepdims=True)
                self.memory[idx]["dA"] = self.memory[idx]["W"].T @ self.memory[idx]["dZ"]  ## dA = dX

            else:
                self.memory[idx]["dZ"] = self.dactivation(self.memory[idx]["activation"])(self.memory[str(i + 1)]["dA"],
                                                                                          self.memory[idx]["Z"])
                self.memory[idx]["dW"] = self.memory[idx]["dZ"] @ self.memory[str(i - 1)]["A"].T + self.dregularization(
                    self.memory[idx]["W"])
                self.memory[idx]["db"] = np.sum(self.memory[idx]["dZ"], axis=1, keepdims=True)
                self.memory[idx]["dA"] = self.memory[idx]["W"].T @ self.memory[idx]["dZ"]

        for i in range(len(self.memory)):
            idx = str(i)
            self.memory[idx]["W"] -= learning_rate * self.memory[idx]["dW"]
            self.memory[idx]["b"] -= learning_rate * self.memory[idx]["db"]

            # dA3 = self.A2-self.ytrain #derivative of loss function

        # dZ2 = dActivationFunction(dA3, self.Z2)
        # dW2 = dZ2 @ self.A1.T
        # db2 = np.sum(dZ2, axis=1, keepdims=True)
        # dA2 = self.W2.T @ dZ2

        # dZ1 = dActivationFunction(dA2, self.Z1)
        # dW1 = dZ1 @ self.A0.T
        # db1 = np.sum(dZ1, axis=1, keepdims=True)
        # dA1 = self.W1.T @ dZ1

    def regularization(self):
        if self.reg == "Frob":
            weights = 0
            for i in range(len(self.memory)):
                idx = str(i)
                weights += np.sum(np.square(self.memory[idx]["W"]))
            return self.reg_parameter / 2 * weights
        else:
            raise ValueError("Regularization is not supported")

    def dregularization(self, w):
        if self.reg == "Frob":
            return self.reg_parameter * w
        else:
            raise ValueError("Regularization is not supported")

    def multiclassCrossEntropy(self):
        probs = -np.log(self.output[np.squeeze(self.ytrain), range(self.N)])
        return np.mean(probs)

    def binaryCrossEntropy(self):
        L = -(self.ytrain * np.log(self.output) + (1 - self.ytrain) * np.log(1 - self.output))
        return np.mean(L)

    def fit(self, nbIter, learning_rate):
        cost = np.zeros(nbIter)
        for i in range(nbIter):
            self.feedforward(self.Xtrain)

            if len(self.output) == 1:
                cost[i] = self.binaryCrossEntropy()
            else:
                cost[i] = self.multiclassCrossEntropy()

            cost[i] += self.regularization()

            self.backpropagate(learning_rate)

        plt.figure()
        plt.scatter(np.arange(nbIter), cost, s=3, color='blue')
        plt.ylabel("cross entropy")
        plt.xlabel("iteration")
        plt.show()

    def predict(self, X):

        """"
        X = (N,D)
        """

        self.feedforward(X.T)
        ypred = np.zeros(self.output.shape, dtype=int)

        # binary classificatiom
        if ypred.shape[0] == 1:
            ypred[0, self.output[0, :] > 0.5] = 1
            return np.squeeze(ypred)
        else:
            ypred = np.argmax(self.output, axis=0)
            return ypred

    def getAccuracy(self, ypred, ytest):
        return np.mean(ypred == ytest)
