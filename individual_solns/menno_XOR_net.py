import numpy as np
from functools import lru_cache


class XOR_net:
    def __init__(self, activation="sigmoid", weights=None):
        FUNC_TABLE = {"sigmoid": self.sigmoid, "tanh": self.tanh}
        DFUNC_TABLE = {"sigmoid": self.dsigmoid, "tanh": self.dtanh}
        self.weights = self.random_weights() if weights is None else weights
        # Storage for skipping the calculation of some variables
        # 0, 1: inputs of NN
        # 2, 3: values of the hidden layer pre activation
        # 4, 5: values of the hidden layer post activation
        # 6: value of the final node pre activation
        # 7: value of the final node post activation
        self.store = np.zeros(8)
        self.actf = FUNC_TABLE[activation]  # activation function
        self.dactf = DFUNC_TABLE[activation]  # derivative of activation function
        self.inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.outputs = np.array([0, 1, 1, 0])

    def simulate(self, inpt):
        self.store[:2] = inpt
        self.store[2:4] = np.dot(self.weights[:6].reshape((2, 3)), np.append(inpt, 1))
        self.store[4:6] = [self.actf(el) for el in self.store[2:4]]
        self.store[6] = np.dot(self.weights[6:], np.append(self.store[4:6], 1))
        self.store[7] = self.actf(self.store[6])
        return self

    def output(self, discrete=False):
        if discrete:
            return 1 if self.store[-1] >= 0.5 else 0
        else:
            return self.store[-1]

    def mse(self):
        return sum(
            [
                (self.simulate(inpt).output() - output) ** 2
                for inpt, output in zip(self.inputs, self.outputs)
            ]
        )

    def grdmse(self):
        par_derivs = np.zeros((9))
        for inpt, output in zip(self.inputs, self.outputs):
            self.simulate(inpt)
            temp = 2 * (self.store[-1] - output) * self.dactf(self.store[6])

            par_derivs[6:] += temp * np.append(self.store[4:6], 1)
            par_derivs[3:6] += (
                temp
                * self.weights[7]
                * self.dactf(self.store[3])
                * np.append(self.store[:2], 1)
            )
            par_derivs[:3] += (
                temp
                * self.weights[6]
                * self.dactf(self.store[2])
                * np.append(self.store[:2], 1)
            )
        return par_derivs

    def update_weights(self, eta=0.01):
        self.weights -= eta * self.grdmse()

    def random_weights(self):
        weights = np.zeros((9))
        weights[:6] = np.random.normal(0, np.sqrt(2 / (2 + 2)), (6))
        weights[6:] = np.random.normal(0, np.sqrt(2 / (2 + 1)), (3))
        return weights

    def print_test(self, discrete=True):
        for inpt, output in zip(self.inputs, self.outputs):
            print(
                f"{inpt[0]} ^ {inpt[1]} = {self.simulate(inpt).output(discrete):{'.3f' if not discrete else ''}} [{output}]"
            )
        print(f"final mse: {self.mse()}")

    def is_xor(self):
        for inpt, output in zip(self.inputs, self.outputs):
            if self.simulate(inpt).output(True) != output:
                return False
        return True

    @lru_cache()
    def sigmoid(self, x):
        return (1 + np.exp(-x)) ** -1

    @lru_cache()
    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    @lru_cache()
    def tanh(self, x):
        return np.tanh(x)

    @lru_cache()
    def dtanh(self, x):
        return np.cosh(x) ** -2


# Lazy approach
for i in range(10):
    satisfied = False
    n = 0
    while not satisfied:
        n += 1
        nn = XOR_net()
        satisfied = nn.is_xor()
    nn.print_test(False)
    print(n, nn.weights.reshape((3, 3)), sep="\n")


n_loops = 10000
# train network with sigmoid
nn = XOR_net(activation="sigmoid")
for i in range(n_loops):
    nn.update_weights(0.1)
    if not i % (n_loops // 10):
        print(f"Epoch {i:{int(np.log10(n_loops))}}: MSE = {nn.mse():.4f}")
nn.print_test()

# train network with tanh
nn = XOR_net(activation="tanh")
for i in range(n_loops):
    nn.update_weights(0.01)
    if not i % (n_loops // 10):
        print(f"Epoch {i:{int(np.log10(n_loops))}}: MSE = {nn.mse():.4f}")
nn.print_test()
