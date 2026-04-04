import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = list(params)

    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data, dtype=np.float32)

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [np.zeros_like(p.data, dtype=np.float32) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            grad = p.grad

            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.data

            if self.momentum != 0.0:
                self.velocity[i] = self.momentum * self.velocity[i] + grad
                grad = self.velocity[i]

            p.data -= self.lr * grad


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = [np.zeros_like(p.data, dtype=np.float32) for p in self.params]
        self.v = [np.zeros_like(p.data, dtype=np.float32) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1

        for i, p in enumerate(self.params):
            grad = p.grad

            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.data

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
