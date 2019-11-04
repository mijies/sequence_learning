import sys
sys.path.append('..')
from common.util import np

# DONE
class Adam():
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter  = 0
        self.m = None
        self.v = None

    
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))

        self.iter += 1 # reduce learning rate per iteration
        lr = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for i in range(len(params)):
            # self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            # self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) + grads[i]**2
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i]**2 - self.v[i])

            params[i] -= lr * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)

