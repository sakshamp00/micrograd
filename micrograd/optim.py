# This file contains the SGD class, which implements Stochastic Gradient Descent.
# It updates model parameters by moving them in the opposite direction of the gradient of the loss.
class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad
    
    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0.0