from nanaograd.nn.optimizers._optimzer import Optimizer


class SGD(Optimizer):

    def __init__(self,parameters, lr, momentum):
        super().__init__(parameters,lr,momentum)
        self.momentum = momentum
        self.lr = lr
        self.velocities = [0] * len(self.parameters)

    def step(self):
        for i in range(len(self.parameters)):
            p,v = self.parameters[i], self.velocities[i]
            self.velocities[i] = self.momentum * v + ( 1 - self.momentum ) * p.grad
            p.data -= self.lr * v

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad = 0
            
    def __call__(self,true_y, pred_y):
        return self.step()
