from nanaograd.nn.nn import Module


class Optimizer(Module):
    def __init__(self,parameters, lr=0.001, momentum=0.9):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum

    def forward(self):
        pass

    def step(self):
        raise NotImplementedError('Step method must be implemented')
