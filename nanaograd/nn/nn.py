from nanaograd.engine import Value
import random 

class Module:

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.grad = 0

    def parameters(self):
        return []

    def forward(self,*args, **kwargs):
        raise NotImplementedError("Module base class's forward method must be implemented")

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class Neuron(Module):

    def __init__(self,n_inputs):
        self.w = [Value(random.uniform(-1,1), label=f'w{i}') for i in range(n_inputs)]
        self.b = Value(random.uniform(-1,1), label='b')

    def forward(self, x):
        act = sum((wi * xi for wi,xi in zip(self.w,x)), self.b)
        out = act.tanh()
        return out

    def __repr__(self):
        return f'number of weights: {len(self.w)}'

    def parameters(self):
        return self.w + [self.b]

class Layer(Module):

    def __init__(self,n_inputs,n_output):
        self.neurons = [Neuron(n_inputs) for _ in range(n_output)]

    def forward(self,x):
        outs = [neuron(x) for neuron in self.neurons]
        return outs[0] if len(self.neurons) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __call__(self,x):
        return self.forward(x)

class MLP(Module):

    def __init__(self, n_inputs, n_outs):
        sz = [n_inputs] + n_outs
        print(f'sz{sz}')
        self.layers = [Layer(sz[i],sz[i+1]) for i in range(len(n_outs))]
        for l in self.layers:
            print(l.neurons)

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __call__(self,x):
        return self.forward(x)