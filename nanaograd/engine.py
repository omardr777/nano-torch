import math

class Value:
    def __init__(self,data, _children=(), _op="", label=''):
        self.data = data
        self._op = _op
        self.label = label
        self._prev = set(_children)
        self.grad = 0.0
        self._backward = lambda:None

    def __repr__(self):
        return f"Value(data={self.data}, label={self.label})"
    
    def __add__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out =  Value(self.data + other.data,(self,other), "+")
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data,(self,other),'*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self,other):
        assert isinstance(other,(int,float))
        out = Value(self.data ** other, (self,), f'**{other}')
        
        def _backward():
            self.grad += other * self.data**(other -1) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self,other):
        return self * other **-1

    def __rmul__(self,other):
        return self*other

    def __radd__(self,other):
        return self + other

    def __neg__(self):
        return self * -1
    
    def __sub__(self,other):
        return self + (-other)

    def __rsub__(self,other):
        return -self+other
        
    def backward(self):
        self.grad = 1.0
        topo = []
        visited = set()
        def build_topo(root):
            if root not in visited:
                visited.add(root)
                for child in root._prev:
                    build_topo(child)
                topo.append(root)
        build_topo(self)
        for node in reversed(topo):
            node._backward()

    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1 )/(math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def relu(self):
        x = self.data
        out = Value(x if x > 0 else 0,(self,), 'relu')

        def _backward():
            self.grad += (1 if x > 0 else 0) * out.grad

        out._backward = _backward
        return out