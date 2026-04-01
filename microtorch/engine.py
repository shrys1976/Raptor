from audioop import reverse
import numpy as np

# Tensor core, fundamental graph node
class Tensor:
    def __init__(self,data, _children = (), _op = '', requires_grad = True):
        if not isinstance(data,np.ndarray):
            data = np.array(data,dtype=np.float32)

        self.data =data
        self.grad = np.zeros_like(self.data, dtype=np.float32)
        self.requires_grad = requires_grad

        self._prev = set(_children)
        self._op = _op
        self._backward = lambda : None

    def __repr__(self):
        return f"Tensor(data = {self.data}, grad ={self.grad})"

#topological backward pass 
#starts from output tensor, visit all ancestors
#call each node's backward rule (_backward) in rev topoligical order
    def backward(self):

        topo = []
        visited = set()


        def build(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build(child)
                topo.append(node)

        build(self)             

        #grad = 1
        self.grad = np.ones_like(self.data, dtype = np.float32)

        for node in reversed(topo):
            node._backward()           



    def __add__(self,other):
        other = other if isinstance(other, Tensor)else Tensor(other, requires_grad=False)

        out = Tensor(

            self.data+other.data,
            (self,other),
            '+',
            self.requires_grad or other.requires_grad,
        )

        def _backward():
            if self.requires_grad: self.grad +=out.grad
            if other.requires_grad: other.grad += out.grad

            out._backward = _backward
            return out


    def __mul__(self,other):

        other = other if isinstance(other,Tensor) else Tensor(other,requires_grad=False)

        out = Tensor(

            self.data*other.data,
            (self,other),
            '*',
            self.requires_grad or other.requires_grad,
        )

        def _backward():
            if self.requires_grad:
                self.grad+= other.data * out.grad
            if other.requires_grad:
                other.grad += self.data  * out.grad 

            out._backward = _backward
            return out    


    def __rmul__(self,other):
        return self*other



    def __radd__(self,other):
        return self+other

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self,other):
        return self*-1

    def __rsub__(self,other):
        return other +(-self)                                