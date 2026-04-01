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
               

