import numpy as np
from .engine import Tensor

def relu(x):
    out = Tensor(

        np.maximum(0,x.data),
        (x,),
        "relu",
        x.requires_grad,
    )

    def _backward():
        if x.requires_grad:
            x.grad+=(x.data>0).astype(np.float32)*out.grad


    out._backward = _backward
    return out        



def sigmoid(x):
    sig  = 1.0/(1.0 + np.exp(-x.data))

    out = Tensor(
        sig,
        (x,),
        "sigmoid",
        x.requires_grad,
    )    

    def _backward():
        if x.requires_grad:
            x.grad += sig*(1.0-sig)*out.grad

    out._backward = _backward
    return out       


def tanh(x):
    t = np.tanh(x.data)

    out = Tensor(

        t,
        (x,),
        "tanh",
        x.requires_grad,  
      )

    def _backward():
        if x.requires_grad:
            x.grad += (1.0-t**2)*out.grad 

    out._backward = _backward
    return out           
