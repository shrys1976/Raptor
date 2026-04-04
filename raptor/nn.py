from .engine import Tensor
from .import ops
import numpy as np

class Module:
    def parameters(self):
        params = []

        for value in self .__dict__.values():
            if isinstance(value, Tensor):
                params.append(value)


            elif isinstance(value, Module):
                params.extend(value.parameters())

            elif isinstance(value,(list,tuple)):
                for item in value:
                    if isinstance(item, Tensor):
                        params.append(item)
                    elif isinstance(item,Module):
                        params.extend(item.parameters())


        return params


# clears old accumulating gradients before running backward pass
    def zero_grad(self):
            for p in  self.parameters():
                p.grad =np.zeros_like(p.data, dtype = np.float32)


    def forward(self,*inputs):
        raise NotImplementedError


    def __call__(self, *inputs):
        return self.forward(*inputs)                            



class Linear(Module):
    def __init__(self, in_features, out_features, bias =True):
        limit  = np.sqrt(6.0/in_features)

        self.weight = Tensor(
            np.random.uniform(-limit, limit, (out_features, in_features)).astype(np.float32)

        )

        self.bias =( Tensor(np.zeros(out_features, dtype = np.float32))
        if bias else None

        )        

    def forward(self,x):
        out = x @ self.weight.transpose()


        if self.bias is not None:
            out = out + self.bias

        return out       

class ReLU(Module):
    def forward(self,x):
        return ops.relu(x)

        
class Sigmoid(Module):
    def forward(self, x):
        return ops.sigmoid(x)


class Tanh(Module):
    def forward(self, x):
        return ops.tanh(x)

class Sequential(Module):
    def __init__(self, *layers):
        self.layers = layers

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
        return x    

class MSELoss(Module):
    def forward(self, pred, target):
        diff = pred - target
        return (diff*diff).mean() # mean sqaured error


class CrossEntropyLoss(Module):
    def forward(self, logits, target):
        logits_data = logits.data

        if logits_data.ndim == 1:
            logits_data = logits_data.reshape(1, -1)

        target = np.array(target)
        if target.ndim == 0:
            target = target.reshape(1)

        shifted = logits_data - np.max(logits_data, axis=1, keepdims=True)
        exp_shifted = np.exp(shifted)
        probs = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)

        batch_size = logits_data.shape[0]
        loss_data = -np.log(probs[np.arange(batch_size), target])

        out = Tensor(
            loss_data.mean(),
            (logits,),
            "cross_entropy",
            logits.requires_grad,
        )

        def _backward():
            if logits.requires_grad:
                grad = probs.copy()
                grad[np.arange(batch_size), target] -= 1.0
                grad = grad / batch_size

                if logits.data.ndim == 1:
                    grad = grad.reshape(logits.data.shape)

                logits.grad += grad * out.grad

        out._backward = _backward
        return out
