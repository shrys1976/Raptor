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


        
                                                                
