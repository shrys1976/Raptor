
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
            if self.requires_grad: 
                #grad_self = self.data + other.data
                self.grad += Tensor._unbroadcast(out.grad, self.data.shape)
            if other.requires_grad: 
               # grad_other = self.data + other.data
                other.grad +=Tensor._unbroadcast(out.grad, other.data.shape)

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
                grad_self = other.data * out.grad
               
                self.grad+=Tensor._unbroadcast(grad_self, self.data.shape)
            if other.requires_grad:
                grad_other = self.data * out.grad
                self.grad += Tensor._unbroadcast(grad_other, other.data.shape) 

        out._backward = _backward
        return out    


    def __rmul__(self,other):
        return self*other


    def __radd__(self,other):
        return self+other

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return self*-1

    def __rsub__(self,other):
        return other +(-self)


    def __truediv__(self, other):
        other  = other if isinstance(other,Tensor) else Tensor(other, requires_grad = False)       

        out = Tensor(

            self.data/other.data,
            (self, other),
            '/',
            self.requires_grad or other.requires_grad,
        )                                 

        def _backward():
            if self.requires_grad:
                grad_self =  (1.0/other.data)*out.grad
                self.grad += Tensor._unbroadcast(grad_self, self.data.shape)

            if other.requires_grad:
                grad_other =  (-self.data/(other.data**2))*out.grad
                other.grad += Tensor._unbroadcast(grad_other, other.data.shape)

        out._backward = _backward
        return out


    def __rtruediv__(self, other):
        
         other  = other if isinstance(other,Tensor) else Tensor(other, requires_grad = False)       
         return other/self



    def __pow__(self,power):
        assert isinstance(power,(int, float)), "only scalar powers supported"

        out = Tensor(

            self.data ** power,
            (self,),
            f"**{power}",
            self.requires_grad,
        )

        def _backward():
            if self.requires_grad:
                self.grad += (power*(self.data ** (power-1)))*out.grad


        out._backward  = _backward
        return out                



    def sum(self, axis =None, keepdims  =  False): # performs on all axes, dims are not retained after op
        out = Tensor(

            self.data.sum(axis = axis, keepdims = keepdims),
            (self,),
            "sum",
            self.requires_grad,
        )         

        def _backward():
            if not self.requires_grad:
                return

            grad = out.grad

            if axis is None:
                self.grad += np.ones_like(self.data)*grad

            else:
                if not keepdims:
                    if isinstance(axis,int):
                        axes = (axis,)
                    else:
                        axes = axis

                    for ax in sorted(axes):
                        grad = np.expand_dims(grad,ax)

                self.grad += np.ones_like(self.data)*grad 

        out._backward = _backward
        return out                                                                              



    def __matmul__(self, other):
        
            other = other if isinstance(other, Tensor) else Tensor(other,requires_grad=False)


            out = Tensor(

                self.data @ other.data,
                (self, other),
                '@',
                self.requires_grad or other.requires_grad,
            )     

            def _backward():
                if self.requires_grad:
                    self.grad+= out.grad @ other.data.T
                if other.requires_grad:
                    other.grad+= self.data.T @ out.grad    

            out._backward = _backward
            return out           


    # helper to undo broadcasting while calculating the backprop gradient
    # converts the tensor shape back to what it was before broadcasting during forward pass
    @staticmethod
    def _unbroadcast(grad, shape):
        while len(grad.shape)> len(shape): # check if grad has more dim than shape
                                            # means that broadcasting inserted dims to the left
            grad = grad.sum(axis= 0)

        for i, dim in enumerate(shape):
            if dim == 1 :# sum across dims where org shape had 1
                grad = grad.sum(axis = i, keepdims =  True)

        return grad                                