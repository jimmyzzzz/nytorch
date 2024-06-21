# Introduction

Nytorch enriches PyTorch with advanced particle operations through nytorch.NytoModule, seamlessly integrated with torch.nn.Module. It enhances PyTorch without redundancy, allowing effortless integration by simply inheriting from nytorch.NytoModule instead of torch.nn.Module. This ensures full compatibility with existing methods while unlocking powerful new functionalities.

Below is a simple example:

```python
import nytorch
import torch

class NytoLinear(nytorch.NytoModule):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.Tensor([2.]))
        self.bias = torch.nn.Parameter(torch.Tensor([1.]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias

net: NytoLinear = NytoLinear()
```

Particle operation allows the model to perform operations similar to tensors to obtain a new model.
For example, `net` obtains a new NytoLinear instance through particle operation:

```python
net2: NytoLinear = net + 10
```

Compare the parameters of `net` and `net2`:

```python
print(net.weight)   # Parameter containing: tensor([2.], requires_grad=True)
print(net2.weight)  # Parameter containing: tensor([12.], requires_grad=True)
print(net.bias)     # Parameter containing: tensor([1.], requires_grad=True)
print(net2.bias)    # Parameter containing: tensor([11.], requires_grad=True)
```

We can observe that the result of adding a scalar to the model is equivalent to applying the scalar to all parameters of the model. Similar principles apply to other arithmetic operations such as subtraction or multiplication. When the operands are models, the operation acts on corresponding parameter positions:

```python
net3: NytoLinear = net + net2

print(net3.weight)  # Parameter containing: tensor([14.], requires_grad=True)
print(net3.bias)    # Parameter containing: tensor([12.], requires_grad=True)
```

So, algorithms that might have been difficult to implement in the past can now be implemented elegantly and simply through Nytorch.


# Installation

Nytorch is a tool developed based on PyTorch.
Before installing Nytorch, please ensure that you have installed PyTorch version 1.7 or higher, and are using Python version 3.8 or higher.

You can install Nytorch via PyPi:

```
pip install nytorch
```


# Documentation

For detailed documentation and usage instructions, please visit [ReadTheDocs](https://nytorch.readthedocs.io/en/latest/).


# Author(jimmyzzzz)

gmail: sciencestudyjimmy@gmail.com
github: https://github.com/jimmyzzzz
