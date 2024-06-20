Introduction
======================

In this section, we'll quickly introduce the basic functionalities of Nytorch along with usage examples.
	
Particle Operation
------------------------------------

Nytorch doesn't reinvent the wheel but rather provides **particle operation** functionalities built upon the existing capabilities of PyTorch. When using Nytorch, simply change the inheritance object of the model from torch.nn.Module to nytorch.NytoModule. As nytorch.NytoModule is a subclass of torch.nn.Module, it provides corresponding implementations of all parent class methods.

Below is a simple example::

    import nytorch as nyto
    import torch

    class NytoLinear(nyto.NytoModule):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.Tensor([2.]))
            self.bias = torch.nn.Parameter(torch.Tensor([1.]))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.weight * x + self.bias

    net: NytoLinear = NytoLinear()

Particle operation allows the model to perform operations similar to tensors to obtain a new model.
For example, ``net`` obtains a new NytoLinear instance through particle operation::

	net2: NytoLinear = net + 10

Compare the results of forward propagation between ``net`` and ``net2``::

    x = torch.arange(4)  # tensor([0, 1, 2, 3])
    print(net(x))        # tensor([1., 3., 5., 7.], grad_fn=<MulBackward0>)
    print(net2(x))       # tensor([11., 23., 35., 47.], grad_fn=<MulBackward0>)

Compare the parameters of ``net`` and ``net2``::

    print(net.weight)   # Parameter containing: tensor([2.], requires_grad=True)
    print(net2.weight)  # Parameter containing: tensor([12.], requires_grad=True)
    print(net.bias)     # Parameter containing: tensor([1.], requires_grad=True)
    print(net2.bias)    # Parameter containing: tensor([11.], requires_grad=True)

We can observe that the result of adding a scalar to the model is equivalent to applying the scalar to all parameters of the model. Similar principles apply to other arithmetic operations such as subtraction or multiplication. When the operands are models, the operation acts on corresponding parameter positions::

    net3: NytoLinear = net + net2
    
    print(net3.weight)  # Parameter containing: tensor([14.], requires_grad=True)
    print(net3.bias)    # Parameter containing: tensor([12.], requires_grad=True)

Besides arithmetic particle operations, there are operations to generate a new model from an existing one. For instance, if a model with the same structure but random parameters is needed, the ``randn()`` method can be used. This method returns a model with parameters sampled from a standard normal distribution::

    net_randn: NytoLinear = net.randn()
    
    print(net_randn.weight)  # Parameter containing: tensor([0.3187], requires_grad=True)
    print(net_randn.bias)    # Parameter containing: tensor([0.1877], requires_grad=True)
    
If a model needs to be copied and parameters cloned from the original model, the ``clone()`` method can be used::

    net_clone: NytoLinear = net.clone()

    print(net_clone.weight is net.weight)             # False
    print(torch.equal(net_clone.weight, net.weight))  # True


Benefits of Particle Operation
--------------------------------------------

Why do we need functionalities like particle operation? What benefits does it bring to us? If you're curious about these questions, we have a dedicated section to discuss this topic. In simple terms, such functionalities are for researching and using hybrid evolutionary algorithms and gradient descent algorithms.

Even if you're not familiar with evolutionary algorithms, particle operation is not something new in deep learning. It's used in momentum encoders in MoCo or target networks in DDQN.

Here's another example with momentum. In fact, momentum also utilizes particle operation:

.. math::

   W_t = W_{t-1} + V_t
   
   V_t = \beta V_{t-1} + G_t

The above is the common expression of momentum, where we need to calculate momentum and gradients to update weights. However, momentum can be expressed in the form of particle operation as:

.. math::
   
   W_t = (W_{t-1} + G_t) + \beta (W_{t-1} - W_{t-2})
   
This means that momentum is equivalent to adding :math:`\beta (W_{t-1} - W_{t-2})` to gradient descent.

In code, it would look something like this::

	net1, net0 = gradient_descent(net1) + beta*(net1 - net0), net1

So, algorithms that might have been difficult to implement in the past can now be implemented elegantly and simply through Nytorch.

Below is an example::

    from nytorch import NytoModule
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as f


    # load Iris dataset
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
    x_train = torch.FloatTensor(x_train)
    x_test = torch.FloatTensor(x_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    # Hyperparameter
    BETA = 0.9
    LR = 0.005
    criterion = nn.CrossEntropyLoss()


    def train_module(net: 'MyModel', lr: float) -> 'MyModel':
        new_net = net.clone()
        optimizer = torch.optim.SGD(new_net.parameters(), lr=lr)
        optimizer.zero_grad()
        loss = criterion(new_net(x_train), y_train)
        loss.backward()
        optimizer.step()
        return new_net


    @torch.no_grad()
    def eval_module(net: nn.Module) -> np.float64:
        net.eval()
        _, predicted = torch.max(net(x_test), 1)
        net.train()
        return accuracy_score(y_test, predicted)


    # MyModel class
    class MyModel(NytoModule):
        def __init__(self, in_feat: int, h_size: int, out_feat: int):
            super().__init__()
            self.layer1 = nn.Linear(in_feat, h_size)
            self.layer2 = nn.Linear(h_size, out_feat)

        def forward(self, inpts):
            h_out = f.relu(self.layer1(inpts))
            return self.layer2(h_out)


    # init network
    net0: MyModel = MyModel(4, 12 ,3)
    net1: MyModel = train_module(net0, LR)

    # trainig
    for epoch in range(100):
        momentum: MyModel = BETA * (net1 - net0)
        net1, net0 = train_module(net1, LR) + momentum, net1

        if (epoch+1) % 10 == 0:
            accuracy = eval_module(net1)
            print(f"{epoch=} acc={accuracy:.2f}")







