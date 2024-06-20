Customized operation
========================

The process of particle operation consists of four steps:

	1. Transformation from an instance of NytoModule to an instance of ParamProduct.
	2. Perform operations using the instance of ParamProduct to obtain a new instance of ParamProduct.
	3. Duplicate the instance of NytoModule to obtain a new instance of NytoModule.
	4. Copy the values of the new ParamProduct to the new NytoModule instance.

In steps 3 and 4, 
sometimes they are not necessary because during operations with intermediate variables, 
the already computed ParamProduct is often transformed back and forth into NytoModule, which is meaningless. 
Eliminating these redundant transformation steps can improve computational efficiency and save cache space.

For this purpose, we need to introduce a new tool: **ParamProduct**.

In this chapter, we will cover:

	1. How to use ParamProduct to improve efficiency.
	2. How to use ParamProduct to customize particle operation.
	

ParamProduct
--------------

Let's explore how NytoModule and ParamProduct are transformed into each other.

Here's an example model::
	
    import nytorch as nyto
    import torch

    class Linear(nyto.NytoModule):
        def __init__(self, w: float, b: float) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.Tensor([w]))
            self.bias = torch.nn.Parameter(torch.Tensor([b]))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.weight * x + self.bias
    
    net = Linear(2., 1.)

From NytoModule to ParamProduct::

	product: ParamProduct = net.product()

From ParamProduct to NytoModule::

    new_net: Linear = product.module()

Copying the result of ParamProduct to an existing NytoModule::
	
	net.product_(product)


ParamProduct operation
---------------------------------

In this section, we will explore how instances of ParamProduct perform operations to obtain new instances of ParamProduct. We first look at built-in methods, each corresponding to methods of NytoModule::

	product: ParamProduct = net.product()
	
	pos_product = +product

	neg_product = -product
	
	pow_product1 = product ** 2
	pow_product2 = product ** product

	add_product1 = product + 2
	add_product2 = product + product

	sub_product1 = product - 2
	sub_product2 = product - product

	mul_product1 = product * 2
	mul_product2 = product * product

	truediv_product1 = product / 2
	truediv_product2 = product / product

	randn_product = product.randn()
	
	clone_product = product.clone()

Using ParamProduct has the advantage of speeding up execution. Let's take a common particle operation and its equivalent process::

	net2 = 2 * net1 + 5
	# 1. product_a = net1.product()
	# 2. product_b = 2 * product_a
	# 3. net_b = product_b.module()
	# 4. product_c = net_b.product()
	# 5. product_d = product_c + 5
	# 6. net2 = product_d.module()
	
We notice that the intermediate variable ``net_b`` is entirely redundant. We can achieve the same result with the following code, eliminating unnecessary steps::

	product_a = net1.product()
	product_b = 2 * product_a + 5
	net2 = product_b.module()

This approach not only reduces steps but also eliminates a costly ``module()`` call.

Another benefit of using ParamProduct is the ability to customize particle operations. These operations are divided into two types: unary and binary operations.


Unary operation
-----------------------------

Let's first look at unary operations, which involve only one particle.

Here are some examples::

	# example 1
	product + 10

	# example 2
	2 * product

	# example 3
	product.randn()

By calling the ``unary_operator()`` method of the ParamProduct instance, we can customize unary operations. Here's the function signature for ``unary_operator()``::

    def unary_operator(
        self, 
        fn: Callable[[ParamType, ParamConfig], ParamType]
    ) -> ParamProduct[Tmodule]:

Users need to provide a function that receives a parameter (of ``type torch.nn.Parameter``) and its corresponding ParamConfig instance and returns a new parameter (of type  ``torch.nn.Parameter``)。

Here's an example. Let's create a unary operation that multiplies all parameters in the particle by 2 and adds 5::

    def my_unary_operation(param: 'ParamType', config: 'ParamConfig') -> 'ParamType':
        return torch.nn.Parameter(2*param + 5)
    
    net = Linear(2., 1.)
    product = net.product()

    new_product = product.unary_operator(my_unary_operation)
    new_net = new_product.module()

::

    >>> list(net.parameters())
    [Parameter containing:
     tensor([2.], requires_grad=True),
     Parameter containing:
     tensor([1.], requires_grad=True)]
 
    >>> list(new_net.parameters())
    [Parameter containing:
     tensor([9.], requires_grad=True),
     Parameter containing:
     tensor([7.], requires_grad=True)]

.. note::

	In writing the function fn,
	gradient calculation does not need to be disabled because ``torch.no_grad`` is used within the ``unary_operator()`` method to disable gradient calculation.
	
Given that some parameters may be marked as non-operational parameters, we should adjust our behavior during particle operation based on the information carried by the incoming ParamConfig instance::

    def my_unary_operation(param: 'ParamType', config: 'ParamConfig') -> 'ParamType':
        if config.operational:
            return torch.nn.Parameter(2*param + 5)
        elif config.clone:
            return torch.nn.Parameter(param.clone())
        return param
    
    net = Linear(2., 1.)
    product = net.product()
    
    # Some parameters are marked as non-operational parameters.
    net.set_param_config(operational=False, name='bias')

    new_product = product.unary_operator(my_unary_operation)
    new_net = new_product.module()

::

    >>> list(net.parameters())
    [Parameter containing:
     tensor([2.], requires_grad=True),
     Parameter containing:
     tensor([1.], requires_grad=True)]
 
    >>> list(new_net.parameters())
    [Parameter containing:
     tensor([9.], requires_grad=True),
     Parameter containing:
     tensor([1.], requires_grad=True)]

As we can see, by modifying the contents of the ParamConfig instance, we can label parameters to have more flexibility in customizing particle operations.

Here's another example. Let's create a unary operation that generates random parameters but only operates on parameters marked as ``is_weight=True``, while parameters marked as ``is_weight=False`` will be set to zero.

First, let's create a model::

    from torch import nn
    import nytorch as nyto
    import torch

    class CNN(nyto.NytoModule):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Sequential(nn.Conv2d(in_channels = 1, 
                                                 out_channels = 16, 
                                                 kernel_size = 5, 
                                                 stride = 1, 
                                                 padding = 2),
                                       nn.ReLU(),
                                       nn.MaxPool2d(kernel_size = 2))
            self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 1, 2),
                                       nn.ReLU(),
                                       nn.MaxPool2d(2))
            self.output_layer = nn.Linear(32*7*7, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.view(x.size(0), -1)
            output = self.output_layer(x)
            return output, x
    
    net = CNN()

Next, let's get the parameter IDs for weights::

    weight_set: set['ParamID'] = {
        net.get_param_id(sub_param)
        for name, sub_param in net.named_parameters()
        if name.split('.')[-1] == 'weight'}

Then, we add a new attribute ``is_weight`` to the corresponding ParamConfig instance::

    def making_weight_and_bias(pid: 'ParamID', config: 'ParamConfig') -> None:
        if pid in weight_set:
            config.is_weight = True
        else:
            config.is_weight = False
            
    net.apply_param_config(making_weight_and_bias)

Let's check the modifications::

    def print_config(pid: 'ParamID', config: 'ParamConfig') -> None:
        print(f"{pid=} {config=}")

::

    >>> net.apply_param_config(print_config)
    pid=0 config=ParamConfig(operational=True, clone=True, is_weight=True)
    pid=1 config=ParamConfig(operational=True, clone=True, is_weight=False)
    pid=2 config=ParamConfig(operational=True, clone=True, is_weight=True)
    pid=3 config=ParamConfig(operational=True, clone=True, is_weight=False)
    pid=4 config=ParamConfig(operational=True, clone=True, is_weight=True)
    pid=5 config=ParamConfig(operational=True, clone=True, is_weight=False)

Finally, let's run the custom unary operation::

    def randn_weight(param: 'ParamType', config: 'ParamConfig') -> 'ParamType':
        if config.is_weight:
            return nn.Parameter(torch.randn_like(param))
        return nn.Parameter(torch.zeros_like(param))

    new_net = net.product().unary_operator(randn_weight).module()

Let's see some of the results::

    >>> new_net.output_layer.weight
    Parameter containing:
    tensor([[ 0.9797, -0.2713,  0.6872,  ..., -0.1385, -0.2651,  2.5661],
            [-0.1535,  1.5814, -0.6361,  ..., -1.9001,  0.4541, -0.8917],
            [-1.8478,  0.7187,  2.2011,  ...,  0.2117, -0.1923, -1.6886],
            ...,
            [-0.5017, -1.1098, -0.4653,  ..., -2.0727,  0.9889,  0.7774],
            [-0.0027,  1.3248,  0.3038,  ..., -1.0170, -0.3165,  1.2529],
            [ 1.4229, -0.3351,  0.1424,  ..., -0.0538, -0.0118, -0.0574]],
           requires_grad=True)
    
    >>> new_net.output_layer.bias
    Parameter containing:
    tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], requires_grad=True)
    

Binary operation
---------------------

The usage of binary operations is similar to unary operations; it involves two particles participating in the operation.

Below are examples::

	# example 1
	product + product

	# example 2
	product - product

You can define a binary operation by calling the ``binary_operator()`` method of an instance of ParamProduct. Here's the function signature for ``binary_operator()``::

    def binary_operator(
        self,
        other: ParamProduct[Tmodule],
        fn: Callable[[ParamType, ParamType, ParamConfig], ParamType]
    ) -> ParamProduct[Tmodule]:

Users need to pass in a ParamProduct and a function. The function receives two parameters (of type ``torch.nn.Parameter``). The first parameter comes from itself, and the second parameter comes from the passed-in particle. Additionally, there's an instance of ParamConfig corresponding to the parameter. The function needs to return a new Parameter (of type ``torch.nn.Parameter``).

.. note::

	You don't need to disable gradient calculation when writing fn because ``torch.no_grad`` is already used in the ``binary_operator()`` method to disable gradient calculation.

Here's an example where we create a binary operation that subtracts the parameters from two particles.

First, the model::

    import nytorch as nyto
    import torch

    class Linear(nyto.NytoModule):
        def __init__(self, w: float, b: float) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.Tensor([w]))
            self.bias = torch.nn.Parameter(torch.Tensor([b]))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.weight * x + self.bias
    
    net1 = Linear(10, 5)
    net2 = net1.clone_from(Linear(3, 2))

Then, we execute the custom binary operation::

    def my_sub_operator(left: 'ParamType', 
                        right: 'ParamType', 
                        config: 'ParamConfig') -> 'ParamType':
        return torch.nn.Parameter(left - right)
    
    net3_product = net1.product().binary_operator(net2.product(), my_sub_operator)
    net3 = net3_product.module()

View parameters::

    >>> list(net3.named_parameters())
    [('weight',
      Parameter containing:
      tensor([7.], requires_grad=True)),
     ('bias',
      Parameter containing:
      tensor([3.], requires_grad=True))]







