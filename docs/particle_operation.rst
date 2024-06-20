Particle operation
==================

First, let's clarify what a particle is. The original definition of a particle is the collection of all parameters of a model. However, a more common and extended meaning refers to a Module instance and the ensemble of its sub-modules. Here, when we refer to a particle, we are talking about this extended meaning.

Operations based on particles are termed **particle operation**.
We have two types of particle operations:

	1. Operations between particles and numbers.
	2. Operations between particles.

nytorch.NytoModule inherits from torch.nn.Module. We can enable the functionality of particle operations by inheriting from NytoModule::

    import nytorch as nyto
    import torch

    class NytoLinear(nyto.NytoModule):
        def __init__(self, w: torch.Tensor, b: torch.Tensor) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(w)
            self.bias = torch.nn.Parameter(b)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.weight * x + self.bias


Operations Between Particles and Numbers
------------------------------------------

Operations between particles and numbers apply the operation to each parameter of the particle and return the result as a new particle. For example::

	net = NytoLinear(torch.Tensor([2.]), torch.Tensor([1.]))
	new_net = net + 10

::

	>>> net.weight
	Parameter containing:
	tensor([2.], requires_grad=True)

	>>> new_net.weight
	Parameter containing:
	tensor([12.], requires_grad=True)

	>>> net.bias
	Parameter containing:
	tensor([1.], requires_grad=True)
	
	>>> new_net.bias
	Parameter containing:
	tensor([11.], requires_grad=True)

The following operations are supported::

	pos_net = +net
	neg_net = -net
	pow_net = net ** 2
	add_net = net + 2
	sub_net = net - 2
	mul_net = net * 2
	truediv_net = net / 2
	rpow_net = 2 ** net
	radd_net = 2 + net
	rsub_net = 2 - net
	rmul_net = net * 2
	rtruediv_net = 2 / net

	# Inplace operators
	ipow_net **= 2
	iadd_net += 2
	isub_net -= 2
	imul_net *= 2
	itruediv_net /= 2
	

Operations Between Particles
-------------------------------

Operations between particles apply the operation to the corresponding parameters of the two particles and return the result as a new particle. For example::

	net = NytoLinear(torch.Tensor([2.]), torch.Tensor([1.]))
	new_net = net + net

::

	>>> net.weight
	Parameter containing:
	tensor([2.], requires_grad=True)

	>>> new_net.weight
	Parameter containing:
	tensor([4.], requires_grad=True)

	>>> net.bias
	Parameter containing:
	tensor([1.], requires_grad=True)
	
	>>> new_net.bias
	Parameter containing:
	tensor([2.], requires_grad=True)
	
The following operations are supported::

	pow_net = net ** net
	add_net = net + net
	sub_net = net - net
	mul_net = net * net
	truediv_net = net / net

inplace operators::

	ipow_net **= net
	iadd_net += net
	isub_net -= net
	imul_net *= net
	itruediv_net /= net

.. note::

	It's essential to note that not all combinations of particles can undergo particle operation.
	They must belong to the same **species**.
	Details about species are discussed in the next subsection.
	
	
Species
------------------------

Here, we introduce a new concept. If two particles are derived from the same particle through particle operation or one particle is derived from another particle through particle operation, we say the two particles belong to the same **species**, and we call the collection of particles belonging to the same species a **swarm**.

In other words, whenever we create a new particle through the constructor of NytoModule, we essentially create a new species, and the new particle belongs to this new species. Particle operations can only occur between particles belonging to the same species. If particle operations occur between particles from different species, it will lead to an error::

	net1 = NytoLinear(torch.Tensor([1.]), torch.Tensor([2.]))
	net2 = NytoLinear(torch.Tensor([3.]), torch.Tensor([4.]))
	net3 = net1 + 10

::

	>>> net1 + net2
	AssertionError

	>>> net1 + net3
	NytoLinear()
	
This is because particles from different species cannot guarantee the same structure or shape. While it's possible to check for structure or shape consistency during each particle operation, it's not a common scenario and incurs high computational costs. Hence, we prioritize the efficiency of particle operations, and this approach is not employed.
	
However, sometimes it might be necessary to perform particle operations between particles from different species. In such cases, one can first copy the parameters of a particle to another particle from a different species and then perform particle operations::

	net3.load_state_dict(net2.state_dict())

::

	>>> net1 + net3
	NytoLinear()


Clone and Detach
-----------------

Here, we introduce two related methods: ``clone()`` and ``detach()`` .

``clone()`` returns a new particle with cloned parameters from the original particle, and they belong to the same species::

	net = NytoLinear(torch.Tensor([1.]), torch.Tensor([2.]))
	net_clone = net1.clone()

::

	>>> net.weight is net_clone.weight
	False
	
	>>> net.bias is net_clone.bias
	False

	>>> torch.equal(net.weight, net_clone.weight)
	True
	
	>>> torch.equal(net.bias, net_clone.bias)
	True
	
	>>> net + net_clone
	NytoLinear()

``detach()`` returns a new particle with parameters referencing the original particle, and they belong to different species::

	net = NytoLinear(torch.Tensor([1.]), torch.Tensor([2.]))
	net_detach = net1.clone()

::

	>>> net.weight is net_detach.weight
	True
	
	>>> net.bias is net_detach.bias
	True
	
	>>> net + net_detach
	AssertionError
	

clone_from
-------------

If there's a need to clone particles from another species to the current species, one can use either of the following approaches::

	net1 = NytoLinear(torch.Tensor([1.]),
	                  torch.Tensor([2.]))
	net2 = NytoLinear(torch.Tensor([3.]),
	                  torch.Tensor([4.]))
	                  
	net3 = net1.clone()
	net3.load_state_dict(net2.state_dict())
	
or::

	net1 = NytoLinear(torch.Tensor([1.]),
	                  torch.Tensor([2.]))
	net2 = NytoLinear(torch.Tensor([3.]),
	                  torch.Tensor([4.]))
	
	net3 = net1.clone_from(net2)

Both approaches are equivalent logically.


Randn
----------

Sometimes, it's necessary to introduce randomness. In such cases, one can use the randn() method, which returns a particle with parameters drawn from a standard normal distribution, and they belong to the same species as the original particle::

	net = NytoLinear(torch.Tensor([1.]), torch.Tensor([2.]))
	net_randn = net.randn()

::

	>>> net_randn.weight
	Parameter containing:
	tensor([1.2006], requires_grad=True)
	
	>>> net_randn.bias
	Parameter containing:
	tensor([-1.6793], requires_grad=True)

	>>> net + net_randn
	NytoLinear()


Particle Operation on Submodules
----------------------------------------

.. note::

	We usually consider a module instance and its submodules as a particle,
	because they need to work together to complete a forward pass.
	In Nytorch, there are two definitions for the **root module**:
	
		1. A particle has only one root module.
		2. The module that can traverse all modules in the particle starting from itself.
	
	.. image:: ./image/root_module.png
		:width: 500

Usually, we perform particle operations on the **root module**.
But what happens if we perform particle operations on submodules?

Consider the following example::

    class Layer(nyto.NytoModule):
        def __init__(self, in_size, out_size):
            super().__init__()
            self.lin = nn.Linear(in_size, out_size)
    
        def forward(self, x):
            return self.lin(x)
    
    
    class ResLayer(nyto.NytoModule):
        def __init__(self, in_size, out_size):
            super().__init__()
            self.sub_moudle = Layer(in_size, out_size)
    
        def forward(self, x):
            return self.sub_moudle(x) + x
    
    root_module = ResLayer(12, 2)
    sub_moudle = root_module.sub_moudle

In this example, we have a root module and a submodule. If we perform particle operations on both, new particles corresponding to each module are generated::

    new_root_module = root_module + 10
    new_sub_moudle = sub_moudle + 10

::

    >>> new_root_module
    ResLayer(
      (sub_moudle): Layer(
        (lin): Linear(in_features=12, out_features=2, bias=True)
      )
    )

    >>> new_sub_moudle
    Layer(
      (lin): Linear(in_features=12, out_features=2, bias=True)
    )

    >>> new_root_module.sub_moudle is new_sub_moudle
    False

The resulting new submodule is independent of the new root module.









