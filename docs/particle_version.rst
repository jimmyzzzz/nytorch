Version Manager
===============================

In Nytorch, two main functionalities are implemented:

	1. Particle operation
	2. Version manager
	
We have already discussed particle operation in previous sections. Now, let's delve into the version manager. So, what does the version manager do?

Consider the following example::

    import nytorch as nyto
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class LSTMTagger(nyto.NytoModule):
        def __init__(self, word_embeddings, embedding_dim, hidden_dim, tagset_size):
            super().__init__()
            self.word_embeddings = word_embeddings
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
            
            self.set_param_config(operational=False,
                                  clone=False,
                                  name='word_embeddings')

        def forward(self, sentence):
            embeds = self.word_embeddings(sentence)
            lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
            tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
            tag_scores = F.softmax(tag_space, dim=1)
            return tag_scores
   
In this example, we create a model based on LSTM. Given a sentence, it first obtains embeddings, then passes through an LSTM and a classifier to return the class of each word. Here, ``word_embeddings`` is pretrained elsewhere and needs to be provided during model creation::

    word_embeddings = nn.Embedding(32, 6).requires_grad_(False)
    net1 = LSTMTagger(word_embeddings, 6, 6, 3)
    net2 = net1.randn()
    net3 = net1.randn()

At this point, ``net1`` and ``net2`` should have the same shape of parameters with the same parameter IDs and attributes.

Now, suppose we need to adjust the model structure, such as replacing the existing ``word_embeddings`` with a new one that supports 64 words instead of 32::

	new_word_embeddings = nn.Embedding(64, 6).requires_grad_(False)
	net1.word_embeddings = new_word_embeddings

Considering that such adjustments should be applied to all particles of the species, we would have to manually update each particle::
	
	net2.word_embeddings = new_word_embeddings
	net3.word_embeddings = new_word_embeddings

However, this approach is inefficient. Instead, Nytorch provides a series of automatic mechanisms to handle such situations, which is the topic of this chapter.


Version
---------

Before delving into the details, let's define what a version is. We consider a version as the metadata state of a particle's module, buffer, and parameter at a certain point in time. When the metadata is modified, the state before modification is regarded as the old version, and the state after modification is regarded as the latest version.

What constitutes a modification to the metadata? From the perspective of a particle, we consider the following six events as modifications:

	1. Adding a module
	2. Deleting a module
	3. Adding a buffer
	4. Deleting a buffer
	5. Adding a parameter
	6. Deleting a parameter

Let's look at a specific example.
We can access the version instance pointed to by a particle using ``_version_kernel``::

    from torch import nn
    import nytorch as nyto
    import torch

    class Linear(nyto.NytoModule):
        def __init__(self, w: float, b: float) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.Tensor([w]))
            self.bias = nn.Parameter(torch.Tensor([b]))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.weight * x + self.bias

    net1 = Linear(1, 2)
    net2 = net1.randn()
    
    version_before_del = net1._version_kernel
   
When the metadata is modified,
the system records relevant information and saves it to a version instance.
Simultaneously, a new version instance is created as the latest version::

	del net1.weight

	version_after_del = net1._version_kernel
	
::

	>>> version_before_del is not version_after_del
	True
	
At this point, we observe that the metadata of ``net1`` has changed, and the version instance it points to has also changed. However, if we inspect ``net2``, we find that both its metadata and the version instance it points to remain unchanged::

	>>> hasattr(net2, 'weight')
	True

	>>> version_before_del is net2._version_kernel
	True

To upgrade ``net2`` to the latest version, we can call the ``touch()`` method. Nytorch automatically updates the particle to the latest version based on the previously recorded modification information::

	net2.touch()
	
::

	>>> hasattr(net2, 'weight')
	False

	>>> version_after_del is net2._version_kernel
	True

In practical use, frequent use of ``touch()`` is unnecessary because whenever a particle operation or metadata modification occurs, ``touch()`` is automatically invoked to ensure the particle is at the latest version.


Update Behavior
--------------------

Next, we'll discuss the differences in update behavior for adding and deleting modules, buffers, and parameters between the particle initiating the event and other particles of the same species.

Here, we define two new terms: **event initiator** and **event recipient**.

The event initiator is the particle where the modification event occurs, while the event recipient is another particle of the same species as the event initiator.
Consider the following example,
where ``net1`` is the event initiator and ``net2`` is the event recipient::

    net1 = Linear(10, 5)
    net2 = net1.randn()
    
    del net1.weight
    
    net2.touch()
    
For certain modification events, such as adding a parameter, the update behavior differs between the event initiator and the event recipient. For example, when adding a parameter, the event initiator adds the parameter itself, while the event recipient adds a clone of the paramete::

    net1 = Linear(1, 2)
    net2 = net1.randn()

    add_parameter = nn.Parameter(torch.randn(1))
    net1.add_parameter = add_parameter
    net2.touch()

::

    >>> net1.add_parameter is add_parameter
    True
    
    >>> net2.add_parameter is add_parameter
    False

    >>> torch.equal(net2.add_parameter, add_parameter)
    True

Similarly, for adding modules, the event initiator adds the module itself, while the event recipient adds a clone of the module::

    add_linear = nn.Linear(3, 4)
    net1.add_linear = add_linear
    net2.touch()
	
::

    >>> net1.add_linear is add_linear
    True
    
    >>> net2.add_linear is add_linear
    False
	
    >>> torch.equal(net2.add_linear.weight, add_linear.weight)
    True
	
    >>> torch.equal(net2.add_linear.bias, add_linear.bias)
    True

However, for adding buffers, both the event initiator and event receiver add a reference to the same buffer::

	add_tensor = torch.randn(3, 3)
	net1.register_buffer("add_tensor", add_tensor)
	net2.touch()
	
::

	>>> net1.add_tensor is add_tensor
	True
	
	>>> net2.add_tensor is add_tensor
	True

As for deleting modules, buffers, and parameters, there's no difference in behavior between event initiators and event receivers.


Avoiding Update Behavior
-------------------------

In nytorch, any version update to a NytoModule instance is mandatory. 
This means that in nytorch, 
you cannot avoid a version update after making any of the following six events as modifications:

    1. Adding a module
    2. Deleting a module
    3. Adding a buffer
    4. Deleting a buffer
    5. Adding a parameter
    6. Deleting a parameter

However, we may want to avoid triggering a version update in certain situations. 
Below is a specific example::

    import torch
    import torch.nn as nn
    import nytorch as nyto

    class MyModule(nyto.NytoModule):
        def __init__(self, rate=0.5):
            super().__init__()
            self.layer = nn.Linear(4, 1)
            self.register_buffer("mean", torch.zeros(1))
            self.rate = rate

        def forward(self, x):
            y = self.layer(x)
            self.update_mean(y)
            return y

        def update_mean(self, y):
            with torch.no_grad():
                self.mean = (1-self.rate)*self.mean + self.rate*y.mean()
    
    model = MyModule()
    model_clone = model.clone()

We need to record the output mean in the buffer each time. 
Our expectation is that different particles of the same species have their own independent mean records. 
However, you will find that when one of the particles updates the mean, 
all particles of the same species are updated::

    >>> model.mean
    tensor([0.])
    >>> model_clone.mean
    tensor([0.])
    >>> y = model(torch.randn(10, 4))
    >>> model.mean
    tensor([0.2386])
    >>> model_clone.touch().mean
    tensor([0.2386])

For this, we have two different solutions.

Solution 1: Move the buffer to a ``torch.nn.Module``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first solution is to move the buffer that needs to be modified to a ``torch.nn.Module``. 
This way, when the buffer is modified, 
it will not trigger a version update::
    
    import torch
    import torch.nn as nn
    import nytorch as nyto

    class MeanRecord(nn.Module):
        def __init__(self, rate=0.5):
            super().__init__()
            self.register_buffer("mean", torch.zeros(1))
            self.rate = rate

        def update_mean(self, output):
            with torch.no_grad():
                self.mean = (1-self.rate)*self.mean + self.rate*output.mean()

    class MyModule(nyto.NytoModule):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(4, 1)
            self.mean_record = MeanRecord()

        def forward(self, x):
            y = self.layer(x)
            self.mean_record.update_mean(y)
            return y

In the example above, 
we define a ``MeanRecord`` class to record the mean of the ``MyModule`` and store the mean in the buffer. 
This way, different particles of the same species have their own independent mean records::

    >>> model.mean_record.mean
    tensor([0.])
    >>> model_clone.mean_record.mean
    tensor([0.])
    >>> y = model(torch.randn(10, 4))
    >>> model.mean_record.mean
    tensor([0.2386])
    >>> model_clone.touch().mean_record.mean
    tensor([0.])

Solution 2: Use ``super().register_buffer`` to modify the buffer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The second solution is to use ``super().register_buffer`` to modify the buffer, 
which can also avoid triggering a version update::

    import torch
    import torch.nn as nn
    import nytorch as nyto

    class MyModule(nyto.NytoModule):
        def __init__(self, rate=0.5):
            super().__init__()
            self.layer = nn.Linear(4, 1)
            self.register_buffer("mean", torch.zeros(1))
            self.rate = rate

        def forward(self, x):
            y = self.layer(x)
            self.update_mean(y)
            return y

        def update_mean(self, y):
            with torch.no_grad():
                new_mean = (1-self.rate)*self.mean + self.rate*y.mean()
                super().register_buffer("mean", new_mean)

.. warning::

    Although users can also avoid version updates by modifying parameters and modules using the above methods, 
    it is not recommended to do so, 
    as it carries the risk of causing incorrect behavior.

































