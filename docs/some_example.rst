Example Code
==================

In previous chapters, 
various features and techniques of Nytorch have been discussed. 
This chapter provides a practical example to demonstrate its application.

Purpose
-------------

This chapter serves two main purposes:

    1. **Introduction of ParticleModule**: Earlier tutorials excluded ParticleModule for brevity, which is insufficient in practical training scenarios. Therefore, this chapter introduces its integration.
    2. **Combining Evolutionary Algorithm and Gradient Descent**: Nytorch facilitates the integration of these methods. Here, both algorithms are employed in training: Gradient Descent optimizes parameters in most iterations, while periodically, every 5 iterations, the Evolutionary Algorithm adjusts a subset of the swarm to explore better solutions efficiently.

For the Evolutionary Algorithm phase, 
we adopt an approach similar to Accelerated Particle Swarm Optimization, 
updating models based on:

.. math::

	W_{i,t} = (1 - \alpha) W_{i,t-1} + \alpha W_{g,t-1}

where:

    * :math:`W_{i,t}` is particle i at time *t*.
    * :math:`W_{i,t-1}` is particle i at time *t-1*.
    * :math:`W_{g,t-1}` is the best-known particle in the swarm at time *t-1*.
    * :math:`\alpha` is a scalar.

To optimize distributed models across nodes with high communication costs, adjustments include:

	1. Reduce communication frequency
	2. Optimizing only a subset of the swarm per iteration.
	
Reducing communication frequency involves periodic use of Evolutionary Algorithm, 
such as every 5 iterations, 
while Gradient Descent optimizes in other iterations. 
Optimizing a subset of the swarm involves selecting random particles for optimization, enhancing efficiency.


Example Content
--------------------------

Let's begin by configuring the training parameters::

	from nytorch import NytoModule, ParticleModule
	from nytorch.particle_module import PMProduct
	from random import choices, random
	import torch
	import torch.nn as nn
	import torch.nn.functional as F
	import torch.optim as optim
	from torch.utils.data import Subset, DataLoader, random_split
	from torchvision import datasets, transforms


	BATCH_SIZE = 64
	TRAIN_BATCH_NUM = 256
	TEST_BATCH_NUM = 16
	POOL_SIZE = 12
	SWARM_SIZE = 6
	LR = 0.01
	ALPHA = 0.5
	SWARM_INTERVAL = 16
	PRINT_INTERVAL = 16
	DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
We use the MNIST dataset for demonstration, selecting only a subset for the example::

    full_train_dataset = datasets.MNIST('mnist', 
                                        train=True, 
                                        download=True, 
                                        transform=transforms.Compose([transforms.ToTensor(),
                                                                      transforms.Normalize((0.1307,), (0.3081,))]))
    full_test_dataset = datasets.MNIST('mnist', 
                                       train=False, 
                                       transform=transforms.Compose([transforms.ToTensor(),
                                                                     transforms.Normalize((0.1307,), (0.3081,))]))


    train_size = TRAIN_BATCH_NUM * BATCH_SIZE
    test_size = TEST_BATCH_NUM * BATCH_SIZE
    train_dataset, _ = random_split(full_train_dataset, [train_size, len(full_train_dataset)-train_size])
    test_dataset, _ = random_split(full_test_dataset, [test_size, len(full_test_dataset)-test_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

Next, we define the model::

    class ConvNet(NytoModule):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 10, 5)
            self.conv2 = nn.Conv2d(10, 20, 3)
            self.fc = nn.Linear(20*10*10, 10)

        def forward(self, x):
            in_size = x.size(0)
            out = self.conv1(x)
            out = F.relu(out)
            out = F.max_pool2d(out, 2, 2)
            out = self.conv2(out)
            out = F.relu(out)
            out = out.view(in_size,-1)
            out = self.fc(out)
            out = F.log_softmax(out,dim=1)
            return out
		
		
    class ConvModel:
        @classmethod
        def from_product(cls, product, device):
            assert isinstance(product, PMProduct)
            return cls(product.module(), device)

        def __init__(self, particle, device):
            assert isinstance(particle, ParticleModule)
            self.device = device
            self.particle = particle
            self.optimizer = optim.SGD(self.particle.parameters(), lr=LR)
            self.particle.to(self.device)

        def product(self):
            return self.particle.product()

        def train(self, data, target):
            data, target = data.to(self.device), target.to(self.device)
            self.particle.train()
            self.optimizer.zero_grad()
            loss = F.nll_loss(self.particle(data), target)
            loss.backward()
            self.optimizer.step()
            return loss.item()

        def test(self, data, target):
            data, target = data.to(self.device), target.to(self.device)
            self.particle.eval()
            with torch.no_grad():
                output = self.particle(data)
                loss = F.nll_loss(output, target, reduction='sum').item()
                pred = output.max(1, keepdim=True)[1]
                correct = pred.eq(target.view_as(pred)).sum().item()
                return loss, correct

We also create a decorator for ConvNet called ConvModel, 
which wraps the optimizer and training/testing methods. 
The ``product`` method returns a PMProduct instance for particle operations,
and the ``from_product`` method transforms the PMProduct instance back to ConvModel after particle operations.

Since we are using a swarm-based algorithm, 
we need some swarm operations during training,
which we wrap into functions:::

    def create_pool(size, device):
        assert size >= 2
        pool = [ParticleModule(ConvNet()) for _ in range(size)]
        p0 = pool[0]
        return [ConvModel(p0.clone_from(p), device) for p in pool[1:]] + [ConvModel(p0, device)]


    def test_model(model, test_loader):
        test_loss = 0
        total_correct = 0
        for data, target in test_loader:
            loss, correct = model.test(data, target)
            test_loss += loss
            total_correct += correct
            
        test_loss /= len(test_loader.dataset)
        test_acc = total_correct / len(test_loader.dataset)
        return test_loss, test_acc


    def swarm_algorithm(pool, swarm_size, loss_list, alpha):
        assert 0 < swarm_size <= len(pool) == len(loss_list)
        assert 1 > alpha > 0
        idx_list = choices(list(range(len(pool))), k=swarm_size)
        idx_loss_list = [(idx, loss_list[idx]) for idx in idx_list]
        idx_loss_list = sorted(idx_loss_list, key=lambda idx_loss: idx_loss[1])

        best_seed_idx, _ = idx_loss_list[0]
        for i, (idx, loss) in enumerate(idx_loss_list):
            if idx == best_seed_idx: continue
            seed0 = pool[best_seed_idx].product()
            seed1 = pool[idx].product()
            new_product = alpha*seed0 + (1-alpha)*seed1
            pool[idx] = ConvModel.from_product(new_product, pool[idx].device)


    def train_pool(pool, train_loader, test_loader, swarm_size, swarm_interval=4, alpha=0.5, print_interval=8):
        assert len(pool) >= swarm_size >= 2
        assert swarm_interval > 0
        assert 1 > alpha > 0
        assert print_interval > 0        
        
        for batch_idx, (data, target) in enumerate(train_loader):
            loss_list = [model.train(data, target) for model in pool]

            if (batch_idx+1)%swarm_interval == 0:
                swarm_algorithm(pool, swarm_size, loss_list, alpha)
                
            if batch_idx==0 or (batch_idx+1)%print_interval == 0:
                print(f"batch: {batch_idx:>3} Accuracy: ", end='')
                for idx, model in enumerate(pool):
                    _, acc = test_model(model, test_loader)
                    print(f"[{idx}]{acc:.2f}", end=' ')
                print()

We pay special attention to the techniques used in ``create_pool`` and ``swarm_algorithm``. 
In ``create_pool``, we use ParticleModule to wrap NytoModule to eliminate circular references and reduce memory pressure. 
In ``swarm_algorithm``, 
we use the ``product`` method to transform to PMProduct instances for particle operations, 
and then transform back to ParticleModule instances in a batch to reduce unnecessary conversions.

Finally, we start training::

    if __name__ == '__main__':
        pool = create_pool(POOL_SIZE, DEVICE)
        train_pool(pool, 
                   train_loader, 
                   test_loader, 
                   SWARM_SIZE,
                   SWARM_INTERVAL, 
                   ALPHA, 
                   PRINT_INTERVAL)
        
        print("End")
        print("Accuracy: ", end='')
        for idx, model in enumerate(pool):
            _, acc = test_model(model, test_loader)
            print(f"[{idx}]{acc:.2f}", end=' ')

Below is the output of the program::

	batch:   0 Accuracy: [0]0.11 [1]0.06 [2]0.09 [3]0.07 [4]0.08 [5]0.07 [6]0.11 [7]0.17 [8]0.15 [9]0.13 [10]0.13 [11]0.20 
	batch:  15 Accuracy: [0]0.37 [1]0.43 [2]0.51 [3]0.60 [4]0.63 [5]0.36 [6]0.22 [7]0.44 [8]0.62 [9]0.62 [10]0.58 [11]0.63 
	batch:  31 Accuracy: [0]0.47 [1]0.47 [2]0.61 [3]0.69 [4]0.73 [5]0.57 [6]0.55 [7]0.56 [8]0.70 [9]0.55 [10]0.69 [11]0.63 
	batch:  47 Accuracy: [0]0.70 [1]0.67 [2]0.72 [3]0.83 [4]0.84 [5]0.69 [6]0.77 [7]0.67 [8]0.81 [9]0.77 [10]0.80 [11]0.75 
	batch:  63 Accuracy: [0]0.79 [1]0.75 [2]0.80 [3]0.80 [4]0.84 [5]0.80 [6]0.78 [7]0.78 [8]0.81 [9]0.78 [10]0.73 [11]0.82 
	batch:  79 Accuracy: [0]0.81 [1]0.84 [2]0.84 [3]0.84 [4]0.87 [5]0.82 [6]0.84 [7]0.82 [8]0.84 [9]0.82 [10]0.87 [11]0.87 
	batch:  95 Accuracy: [0]0.85 [1]0.83 [2]0.85 [3]0.85 [4]0.88 [5]0.83 [6]0.84 [7]0.80 [8]0.85 [9]0.85 [10]0.86 [11]0.86 
	batch: 111 Accuracy: [0]0.87 [1]0.88 [2]0.87 [3]0.88 [4]0.89 [5]0.84 [6]0.87 [7]0.85 [8]0.82 [9]0.87 [10]0.89 [11]0.89 
	batch: 127 Accuracy: [0]0.87 [1]0.86 [2]0.86 [3]0.87 [4]0.87 [5]0.85 [6]0.87 [7]0.87 [8]0.86 [9]0.85 [10]0.87 [11]0.87 
	batch: 143 Accuracy: [0]0.87 [1]0.86 [2]0.85 [3]0.84 [4]0.87 [5]0.83 [6]0.87 [7]0.86 [8]0.87 [9]0.87 [10]0.87 [11]0.87 
	batch: 159 Accuracy: [0]0.86 [1]0.83 [2]0.82 [3]0.84 [4]0.88 [5]0.87 [6]0.87 [7]0.83 [8]0.87 [9]0.83 [10]0.88 [11]0.88 
	batch: 175 Accuracy: [0]0.89 [1]0.89 [2]0.89 [3]0.88 [4]0.90 [5]0.90 [6]0.90 [7]0.90 [8]0.90 [9]0.90 [10]0.90 [11]0.90 
	batch: 191 Accuracy: [0]0.89 [1]0.88 [2]0.89 [3]0.88 [4]0.89 [5]0.89 [6]0.89 [7]0.89 [8]0.89 [9]0.88 [10]0.89 [11]0.89 
	batch: 207 Accuracy: [0]0.90 [1]0.90 [2]0.90 [3]0.81 [4]0.90 [5]0.90 [6]0.90 [7]0.90 [8]0.90 [9]0.90 [10]0.90 [11]0.90 
	batch: 223 Accuracy: [0]0.90 [1]0.90 [2]0.90 [3]0.88 [4]0.90 [5]0.90 [6]0.90 [7]0.90 [8]0.90 [9]0.90 [10]0.90 [11]0.90 
	batch: 239 Accuracy: [0]0.90 [1]0.90 [2]0.90 [3]0.88 [4]0.90 [5]0.90 [6]0.90 [7]0.90 [8]0.89 [9]0.89 [10]0.90 [11]0.90 
	batch: 255 Accuracy: [0]0.90 [1]0.90 [2]0.90 [3]0.87 [4]0.90 [5]0.90 [6]0.90 [7]0.90 [8]0.90 [9]0.90 [10]0.90 [11]0.90 
	End
	Accuracy: [0]0.90 [1]0.90 [2]0.90 [3]0.87 [4]0.90 [5]0.90 [6]0.90 [7]0.90 [8]0.90 [9]0.90 [10]0.90 [11]0.90 

As training progresses, particle performance converges,
demonstrating the Evolutionary Algorithm's efficacy. Initially impactful, 
its influence diminishes as parameters converge.

By slowing Evolutionary Algorithm convergence, 
particles explore better solutions, 
though computational overhead increases.

Summary
--------

This chapter detailed Nytorch usage for model training, 
emphasizing Gradient Descent and Evolutionary Algorithm optimization. 
Techniques included encapsulating NytoModule with ParticleModule and using PMProduct for particle operations, 
fostering deeper Nytorch application insights.













