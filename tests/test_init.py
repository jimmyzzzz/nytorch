from collections import OrderedDict
from .utils import MyModule, MySubModule, UserData
from nytorch import NytoModule
from torch import nn
import unittest
import nytorch as nyto
import torch

class TestInit(unittest.TestCase):
    def test_init(self):
        param0: nn.Parameter = nn.Parameter(torch.randn(2, 2))
        param1: nn.Parameter = nn.Parameter(torch.randn(3, 3))
        buffer0: torch.Tensor = torch.randn(1)
        buffer1: torch.Tensor = torch.randn(1)
        lin: nn.Linear = nn.Linear(2, 3)
        data0: UserData = UserData()
        data1: UserData = UserData()
        
        sub_module: MySubModule = MySubModule(param0, lin, buffer0, data0)
        module: MyModule = MyModule(param1, sub_module, buffer1, data1)
        
        self.assertIs(module._particle_kernel, module.sub_module._particle_kernel)
        self.assertNotEqual(module._module_id, module.sub_module._module_id)
        
        self.assertIs(module.param1, param1)
        self.assertIs(module.sub_module, sub_module)
        self.assertIs(module.buffer1, buffer1)
        self.assertIs(module.data1, data1)
        
        self.assertEqual(module._parameters, OrderedDict([("param1", param1)]))
        self.assertEqual(module._modules, OrderedDict([("sub_module", sub_module)]))
        self.assertEqual(module._buffers, OrderedDict([("buffer1", buffer1)]))
        
        self.assertIs(module.sub_module.param0, param0)
        self.assertIs(module.sub_module.lin, lin)
        self.assertIs(module.sub_module.buffer0, buffer0)
        self.assertIs(module.sub_module.data0, data0)
        
        self.assertEqual(module.sub_module._parameters, OrderedDict([("param0", param0)]))
        self.assertEqual(module.sub_module._modules, OrderedDict([("lin", lin)]))
        self.assertEqual(module.sub_module._buffers, OrderedDict([("buffer0", buffer0)]))


class TestInit2(unittest.TestCase):
    def test_init(self):
        class DummyModule(NytoModule):
            pass
        
        param0: nn.Parameter = nn.Parameter(torch.randn(2, 2))
        param1: nn.Parameter = nn.Parameter(torch.randn(3, 3))
        buffer0: torch.Tensor = torch.randn(1)
        buffer1: torch.Tensor = torch.randn(1)
        lin: nn.Linear = nn.Linear(2, 3)
        data0: UserData = UserData()
        data1: UserData = UserData()
        
        sub_module: DummyModule = DummyModule()
        sub_module.register_parameter("param0", param0)
        sub_module.add_module("lin", lin)
        sub_module.register_buffer("buffer0", buffer0)
        sub_module.data0 = data0
        
        module: DummyModule = DummyModule()
        module.register_parameter("param1", param1)
        module.add_module("sub_module", sub_module)
        module.register_buffer("buffer1", buffer1)
        module.data1 = data1
        
        sub_module: MySubModule = MySubModule(param0, lin, buffer0, data0)
        module: MyModule = MyModule(param1, sub_module, buffer1, data1)
        
        self.assertIs(module._particle_kernel, module.sub_module._particle_kernel)
        self.assertNotEqual(module._module_id, module.sub_module._module_id)
        
        self.assertIs(module.param1, param1)
        self.assertIs(module.sub_module, sub_module)
        self.assertIs(module.buffer1, buffer1)
        self.assertIs(module.data1, data1)
        
        self.assertEqual(module._parameters, OrderedDict([("param1", param1)]))
        self.assertEqual(module._modules, OrderedDict([("sub_module", sub_module)]))
        self.assertEqual(module._buffers, OrderedDict([("buffer1", buffer1)]))
        
        self.assertIs(module.sub_module.param0, param0)
        self.assertIs(module.sub_module.lin, lin)
        self.assertIs(module.sub_module.buffer0, buffer0)
        self.assertIs(module.sub_module.data0, data0)
        
        self.assertEqual(module.sub_module._parameters, OrderedDict([("param0", param0)]))
        self.assertEqual(module.sub_module._modules, OrderedDict([("lin", lin)]))
        self.assertEqual(module.sub_module._buffers, OrderedDict([("buffer0", buffer0)]))
