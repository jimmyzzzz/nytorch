from collections import OrderedDict
from .utils import MyModule, MySubModule, UserData
from nytorch import NytoModule
from torch import nn
import unittest
import nytorch as nyto
import torch


class TestResetAttar(unittest.TestCase):
    def test_reset_param(self):
        param0: nn.Parameter = nn.Parameter(torch.randn(2, 2))
        param1: nn.Parameter = nn.Parameter(torch.randn(3, 3))
        buffer0: torch.Tensor = torch.randn(1)
        buffer1: torch.Tensor = torch.randn(1)
        lin: nn.Linear = nn.Linear(2, 3)
        data0: UserData = UserData()
        data1: UserData = UserData()
        
        sub_module: MySubModule = MySubModule(param0, lin, buffer0, data0)
        module: MyModule = MyModule(param1, sub_module, buffer1, data1)
        
        reset_param0: nn.Parameter = nn.Parameter(torch.randn(2, 2))
        reset_param1: nn.Parameter = nn.Parameter(torch.randn(3, 3))
        
        sub_module.param0 = reset_param0
        module.param1 = reset_param1
        
        self.assertIs(module._particle_kernel, module.sub_module._particle_kernel)
        self.assertNotEqual(module._module_id, module.sub_module._module_id)
        
        self.assertIs(module.param1, reset_param1)
        self.assertIs(module.sub_module, sub_module)
        self.assertIs(module.buffer1, buffer1)
        self.assertIs(module.data1, data1)
        
        self.assertEqual(module._parameters, OrderedDict([("param1", reset_param1)]))
        self.assertEqual(module._modules, OrderedDict([("sub_module", sub_module)]))
        self.assertEqual(module._buffers, OrderedDict([("buffer1", buffer1)]))
        
        self.assertIs(module.sub_module.param0, reset_param0)
        self.assertIs(module.sub_module.lin, lin)
        self.assertIs(module.sub_module.buffer0, buffer0)
        self.assertIs(module.sub_module.data0, data0)
        
        self.assertEqual(module.sub_module._parameters, OrderedDict([("param0", reset_param0)]))
        self.assertEqual(module.sub_module._modules, OrderedDict([("lin", lin)]))
        self.assertEqual(module.sub_module._buffers, OrderedDict([("buffer0", buffer0)]))
        
    def test_reset_buffer(self):
        param0: nn.Parameter = nn.Parameter(torch.randn(2, 2))
        param1: nn.Parameter = nn.Parameter(torch.randn(3, 3))
        buffer0: torch.Tensor = torch.randn(1)
        buffer1: torch.Tensor = torch.randn(1)
        lin: nn.Linear = nn.Linear(2, 3)
        data0: UserData = UserData()
        data1: UserData = UserData()
        
        sub_module: MySubModule = MySubModule(param0, lin, buffer0, data0)
        module: MyModule = MyModule(param1, sub_module, buffer1, data1)
        
        reset_buffer0: torch.Tensor = torch.randn(1)
        reset_buffer1: torch.Tensor = torch.randn(1)
        
        sub_module.buffer0 = reset_buffer0
        module.buffer1 = reset_buffer1
        
        self.assertIs(module._particle_kernel, module.sub_module._particle_kernel)
        self.assertNotEqual(module._module_id, module.sub_module._module_id)
        
        self.assertIs(module.param1, param1)
        self.assertIs(module.sub_module, sub_module)
        self.assertIs(module.buffer1, reset_buffer1)
        self.assertIs(module.data1, data1)

        self.assertEqual(module._parameters, OrderedDict([("param1", param1)]))
        self.assertEqual(module._modules, OrderedDict([("sub_module", sub_module)]))
        self.assertEqual(module._buffers, OrderedDict([("buffer1", reset_buffer1)]))
        
        self.assertIs(module.sub_module.param0, param0)
        self.assertIs(module.sub_module.lin, lin)
        self.assertIs(module.sub_module.buffer0, reset_buffer0)
        self.assertIs(module.sub_module.data0, data0)
        
        self.assertEqual(module.sub_module._parameters, OrderedDict([("param0", param0)]))
        self.assertEqual(module.sub_module._modules, OrderedDict([("lin", lin)]))
        self.assertEqual(module.sub_module._buffers, OrderedDict([("buffer0", reset_buffer0)]))
        
    def test_reset_module1(self):
        param0: nn.Parameter = nn.Parameter(torch.randn(2, 2))
        param1: nn.Parameter = nn.Parameter(torch.randn(3, 3))
        buffer0: torch.Tensor = torch.randn(1)
        buffer1: torch.Tensor = torch.randn(1)
        lin: nn.Linear = nn.Linear(2, 3)
        data0: UserData = UserData()
        data1: UserData = UserData()
        
        sub_module: MySubModule = MySubModule(param0, lin, buffer0, data0)
        module: MyModule = MyModule(param1, sub_module, buffer1, data1)
        
        reset_lin: nn.Linear = nn.Linear(2, 3)
        sub_module.lin = reset_lin
        
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
        self.assertIs(module.sub_module.lin, reset_lin)
        self.assertIs(module.sub_module.buffer0, buffer0)
        self.assertIs(module.sub_module.data0, data0)
        
        self.assertEqual(module.sub_module._parameters, OrderedDict([("param0", param0)]))
        self.assertEqual(module.sub_module._modules, OrderedDict([("lin", reset_lin)]))
        self.assertEqual(module.sub_module._buffers, OrderedDict([("buffer0", buffer0)]))
        
    def test_reset_module2(self):
        param0: nn.Parameter = nn.Parameter(torch.randn(2, 2))
        param1: nn.Parameter = nn.Parameter(torch.randn(3, 3))
        buffer0: torch.Tensor = torch.randn(1)
        buffer1: torch.Tensor = torch.randn(1)
        lin: nn.Linear = nn.Linear(2, 3)
        data0: UserData = UserData()
        data1: UserData = UserData()
        
        sub_module: MySubModule = MySubModule(param0, lin, buffer0, data0)
        module: MyModule = MyModule(param1, sub_module, buffer1, data1)
        
        reset_param0: nn.Parameter = nn.Parameter(torch.randn(2, 2))
        reset_lin: nn.Linear = nn.Linear(2, 3)
        reset_buffer0: torch.Tensor = torch.randn(1)
        reset_data0: UserData = UserData()
        reset_sub_module: MySubModule = MySubModule(reset_param0, reset_lin, reset_buffer0, reset_data0)
        module.sub_module = reset_sub_module
        
        self.assertIs(module._particle_kernel, module.sub_module._particle_kernel)
        self.assertNotEqual(module._module_id, module.sub_module._module_id)
        
        self.assertIs(module.param1, param1)
        self.assertIs(module.sub_module, reset_sub_module)
        self.assertIs(module.buffer1, buffer1)
        self.assertIs(module.data1, data1)
        
        self.assertEqual(module._parameters, OrderedDict([("param1", param1)]))
        self.assertEqual(module._modules, OrderedDict([("sub_module", reset_sub_module)]))
        self.assertEqual(module._buffers, OrderedDict([("buffer1", buffer1)]))
        
        self.assertIs(module.sub_module.param0, reset_param0)
        self.assertIs(module.sub_module.lin, reset_lin)
        self.assertIs(module.sub_module.buffer0, reset_buffer0)
        self.assertIs(module.sub_module.data0, reset_data0)
        
        self.assertEqual(module.sub_module._parameters, OrderedDict([("param0", reset_param0)]))
        self.assertEqual(module.sub_module._modules, OrderedDict([("lin", reset_lin)]))
        self.assertEqual(module.sub_module._buffers, OrderedDict([("buffer0", reset_buffer0)]))

