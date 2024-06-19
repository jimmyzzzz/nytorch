from collections import OrderedDict
from .utils import TestModule, TestSubModule, UserData
from nytorch import NytoModule
from nytorch.module import ParamProduct
from torch import nn
import unittest
import nytorch as nyto
import torch


class TestProductConvert(unittest.TestCase):
    def test_product_convert1(self):
        param0: nn.Parameter = nn.Parameter(torch.randn(2, 2))
        param1: nn.Parameter = nn.Parameter(torch.randn(3, 3))
        buffer0: torch.Tensor = torch.randn(1)
        buffer1: torch.Tensor = torch.randn(1)
        lin: nn.Linear = nn.Linear(2, 3)
        data0: UserData = UserData()
        data1: UserData = UserData()
        
        sub_module: TestSubModule = TestSubModule(param0, lin, buffer0, data0)
        root: TestModule = TestModule(param1, sub_module, buffer1, data1)
        
        root_product: ParamProduct = root.product()
        param_set: set[nn.Parameter] = {param0, param1, lin.weight, lin.bias}
        self.assertEqual(param_set, set(root_product.params.values()))
        
        root_copy: RootModule = root_product.module()
        
        self.assertIsNot(root, root_copy)
        self.assertTrue(torch.equal(root.param1, root_copy.param1))
        self.assertIs(root.buffer1, root_copy.buffer1)
        self.assertIs(root.data1, root_copy.data1)
        
        self.assertIsNot(root.sub_module, root_copy.sub_module)
        self.assertTrue(torch.equal(root.sub_module.param0, root_copy.sub_module.param0))
        self.assertIs(root.sub_module.buffer0, root_copy.sub_module.buffer0)
        self.assertIs(root.sub_module.data0, root_copy.sub_module.data0)
        
        self.assertIsNot(root.sub_module.lin, root_copy.sub_module.lin)
        self.assertTrue(torch.equal(root.sub_module.lin.weight, root_copy.sub_module.lin.weight))
        self.assertTrue(torch.equal(root.sub_module.lin.bias, root_copy.sub_module.lin.bias))
        
    def test_product_convert2(self):
        param0: nn.Parameter = nn.Parameter(torch.randn(2, 2))
        param1: nn.Parameter = nn.Parameter(torch.randn(3, 3))
        buffer0: torch.Tensor = torch.randn(1)
        buffer1: torch.Tensor = torch.randn(1)
        lin: nn.Linear = nn.Linear(2, 3)
        data0: UserData = UserData()
        data1: UserData = UserData()
        
        sub_module: TestSubModule = TestSubModule(param0, lin, buffer0, data0)
        root: TestModule = TestModule(param1, sub_module, buffer1, data1)
        
        sub_module_product: ParamProduct = sub_module.product()
        param_set: set[nn.Parameter] = {param0, param1, lin.weight, lin.bias}
        self.assertEqual(param_set, set(sub_module_product.params.values()))
        
        sub_module_copy: RootModule = sub_module_product.module()
        root_copy: RootModule = sub_module_copy._particle_kernal.data.modules[nyto.mtype.ROOT_MODULE_ID]
        
        self.assertIsNot(root, root_copy)
        self.assertTrue(torch.equal(root.param1, root_copy.param1))
        self.assertIs(root.buffer1, root_copy.buffer1)
        self.assertIs(root.data1, root_copy.data1)
        
        self.assertIsNot(root.sub_module, root_copy.sub_module)
        self.assertTrue(torch.equal(root.sub_module.param0, root_copy.sub_module.param0))
        self.assertIs(root.sub_module.buffer0, root_copy.sub_module.buffer0)
        self.assertIs(root.sub_module.data0, root_copy.sub_module.data0)
        
        self.assertIsNot(root.sub_module.lin, root_copy.sub_module.lin)
        self.assertTrue(torch.equal(root.sub_module.lin.weight, root_copy.sub_module.lin.weight))
        self.assertTrue(torch.equal(root.sub_module.lin.bias, root_copy.sub_module.lin.bias))
