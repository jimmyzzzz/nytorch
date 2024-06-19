from collections import OrderedDict
from .utils import TestModule, TestSubModule, UserData
from nytorch import NytoModule
from torch import nn
import unittest
import nytorch as nyto
import torch


class TestConvertParam(unittest.TestCase):
    def test_param_to_module(self):
        param0: nn.Parameter = nn.Parameter(torch.randn(2, 2))
        buffer0: torch.Tensor = torch.randn(1)
        lin: nn.Linear = nn.Linear(2, 3)
        data0: UserData = UserData()
        module: TestSubModule = TestSubModule(param0, lin, buffer0, data0)

        vkernal: nyto.kernal.VersionKernal = module._version_kernal
        self.assertTrue(vkernal.is_newest)
        
        with self.assertRaises(Exception): 
            module.param0 = nn.Linear(2, 3)
            
        self.assertIs(module._version_kernal, vkernal)
            
    def test_param_to_buffer(self):
        param0: nn.Parameter = nn.Parameter(torch.randn(2, 2))
        buffer0: torch.Tensor = torch.randn(1)
        lin: nn.Linear = nn.Linear(2, 3)
        data0: UserData = UserData()
        module: TestSubModule = TestSubModule(param0, lin, buffer0, data0)
        
        vkernal: nyto.kernal.VersionKernal = module._version_kernal
        self.assertTrue(vkernal.is_newest)
        
        with self.assertRaises(Exception): 
            module.register_buffer("param0", torch.randn(1))
            
        self.assertIs(module._version_kernal, vkernal)
        
    def test_param_to_value(self):
        param0: nn.Parameter = nn.Parameter(torch.randn(2, 2))
        buffer0: torch.Tensor = torch.randn(1)
        lin: nn.Linear = nn.Linear(2, 3)
        data0: UserData = UserData()
        module: TestSubModule = TestSubModule(param0, lin, buffer0, data0)

        vkernal: nyto.kernal.VersionKernal = module._version_kernal
        self.assertTrue(vkernal.is_newest)
        
        with self.assertRaises(Exception):
            module.param0 = UserData()
            
        self.assertIs(module._version_kernal, vkernal)
        
    def test_param_to_none(self):
        param0: nn.Parameter = nn.Parameter(torch.randn(2, 2))
        buffer0: torch.Tensor = torch.randn(1)
        lin: nn.Linear = nn.Linear(2, 3)
        data0: UserData = UserData()
        module: TestSubModule = TestSubModule(param0, lin, buffer0, data0)

        module.param0 = None
        
        self.assertIs(module.param0, None)
        self.assertIs(module.lin, lin)
        self.assertIs(module.buffer0, buffer0)
        self.assertIs(module.data0, data0)
        
        self.assertEqual(module._parameters, OrderedDict([("param0", None)]))
        self.assertEqual(module._modules, OrderedDict([("lin", lin)]))
        self.assertEqual(module._buffers, OrderedDict([("buffer0", buffer0)]))

        
class TestConvertModule(unittest.TestCase):
    def test_module_to_param(self):
        param0: nn.Parameter = nn.Parameter(torch.randn(2, 2))
        buffer0: torch.Tensor = torch.randn(1)
        lin: nn.Linear = nn.Linear(2, 3)
        data0: UserData = UserData()
        module: TestSubModule = TestSubModule(param0, lin, buffer0, data0)
        
        other_param: nn.Parameter = nn.Parameter(torch.randn(2, 2))
        module.lin = other_param
        
        self.assertIs(module.param0, param0)
        self.assertIs(module.lin, other_param)
        self.assertIs(module.buffer0, buffer0)
        self.assertIs(module.data0, data0)
        
        self.assertEqual(module._parameters, OrderedDict([("param0", param0), ("lin", other_param)]))
        self.assertEqual(module._modules, OrderedDict())
        self.assertEqual(module._buffers, OrderedDict([("buffer0", buffer0)]))
            
    def test_module_to_buffer(self):
        param0: nn.Parameter = nn.Parameter(torch.randn(2, 2))
        buffer0: torch.Tensor = torch.randn(1)
        lin: nn.Linear = nn.Linear(2, 3)
        data0: UserData = UserData()
        module: TestSubModule = TestSubModule(param0, lin, buffer0, data0)
        
        vkernal: nyto.kernal.VersionKernal = module._version_kernal
        self.assertTrue(vkernal.is_newest)
        
        with self.assertRaises(Exception): 
            module.register_buffer("lin", torch.randn(1))
            
        self.assertIs(module._version_kernal, vkernal)
        
    def test_module_to_value(self):
        param0: nn.Parameter = nn.Parameter(torch.randn(2, 2))
        buffer0: torch.Tensor = torch.randn(1)
        lin: nn.Linear = nn.Linear(2, 3)
        data0: UserData = UserData()
        module: TestSubModule = TestSubModule(param0, lin, buffer0, data0)
        
        vkernal: nyto.kernal.VersionKernal = module._version_kernal
        self.assertTrue(vkernal.is_newest)
        
        with self.assertRaises(Exception): 
            module.lin = UserData()
            
        self.assertIs(module._version_kernal, vkernal)
        
    def test_module_to_none(self):
        param0: nn.Parameter = nn.Parameter(torch.randn(2, 2))
        buffer0: torch.Tensor = torch.randn(1)
        lin: nn.Linear = nn.Linear(2, 3)
        data0: UserData = UserData()
        module: TestSubModule = TestSubModule(param0, lin, buffer0, data0)

        module.lin = None
        
        self.assertIs(module.param0, param0)
        self.assertIs(module.lin, None)
        self.assertIs(module.buffer0, buffer0)
        self.assertIs(module.data0, data0)
        
        self.assertEqual(module._parameters, OrderedDict([("param0", param0)]))
        self.assertEqual(module._modules, OrderedDict([("lin", None)]))
        self.assertEqual(module._buffers, OrderedDict([("buffer0", buffer0)]))
    
    
class TestConvertBuffer(unittest.TestCase):
    def test_buffer_to_module(self):
        param0: nn.Parameter = nn.Parameter(torch.randn(2, 2))
        buffer0: torch.Tensor = torch.randn(1)
        lin: nn.Linear = nn.Linear(2, 3)
        data0: UserData = UserData()
        module: TestSubModule = TestSubModule(param0, lin, buffer0, data0)
        
        other_lin: nn.Linear = nn.Linear(2, 3)
        module.buffer0 = other_lin
        
        self.assertIs(module.param0, param0)
        self.assertIs(module.lin, lin)
        self.assertIs(module.buffer0, other_lin)
        self.assertIs(module.data0, data0)
        
        self.assertEqual(module._parameters, OrderedDict([("param0", param0)]))
        self.assertEqual(module._modules, OrderedDict([("lin", lin), ("buffer0", other_lin)]))
        self.assertEqual(module._buffers, OrderedDict())
    
    def test_buffer_to_param(self):
        param0: nn.Parameter = nn.Parameter(torch.randn(2, 2))
        buffer0: torch.Tensor = torch.randn(1)
        lin: nn.Linear = nn.Linear(2, 3)
        data0: UserData = UserData()
        module: TestSubModule = TestSubModule(param0, lin, buffer0, data0)
        
        other_param: nn.Parameter = nn.Parameter(torch.randn(2, 2))
        module.buffer0 = other_param
        
        self.assertIs(module.param0, param0)
        self.assertIs(module.lin, lin)
        self.assertIs(module.buffer0, other_param)
        self.assertIs(module.data0, data0)
        
        self.assertEqual(module._parameters, OrderedDict([("param0", param0), ("buffer0", other_param)]))
        self.assertEqual(module._modules, OrderedDict([("lin", lin)]))
        self.assertEqual(module._buffers, OrderedDict())
        
    def test_buffer_to_value(self):
        param0: nn.Parameter = nn.Parameter(torch.randn(2, 2))
        buffer0: torch.Tensor = torch.randn(1)
        lin: nn.Linear = nn.Linear(2, 3)
        data0: UserData = UserData()
        module: TestSubModule = TestSubModule(param0, lin, buffer0, data0)
        
        vkernal: nyto.kernal.VersionKernal = module._version_kernal
        self.assertTrue(vkernal.is_newest)
        
        with self.assertRaises(Exception): 
            module.buffer0 = UserData()
            
        self.assertIs(module._version_kernal, vkernal)
        
    def test_buffer_to_none(self):
        param0: nn.Parameter = nn.Parameter(torch.randn(2, 2))
        buffer0: torch.Tensor = torch.randn(1)
        lin: nn.Linear = nn.Linear(2, 3)
        data0: UserData = UserData()
        module: TestSubModule = TestSubModule(param0, lin, buffer0, data0)

        module.buffer0 = None
        
        self.assertIs(module.param0, param0)
        self.assertIs(module.lin, lin)
        self.assertIs(module.buffer0, None)
        self.assertIs(module.data0, data0)
        
        self.assertEqual(module._parameters, OrderedDict([("param0", param0)]))
        self.assertEqual(module._modules, OrderedDict([("lin", lin)]))
        self.assertEqual(module._buffers, OrderedDict([("buffer0", None)]))
        

class TestConvertValue(unittest.TestCase):
    def test_value_to_module(self):
        param0: nn.Parameter = nn.Parameter(torch.randn(2, 2))
        buffer0: torch.Tensor = torch.randn(1)
        lin: nn.Linear = nn.Linear(2, 3)
        data0: UserData = UserData()
        module: TestSubModule = TestSubModule(param0, lin, buffer0, data0)
        
        other_lin: nn.Linear = nn.Linear(2, 3)
        module.data0 = other_lin
        
        self.assertIs(module.param0, param0)
        self.assertIs(module.lin, lin)
        self.assertIs(module.buffer0, buffer0)
        self.assertIs(module.data0, other_lin)
        
        self.assertEqual(module._parameters, OrderedDict([("param0", param0)]))
        self.assertEqual(module._modules, OrderedDict([("lin", lin), ("data0", other_lin)]))
        self.assertEqual(module._buffers, OrderedDict([("buffer0", buffer0)]))
    
    def test_value_to_param(self):
        param0: nn.Parameter = nn.Parameter(torch.randn(2, 2))
        buffer0: torch.Tensor = torch.randn(1)
        lin: nn.Linear = nn.Linear(2, 3)
        data0: UserData = UserData()
        module: TestSubModule = TestSubModule(param0, lin, buffer0, data0)
        
        other_param: nn.Parameter = nn.Parameter(torch.randn(2, 2))
        module.data0 = other_param
        
        self.assertIs(module.param0, param0)
        self.assertIs(module.lin, lin)
        self.assertIs(module.buffer0, buffer0)
        self.assertIs(module.data0, other_param)
        
        self.assertEqual(module._parameters, OrderedDict([("param0", param0), ("data0", other_param)]))
        self.assertEqual(module._modules, OrderedDict([("lin", lin)]))
        self.assertEqual(module._buffers, OrderedDict([("buffer0", buffer0)]))
        
    def test_value_to_buffer(self):
        param0: nn.Parameter = nn.Parameter(torch.randn(2, 2))
        buffer0: torch.Tensor = torch.randn(1)
        lin: nn.Linear = nn.Linear(2, 3)
        data0: UserData = UserData()
        module: TestSubModule = TestSubModule(param0, lin, buffer0, data0)
        
        vkernal: nyto.kernal.VersionKernal = module._version_kernal
        self.assertTrue(vkernal.is_newest)
        
        with self.assertRaises(Exception):
            module.register_buffer("data0", torch.randn(1))
            
        self.assertIs(module._version_kernal, vkernal)
