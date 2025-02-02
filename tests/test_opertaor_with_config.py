from collections import OrderedDict
from .utils import MyModule, MySubModule, UserData
from nytorch import NytoModule
from nytorch.module import ParamProduct
from nytorch.mtype import ParamType, ParamConfig
from torch import nn
import unittest
import nytorch as nyto
import torch


def create_new_root():
    class SubModule(NytoModule):
        def __init__(self, param: ParamType):
            super().__init__()
            self.param: ParamType = param
    
    class RootModule(NytoModule):
        def __init__(self, p0: ParamType, p1: ParamType, p2: ParamType):
            super().__init__()
            self.param: ParamType = p0
            self.sub_module1: SubModule = SubModule(p1)
            self.sub_module2: SubModule = SubModule(p2)

    param0: nn.Parameter = nn.Parameter(torch.Tensor([1.]))
    param1: nn.Parameter = nn.Parameter(torch.Tensor([2.]))
    param2: nn.Parameter = nn.Parameter(torch.Tensor([3.]))
    root: RootModule = RootModule(param0, param1, param2)

    return root, param0, param1, param2


class TestUnaryAddOperate(unittest.TestCase):
    def test_operational_true_clone_true(self):
        root, param0, param1, param2 = create_new_root()
        root.sub_module1.set_param_config(operational=True, clone=True)
        new_root: NytoModule = root + 1
        
        with torch.no_grad():
            self.assertIsNot(new_root.param, param0)
            self.assertTrue(torch.equal(new_root.param, param0+1))
            self.assertIsNot(new_root.sub_module1.param, param1)
            self.assertTrue(torch.equal(new_root.sub_module1.param, param1+1))
            self.assertIsNot(new_root.sub_module2.param, param2)
            self.assertTrue(torch.equal(new_root.sub_module2.param, param2+1))
        
    def test_operational_true_clone_false(self):
        root, param0, param1, param2 = create_new_root()
        root.sub_module1.set_param_config(operational=True, clone=False)
        new_root: NytoModule = root + 1
        
        with torch.no_grad():
            self.assertIsNot(new_root.param, param0)
            self.assertTrue(torch.equal(new_root.param, param0+1))
            self.assertIsNot(new_root.sub_module1.param, param1)
            self.assertTrue(torch.equal(new_root.sub_module1.param, param1+1))
            self.assertIsNot(new_root.sub_module2.param, param2)
            self.assertTrue(torch.equal(new_root.sub_module2.param, param2+1))
        
    
    def test_operational_false_clone_true(self):
        root, param0, param1, param2 = create_new_root()
        root.sub_module1.set_param_config(operational=False, clone=True)
        new_root: NytoModule = root + 1
        
        with torch.no_grad():
            self.assertIsNot(new_root.param, param0)
            self.assertTrue(torch.equal(new_root.param, param0+1))
            self.assertIsNot(new_root.sub_module1.param, param1)
            self.assertTrue(torch.equal(new_root.sub_module1.param, param1))
            self.assertIsNot(new_root.sub_module2.param, param2)
            self.assertTrue(torch.equal(new_root.sub_module2.param, param2+1))
        
    def test_operational_false_clone_false(self):
        root, param0, param1, param2 = create_new_root()
        root.sub_module1.set_param_config(operational=False, clone=False)
        new_root: NytoModule = root + 1
        
        with torch.no_grad():
            self.assertIsNot(new_root.param, param0)
            self.assertTrue(torch.equal(new_root.param, param0+1))
            self.assertIs(new_root.sub_module1.param, param1)
            self.assertIsNot(new_root.sub_module2.param, param2)
            self.assertTrue(torch.equal(new_root.sub_module2.param, param2+1))
            
    def test_operational_true_clone_true2(self):
        root, param0, param1, param2 = create_new_root()
        root.set_param_config(operational=True, clone=True, name="sub_module1")
        new_root: NytoModule = root + 1
        
        with torch.no_grad():
            self.assertIsNot(new_root.param, param0)
            self.assertTrue(torch.equal(new_root.param, param0+1))
            self.assertIsNot(new_root.sub_module1.param, param1)
            self.assertTrue(torch.equal(new_root.sub_module1.param, param1+1))
            self.assertIsNot(new_root.sub_module2.param, param2)
            self.assertTrue(torch.equal(new_root.sub_module2.param, param2+1))
        
    def test_operational_true_clone_false2(self):
        root, param0, param1, param2 = create_new_root()
        root.set_param_config(operational=True, clone=False, name="sub_module1")
        new_root: NytoModule = root + 1
        
        with torch.no_grad():
            self.assertIsNot(new_root.param, param0)
            self.assertTrue(torch.equal(new_root.param, param0+1))
            self.assertIsNot(new_root.sub_module1.param, param1)
            self.assertTrue(torch.equal(new_root.sub_module1.param, param1+1))
            self.assertIsNot(new_root.sub_module2.param, param2)
            self.assertTrue(torch.equal(new_root.sub_module2.param, param2+1))
        
    
    def test_operational_false_clone_true2(self):
        root, param0, param1, param2 = create_new_root()
        root.set_param_config(operational=False, clone=True, name="sub_module1")
        new_root: NytoModule = root + 1
        
        with torch.no_grad():
            self.assertIsNot(new_root.param, param0)
            self.assertTrue(torch.equal(new_root.param, param0+1))
            self.assertIsNot(new_root.sub_module1.param, param1)
            self.assertTrue(torch.equal(new_root.sub_module1.param, param1))
            self.assertIsNot(new_root.sub_module2.param, param2)
            self.assertTrue(torch.equal(new_root.sub_module2.param, param2+1))
        
    def test_operational_false_clone_false2(self):
        root, param0, param1, param2 = create_new_root()
        root.set_param_config(operational=False, clone=False, name="sub_module1")
        new_root: NytoModule = root + 1
        
        with torch.no_grad():
            self.assertIsNot(new_root.param, param0)
            self.assertTrue(torch.equal(new_root.param, param0+1))
            self.assertIs(new_root.sub_module1.param, param1)
            self.assertIsNot(new_root.sub_module2.param, param2)
            self.assertTrue(torch.equal(new_root.sub_module2.param, param2+1))
            
    def test_operational_true_clone_true3(self):
        root, param0, param1, param2 = create_new_root()
        root.sub_module1.set_param_config(operational=True, clone=True, name="param")
        new_root: NytoModule = root + 1
        
        with torch.no_grad():
            self.assertIsNot(new_root.param, param0)
            self.assertTrue(torch.equal(new_root.param, param0+1))
            self.assertIsNot(new_root.sub_module1.param, param1)
            self.assertTrue(torch.equal(new_root.sub_module1.param, param1+1))
            self.assertIsNot(new_root.sub_module2.param, param2)
            self.assertTrue(torch.equal(new_root.sub_module2.param, param2+1))
        
    def test_operational_true_clone_false3(self):
        root, param0, param1, param2 = create_new_root()
        root.sub_module1.set_param_config(operational=True, clone=False, name="param")
        new_root: NytoModule = root + 1
        
        with torch.no_grad():
            self.assertIsNot(new_root.param, param0)
            self.assertTrue(torch.equal(new_root.param, param0+1))
            self.assertIsNot(new_root.sub_module1.param, param1)
            self.assertTrue(torch.equal(new_root.sub_module1.param, param1+1))
            self.assertIsNot(new_root.sub_module2.param, param2)
            self.assertTrue(torch.equal(new_root.sub_module2.param, param2+1))
        
    
    def test_operational_false_clone_true3(self):
        root, param0, param1, param2 = create_new_root()
        root.sub_module1.set_param_config(operational=False, clone=True, name="param")
        new_root: NytoModule = root + 1
        
        with torch.no_grad():
            self.assertIsNot(new_root.param, param0)
            self.assertTrue(torch.equal(new_root.param, param0+1))
            self.assertIsNot(new_root.sub_module1.param, param1)
            self.assertTrue(torch.equal(new_root.sub_module1.param, param1))
            self.assertIsNot(new_root.sub_module2.param, param2)
            self.assertTrue(torch.equal(new_root.sub_module2.param, param2+1))
        
    def test_operational_false_clone_false3(self):
        root, param0, param1, param2 = create_new_root()
        root.sub_module1.set_param_config(operational=False, clone=False, name="param")
        new_root: NytoModule = root + 1
        
        with torch.no_grad():
            self.assertIsNot(new_root.param, param0)
            self.assertTrue(torch.equal(new_root.param, param0+1))
            self.assertIs(new_root.sub_module1.param, param1)
            self.assertIsNot(new_root.sub_module2.param, param2)
            self.assertTrue(torch.equal(new_root.sub_module2.param, param2+1))
        
        
class TestBinaryAddOperate(unittest.TestCase):
    def test_operational_true_clone_true(self):
        root, param0, param1, param2 = create_new_root()
        root_clone = root.clone()
        root.sub_module1.set_param_config(operational=True, clone=True)
        new_root: NytoModule = root + root_clone
        
        with torch.no_grad():
            self.assertIsNot(new_root.param, param0)
            self.assertTrue(torch.equal(new_root.param, param0+param0))
            self.assertIsNot(new_root.sub_module1.param, param1)
            self.assertTrue(torch.equal(new_root.sub_module1.param, param1+param1))
            self.assertIsNot(new_root.sub_module2.param, param2)
            self.assertTrue(torch.equal(new_root.sub_module2.param, param2+param2))
        
    def test_operational_true_clone_false(self):
        root, param0, param1, param2 = create_new_root()
        root_clone = root.clone()
        root.sub_module1.set_param_config(operational=True, clone=False)
        new_root: NytoModule = root + root_clone
        
        with torch.no_grad():
            self.assertIsNot(new_root.param, param0)
            self.assertTrue(torch.equal(new_root.param, param0+param0))
            self.assertIsNot(new_root.sub_module1.param, param1)
            self.assertTrue(torch.equal(new_root.sub_module1.param, param1+param1))
            self.assertIsNot(new_root.sub_module2.param, param2)
            self.assertTrue(torch.equal(new_root.sub_module2.param, param2+param2))
        
    
    def test_operational_false_clone_true(self):
        root, param0, param1, param2 = create_new_root()
        root_clone = root.clone()
        root.sub_module1.set_param_config(operational=False, clone=True)
        new_root: NytoModule = root + root_clone
        
        with torch.no_grad():
            self.assertIsNot(new_root.param, param0)
            self.assertTrue(torch.equal(new_root.param, param0+param0))
            self.assertIsNot(new_root.sub_module1.param, param1)
            self.assertTrue(torch.equal(new_root.sub_module1.param, param1))
            self.assertIsNot(new_root.sub_module2.param, param2)
            self.assertTrue(torch.equal(new_root.sub_module2.param, param2+param2))
        
    def test_operational_false_clone_false(self):
        root, param0, param1, param2 = create_new_root()
        root_clone = root.clone()
        root.sub_module1.set_param_config(operational=False, clone=False)
        new_root: NytoModule = root + root_clone
        
        with torch.no_grad():
            self.assertIsNot(new_root.param, param0)
            self.assertTrue(torch.equal(new_root.param, param0+param0))
            self.assertIs(new_root.sub_module1.param, param1)
            self.assertIsNot(new_root.sub_module2.param, param2)
            self.assertTrue(torch.equal(new_root.sub_module2.param, param2+param2))
            
    def test_operational_true_clone_true2(self):
        root, param0, param1, param2 = create_new_root()
        root_clone = root.clone()
        root.set_param_config(operational=True, clone=True, name="sub_module1")
        new_root: NytoModule = root + root_clone
        
        with torch.no_grad():
            self.assertIsNot(new_root.param, param0)
            self.assertTrue(torch.equal(new_root.param, param0+param0))
            self.assertIsNot(new_root.sub_module1.param, param1)
            self.assertTrue(torch.equal(new_root.sub_module1.param, param1+param1))
            self.assertIsNot(new_root.sub_module2.param, param2)
            self.assertTrue(torch.equal(new_root.sub_module2.param, param2+param2))
        
    def test_operational_true_clone_false2(self):
        root, param0, param1, param2 = create_new_root()
        root_clone = root.clone()
        root.set_param_config(operational=True, clone=False, name="sub_module1")
        new_root: NytoModule = root + root_clone
        
        with torch.no_grad():
            self.assertIsNot(new_root.param, param0)
            self.assertTrue(torch.equal(new_root.param, param0+param0))
            self.assertIsNot(new_root.sub_module1.param, param1)
            self.assertTrue(torch.equal(new_root.sub_module1.param, param1+param1))
            self.assertIsNot(new_root.sub_module2.param, param2)
            self.assertTrue(torch.equal(new_root.sub_module2.param, param2+param2))
        
    
    def test_operational_false_clone_true2(self):
        root, param0, param1, param2 = create_new_root()
        root_clone = root.clone()
        root.set_param_config(operational=False, clone=True, name="sub_module1")
        new_root: NytoModule = root + root_clone
        
        with torch.no_grad():
            self.assertIsNot(new_root.param, param0)
            self.assertTrue(torch.equal(new_root.param, param0+param0))
            self.assertIsNot(new_root.sub_module1.param, param1)
            self.assertTrue(torch.equal(new_root.sub_module1.param, param1))
            self.assertIsNot(new_root.sub_module2.param, param2)
            self.assertTrue(torch.equal(new_root.sub_module2.param, param2+param2))
        
    def test_operational_false_clone_false2(self):
        root, param0, param1, param2 = create_new_root()
        root_clone = root.clone()
        root.set_param_config(operational=False, clone=False, name="sub_module1")
        new_root: NytoModule = root + root_clone
        
        with torch.no_grad():
            self.assertIsNot(new_root.param, param0)
            self.assertTrue(torch.equal(new_root.param, param0+param0))
            self.assertIs(new_root.sub_module1.param, param1)
            self.assertIsNot(new_root.sub_module2.param, param2)
            self.assertTrue(torch.equal(new_root.sub_module2.param, param2+param2))
            
    def test_operational_true_clone_true3(self):
        root, param0, param1, param2 = create_new_root()
        root_clone = root.clone()
        root.sub_module1.set_param_config(operational=True, clone=True, name="param")
        new_root: NytoModule = root + root_clone
        
        with torch.no_grad():
            self.assertIsNot(new_root.param, param0)
            self.assertTrue(torch.equal(new_root.param, param0+param0))
            self.assertIsNot(new_root.sub_module1.param, param1)
            self.assertTrue(torch.equal(new_root.sub_module1.param, param1+param1))
            self.assertIsNot(new_root.sub_module2.param, param2)
            self.assertTrue(torch.equal(new_root.sub_module2.param, param2+param2))
        
    def test_operational_true_clone_false3(self):
        root, param0, param1, param2 = create_new_root()
        root_clone = root.clone()
        root.sub_module1.set_param_config(operational=True, clone=False, name="param")
        new_root: NytoModule = root + root_clone
        
        with torch.no_grad():
            self.assertIsNot(new_root.param, param0)
            self.assertTrue(torch.equal(new_root.param, param0+param0))
            self.assertIsNot(new_root.sub_module1.param, param1)
            self.assertTrue(torch.equal(new_root.sub_module1.param, param1+param1))
            self.assertIsNot(new_root.sub_module2.param, param2)
            self.assertTrue(torch.equal(new_root.sub_module2.param, param2+param2))
        
    
    def test_operational_false_clone_true3(self):
        root, param0, param1, param2 = create_new_root()
        root_clone = root.clone()
        root.sub_module1.set_param_config(operational=False, clone=True, name="param")
        new_root: NytoModule = root + root_clone
        
        with torch.no_grad():
            self.assertIsNot(new_root.param, param0)
            self.assertTrue(torch.equal(new_root.param, param0+param0))
            self.assertIsNot(new_root.sub_module1.param, param1)
            self.assertTrue(torch.equal(new_root.sub_module1.param, param1))
            self.assertIsNot(new_root.sub_module2.param, param2)
            self.assertTrue(torch.equal(new_root.sub_module2.param, param2+param2))
        
    def test_operational_false_clone_false3(self):
        root, param0, param1, param2 = create_new_root()
        root_clone = root.clone()
        root.sub_module1.set_param_config(operational=False, clone=False, name="param")
        new_root: NytoModule = root + root_clone
        
        with torch.no_grad():
            self.assertIsNot(new_root.param, param0)
            self.assertTrue(torch.equal(new_root.param, param0+param0))
            self.assertIs(new_root.sub_module1.param, param1)
            self.assertIsNot(new_root.sub_module2.param, param2)
            self.assertTrue(torch.equal(new_root.sub_module2.param, param2+param2))
