from collections import OrderedDict
from .utils import TestModule, TestSubModule, UserData
from nytorch import NytoModule
from nytorch.module import ParamProduct
from nytorch.mtype import ParamType, ParamConfig
from torch import nn
from typing import OrderedDict as OrderedDictType
import unittest
import nytorch as nyto
import torch


def params_equal_check(params1: OrderedDictType[str, ParamType],
                       params2: OrderedDictType[str, ParamType]) -> bool:
    for key, param1 in params1.items():
        param2 = params2[key]
        c1 = param1==param1
        c2 = torch.isnan(param1)&torch.isnan(param1)
        if torch.all(c1 | c2): continue
        return False
    return True


def create_new_root() -> NytoModule:
    param0: nn.Parameter = nn.Parameter(torch.randn(2, 2))
    param1: nn.Parameter = nn.Parameter(torch.randn(3, 3))
    buffer0: torch.Tensor = torch.randn(1)
    buffer1: torch.Tensor = torch.randn(1)
    lin: nn.Linear = nn.Linear(2, 3)
    data0: UserData = UserData()
    data1: UserData = UserData()

    sub_module: TestSubModule = TestSubModule(param0, lin, buffer0, data0)
    root: TestModule = TestModule(param1, sub_module, buffer1, data1)
    
    return root


class TestModuleNegOperate(unittest.TestCase):
    def test_module_neg_operator1(self):
        root: TestModule = create_new_root()
        target_root: TestModule = (-root.product()).module()
        test_root: TestModule = -root
        
        self.assertTrue(params_equal_check(OrderedDict(target_root.named_parameters()),
                                           OrderedDict(test_root.named_parameters())))
        
    def test_module_pos_operator2(self):
        root: TestModule = create_new_root()
        target_root: TestModule = (+root.product()).module()
        test_root: TestModule = +root
        
        self.assertTrue(params_equal_check(OrderedDict(target_root.named_parameters()),
                                           OrderedDict(test_root.named_parameters())))

class TestModulePowOperate(unittest.TestCase):
    def test_module_pow_operator1(self):
        root: TestModule = create_new_root()
        target_root: TestModule = (root.product() ** 1.5).module()
        test_root: TestModule = root ** 1.5
        
        self.assertTrue(params_equal_check(OrderedDict(target_root.named_parameters()),
                                           OrderedDict(test_root.named_parameters())))
        
    def test_module_pow_operator2(self):
        root: TestModule = create_new_root()
        target_root: TestModule = (1.5 ** root.product()).module()
        test_root: TestModule = 1.5 ** root
        
        self.assertTrue(params_equal_check(OrderedDict(target_root.named_parameters()),
                                           OrderedDict(test_root.named_parameters())))
        
    def test_module_pow_operator3(self):
        root1: TestModule = create_new_root()
        root2: TestModule = root1.randn()
        target_root: TestModule = (root1.product() ** root2.product()).module()
        test_root: TestModule = root1 ** root2
        
        self.assertTrue(params_equal_check(OrderedDict(target_root.named_parameters()),
                                           OrderedDict(test_root.named_parameters())))

    def test_module_pow_operator4(self):
        root1: TestModule = create_new_root()
        root2: TestModule = root1.randn()
        
        with self.assertRaises(Exception):
            root1 ** root2.product()
            
        with self.assertRaises(Exception):
            root1.product() ** root2
            
            
class TestModuleAddOperate(unittest.TestCase):
    def test_module_add_operator1(self):
        root: TestModule = create_new_root()
        target_root: TestModule = (root.product() + 1.5).module()
        test_root: TestModule = root + 1.5
        
        self.assertTrue(params_equal_check(OrderedDict(target_root.named_parameters()),
                                           OrderedDict(test_root.named_parameters())))
        
    def test_module_add_operator2(self):
        root: TestModule = create_new_root()
        target_root: TestModule = (1.5 + root.product()).module()
        test_root: TestModule = 1.5 + root
        
        self.assertTrue(params_equal_check(OrderedDict(target_root.named_parameters()),
                                           OrderedDict(test_root.named_parameters())))
        
    def test_module_add_operator3(self):
        root1: TestModule = create_new_root()
        root2: TestModule = root1.randn()
        target_root: TestModule = (root1.product() + root2.product()).module()
        test_root: TestModule = root1 + root2
        
        self.assertTrue(params_equal_check(OrderedDict(target_root.named_parameters()),
                                           OrderedDict(test_root.named_parameters())))

    def test_module_add_operator4(self):
        root1: TestModule = create_new_root()
        root2: TestModule = root1.randn()
        
        with self.assertRaises(Exception):
            root1 + root2.product()
            
        with self.assertRaises(Exception):
            root1.product() + root2
        
        
class TestModuleSubOperate(unittest.TestCase):
    def test_module_sub_operator1(self):
        root: TestModule = create_new_root()
        target_root: TestModule = (root.product() - 1.5).module()
        test_root: TestModule = root - 1.5
        
        self.assertTrue(params_equal_check(OrderedDict(target_root.named_parameters()),
                                           OrderedDict(test_root.named_parameters())))
        
    def test_module_sub_operator2(self):
        root: TestModule = create_new_root()
        target_root: TestModule = (1.5 - root.product()).module()
        test_root: TestModule = 1.5 - root
        
        self.assertTrue(params_equal_check(OrderedDict(target_root.named_parameters()),
                                           OrderedDict(test_root.named_parameters())))
        
    def test_module_sub_operator3(self):
        root1: TestModule = create_new_root()
        root2: TestModule = root1.randn()
        target_root: TestModule = (root1.product() - root2.product()).module()
        test_root: TestModule = root1 - root2
        
        self.assertTrue(params_equal_check(OrderedDict(target_root.named_parameters()),
                                           OrderedDict(test_root.named_parameters())))

    def test_module_sub_operator4(self):
        root1: TestModule = create_new_root()
        root2: TestModule = root1.randn()
        
        with self.assertRaises(Exception):
            root1 - root2.product()
            
        with self.assertRaises(Exception):
            root1.product() - root2
            

class TestModuleMulOperate(unittest.TestCase):
    def test_module_mul_operator1(self):
        root: TestModule = create_new_root()
        target_root: TestModule = (root.product() * 1.5).module()
        test_root: TestModule = root * 1.5
        
        self.assertTrue(params_equal_check(OrderedDict(target_root.named_parameters()),
                                           OrderedDict(test_root.named_parameters())))
        
    def test_module_mul_operator2(self):
        root: TestModule = create_new_root()
        target_root: TestModule = (1.5 * root.product()).module()
        test_root: TestModule = 1.5 * root
        
        self.assertTrue(params_equal_check(OrderedDict(target_root.named_parameters()),
                                           OrderedDict(test_root.named_parameters())))
        
    def test_module_mul_operator3(self):
        root1: TestModule = create_new_root()
        root2: TestModule = root1.randn()
        target_root: TestModule = (root1.product() * root2.product()).module()
        test_root: TestModule = root1 * root2
        
        self.assertTrue(params_equal_check(OrderedDict(target_root.named_parameters()),
                                           OrderedDict(test_root.named_parameters())))

    def test_module_mul_operator4(self):
        root1: TestModule = create_new_root()
        root2: TestModule = root1.randn()
        
        with self.assertRaises(Exception):
            root1 * root2.product()
            
        with self.assertRaises(Exception):
            root1.product() * root2
            
            
class TestModuleTruedivOperate(unittest.TestCase):
    def test_module_truediv_operator1(self):
        root: TestModule = create_new_root()
        target_root: TestModule = (root.product() / 1.5).module()
        test_root: TestModule = root / 1.5
        
        self.assertTrue(params_equal_check(OrderedDict(target_root.named_parameters()),
                                           OrderedDict(test_root.named_parameters())))
        
    def test_module_truediv_operator2(self):
        root: TestModule = create_new_root()
        target_root: TestModule = (1.5 / root.product()).module()
        test_root: TestModule = 1.5 / root
        
        self.assertTrue(params_equal_check(OrderedDict(target_root.named_parameters()),
                                           OrderedDict(test_root.named_parameters())))
        
    def test_module_truediv_operator3(self):
        root1: TestModule = create_new_root()
        root2: TestModule = root1.randn()
        target_root: TestModule = (root1.product() / root2.product()).module()
        test_root: TestModule = root1 / root2
        
        self.assertTrue(params_equal_check(OrderedDict(target_root.named_parameters()),
                                           OrderedDict(test_root.named_parameters())))

    def test_module_truediv_operator4(self):
        root1: TestModule = create_new_root()
        root2: TestModule = root1.randn()
        
        with self.assertRaises(Exception):
            root1 / root2.product()
            
        with self.assertRaises(Exception):
            root1.product() / root2
