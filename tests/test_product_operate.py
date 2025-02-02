from collections import OrderedDict
from .utils import MyModule, MySubModule, UserData
from nytorch import NytoModule
from nytorch.module import ParamProduct
from nytorch.mtype import ParamType, ParamConfig
from torch import nn
from typing import OrderedDict as OrderedDictType
import unittest
import nytorch as nyto
import torch


class TestProductOperateCheck(unittest.TestCase):
    def test_product_operate_check1(self):
        class RootModule(NytoModule):
            def __init__(self):
                super().__init__()
                self.param: nn.Parameter = nn.Parameter(torch.randn(1))
        
        product1: ParamProduct = RootModule().product()
        product2: ParamProduct = RootModule().product()
        
        with self.assertRaises(Exception): 
            product1 + product2
        
        with self.assertRaises(Exception): 
            product1 - product2
            
        with self.assertRaises(Exception): 
            product1 * product2
            
        with self.assertRaises(Exception): 
            product1 / product2
            
        with self.assertRaises(Exception): 
            product1 ** product2
    
    def test_product_operate_check2(self):
        class RootModule(NytoModule):
            def __init__(self):
                super().__init__()
                self.param: nn.Parameter = nn.Parameter(torch.randn(1))
        
        root: RootModule = RootModule()
        product1: ParamProduct = root.product()
        product2: ParamProduct = root.detach().product()
        
        with self.assertRaises(Exception): 
            product1 + product2
        
        with self.assertRaises(Exception): 
            product1 - product2
            
        with self.assertRaises(Exception): 
            product1 * product2
            
        with self.assertRaises(Exception): 
            product1 / product2
            
        with self.assertRaises(Exception): 
            product1 ** product2
            
    def test_product_operate_check3(self):
        class RootModule(NytoModule):
            def __init__(self):
                super().__init__()
                self.param: nn.Parameter = nn.Parameter(torch.randn(1))
        
        root: RootModule = RootModule()
        product: ParamProduct = root.product()
        
        with self.assertRaises(Exception): 
            product + root
        
        with self.assertRaises(Exception): 
            product - root
            
        with self.assertRaises(Exception): 
            product * root
            
        with self.assertRaises(Exception): 
            product / root
            
        with self.assertRaises(Exception): 
            product ** root
        

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

    sub_module: MySubModule = MySubModule(param0, lin, buffer0, data0)
    root: MyModule = MyModule(param1, sub_module, buffer1, data1)
    
    return root


class TestProductOperateMethod(unittest.TestCase):
    def test_product_unary_operator(self):
        root: MyModule = create_new_root()
        add_one_root: MyModule = (root.product()
                                        .unary_operator(lambda param, conf: param+1)
                                        .module())
        
        with torch.no_grad():
            add_one_dict = OrderedDict(
                (name, param+1) for name, param in root.named_parameters())
        
        self.assertTrue(params_equal_check(OrderedDict(add_one_root.named_parameters()),
                                           add_one_dict))

    def test_product_binary_operator(self):
        root1: MyModule = create_new_root()
        root2: MyModule = root1.clone()
        add_root: MyModule = (root1.product() 
                                     .binary_operator(root2.product(), 
                                                      lambda param1, param2, conf: param1+param2)
                                     .module())
        
        with torch.no_grad():
            root1_params: OrderedDict[str, ParamType] = OrderedDict(root1.named_parameters())
            root2_params: OrderedDict[str, ParamType] = OrderedDict(root2.named_parameters())
            
            add_dict = OrderedDict()
            for key in root1_params:
                add_dict[key] = root1_params[key] + root2_params[key]

        self.assertTrue(params_equal_check(OrderedDict(add_root.named_parameters()),
                                           add_dict))

        
class TestProductNegOperate(unittest.TestCase):
    def test_product_pos_operator1(self):
        root: MyModule = create_new_root()
        new_root: MyModule = (+root.product()).module()
        
        with torch.no_grad():
            manual_dict = OrderedDict((name, param) for name, param in root.named_parameters())

        self.assertTrue(params_equal_check(OrderedDict(new_root.named_parameters()),
                                           manual_dict))
        
    def test_product_neg_operator1(self):
        root: MyModule = create_new_root()
        new_root: MyModule = (-root.product()).module()
        
        with torch.no_grad():
            manual_dict = OrderedDict((name, -param) for name, param in root.named_parameters())
        
        self.assertTrue(params_equal_check(OrderedDict(new_root.named_parameters()),
                                           manual_dict))

        
class TestProductPowOperate(unittest.TestCase):
    def test_product_pow_operator1(self):
        root: MyModule = create_new_root()
        new_root: MyModule = (root.product() ** 1.2).module()
        
        with torch.no_grad():
            manual_dict = OrderedDict((name, param**1.2) for name, param in root.named_parameters())
        
        self.assertTrue(params_equal_check(OrderedDict(new_root.named_parameters()),
                                           manual_dict))

    def test_product_pow_operator2(self):
        root: MyModule = create_new_root()
        new_root: MyModule = (1.2 ** root.product()).module()
        
        with torch.no_grad():
            manual_dict = OrderedDict((name, 1.2**param) for name, param in root.named_parameters())
        
        self.assertTrue(params_equal_check(OrderedDict(new_root.named_parameters()),
                                           manual_dict))
    
    def test_product_pow_operator3(self):
        root1: MyModule = create_new_root()
        root2: MyModule = root1.randn()
        new_root: MyModule = (root1.product() ** root2.product()).module()
        
        with torch.no_grad():
            manual_dict = OrderedDict()
            root1_params: OrderedDict[str, ParamType] = OrderedDict(root1.named_parameters())
            root2_params: OrderedDict[str, ParamType] = OrderedDict(root2.named_parameters())
            for key in root1_params:
                manual_dict[key] = root1_params[key] ** root2_params[key]
        
        self.assertTrue(params_equal_check(OrderedDict(new_root.named_parameters()),
                                           manual_dict))
        
        
class TestProductAddOperate(unittest.TestCase):
    def test_product_add_operator1(self):
        root: MyModule = create_new_root()
        new_root: MyModule = (root.product() + 1).module()
        
        with torch.no_grad():
            manual_dict = OrderedDict((name, param+1) for name, param in root.named_parameters())
        
        self.assertTrue(params_equal_check(OrderedDict(new_root.named_parameters()),
                                           manual_dict))

    def test_product_add_operator2(self):
        root: MyModule = create_new_root()
        new_root: MyModule = (1 + root.product()).module()
        
        with torch.no_grad():
            manual_dict = OrderedDict((name, 1+param) for name, param in root.named_parameters())
        
        self.assertTrue(params_equal_check(OrderedDict(new_root.named_parameters()),
                                           manual_dict))
    
    def test_product_add_operator3(self):
        root1: MyModule = create_new_root()
        root2: MyModule = root1.randn()
        new_root: MyModule = (root1.product() + root2.product()).module()
        
        with torch.no_grad():
            manual_dict = OrderedDict()
            root1_params: OrderedDict[str, ParamType] = OrderedDict(root1.named_parameters())
            root2_params: OrderedDict[str, ParamType] = OrderedDict(root2.named_parameters())
            for key in root1_params:
                manual_dict[key] = root1_params[key] + root2_params[key]
        
        self.assertTrue(params_equal_check(OrderedDict(new_root.named_parameters()),
                                           manual_dict))


class TestProductSubOperate(unittest.TestCase):
    def test_product_sub_operator1(self):
        root: MyModule = create_new_root()
        new_root: MyModule = (root.product() - 1).module()
        
        with torch.no_grad():
            manual_dict = OrderedDict((name, param-1) for name, param in root.named_parameters())
        
        self.assertTrue(params_equal_check(OrderedDict(new_root.named_parameters()),
                                           manual_dict))

    def test_product_sub_operator2(self):
        root: MyModule = create_new_root()
        new_root: MyModule = (1 - root.product()).module()
        
        with torch.no_grad():
            manual_dict = OrderedDict((name, 1-param) for name, param in root.named_parameters())
        
        self.assertTrue(params_equal_check(OrderedDict(new_root.named_parameters()),
                                           manual_dict))
    
    def test_product_sub_operator3(self):
        root1: MyModule = create_new_root()
        root2: MyModule = root1.randn()
        new_root: MyModule = (root1.product() - root2.product()).module()
        
        with torch.no_grad():
            manual_dict = OrderedDict()
            root1_params: OrderedDict[str, ParamType] = OrderedDict(root1.named_parameters())
            root2_params: OrderedDict[str, ParamType] = OrderedDict(root2.named_parameters())
            for key in root1_params:
                manual_dict[key] = root1_params[key] - root2_params[key]
        
        self.assertTrue(params_equal_check(OrderedDict(new_root.named_parameters()),
                                           manual_dict))
        
        
class TestProductMulOperate(unittest.TestCase):
    def test_product_mul_operator1(self):
        root: MyModule = create_new_root()
        new_root: MyModule = (root.product() * 1.5).module()
        
        with torch.no_grad():
            manual_dict = OrderedDict((name, param*1.5) for name, param in root.named_parameters())
        
        self.assertTrue(params_equal_check(OrderedDict(new_root.named_parameters()),
                                           manual_dict))

    def test_product_mul_operator2(self):
        root: MyModule = create_new_root()
        new_root: MyModule = (1.5 * root.product()).module()
        
        with torch.no_grad():
            manual_dict = OrderedDict((name, 1.5*param) for name, param in root.named_parameters())
        
        self.assertTrue(params_equal_check(OrderedDict(new_root.named_parameters()),
                                           manual_dict))
    
    def test_product_mul_operator3(self):
        root1: MyModule = create_new_root()
        root2: MyModule = root1.randn()
        new_root: MyModule = (root1.product() * root2.product()).module()
        
        with torch.no_grad():
            manual_dict = OrderedDict()
            root1_params: OrderedDict[str, ParamType] = OrderedDict(root1.named_parameters())
            root2_params: OrderedDict[str, ParamType] = OrderedDict(root2.named_parameters())
            for key in root1_params:
                manual_dict[key] = root1_params[key] * root2_params[key]
        
        self.assertTrue(params_equal_check(OrderedDict(new_root.named_parameters()),
                                           manual_dict))
        
        
class TestProductTruedivOperate(unittest.TestCase):
    def test_product_truediv_operator1(self):
        root: MyModule = create_new_root()
        new_root: MyModule = (root.product() / 1.5).module()
        
        with torch.no_grad():
            manual_dict = OrderedDict((name, param/1.5) for name, param in root.named_parameters())
        
        self.assertTrue(params_equal_check(OrderedDict(new_root.named_parameters()),
                                           manual_dict))

    def test_product_truediv_operator2(self):
        root: MyModule = create_new_root()
        new_root: MyModule = (1.5 / root.product()).module()
        
        with torch.no_grad():
            manual_dict = OrderedDict((name, 1.5/param) for name, param in root.named_parameters())
        
        self.assertTrue(params_equal_check(OrderedDict(new_root.named_parameters()),
                                           manual_dict))
    
    def test_product_truediv_operator3(self):
        root1: MyModule = create_new_root()
        root2: MyModule = root1.randn()
        new_root: MyModule = (root1.product() / root2.product()).module()
        
        with torch.no_grad():
            manual_dict = OrderedDict()
            root1_params: OrderedDict[str, ParamType] = OrderedDict(root1.named_parameters())
            root2_params: OrderedDict[str, ParamType] = OrderedDict(root2.named_parameters())
            for key in root1_params:
                manual_dict[key] = root1_params[key] / root2_params[key]
        
        self.assertTrue(params_equal_check(OrderedDict(new_root.named_parameters()),
                                           manual_dict))
