from collections import OrderedDict
from .utils import TestModule, TestSubModule, UserData
from nytorch import NytoModule
from nytorch.module import ParamProduct
from nytorch.mtype import ModuleID, ParamConfig, ParamDict, ParamType
from torch import nn
import unittest
import nytorch as nyto
import torch


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


class TestConfig(unittest.TestCase):
    def test_config_setting(self):
        def set_config(pid: ModuleID, conf: ParamConfig):
            conf.pid = pid
        
        root: NytoModule = create_new_root()
        root.apply_param_config(set_config)
        root_product: ParamProduct = root.product()
        params_dict: ParamDict = root_product.pdata.params

        def test_unary_operator(param: ParamType, conf: ParamConfig) -> ParamType:
            self.assertIs(param, params_dict[conf.pid])
            return param
        
        def test_binary_operator(param1: ParamType, param2: ParamType, conf: ParamConfig) -> ParamType:
            self.assertIs(param1, params_dict[conf.pid])
            return param1
        
        root.product().unary_operator(test_unary_operator)
        root.product().binary_operator(root.clone().product(), test_binary_operator)
        
    def set_error_name(self):
        def set_config(pid: ModuleID, conf: ParamConfig):
            conf.pid = pid
        root: NytoModule = create_new_root()
        root.param1 = None
        root.sub_module.lin = None
        
        with self.assertRaises(Exception): 
            root.apply_param_config(set_config, "error_name")
            
        with self.assertRaises(Exception): 
            root.apply_param_config(set_config, "param1")
            
        with self.assertRaises(Exception):
            root.sub_module.apply_param_config(set_config, "lin")
    
