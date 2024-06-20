from collections import OrderedDict
from nytorch import NytoModule
from torch import nn
from typing import Optional
from .utils import TestModule, TestSubModule, UserData
from unittest.mock import patch

import numpy as np
import nytorch as nyto
import random
import torch
import unittest
import warnings


class TestClone(unittest.TestCase):
    def test_clone_root(self):
        param0: nn.Parameter = nn.Parameter(torch.randn(2, 2))
        param1: nn.Parameter = nn.Parameter(torch.randn(3, 3))
        buffer0: torch.Tensor = torch.randn(1)
        buffer1: torch.Tensor = torch.randn(1)
        lin: nn.Linear = nn.Linear(2, 3)
        data0: UserData = UserData()
        data1: UserData = UserData()
        
        sub_module: TestSubModule = TestSubModule(param0, lin, buffer0, data0)
        root: TestModule = TestModule(param1, sub_module, buffer1, data1)
        
        root_clone: TestModule = root.clone()
        self.assertIsNot(root_clone, root)
        self.assertIsNot(root_clone.param1, root.param1)
        self.assertTrue(torch.equal(root_clone.param1, root.param1))
        self.assertIs(root_clone.buffer1, root.buffer1)
        self.assertIs(root_clone.data1, root.data1)
        
        sub_module_clone: TestSubModule = root_clone.sub_module
        self.assertIsNot(sub_module_clone, sub_module)
        self.assertIsNot(sub_module_clone.param0, sub_module.param0)
        self.assertTrue(torch.equal(sub_module_clone.param0, sub_module.param0))
        self.assertIs(sub_module_clone.buffer0, sub_module.buffer0)
        self.assertIs(sub_module_clone.data0, sub_module.data0)
        
        self.assertIs(root_clone._particle_kernal, sub_module_clone._particle_kernal)
        self.assertIsNot(root_clone._particle_kernal, root._particle_kernal)
        self.assertIs(root_clone._version_kernal, root._version_kernal)
        self.assertEqual(root_clone._module_id, root._module_id)
        self.assertEqual(sub_module_clone._module_id, sub_module._module_id)
        
    def test_clone_sub_module(self):
        param0: nn.Parameter = nn.Parameter(torch.randn(2, 2))
        param1: nn.Parameter = nn.Parameter(torch.randn(3, 3))
        buffer0: torch.Tensor = torch.randn(1)
        buffer1: torch.Tensor = torch.randn(1)
        lin: nn.Linear = nn.Linear(2, 3)
        data0: UserData = UserData()
        data1: UserData = UserData()
        
        sub_module: TestSubModule = TestSubModule(param0, lin, buffer0, data0)
        root: TestModule = TestModule(param1, sub_module, buffer1, data1)
        
        sub_module_clone: TestSubModule = sub_module.clone()
        self.assertIsNot(sub_module_clone, sub_module)
        self.assertIsNot(sub_module_clone.param0, sub_module.param0)
        self.assertTrue(torch.equal(sub_module_clone.param0, sub_module.param0))
        self.assertIs(sub_module_clone.buffer0, sub_module.buffer0)
        self.assertIs(sub_module_clone.data0, sub_module.data0)
        
        root_clone: TestModule = sub_module_clone._particle_kernal.data.modules[nyto.mtype.ROOT_MODULE_ID]
        self.assertIsNot(root_clone, root)
        self.assertIsNot(root_clone.param1, root.param1)
        self.assertTrue(torch.equal(root_clone.param1, root.param1))
        self.assertIs(root_clone.buffer1, root.buffer1)
        self.assertIs(root_clone.data1, root.data1)
        
        self.assertIs(root_clone._particle_kernal, sub_module_clone._particle_kernal)
        self.assertIsNot(root_clone._particle_kernal, root._particle_kernal)
        self.assertIs(root_clone._version_kernal, root._version_kernal)
        self.assertEqual(root_clone._module_id, root._module_id)
        self.assertEqual(sub_module_clone._module_id, sub_module._module_id)

        
class TestCloneFrom(unittest.TestCase):
    def test_clone_from(self):
        class MySubModule(NytoModule):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.randn(2, 2))
        
        class MyRoot(NytoModule):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.randn(2, 2))
                self.submodule = MySubModule()
                
        root1 = MyRoot()
        root2 = MyRoot()
        
        root3 = root1.clone_from(root2)
        self.assertIs(root1._version_kernal, root3._version_kernal)
        self.assertTrue(torch.equal(root3.param, root2.param))
        self.assertTrue(torch.equal(root3.submodule.param, root2.submodule.param))


class TestRand(unittest.TestCase):
    def test_rand_root(self):
        param0: nn.Parameter = nn.Parameter(torch.rand(2, 2))
        param1: nn.Parameter = nn.Parameter(torch.rand(3, 3))
        buffer0: torch.Tensor = torch.rand(1)
        buffer1: torch.Tensor = torch.rand(1)
        lin: nn.Linear = nn.Linear(2, 3)
        data0: UserData = UserData()
        data1: UserData = UserData()
        
        sub_module: TestSubModule = TestSubModule(param0, lin, buffer0, data0)
        root: TestModule = TestModule(param1, sub_module, buffer1, data1)
        
        root_rand: TestModule = root.rand()
        self.assertIsNot(root_rand, root)
        self.assertIsNot(root_rand.param1, root.param1)
        self.assertTrue(root_rand.param1.shape, root.param1.shape)
        self.assertIs(root_rand.buffer1, root.buffer1)
        self.assertIs(root_rand.data1, root.data1)
        
        sub_module_rand: TestSubModule = root_rand.sub_module
        self.assertIsNot(sub_module_rand, sub_module)
        self.assertIsNot(sub_module_rand.param0, sub_module.param0)
        self.assertEqual(sub_module_rand.param0.shape, sub_module.param0.shape)
        self.assertIs(sub_module_rand.buffer0, sub_module.buffer0)
        self.assertIs(sub_module_rand.data0, sub_module.data0)
        
        self.assertIs(root_rand._particle_kernal, sub_module_rand._particle_kernal)
        self.assertIsNot(root_rand._particle_kernal, root._particle_kernal)
        self.assertIs(root_rand._version_kernal, root._version_kernal)
        self.assertEqual(root_rand._module_id, root._module_id)
        self.assertEqual(sub_module_rand._module_id, sub_module._module_id)   
    
    def test_rand_sub_module(self):
        param0: nn.Parameter = nn.Parameter(torch.rand(2, 2))
        param1: nn.Parameter = nn.Parameter(torch.rand(3, 3))
        buffer0: torch.Tensor = torch.rand(1)
        buffer1: torch.Tensor = torch.rand(1)
        lin: nn.Linear = nn.Linear(2, 3)
        data0: UserData = UserData()
        data1: UserData = UserData()
        
        sub_module: TestSubModule = TestSubModule(param0, lin, buffer0, data0)
        root: TestModule = TestModule(param1, sub_module, buffer1, data1)
        
        sub_module_rand: TestSubModule = sub_module.rand()
        self.assertIsNot(sub_module_rand, sub_module)
        self.assertIsNot(sub_module_rand.param0, sub_module.param0)
        self.assertTrue(sub_module_rand.param0.shape, sub_module.param0.shape)
        self.assertIs(sub_module_rand.buffer0, sub_module.buffer0)
        self.assertIs(sub_module_rand.data0, sub_module.data0)
        
        root_rand: TestModule = sub_module_rand._particle_kernal.data.modules[nyto.mtype.ROOT_MODULE_ID]
        self.assertIsNot(root_rand, root)
        self.assertIsNot(root_rand.param1, root.param1)
        self.assertTrue(root_rand.param1.shape, root.param1.shape)
        self.assertIs(root_rand.buffer1, root.buffer1)
        self.assertIs(root_rand.data1, root.data1)
        
        self.assertIs(root_rand._particle_kernal, sub_module_rand._particle_kernal)
        self.assertIsNot(root_rand._particle_kernal, root._particle_kernal)
        self.assertIs(root_rand._version_kernal, root._version_kernal)
        self.assertEqual(root_rand._module_id, root._module_id)
        self.assertEqual(sub_module_rand._module_id, sub_module._module_id)


class TestRandn(unittest.TestCase):
    def test_randn_root(self):
        param0: nn.Parameter = nn.Parameter(torch.randn(2, 2))
        param1: nn.Parameter = nn.Parameter(torch.randn(3, 3))
        buffer0: torch.Tensor = torch.randn(1)
        buffer1: torch.Tensor = torch.randn(1)
        lin: nn.Linear = nn.Linear(2, 3)
        data0: UserData = UserData()
        data1: UserData = UserData()
        
        sub_module: TestSubModule = TestSubModule(param0, lin, buffer0, data0)
        root: TestModule = TestModule(param1, sub_module, buffer1, data1)
        
        root_randn: TestModule = root.randn()
        self.assertIsNot(root_randn, root)
        self.assertIsNot(root_randn.param1, root.param1)
        self.assertTrue(root_randn.param1.shape, root.param1.shape)
        self.assertIs(root_randn.buffer1, root.buffer1)
        self.assertIs(root_randn.data1, root.data1)
        
        sub_module_randn: TestSubModule = root_randn.sub_module
        self.assertIsNot(sub_module_randn, sub_module)
        self.assertIsNot(sub_module_randn.param0, sub_module.param0)
        self.assertEqual(sub_module_randn.param0.shape, sub_module.param0.shape)
        self.assertIs(sub_module_randn.buffer0, sub_module.buffer0)
        self.assertIs(sub_module_randn.data0, sub_module.data0)
        
        self.assertIs(root_randn._particle_kernal, sub_module_randn._particle_kernal)
        self.assertIsNot(root_randn._particle_kernal, root._particle_kernal)
        self.assertIs(root_randn._version_kernal, root._version_kernal)
        self.assertEqual(root_randn._module_id, root._module_id)
        self.assertEqual(sub_module_randn._module_id, sub_module._module_id)   
    
    def test_randn_sub_module(self):
        param0: nn.Parameter = nn.Parameter(torch.randn(2, 2))
        param1: nn.Parameter = nn.Parameter(torch.randn(3, 3))
        buffer0: torch.Tensor = torch.randn(1)
        buffer1: torch.Tensor = torch.randn(1)
        lin: nn.Linear = nn.Linear(2, 3)
        data0: UserData = UserData()
        data1: UserData = UserData()
        
        sub_module: TestSubModule = TestSubModule(param0, lin, buffer0, data0)
        root: TestModule = TestModule(param1, sub_module, buffer1, data1)
        
        sub_module_randn: TestSubModule = sub_module.randn()
        self.assertIsNot(sub_module_randn, sub_module)
        self.assertIsNot(sub_module_randn.param0, sub_module.param0)
        self.assertTrue(sub_module_randn.param0.shape, sub_module.param0.shape)
        self.assertIs(sub_module_randn.buffer0, sub_module.buffer0)
        self.assertIs(sub_module_randn.data0, sub_module.data0)
        
        root_randn: TestModule = sub_module_randn._particle_kernal.data.modules[nyto.mtype.ROOT_MODULE_ID]
        self.assertIsNot(root_randn, root)
        self.assertIsNot(root_randn.param1, root.param1)
        self.assertTrue(root_randn.param1.shape, root.param1.shape)
        self.assertIs(root_randn.buffer1, root.buffer1)
        self.assertIs(root_randn.data1, root.data1)
        
        self.assertIs(root_randn._particle_kernal, sub_module_randn._particle_kernal)
        self.assertIsNot(root_randn._particle_kernal, root._particle_kernal)
        self.assertIs(root_randn._version_kernal, root._version_kernal)
        self.assertEqual(root_randn._module_id, root._module_id)
        self.assertEqual(sub_module_randn._module_id, sub_module._module_id)
        
        
class TestDetach(unittest.TestCase):
    def test_detach_root(self):
        param0: nn.Parameter = nn.Parameter(torch.randn(2, 2))
        param1: nn.Parameter = nn.Parameter(torch.randn(3, 3))
        buffer0: torch.Tensor = torch.randn(1)
        buffer1: torch.Tensor = torch.randn(1)
        lin: nn.Linear = nn.Linear(2, 3)
        data0: UserData = UserData()
        data1: UserData = UserData()
        
        sub_module: TestSubModule = TestSubModule(param0, lin, buffer0, data0)
        root: TestModule = TestModule(param1, sub_module, buffer1, data1)
        
        root_detach: TestModule = root.detach()
        self.assertIsNot(root_detach, root)
        self.assertIs(root_detach.param1, root.param1)
        self.assertIs(root_detach.buffer1, root.buffer1)
        self.assertIs(root_detach.data1, root.data1)
        
        sub_module_detach: TestSubModule = root_detach.sub_module
        self.assertIsNot(sub_module_detach, sub_module)
        self.assertIs(sub_module_detach.param0, sub_module.param0)
        self.assertIs(sub_module_detach.buffer0, sub_module.buffer0)
        self.assertIs(sub_module_detach.data0, sub_module.data0)
        
        self.assertIs(root_detach._particle_kernal, sub_module_detach._particle_kernal)
        self.assertIsNot(root_detach._particle_kernal, root._particle_kernal)
        self.assertIsNot(root_detach._version_kernal, root._version_kernal)
        self.assertEqual(root_detach._module_id, root._module_id)
        self.assertEqual(sub_module_detach._module_id, sub_module._module_id)
        
    def test_detach_sub_module(self):
        param0: nn.Parameter = nn.Parameter(torch.randn(2, 2))
        param1: nn.Parameter = nn.Parameter(torch.randn(3, 3))
        buffer0: torch.Tensor = torch.randn(1)
        buffer1: torch.Tensor = torch.randn(1)
        lin: nn.Linear = nn.Linear(2, 3)
        data0: UserData = UserData()
        data1: UserData = UserData()
        
        sub_module: TestSubModule = TestSubModule(param0, lin, buffer0, data0)
        root: TestModule = TestModule(param1, sub_module, buffer1, data1)
        
        sub_module_detach: TestSubModule = sub_module.detach()
        self.assertIsNot(sub_module_detach, sub_module)
        self.assertIs(sub_module_detach.param0, sub_module.param0)
        self.assertIs(sub_module_detach.buffer0, sub_module.buffer0)
        self.assertIs(sub_module_detach.data0, sub_module.data0)
        
        root_detach: TestModule = sub_module_detach._particle_kernal.data.modules[nyto.mtype.ROOT_MODULE_ID]
        self.assertIsNot(root_detach, root)
        self.assertIs(root_detach.param1, root.param1)
        self.assertIs(root_detach.buffer1, root.buffer1)
        self.assertIs(root_detach.data1, root.data1)
        
        self.assertIs(root_detach._particle_kernal, sub_module_detach._particle_kernal)
        self.assertIsNot(root_detach._particle_kernal, root._particle_kernal)
        self.assertIsNot(root_detach._version_kernal, root._version_kernal)
        self.assertEqual(root_detach._module_id, root._module_id)
        self.assertEqual(sub_module_detach._module_id, sub_module._module_id)


class TestGetParamId(unittest.TestCase):
    def test_get_param_id(self):
        class MySubModule(NytoModule):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.randn(2, 2))
        
        class MyRoot(NytoModule):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.randn(2, 2))
                self.submodule = MySubModule()
                
        root = MyRoot()
        root_param_id = root.get_param_id(root.param)
        submodule_param_id = root.get_param_id(root.submodule.param)
        
        self.assertIs(root._particle_kernal.data.params[root_param_id], root.param)
        self.assertIs(root._particle_kernal.data.params[submodule_param_id], root.submodule.param)
        
    def test_get_param_id2(self):
        class MyModule(NytoModule):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.randn(2, 2))
        
        root = MyModule()
        with self.assertRaises(Exception): 
            root.get_param_id(nn.Parameter(torch.randn(2, 2)))
            
            
class TestIsRoot(unittest.TestCase):
    def test_is_root(self):
        class MySubModule(NytoModule):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.randn(2, 2))
        
        class MyRoot(NytoModule):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.randn(2, 2))
                self.submodule = MySubModule()
        
        root = MyRoot()
        self.assertTrue(root.is_root)
        self.assertFalse(root.submodule.is_root)
        
