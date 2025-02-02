from collections import OrderedDict
from .utils import MyModule, MySubModule, UserData
from nytorch import NytoModule
from torch import nn
import unittest
import nytorch as nyto
import torch


class TestAddAttar(unittest.TestCase):
    def test_add_param(self):
        class SubModule(NytoModule):
            pass
        
        class RootModule(NytoModule):
            def __init__(self):
                super().__init__()
                self.sub_module: SubModule = SubModule()
        
        root: RootModule = RootModule()
        root_clone: RootModule = root.clone()
        root_randn: RootModule = root.randn()
        root_detach: RootModule = root.detach()
        
        root.param0: nn.Parameter = nn.Parameter(torch.randn(1))
        root.sub_module.param1: nn.Parameter = nn.Parameter(torch.randn(1))
        root_clone.touch()
        root_randn.touch()
        root_detach.touch()
        
        self.assertTrue(hasattr(root_clone, "param0"))
        self.assertTrue(hasattr(root_randn, "param0"))
        self.assertFalse(hasattr(root_detach, "param0"))
        self.assertTrue(hasattr(root_clone.sub_module, "param1"))
        self.assertTrue(hasattr(root_randn.sub_module, "param1"))
        self.assertFalse(hasattr(root_detach.sub_module, "param1"))
        
        self.assertIsNot(root.param0, root_clone.param0)
        self.assertIsNot(root.param0, root_randn.param0)
        self.assertTrue(torch.equal(root.param0, root_clone.param0))
        self.assertTrue(torch.equal(root.param0, root_randn.param0))
        self.assertIsNot(root.sub_module.param1, root_clone.sub_module.param1)
        self.assertIsNot(root.sub_module.param1, root_randn.sub_module.param1)
        self.assertTrue(torch.equal(root.sub_module.param1, root_clone.sub_module.param1))
        self.assertTrue(torch.equal(root.sub_module.param1, root_randn.sub_module.param1))

    def test_add_module(self):
        class SubModule(NytoModule):
            pass
        
        class RootModule(NytoModule):
            def __init__(self):
                super().__init__()
                self.sub_module: SubModule = SubModule()
        
        root: RootModule = RootModule()
        root_clone: RootModule = root.clone()
        root_randn: RootModule = root.randn()
        root_detach: RootModule = root.detach()
        
        root.lin0: MySubModule = MySubModule(param=nn.Parameter(torch.randn(2, 2)),
                                                 lin=nn.Linear(3, 2),
                                                 buffer=torch.randn(1),
                                                 data=UserData())
        root.sub_module.lin1: MySubModule = MySubModule(param=nn.Parameter(torch.randn(2, 2)),
                                                            lin=nn.Linear(3, 2),
                                                            buffer=torch.randn(1),
                                                            data=UserData())
        root_clone.touch()
        root_randn.touch()
        root_detach.touch()
        
        self.assertTrue(hasattr(root_clone, "lin0"))
        self.assertTrue(hasattr(root_randn, "lin0"))
        self.assertFalse(hasattr(root_detach, "lin0"))
        self.assertTrue(hasattr(root_clone.sub_module, "lin1"))
        self.assertTrue(hasattr(root_randn.sub_module, "lin1"))
        self.assertFalse(hasattr(root_detach.sub_module, "lin1"))
        
        self.assertIsNot(root.lin0, root_clone.lin0)
        self.assertIsNot(root.lin0, root_randn.lin0)
        self.assertIsNot(root.sub_module.lin1, root_clone.sub_module.lin1)
        self.assertIsNot(root.sub_module.lin1, root_randn.sub_module.lin1)
        
        self.assertTrue(hasattr(root_clone.lin0, "param0"))
        self.assertTrue(hasattr(root_clone.lin0, "lin"))
        self.assertTrue(hasattr(root_clone.lin0, "buffer0"))
        self.assertTrue(hasattr(root_clone.lin0, "data0"))
        self.assertTrue(hasattr(root_randn.lin0, "param0"))
        self.assertTrue(hasattr(root_randn.lin0, "lin"))
        self.assertTrue(hasattr(root_randn.lin0, "buffer0"))
        self.assertTrue(hasattr(root_randn.lin0, "data0"))
        
        self.assertIsNot(root.lin0.param0, root_clone.lin0.param0)
        self.assertTrue(torch.equal(root.lin0.param0, root_clone.lin0.param0))
        self.assertIsNot(root.lin0.lin, root_clone.lin0.lin)
        self.assertIsNot(root.lin0.lin.weight, root_clone.lin0.lin.weight)
        self.assertTrue(torch.equal(root.lin0.lin.weight, root_clone.lin0.lin.weight))
        self.assertIsNot(root.lin0.lin.bias, root_clone.lin0.lin.bias)
        self.assertTrue(torch.equal(root.lin0.lin.bias, root_clone.lin0.lin.bias))
        self.assertIs(root.lin0.buffer0, root_clone.lin0.buffer0)
        self.assertIs(root.lin0.data0, root_clone.lin0.data0)
        
        self.assertIsNot(root.lin0.param0, root_randn.lin0.param0)
        self.assertTrue(torch.equal(root.lin0.param0, root_randn.lin0.param0))
        self.assertIsNot(root.lin0.lin, root_randn.lin0.lin)
        self.assertIsNot(root.lin0.lin.weight, root_randn.lin0.lin.weight)
        self.assertTrue(torch.equal(root.lin0.lin.weight, root_randn.lin0.lin.weight))
        self.assertIsNot(root.lin0.lin.bias, root_randn.lin0.lin.bias)
        self.assertTrue(torch.equal(root.lin0.lin.bias, root_randn.lin0.lin.bias))
        self.assertIs(root.lin0.buffer0, root_randn.lin0.buffer0)
        self.assertIs(root.lin0.data0, root_randn.lin0.data0)
        
        self.assertTrue(hasattr(root_clone.sub_module.lin1, "param0"))
        self.assertTrue(hasattr(root_clone.sub_module.lin1, "lin"))
        self.assertTrue(hasattr(root_clone.sub_module.lin1, "buffer0"))
        self.assertTrue(hasattr(root_clone.sub_module.lin1, "data0"))
        self.assertTrue(hasattr(root_randn.sub_module.lin1, "param0"))
        self.assertTrue(hasattr(root_randn.sub_module.lin1, "lin"))
        self.assertTrue(hasattr(root_randn.sub_module.lin1, "buffer0"))
        self.assertTrue(hasattr(root_randn.sub_module.lin1, "data0"))
        
        self.assertIsNot(root.sub_module.lin1.param0, root_clone.sub_module.lin1.param0)
        self.assertTrue(torch.equal(root.sub_module.lin1.param0, root_clone.sub_module.lin1.param0))
        self.assertIsNot(root.sub_module.lin1.lin, root_clone.sub_module.lin1.lin)
        self.assertIsNot(root.sub_module.lin1.lin.weight, root_clone.sub_module.lin1.lin.weight)
        self.assertTrue(torch.equal(root.sub_module.lin1.lin.weight, root_clone.sub_module.lin1.lin.weight))
        self.assertIsNot(root.sub_module.lin1.lin.bias, root_clone.sub_module.lin1.lin.bias)
        self.assertTrue(torch.equal(root.sub_module.lin1.lin.bias, root_clone.sub_module.lin1.lin.bias))
        self.assertIs(root.sub_module.lin1.buffer0, root_clone.sub_module.lin1.buffer0)
        self.assertIs(root.sub_module.lin1.data0, root_clone.sub_module.lin1.data0)
        
        self.assertIsNot(root.sub_module.lin1.param0, root_randn.sub_module.lin1.param0)
        self.assertTrue(torch.equal(root.sub_module.lin1.param0, root_randn.sub_module.lin1.param0))
        self.assertIsNot(root.sub_module.lin1.lin, root_randn.sub_module.lin1.lin)
        self.assertIsNot(root.sub_module.lin1.lin.weight, root_randn.sub_module.lin1.lin.weight)
        self.assertTrue(torch.equal(root.sub_module.lin1.lin.weight, root_randn.sub_module.lin1.lin.weight))
        self.assertIsNot(root.sub_module.lin1.lin.bias, root_randn.sub_module.lin1.lin.bias)
        self.assertTrue(torch.equal(root.sub_module.lin1.lin.bias, root_randn.sub_module.lin1.lin.bias))
        self.assertIs(root.sub_module.lin1.buffer0, root_randn.sub_module.lin1.buffer0)
        self.assertIs(root.sub_module.lin1.data0, root_randn.sub_module.lin1.data0)
        
    def test_add_buffer(self):
        class SubModule(NytoModule):
            pass
        
        class RootModule(NytoModule):
            def __init__(self):
                super().__init__()
                self.sub_module: SubModule = SubModule()
        
        root: RootModule = RootModule()
        root_clone: RootModule = root.clone()
        root_randn: RootModule = root.randn()
        root_detach: RootModule = root.detach()
        
        root.register_buffer("buffer0", torch.randn(1))
        root.sub_module.register_buffer("buffer1", torch.randn(1))
        root_clone.touch()
        root_randn.touch()
        root_detach.touch()
        
        self.assertTrue(hasattr(root_clone, "buffer0"))
        self.assertTrue(hasattr(root_randn, "buffer0"))
        self.assertFalse(hasattr(root_detach, "buffer0"))
        self.assertTrue(hasattr(root_clone.sub_module, "buffer1"))
        self.assertTrue(hasattr(root_randn.sub_module, "buffer1"))
        self.assertFalse(hasattr(root_detach.sub_module, "buffer1"))
        
        self.assertIs(root.buffer0, root_clone.buffer0)
        self.assertIs(root.buffer0, root_randn.buffer0)
        self.assertIs(root.sub_module.buffer1, root_clone.sub_module.buffer1)
        self.assertIs(root.sub_module.buffer1, root_randn.sub_module.buffer1)
    
    
class TestDelAttar(unittest.TestCase):
    def test_del_param(self):
        class SubModule(NytoModule):
            def __init__(self):
                super().__init__()
                self.param0: nn.Parameter = nn.Parameter(torch.randn(1))
        
        class RootModule(NytoModule):
            def __init__(self):
                super().__init__()
                self.sub_module: SubModule = SubModule()
                self.param1: nn.Parameter = nn.Parameter(torch.randn(1))
        
        root: RootModule = RootModule()
        root_clone: RootModule = root.clone()
        root_randn: RootModule = root.randn()
        root_detach: RootModule = root.detach()
        
        del root.param1
        del root.sub_module.param0
        root_clone.touch()
        root_randn.touch()
        root_detach.touch()
        
        self.assertFalse(hasattr(root_clone, "param1"))
        self.assertFalse(hasattr(root_randn, "param1"))
        self.assertTrue(hasattr(root_detach, "param1"))
        self.assertFalse(hasattr(root_clone.sub_module, "param0"))
        self.assertFalse(hasattr(root_randn.sub_module, "param0"))
        self.assertTrue(hasattr(root_detach.sub_module, "param0"))
        
    def test_del_module(self):
        class SubModule(NytoModule):
            def __init__(self):
                super().__init__()
                self.lin0: nn.Linear = nn.Linear(3, 2)
        
        class RootModule(NytoModule):
            def __init__(self):
                super().__init__()
                self.sub_module: SubModule = SubModule()
                self.lin1: nn.Linear = nn.Linear(3, 2)
        
        root: RootModule = RootModule()
        root_clone: RootModule = root.clone()
        root_randn: RootModule = root.randn()
        root_detach: RootModule = root.detach()
        
        del root.lin1
        del root.sub_module.lin0
        root_clone.touch()
        root_randn.touch()
        root_detach.touch()
        
        self.assertFalse(hasattr(root_clone, "lin1"))
        self.assertFalse(hasattr(root_randn, "lin1"))
        self.assertTrue(hasattr(root_detach, "lin1"))
        self.assertFalse(hasattr(root_clone.sub_module, "lin0"))
        self.assertFalse(hasattr(root_randn.sub_module, "lin0"))
        self.assertTrue(hasattr(root_detach.sub_module, "lin0"))
        
    def test_del_buffer(self):
        class SubModule(NytoModule):
            def __init__(self):
                super().__init__()
                self.register_buffer("buffer0", torch.Tensor([0.]))
        
        class RootModule(NytoModule):
            def __init__(self):
                super().__init__()
                self.sub_module: SubModule = SubModule()
                self.register_buffer("buffer1", torch.Tensor([1.]))
        
        root: RootModule = RootModule()
        root_clone: RootModule = root.clone()
        root_randn: RootModule = root.randn()
        root_detach: RootModule = root.detach()
        
        del root.buffer1
        del root.sub_module.buffer0
        root_clone.touch()
        root_randn.touch()
        root_detach.touch()
        
        self.assertFalse(hasattr(root_clone, "buffer1"))
        self.assertFalse(hasattr(root_randn, "buffer1"))
        self.assertTrue(hasattr(root_detach, "buffer1"))
        self.assertFalse(hasattr(root_clone.sub_module, "buffer0"))
        self.assertFalse(hasattr(root_randn.sub_module, "buffer0"))
        self.assertTrue(hasattr(root_detach.sub_module, "buffer0"))
        
