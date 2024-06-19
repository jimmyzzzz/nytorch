from collections import OrderedDict
from nytorch import NytoModule
from torch import nn
from typing import Optional
from .utils import TestModule, TestSubModule, UserData

import nytorch as nyto
import unittest
import warnings
import torch

class TestReference(unittest.TestCase):
    def test_reference_same_module(self):
        class AModule(NytoModule):
            pass
        
        class BModule(NytoModule):
            def __init__(self, a: AModule) -> None:
                super().__init__()
                self.a: AModule = a
                
        class CModule(NytoModule):
            def __init__(self, a: AModule) -> None:
                super().__init__()
                self.a: AModule = a
                
        class RootModule(NytoModule):
            def __init__(self, b: BModule, c: CModule) -> None:
                super().__init__()
                self.b: BModule = b
                self.c: CModule = c
        
        with warnings.catch_warnings(record=True) as w:
            a: AModule = AModule()
            b: BModule = BModule(a)
            c: CModule = CModule(a)
            self.assertIn("modules from different particles refer to the same module", str(w[-1].message))
            
            root: RootModule = RootModule(b, c)
            self.assertIs(root._particle_kernal, root.b._particle_kernal)
            self.assertIs(root._particle_kernal, root.c._particle_kernal)
            self.assertIs(root._particle_kernal, root.b.a._particle_kernal)
            self.assertEqual(root._module_id, nyto.mtype.ROOT_MODULE_ID)
            
            def assertModuleIDNotEquals(n, m_ls):
                for m in m_ls: self.assertNotEqual(n._module_id, m._module_id)
            assertModuleIDNotEquals(root, [root.b, root.c, root.b.a])
            assertModuleIDNotEquals(root.b, [root, root.c, root.b.a])
            assertModuleIDNotEquals(root.c, [root.b, root, root.b.a])
            assertModuleIDNotEquals(root.b.a, [root.b, root.c, root])
            
    def test_reference_root(self):
        class RootModule(NytoModule):
            def __init__(self) -> None:
                super().__init__()
                self.sub_module: SubModule = SubModule()
                self.sub_module.root = self
        
        class SubModule(NytoModule):
            def __init__(self) -> None:
                super().__init__()
                self.root: Optional[RootModule] = None
        
        with warnings.catch_warnings(record=True) as w:
            root = RootModule()
            self.assertIn("circular references to the root module", str(w[-1].message))
