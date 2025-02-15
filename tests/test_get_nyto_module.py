from torch import nn
import unittest
import nytorch as nyto


class MyModule(nyto.NytoModule):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x


class TestGetNytoModule(unittest.TestCase):
    def test_directly_manipulate(self):
        model = nyto.ParticleModule(MyModule())
        
        with self.assertRaises(Exception):
            model.root_module.add_lin = nn.Linear(3, 2)
            
    def test_GetNytoModule1(self):
        my_module = MyModule()
        model = nyto.ParticleModule(my_module)
        
        with nyto.GetNytoModule(model) as root_module:
            self.assertIs(my_module, root_module)
            root_module.add_lin = nn.Linear(3, 2)
        
        self.assertTrue(hasattr(model.root_module, 'add_lin'))

        with self.assertRaises(Exception):
            model.root_module.add_lin = nn.Linear(3, 2)
            
    def test_GetNytoModule2(self):
        model = nyto.ParticleModule(MyModule())
        model_clone = model.clone()
        
        with nyto.GetNytoModule(model) as root_module:
            root_module.add_lin = nn.Linear(3, 2)
            
        with nyto.GetNytoModule(model_clone) as root_module:
            root_module.touch()
            
        self.assertTrue(hasattr(model_clone.root_module, 'add_lin'))