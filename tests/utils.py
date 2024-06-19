from nytorch import NytoModule
from torch import nn
import torch


class UserData:
    pass


class TestSubModule(NytoModule):
    def __init__(self, param: nn.Parameter, lin: nn.Linear, buffer: torch.Tensor, data: UserData):
        super().__init__()
        self.param0: nn.Parameter = param
        self.lin: nn.Linear = lin
        self.register_buffer('buffer0', buffer)
        self.data0: UserData = data
        
    def forward(self, x):
        return self.lin(x) + self.param0
    

class TestModule(NytoModule):
    def __init__(self, param: nn.Parameter, sub_module: NytoModule, buffer: torch.Tensor, data: UserData):
        super().__init__()
        self.param1: nn.Parameter = param
        self.sub_module: NytoModule = sub_module
        self.register_buffer('buffer1', buffer)
        self.data1: UserData = data
    
    def forward(self, x):
        return self.sub_module(x) + self.param1
