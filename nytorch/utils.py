from collections import OrderedDict
from .mtype import MetaDict, Module, ModuleDict, ModuleID, ModuleMeta, ParamDict, ParamType
from typing import Optional, Set
from torch import nn
import torch


def copy_module(module: Module) -> Module:
    """
    Create a shallow copy of the module.

    Args:
        module (Module): The module to be copied.

    Returns:
        Module: The shallow copy of the module.
    """
    if module is None: return module
    
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()
    return clone


def copy_modules(modules: ModuleDict) -> ModuleDict:
    """
    Create shallow copies of modules.

    Args:
        modules (ModuleDict): The modules to be copied.

    Returns:
        ModuleDict: Shallow copies of the modules.
    """
    return OrderedDict((mid, copy_module(mod)) for mid, mod in modules.items())
    

def make_modules_ref(modules: ModuleDict, params: ParamDict, metas: MetaDict, targets: Optional[Set[ModuleID]]=None) -> None:
    r"""Establish references between a set of modules and parameters according to correct structure.
    
    Args:
        modules (ModuleDict):
            Modules needing to establish references.
        params (ParamDict):
            Parameters needing to establish references.
        metas (MetaDict):
            Records of the correct reference structure for modules and parameters.
        targets (Optional[set[ModuleID]]):
            Choose which modules to establish references for.
    
    Examples:
    
    .. code-block:: python
    
        class MyNet(nn.Module):
            def __init__(self):
                super().__init__()
            
        def new_param(*args):
            return nn.Parameter(torch.randn(*args))
        
        net0, net1 = MyNet(), MyNet()
        param0, param1, param2 = new_param(1), new_param(2), new_param(3)
        
        modules = OrderedDict([(0, net0), 
                               (1, net1)])
        params = OrderedDict([(0, param0), 
                              (1, param1), 
                              (2, param2)])
        metas = OrderedDict([(0, ModuleMeta(sub_modules=OrderedDict([("m1", 1)]), 
                                            sub_params=OrderedDict([("p0", 0)]))), 
                             (1, ModuleMeta(sub_modules=OrderedDict(), 
                                            sub_params=OrderedDict([("p1", 1), 
                                                                    ("p2", 2)])))])
                                            
        >>> make_modules_ref(modules, params, metas)
        >>> assert net0.m1 is net1
        >>> assert net0.p0 is param0
        >>> assert net1.p1 is param1
        >>> assert net1.p2 is param2
    """
    if targets is None:
        targets = set(modules.keys())
    assert set(modules.keys()) == set(metas.keys())
    assert len(targets - set(metas.keys())) == 0
    for mid in targets:
        module: Module = modules[mid]
        mmeta: ModuleMeta = metas[mid]
        module._modules = OrderedDict((sub_name, sub_mid) if sub_mid is None else (sub_name, modules[sub_mid])
                                      for sub_name, sub_mid in mmeta.sub_modules.items())
        module._parameters = OrderedDict((sub_name, sub_pid) if sub_pid is None else (sub_name, params[sub_pid])
                                         for sub_name, sub_pid in mmeta.sub_params.items())

        if hasattr(module, 'flatten_parameters'):
            modules[mid] = module._apply(lambda x: x)


def clone_param(param: ParamType) -> ParamType:
    """
    Create a deep copy of the parameter.

    Args:
        param (ParamType): The parameter to be copied.

    Returns:
        ParamType: The deep copy of the parameter.
    """
    with torch.no_grad():
        return nn.Parameter(param.clone().requires_grad_(param.requires_grad))


def clone_params(params: ParamDict) -> ParamDict:
    """
    Create deep copies of parameters.

    Args:
        params (ParamDict): The parameters to be copied.

    Returns:
        ParamDict: Deep copies of the parameters.
    """
    with torch.no_grad():
        return OrderedDict((k, v) if v is None else 
                           (k, nn.Parameter(v.clone().requires_grad_(v.requires_grad)))
                           for k, v in params.items())
