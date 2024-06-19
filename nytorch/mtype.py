
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from torch import nn
from typing import Optional
from typing import OrderedDict as OrderedDictType
import torch

ModuleID = int
ParamID = int
Module = nn.Module
ParamType = nn.Parameter
ROOT_MODULE_ID: ModuleID = 0


@dataclass
class ModuleMeta:
    """     
    Maintains mappings of module and parameter attribute names to their respective IDs.

    Attributes:
        sub_modules (OrderedDict[str, Optional[ModuleID]]): 
            Mapping of attribute names to submodule IDs.
        sub_params (OrderedDict[str, Optional[ParamID]]): 
            Mapping of attribute names to parameter IDs.
    """
    
    __slots__ = "sub_modules", "sub_params"

    sub_modules: OrderedDict[str, Optional[ModuleID]]
    sub_params: OrderedDict[str, Optional[ParamID]]
    
    def copy(self) -> ModuleMeta:
        """
        Creates a deep copy of the current ModuleMeta instance.

        Returns:
            ModuleMeta: A deep copy of this ModuleMeta.
        """
        return ModuleMeta(self.sub_modules.copy(), self.sub_params.copy())


@dataclass
class ParamConfig:
    r"""
    Stores configuration settings for parameter operations in particles.

    By modifying these settings, users can influence particle behavior. 
    Additional attributes can be added to support custom particle operations, 
    which will define how these custom attributes are used.

    Attributes:
        operational (bool): 
            Indicates if the parameter is included in particle operations.
        clone (bool): 
            Indicates if the parameter should be cloned when not included in particle operations.
    """

    operational: bool = True
    clone: bool = True
    
    def __repr__(self) -> str:
        """
        Returns a string representation of ParamConfig.

        Returns:
            str: A string representation of the current ParamConfig instance.
        """
        return f"ParamConfig({', '.join([k + '=' + v.__repr__() for k, v in self.__dict__.items()])})"

    def copy(self) -> ParamConfig:
        """
        Creates a deep copy of the current ParamConfig instance.

        Returns:
            ParamConfig: A deep copy of this ParamConfig.
        """
        return ParamConfig(self.operational, self.clone)


ModuleDict = OrderedDictType[ModuleID, Module]
ParamDict = OrderedDictType[ParamID, ParamType]
MetaDict = OrderedDictType[ModuleID, ModuleMeta]
ConfigDict = OrderedDictType[ParamID, ParamConfig]

