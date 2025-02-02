from __future__ import annotations
from collections import OrderedDict
from .base import NytoModuleBase, ParticleDataImp, VersionDataImp
from .kernel import ParticleData, ParticleKernel, ParticleUpdater, VersionData
from .mtype import ConfigDict, MetaDict, Module, ModuleDict, ModuleID, ParamDict, ParamID, ParamType, ROOT_MODULE_ID
from .utils import copy_modules, clone_param, clone_params, make_modules_ref
from typing import Optional
from typing_extensions import Self
import torch


class AddModuleParticleUpdater(ParticleUpdater[VersionDataImp, ParticleDataImp]):

    module_id: ModuleID
    attr_name: str
    add_module_id: ModuleID
    add_modules: ModuleDict
    add_params: ParamDict
    remove_modules: set[ModuleID]
    remove_params: set[ParamID]
    next_version_modules_meta: Optional[MetaDict]
    owner: Optional[ParticleData]
    
    def __init__(self,
                 module_id: ModuleID,
                 attr_name: str,
                 add_module_id: ModuleID,
                 add_modules: ModuleDict,
                 add_params: ParamDict,
                 remove_modules: set[ModuleID],
                 remove_params: set[ParamID]) -> None:
        self.module_id = module_id
        self.attr_name = attr_name
        self.add_module_id = add_module_id
        self.add_modules = add_modules
        self.add_params = add_params
        self.remove_modules = remove_modules
        self.remove_params = remove_params
        self.next_version_modules_meta = None
        self.owner = None
        
    def set_version_data(self, vdata: VersionDataImp) -> Self:
        return self
    
    def set_next_version_data(self, vdata: VersionDataImp) -> Self:
        self.next_version_modules_meta = vdata.meta
        return self
    
    def set_particle_data(self, pdata: ParticleDataImp) -> Self:
        self.owner = pdata
        return self
    
    def run(self, pdata: ParticleDataImp) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if self.attr_name in d:
                    if isinstance(d, dict):
                        del d[self.attr_name]
                    else:
                        d.discard(self.attr_name)

        if self.owner is pdata:
            pdata.add_modules(self.add_modules)
            pdata.add_params(self.add_params)
            pdata.remove_modules(self.remove_modules)
            pdata.remove_params(self.remove_params)
            Module.__setattr__(pdata.modules[self.module_id],
                               self.attr_name,
                               pdata.modules[self.add_module_id])
        else:
            assert self.next_version_modules_meta is not None
            pdata.add_modules(copy_modules(self.add_modules))
            pdata.add_params(clone_params(self.add_params))
            pdata.remove_modules(self.remove_modules)
            pdata.remove_params(self.remove_params)
            
            this = pdata.modules[self.module_id]
            remove_from(this.__dict__, this._parameters, this._buffers, this._non_persistent_buffers_set)
            make_modules_ref(pdata.modules,
                             pdata.params,
                             self.next_version_modules_meta,
                             {self.module_id}|set(self.add_modules.keys()))
        
        root_module: Module = pdata.modules[ROOT_MODULE_ID]
        assert isinstance(root_module, NytoModuleBase)
        for mid, mod in pdata.modules.items():
            if isinstance(mod, NytoModuleBase):
                mod._module_id = mid
                mod._particle_kernel = root_module._particle_kernel
                

class AddParamParticleUpdater(ParticleUpdater[VersionDataImp, ParticleDataImp]):
    
    module_id: ModuleID
    attr_name: str
    add_param_id: ParamID
    remove_modules: set[ModuleID]
    remove_params: set[ParamID]
    add_param: Optional[ParamType]
    owner: Optional[ParticleDataImp]
        
    def __init__(self,
                 module_id: ModuleID,
                 attr_name: str,
                 add_param_id: ParamID,
                 remove_modules: set[ModuleID],
                 remove_params: set[ParamID],
                 add_param: Optional[ParamType]=None) -> None:
        self.module_id = module_id
        self.attr_name = attr_name
        self.add_param_id = add_param_id
        self.remove_modules = remove_modules
        self.remove_params = remove_params
        self.add_param = add_param
        self.owner = None
        
    def set_version_data(self, vdata: VersionDataImp) -> Self:
        return self
    
    def set_next_version_data(self, vdata: VersionDataImp) -> Self:
        return self
    
    def set_particle_data(self, pdata: ParticleDataImp) -> Self:
        self.owner = pdata
        return self
    
    def run(self, pdata: ParticleDataImp) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if self.attr_name in d:
                    if isinstance(d, dict):
                        del d[self.attr_name]
                    else:
                        d.discard(self.attr_name)
        
        if self.add_param is not None:
            pdata.params[self.add_param_id] = self.add_param if self.owner is pdata else clone_param(self.add_param)
        
        pdata.remove_modules(self.remove_modules)
        pdata.remove_params(self.remove_params)
        
        this = pdata.modules[self.module_id]
        remove_from(this.__dict__, this._buffers, this._modules, this._non_persistent_buffers_set)
        Module.register_parameter(this, self.attr_name, pdata.params[self.add_param_id])

    
class RegisterBufferParticleUpdater(ParticleUpdater[VersionDataImp, ParticleDataImp]):
    
    module_id: ModuleID
    attr_name: str
    value: Optional[torch.Tensor]
    persistent: bool
    remove_modules: set[ModuleID]
    remove_params: set[ParamID]
    
    def __init__(self,
                 module_id: ModuleID,
                 attr_name: str,
                 value: Optional[torch.Tensor],
                 persistent: bool,
                 remove_modules: set[ModuleID],
                 remove_params: set[ParamID]) -> None:
        self.module_id = module_id
        self.attr_name = attr_name
        self.value = value
        self.persistent = persistent
        self.remove_modules = remove_modules
        self.remove_params = remove_params
        
    def set_version_data(self, vdata: VersionDataImp) -> Self:
        return self
    
    def set_next_version_data(self, vdata: VersionDataImp) -> Self:
        return self
    
    def set_particle_data(self, pdata: ParticleDataImp) -> Self:
        return self
    
    def run(self, pdata: ParticleDataImp) -> None:
        pdata.remove_modules(self.remove_modules)
        pdata.remove_params(self.remove_params)
        
        this = pdata.modules[self.module_id]
        Module.register_buffer(this, self.attr_name, self.value, self.persistent)

    
class SetModuleNoneParticleUpdater(ParticleUpdater[VersionDataImp, ParticleDataImp]):
    
    module_id: ModuleID
    attr_name: str
    remove_modules: set[ModuleID]
    remove_params: set[ParamID]

    def __init__(self, 
                 module_id: ModuleID, 
                 attr_name: str, 
                 remove_modules: set[ModuleID], 
                 remove_params: set[ParamID]) -> None:
        self.module_id = module_id
        self.attr_name = attr_name
        self.remove_modules = remove_modules
        self.remove_params = remove_params
        
    def set_version_data(self, vdata: VersionDataImp) -> Self:
        return self
    
    def set_next_version_data(self, vdata: VersionDataImp) -> Self:
        return self
    
    def set_particle_data(self, pdata: ParticleDataImp) -> Self:
        return self
    
    def run(self, pdata: ParticleDataImp) -> None:
        pdata.remove_modules(self.remove_modules)
        pdata.remove_params(self.remove_params)
        Module.__setattr__(pdata.modules[self.module_id], self.attr_name, None)  # type: ignore
        

class SetParamNoneParticleUpdater(ParticleUpdater[VersionDataImp, ParticleDataImp]):
    
    module_id: ModuleID
    attr_name: str
    remove_modules: set[ModuleID]
    remove_params: set[ParamID]

    def __init__(self, 
                 module_id: ModuleID, 
                 attr_name: str, 
                 remove_modules: set[ModuleID],
                 remove_params: set[ParamID]) -> None:
        self.module_id = module_id
        self.attr_name = attr_name
        self.remove_modules = remove_modules
        self.remove_params = remove_params
        
    def set_version_data(self, vdata: VersionDataImp) -> Self:
        return self
    
    def set_next_version_data(self, vdata: VersionDataImp) -> Self:
        return self
    
    def set_particle_data(self, pdata: ParticleDataImp) -> Self:
        return self
    
    def run(self, pdata: ParticleDataImp) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if self.attr_name in d:
                    if isinstance(d, dict):
                        del d[self.attr_name]
                    else:
                        d.discard(self.attr_name)
        
        pdata.remove_modules(self.remove_modules)
        pdata.remove_params(self.remove_params)
        this = pdata.modules[self.module_id]
        remove_from(this.__dict__, this._buffers, this._modules, this._non_persistent_buffers_set)
        Module.register_parameter(this, self.attr_name, None)
        
        
class DelModuleParticleUpdater(ParticleUpdater[VersionDataImp, ParticleDataImp]):
    
    module_id: ModuleID
    attr_name: str
    remove_modules: set[ModuleID]
    remove_params: set[ParamID]

    def __init__(self, 
                 module_id: ModuleID, 
                 attr_name: str, 
                 remove_modules: set[ModuleID], 
                 remove_params: set[ParamID]) -> None:
        self.module_id = module_id
        self.attr_name = attr_name
        self.remove_modules = remove_modules
        self.remove_params = remove_params
        
    def set_version_data(self, vdata: VersionDataImp) -> Self:
        return self
    
    def set_next_version_data(self, vdata: VersionDataImp) -> Self:
        return self
    
    def set_particle_data(self, pdata: ParticleDataImp) -> Self:
        return self
    
    def run(self, pdata: ParticleDataImp) -> None:
        pdata.remove_modules(self.remove_modules)
        pdata.remove_params(self.remove_params)
        Module.__delattr__(pdata.modules[self.module_id], self.attr_name)
        

class DelParamParticleUpdater(ParticleUpdater[VersionDataImp, ParticleDataImp]):
    
    module_id: ModuleID
    attr_name: str
    remove_params: set[ParamID]

    def __init__(self, 
                 module_id: ModuleID, 
                 attr_name: str, 
                 remove_params: set[ParamID]) -> None:
        self.module_id = module_id
        self.attr_name = attr_name
        self.remove_params = remove_params
        
    def set_version_data(self, vdata: VersionDataImp) -> Self:
        return self
    
    def set_next_version_data(self, vdata: VersionDataImp) -> Self:
        return self
    
    def set_particle_data(self, pdata: ParticleDataImp) -> Self:
        return self
    
    def run(self, pdata: ParticleDataImp) -> None:
        pdata.remove_params(self.remove_params)
        Module.__delattr__(pdata.modules[self.module_id], self.attr_name)
        
        
class DelBufferParticleUpdater(ParticleUpdater[VersionDataImp, ParticleDataImp]):
    
    module_id: ModuleID
    attr_name: str

    def __init__(self, module_id: ModuleID, attr_name: str) -> None:
        self.module_id = module_id
        self.attr_name = attr_name
        
    def set_version_data(self, vdata: VersionDataImp) -> Self:
        return self
    
    def set_next_version_data(self, vdata: VersionDataImp) -> Self:
        return self
    
    def set_particle_data(self, pdata: ParticleDataImp) -> Self:
        return self
    
    def run(self, pdata: ParticleDataImp) -> None:
        Module.__delattr__(pdata.modules[self.module_id], self.attr_name)
    
