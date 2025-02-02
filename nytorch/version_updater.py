from __future__ import annotations
from collections import ChainMap, OrderedDict
from collections.abc import Iterable
from .base import ParticleDataImp, VersionDataImp
from .kernel import ParticleData, ParticleUpdater, VersionData, VersionUpdater
from .mtype import ConfigDict, MetaDict, Module, ModuleDict, ModuleID, ModuleMeta, ParamConfig, ParamDict, ParamID, ParamType
from .particle_updater import AddModuleParticleUpdater, AddParamParticleUpdater, DelBufferParticleUpdater, DelModuleParticleUpdater, DelParamParticleUpdater, RegisterBufferParticleUpdater, SetModuleNoneParticleUpdater, SetParamNoneParticleUpdater
from typing import Generic, Optional, TypeVar
from typing_extensions import Self
import torch


class AddModuleVersionUpdater(VersionUpdater[VersionDataImp, ParticleDataImp]):
    r"""
    Updater for VersionKernel instances that adds a module to a particle.

    This class facilitates the addition of a module to a particle by updating both version and particle data.

    Args:
        module_id (ModuleID): 
            ID of the module where the new module is added.
        attr_name (str): 
            Name of the attribute where the new module is added.
        add_module (Module): 
            The module to be added.
            
    Attributes:
        module_id (ModuleID): 
            ID of the module in the particle where the module is added.
        attr_name (str): 
            Name of the attribute where the module is added.
        add_module (Module): 
            The module to be added to the particle.
        owner_modules (Optional[ModuleDict]): 
            Modules of the particle initiating the event.
        owner_params (Optional[ParamDict]): 
            Parameters of the particle initiating the event.
    
    
    Example::
        
        class MyNet(NytoModule):
            ...
        
        net0 = MyNet()
        net1 = MyNet()
        particle_kernel: ParticleKernel = net0._particle_kernel
        particle_kernel.version_update(AddModuleVersionUpdater(net0._module_id, 
                                                               "attar_net1", 
                                                               net1))
        
        >>> net0.attar_net1 is net1
        True
    """

    module_id: ModuleID
    attr_name: str
    add_module: Module
    owner_modules: Optional[ModuleDict]
    owner_params: Optional[ParamDict]
    
    def __init__(self, module_id: ModuleID, attr_name: str, add_module: Module) -> None:
        r"""
        Initialize the AddModuleVersionUpdater.

        Args:
            module_id (ModuleID): 
                ID of the module where the new module is added.
            attr_name (str): 
                Name of the attribute where the new module is added.
            add_module (Module): 
                The module to be added.
        """
        self.module_id = module_id
        self.attr_name = attr_name
        self.add_module = add_module
        self.owner_modules = None
        self.owner_params = None
        
    def set_version_data(self, vdata: VersionDataImp) -> Self:
        return self
    
    def set_particle_data(self, pdata: ParticleDataImp) -> Self:
        self.owner_modules, self.owner_params = pdata.modules, pdata.params
        return self
    
    def run(self, vdata: VersionDataImp) -> tuple[VersionDataImp, ParticleUpdater]:
        K = TypeVar("K", ModuleID, ParamID)
        V = TypeVar("V", Module, ParamType)
        class UniqueDict(Generic[K, V]):
            def __init__(self, data: OrderedDict[K, V]) -> None:
                self.data: OrderedDict[K, V] = data
                self.add_data: OrderedDict[K, V] = OrderedDict()
                
            def _next_key(self) -> K:
                if len(ChainMap(self.data, self.add_data)) == 0:
                    return 0
                return max(ChainMap(self.data, self.add_data).keys())+1
            
            def add_value(self, value: Optional[V]) -> None:
                if value is None: return
                for k, v in self.data.items():
                    if value is v: return
                self.add_data[self._next_key()] = value
                
            def add_values(self, values: Iterable[Optional[V]]) -> None:
                for v in values: self.add_value(v)
                
            def value_to_key(self, value: V) -> K:
                for k, v in ChainMap(self.data, self.add_data).items():
                    if value is v: return k
                raise ValueError(f"can not found {value}")
        
        assert self.owner_modules is not None
        assert self.owner_params is not None
        unq_modules = UniqueDict[ModuleID, Module](self.owner_modules)
        unq_params = UniqueDict[ParamID, ParamType](self.owner_params)
        for mod in self.add_module.modules():
            unq_modules.add_value(mod)
            unq_params.add_values(mod._parameters.values())
        
        add_modules_meta: MetaDict = OrderedDict(
            (unq_modules.value_to_key(mod), 
             ModuleMeta(sub_modules=OrderedDict((sub_name, sub_mod) 
                                                if sub_mod is None else 
                                                (sub_name, unq_modules.value_to_key(sub_mod))
                                                for sub_name, sub_mod in mod._modules.items()),
                        sub_params=OrderedDict((sub_name, sub_param)
                                               if sub_param is None else
                                               (sub_name, unq_params.value_to_key(sub_param))
                                               for sub_name, sub_param in mod._parameters.items())))
            for mid, mod in unq_modules.add_data.items())
        add_params_config: ConfigDict = OrderedDict((pid, ParamConfig()) for pid, param in unq_params.add_data.items())
        add_modules_id: ModuleID = unq_modules.value_to_key(self.add_module)
        
        new_vdata: VersionDataImp = vdata.copy()
        if self.attr_name in new_vdata.meta[self.module_id].sub_params:
            del new_vdata.meta[self.module_id].sub_params[self.attr_name]
        new_vdata.meta[self.module_id].sub_modules[self.attr_name] = add_modules_id
        new_vdata.meta.update(add_modules_meta)
        new_vdata.config.update(add_params_config)
        remove_modules: set[ModuleID] = new_vdata.get_garbage_modules()
        new_vdata.remove_modules(remove_modules)
        remove_params: set[ParamID] = new_vdata.get_garbage_params()
        new_vdata.remove_params(remove_params)
        
        return (new_vdata,
                AddModuleParticleUpdater(self.module_id,
                                         self.attr_name,
                                         add_modules_id,
                                         unq_modules.add_data,
                                         unq_params.add_data,
                                         remove_modules,
                                         remove_params))


class AddParamVersionUpdater(VersionUpdater[VersionDataImp, ParticleDataImp]):
    r"""
    Updater for VersionKernel instances that adds a parameter to a particle.

    Args:
        module_id (ModuleID): 
            ID of the module where the new parameter is added.
        attr_name (str): 
            Name of the attribute where the new parameter is added.
        add_param (ParamType): 
            The parameter to be added.
            
    Attributes:
        module_id (ModuleID): 
            ID of the module in the particle where the parameter is added.
        attr_name (str): 
            Name of the attribute where the parameter is added.
        add_param (ParamType): 
            The parameter to be added to the particle.
        owner_params (Optional[ParamDict]): 
            Parameters of the particle initiating the event.
    
    
    Example::
        
        class MyNet(NytoModule):
            ...
        
        net = MyNet()
        add_param = torch.nn.Parameter(torch.randn(3, 3))

        particle_kernel: ParticleKernel = net._particle_kernel
        particle_kernel.version_update(AddParamVersionUpdater(net._module_id, 
                                                              "attar_add_param", 
                                                              add_param))
        
        >>> net.attar_add_param is add_param
        True
    """

    module_id: ModuleID
    attr_name: str
    add_param: ParamType
    owner_params: Optional[ParamDict]
        
    def __init__(self, module_id: ModuleID, attr_name: str, add_param: ParamType) -> None:
        """
        Initialize the AddParamVersionUpdater.

        Args:
            module_id (ModuleID): 
                ID of the module where the new parameter is added.
            attr_name (str): 
                Name of the attribute where the new parameter is added.
            add_param (ParamType): 
                The parameter to be added.
        """
        self.module_id = module_id
        self.attr_name = attr_name
        self.add_param = add_param
        self.owner_params = None
        
    def set_version_data(self, vdata: VersionDataImp) -> Self:
        return self
    
    def set_particle_data(self, pdata: ParticleDataImp) -> Self:
        self.owner_params = pdata.params
        return self
    
    def run(self, vdata: VersionDataImp) -> tuple[VersionDataImp, ParticleUpdater]:
        def is_in_dict(data, data_dict) -> bool:
            for k, v in data_dict.items():
                if v is data: return True
            return False
            
        if is_in_dict(self.add_param, self.owner_params):
            def value_to_key(data, data_dict):
                for k, v in data_dict.items():
                    if v is data: return k
                raise ValueError(f"can not found {value}")

            pid: ParamID = value_to_key(self.add_param, self.owner_params)
            new_vdata: VersionDataImp = vdata.copy()
            if self.attr_name in new_vdata.meta[self.module_id].sub_modules:
                del new_vdata.meta[self.module_id].sub_modules[self.attr_name]
            new_vdata.meta[self.module_id].sub_params[self.attr_name] = pid
            
            remove_modules: set[ModuleID] = new_vdata.get_garbage_modules()
            new_vdata.remove_modules(remove_modules)
            remove_params: set[ParamID] = new_vdata.get_garbage_params()
            new_vdata.remove_params(remove_params)
            return (new_vdata,
                    AddParamParticleUpdater(self.module_id, 
                                            self.attr_name, 
                                            pid, 
                                            remove_modules, 
                                            remove_params))
        
        def get_next_key(param_dict: ConfigDict) -> ParamID:
            if len(param_dict) == 0:
                return 0
            return max(param_dict.keys()) + 1
        
        next_key: ParamID = get_next_key(vdata.config)
        new_vdata = vdata.copy()
        if self.attr_name in new_vdata.meta[self.module_id].sub_modules:
            del new_vdata.meta[self.module_id].sub_modules[self.attr_name]
        new_vdata.config[next_key] = ParamConfig()
        new_vdata.meta[self.module_id].sub_params[self.attr_name] = next_key
        remove_modules = new_vdata.get_garbage_modules()
        new_vdata.remove_modules(remove_modules)
        remove_params = new_vdata.get_garbage_params()
        new_vdata.remove_params(remove_params)
        return (new_vdata,
                AddParamParticleUpdater(self.module_id, 
                                        self.attr_name, 
                                        next_key, 
                                        remove_modules, 
                                        remove_params, 
                                        self.add_param))


class RegisterBufferVersionUpdater(VersionUpdater[VersionDataImp, ParticleDataImp]):
    r"""
    Updater for VersionKernel instances that registers a buffer to a particle.

    Args:
        module_id (ModuleID): 
            ID of the module where the buffer is added.
        attr_name (str): 
            Name of the attribute where the buffer is added.
        value (Optional[torch.Tensor]): 
            The buffer to be added to the particle.
        persistent (bool): 
            If True, the buffer becomes part of the module and is saved or loaded with it.
            
    Attributes:
        module_id (ModuleID): 
            ID of the module in the particle where the buffer is added.
        attr_name (str): 
            Name of the attribute where the buffer is added.
        value (Optional[torch.Tensor]): 
            The buffer to be added to the particle.
        persistent (bool): 
            If True, the buffer becomes part of the module and is saved or loaded with it.
    
    Example::
        
        class MyNet(NytoModule):
            ...
        
        net = MyNet()
        add_tensor = torch.randn(3, 3)

        particle_kernel: ParticleKernel = net._particle_kernel
        particle_kernel.version_update(RegisterBufferVersionUpdater(net._module_id, 
                                                                    "attar_add_tensor", 
                                                                    add_tensor))
        
        >>> net.attar_add_tensor is add_tensor
        True
    """
    
    module_id: ModuleID
    attr_name: str
    value: Optional[torch.Tensor]
    persistent: bool

    def __init__(self,
                 module_id: ModuleID,
                 attr_name: str,
                 value: Optional[torch.Tensor],
                 persistent: bool) -> None:
        """
        Initialize the RegisterBufferVersionUpdater.

        Args:
            module_id (ModuleID): 
                ID of the module where the buffer is added.
            attr_name (str): 
                Name of the attribute where the buffer is added.
            value (Optional[torch.Tensor]): 
                The buffer to be added to the particle.
            persistent (bool): 
                If True, the buffer becomes part of the module and is saved or loaded with it.
        """
        self.module_id = module_id
        self.attr_name = attr_name
        self.value = value
        self.persistent = persistent
        
    def set_version_data(self, vdata: VersionDataImp) -> Self:
        return self
    
    def set_particle_data(self, pdata: ParticleDataImp) -> Self:
        return self
    
    def run(self, vdata: VersionDataImp) -> tuple[VersionDataImp, ParticleUpdater]:
        new_vdata: VersionDataImp = vdata.copy()
        if self.attr_name in new_vdata.meta[self.module_id].sub_params:
            del new_vdata.meta[self.module_id].sub_params[self.attr_name]
        if self.attr_name in new_vdata.meta[self.module_id].sub_modules:
            del new_vdata.meta[self.module_id].sub_modules[self.attr_name]
        remove_modules: set[ModuleID] = new_vdata.get_garbage_modules()
        new_vdata.remove_modules(remove_modules)
        remove_params: set[ParamID] = new_vdata.get_garbage_params()
        new_vdata.remove_params(remove_params)
        return (new_vdata,
                RegisterBufferParticleUpdater(self.module_id, 
                                              self.attr_name,
                                              self.value,
                                              self.persistent,
                                              remove_modules,
                                              remove_params))


class SetModuleNoneVersionUpdater(VersionUpdater[VersionDataImp, ParticleDataImp]):
    r"""
    Updater for VersionKernel instances that sets a module property in a particle to None.

    Args:
        module_id (ModuleID): 
            ID of the module performing this operation within the particle.
        attr_name (str): 
            Name of the attribute being operated on.
            
    Attributes:
        module_id (ModuleID): 
            ID of the module performing this operation within the particle.
        attr_name (str): 
            Name of the attribute being operated on.
    
    Example::
        
        class MyNet(NytoModule):
            ...
        
        net = MyNet()
        net.attar_module = MyNet()
        particle_kernel: ParticleKernel = net._particle_kernel
        particle_kernel.version_update(SetModuleNoneVersionUpdater(net._module_id, 
                                                                   "attar_module"))
        
        >>> net.attar_module is None
        True
    """

    module_id: ModuleID
    attr_name: str
    
    def __init__(self, module_id: ModuleID, attr_name: str) -> None:
        r"""
        Initialize the SetModuleNoneVersionUpdater.

        Args:
            module_id (ModuleID): 
                ID of the module performing this operation within the particle.
            attr_name (str): 
                Name of the attribute being operated on.
        """
        self.module_id = module_id
        self.attr_name = attr_name
        
    def set_version_data(self, vdata: VersionDataImp) -> Self:
        return self
    
    def set_particle_data(self, pdata: ParticleDataImp) -> Self:
        return self
        
    def run(self, vdata: VersionDataImp) -> tuple[VersionDataImp, ParticleUpdater]:
        new_vdata: VersionDataImp = vdata.copy()
        if self.attr_name in new_vdata.meta[self.module_id].sub_params:
            del new_vdata.meta[self.module_id].sub_params[self.attr_name]
        new_vdata.meta[self.module_id].sub_modules[self.attr_name] = None
        remove_modules: set[ModuleID] = new_vdata.get_garbage_modules()
        new_vdata.remove_modules(remove_modules)
        remove_params: set[ParamID] = new_vdata.get_garbage_params()
        new_vdata.remove_params(remove_params)
        return (new_vdata,
                SetModuleNoneParticleUpdater(self.module_id,
                                             self.attr_name, 
                                             remove_modules, 
                                             remove_params))


class SetParamNoneVersionUpdater(VersionUpdater[VersionDataImp, ParticleDataImp]):
    r"""
    Updater for VersionKernel instances that sets a param property in a particle to None.

    Args:
        module_id (ModuleID): 
            ID of the module performing this operation within the particle.
        attr_name (str): 
            Name of the attribute being operated on.
            
    Attributes:
        module_id (ModuleID): 
            ID of the module performing this operation within the particle.
        attr_name (str): 
            Name of the attribute being operated on.
        
    Example::
        
        class MyNet(NytoModule):
            ...
        
        net = MyNet()
        net.attar_param = torch.nn.Parameter(torch.randn(3, 3))
        particle_kernel: ParticleKernel = net._particle_kernel
        particle_kernel.version_update(SetParamNoneVersionUpdater(net._module_id, 
                                                                  "attar_param"))
        
        >>> net.attar_param is None
        True
    """
    
    module_id: ModuleID
    attr_name: str
    
    def __init__(self, module_id: ModuleID, attr_name: str) -> None:
        r"""
        Initialize the SetParamNoneVersionUpdater.

        Args:
            module_id (ModuleID): 
                ID of the module performing this operation within the particle.
            attr_name (str): 
                Name of the attribute being operated on.
        """
        self.module_id = module_id
        self.attr_name = attr_name
    
    def set_version_data(self, vdata: VersionDataImp) -> Self:
        return self
    
    def set_particle_data(self, pdata: ParticleDataImp) -> Self:
        return self
    
    def run(self, vdata: VersionDataImp) -> tuple[VersionDataImp, ParticleUpdater]:
        new_vdata: VersionDataImp = vdata.copy()
        if self.attr_name in new_vdata.meta[self.module_id].sub_modules:
            del new_vdata.meta[self.module_id].sub_modules[self.attr_name]
        new_vdata.meta[self.module_id].sub_params[self.attr_name] = None
        remove_modules: set[ModuleID] = new_vdata.get_garbage_modules()
        new_vdata.remove_modules(remove_modules)
        remove_params: set[ParamID] = new_vdata.get_garbage_params()
        new_vdata.remove_params(remove_params)
        return (new_vdata, 
                SetParamNoneParticleUpdater(self.module_id, 
                                            self.attr_name, 
                                            remove_modules,
                                            remove_params))
    
class DelModuleVersionUpdater(VersionUpdater[VersionDataImp, ParticleDataImp]):
    r"""
    Updater for VersionKernel instances that deletes a module attribute from a particle.

    Args:
        module_id (ModuleID): 
            ID of the module performing this operation within the particle.
        attr_name (str): 
            Name of the attribute being deleted.
            
    Attributes:
        module_id (ModuleID): 
            ID of the module in particle where this operation is executed.
        attr_name (str): 
            Name of the attribute to be deleted.
    
    Example::
        
        class MyNet(NytoModule):
            ...
        
        net = MyNet()
        net.attar_moduel = MyNet()
        particle_kernel: ParticleKernel = net._particle_kernel
        particle_kernel.version_update(DelModuleVersionUpdater(net._module_id, 
                                                               "attar_moduel"))
        
        >>> hasattr(net, "attar_moduel")
        False
    """
    
    module_id: ModuleID
    attr_name: str
        
    def __init__(self, module_id: ModuleID, attr_name: str) -> None:
        """
        Initialize the DelModuleVersionUpdater.

        Args:
            module_id (ModuleID): 
                ID of the module performing this operation within the particle.
            attr_name (str): 
                Name of the attribute being deleted.
        """
        self.module_id = module_id
        self.attr_name = attr_name
        
    def set_version_data(self, vdata: VersionDataImp) -> Self:
        return self
    
    def set_particle_data(self, pdata: ParticleDataImp) -> Self:
        return self
        
    def run(self, vdata: VersionDataImp) -> tuple[VersionDataImp, ParticleUpdater]:
        new_vdata: VersionDataImp = vdata.copy()
        del new_vdata.meta[self.module_id].sub_modules[self.attr_name]
        remove_modules: set[ModuleID] = new_vdata.get_garbage_modules()
        new_vdata.remove_modules(remove_modules)
        remove_params: set[ParamID] = new_vdata.get_garbage_params()
        new_vdata.remove_params(remove_params)
        return (new_vdata,
                DelModuleParticleUpdater(self.module_id, 
                                         self.attr_name, 
                                         remove_modules, 
                                         remove_params))


class DelParamVersionUpdater(VersionUpdater[VersionDataImp, ParticleDataImp]):
    r"""
    Updater for VersionKernel instances that deletes a parameter attribute from a particle.

    Args:
        module_id (ModuleID): 
            ID of the module performing this operation within the particle.
        attr_name (str): 
            Name of the attribute being deleted.
            
    Attributes:
        module_id (ModuleID): 
            ID of the module in particle where this operation is executed.
        attr_name (str): 
            Name of the attribute to be deleted.
    
    Example::
        
        class MyNet(NytoModule):
            ...
        
        net = MyNet()
        net.attar_param = torch.nn.Parameter(torch.randn(3, 3))
        particle_kernel: ParticleKernel = net._particle_kernel
        particle_kernel.version_update(DelParamVersionUpdater(net._module_id, 
                                                              "attar_moduel"))
        
        >>> hasattr(net, "attar_param")
        False
    """
    
    module_id: ModuleID
    attr_name: str

    def __init__(self, module_id: ModuleID, attr_name: str) -> None:
        """
        Initialize the DelParamVersionUpdater.

        Args:
            module_id (ModuleID): 
                ID of the module performing this operation within the particle.
            attr_name (str): 
                Name of the attribute being deleted.
        """
        self.module_id = module_id
        self.attr_name = attr_name
    
    def set_version_data(self, vdata: VersionDataImp) -> Self:
        return self
    
    def set_particle_data(self, pdata: ParticleDataImp) -> Self:
        return self
    
    def run(self, vdata: VersionDataImp) -> tuple[VersionDataImp, ParticleUpdater]:
        new_vdata: VersionDataImp = vdata.copy()
        del new_vdata.meta[self.module_id].sub_params[self.attr_name]
        remove_params: set[ParamID] = new_vdata.get_garbage_params()
        new_vdata.remove_params(remove_params)
        return (new_vdata, 
                DelParamParticleUpdater(self.module_id, 
                                        self.attr_name, 
                                        remove_params))

                
class DelBufferVersionUpdater(VersionUpdater[VersionDataImp, ParticleDataImp]):
    r"""
    Updater for VersionKernel instances that deletes a buffer attribute from a particle.

    Args:
        module_id (ModuleID): 
            ID of the module performing this operation within the particle.
        attr_name (str): 
            Name of the attribute being deleted.
        
    Attributes:
        module_id (ModuleID): 
            ID of the module in particle where this operation is executed.
        attr_name (str): 
            Name of the attribute to be deleted.
    
    Example::
        
        class MyNet(NytoModule):
            ...
        
        net = MyNet()
        net.register_buffer("attar_buffer", torch.randn(3, 3))
        particle_kernel: ParticleKernel = net._particle_kernel
        particle_kernel.version_update(DelParamVersionUpdater(net._module_id, 
                                                              "attar_buffer"))
        
        >>> hasattr(net, "attar_buffer")
        False
    """

    module_id: ModuleID
    attr_name: str
        
    def __init__(self, module_id: ModuleID, attr_name: str) -> None:
        """
        Initialize the DelBufferVersionUpdater.

        Args:
            module_id (ModuleID): 
                ID of the module performing this operation within the particle.
            attr_name (str): 
                Name of the attribute being deleted.
        """
        self.module_id = module_id
        self.attr_name = attr_name
    
    def set_version_data(self, vdata: VersionDataImp) -> Self:
        return self
    
    def set_particle_data(self, pdata: ParticleDataImp) -> Self:
        return self
                
    def run(self, vdata: VersionDataImp) -> tuple[VersionDataImp, ParticleUpdater]:
        new_vdata: VersionDataImp = vdata.copy()
        return (new_vdata, 
                DelBufferParticleUpdater(self.module_id, 
                                         self.attr_name))
    
