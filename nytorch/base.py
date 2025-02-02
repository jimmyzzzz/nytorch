from __future__ import annotations
from collections import OrderedDict
from .kernel import ParticleData, ParticleKernel, VersionData, VersionKernel
from .mtype import ConfigDict, MetaDict, Module, ModuleDict, ModuleID, ModuleMeta, ParamDict, ParamID, ParamType, ROOT_MODULE_ID
from .utils import copy_modules, make_modules_ref
from typing import Optional, Type


VersionKernelImp = VersionKernel['VersionDataImp', 'ParticleDataImp']
ParticleKernelImp = ParticleKernel['VersionDataImp', 'ParticleDataImp']


def create_new_group(ref: Module) -> ParticleKernelImp:
    """
    Create a new version kernel and a new particle kernel.

    .. note::
    
        The passed `ref` Module must be in a state where metadata is empty,
        meaning it should not have any parameters or submodules.

    Args:
        ref (Module): The reference module to base the new group on.

    Returns:
        ParticleKernelImp: The created particle kernel.
    """
    assert len(list(ref.children())) == 0
    assert len(list(ref.parameters())) == 0
    version_kernel: VersionKernelImp = VersionKernel(VersionDataImp(OrderedDict([(ROOT_MODULE_ID,
                                                                                  ModuleMeta(OrderedDict(),
                                                                                             OrderedDict()))]),
                                                                    OrderedDict()))
    return ParticleKernel(version_kernel,
                          ParticleDataImp(OrderedDict([(ROOT_MODULE_ID, ref)]),
                                          OrderedDict()))

class NytoModuleBase(Module):
    """
    Base class for NytoModule.

    Acts as an interface for `ParticleDataImp` and various updaters.

    Attributes:
        _module_id (ModuleID): 
            ID of the module.
        _particle_kernel (Optional[ParticleKernelImp]): 
            Points to the particle kernel; set to None when wrapped by ParticleModule.
    """
    
    _module_id: ModuleID
    _particle_kernel: Optional[ParticleKernelImp]
    
    def __init__(self) -> None:
        """
        Initialize a NytoModuleBase instance.
        """
        super().__init__()
        self._module_id = 0
        self._particle_kernel = create_new_group(self)
    
    @property
    def _version_kernel(self) -> VersionKernelImp:
        """
        The version kernel pointed by the particle kernel.

        Returns:
            VersionKernelImp: The version kernel.
        
        Raises:
            AssertionError: If `_particle_kernel` is None.
        """
        assert self._particle_kernel is not None
        return self._particle_kernel.version
    
    @property
    def is_root(self) -> bool:
        """
        Check if the current module is the root module.

        Returns:
            bool: True if the module is the root module, False otherwise.
        """
        return self._module_id == ROOT_MODULE_ID


class VersionDataImp(VersionData['VersionDataImp', 'ParticleDataImp']):
    """
    Implementation of VersionData.

    This class stores metadata of particles, or more precisely, metadata of species where particles belong.
    When particles undergo particle operations, this instance needs to be accessed.
    Instances are stored in `VersionKernel.data`.

    The content of an instance is immutable once determined,
    and a copy will be made when new metadata for particles is needed,
    and modifications will be made on the new instance.

    Args:
        meta (MetaDict): 
            Stores references of modules to modules and parameters within modules.
        config (ConfigDict): 
            Stores settings of particle operations for all parameters.
        
    Attributes:
        meta (MetaDict): 
            Stores references of modules to modules and parameters within modules.
        config (ConfigDict): 
            Stores settings of particle operations for all parameters.
    """
    
    __slots__ = "meta", "config"
    
    meta: MetaDict
    config: ConfigDict
    
    def __init__(self, meta: MetaDict, config: ConfigDict) -> None:
        """
        Initialize VersionDataImp with meta and config.

        Args:
            meta (MetaDict): 
                Stores references of modules to modules and parameters within modules.
            config (ConfigDict): 
                Stores settings of particle operations for all parameters.
        """
        self.meta = meta
        self.config = config
        
    def __repr__(self) -> str:
        return f"VersionDataImp(meta={self.meta}, config={self.config})"
    
    def init_kernel(self, kernel: VersionKernel) -> None:
        return
    
    def init_particle(self, pdata: ParticleDataImp) -> None:
        if self.meta.keys() != pdata.modules.keys():
            pdata.modules = OrderedDict((mid, pdata.modules[mid]) for mid in self.meta)
        if self.config.keys() != pdata.params.keys():
            pdata.params = OrderedDict((pid, pdata.params[pid]) for pid in self.config)
    
    def copy(self) -> VersionDataImp:
        return VersionDataImp(
            OrderedDict((k, v.copy()) for k, v in self.meta.items()),
            OrderedDict((k, v.copy()) for k, v in self.config.items()))
    
    def get_sub_modules(self, root: ModuleID) -> set[ModuleID]:
        """
        Retrieve all module IDs (including the root) under the target module.

        Args:
            root (ModuleID): The ModuleID of the target module.

        Returns:
            Set[ModuleID]: A set of all module IDs under the target module.
        """
        visited_modules: set[ModuleID] = set()
        def preorder_traversal(mid: ModuleID):
            if mid in visited_modules: return
            visited_modules.add(mid)
            for sub_mid in self.meta[mid].sub_modules.values():
                if sub_mid is None: continue
                preorder_traversal(sub_mid)
        
        preorder_traversal(root)
        return visited_modules
    
    def get_sub_params(self, root: ModuleID) -> set[ParamID]:
        """
        Retrieve all parameter IDs (including the root) under the target module.

        Args:
            root (ModuleID): The ModuleID of the target module.

        Returns:
            Set[ParamID]: A set of all parameter IDs under the target module.
        """
        sub_modules: set[ModuleID] = self.get_sub_modules(root)
        sub_params: set[ParamID] = set()
        for mid, mmeta in self.meta.items():
            if mid in sub_modules:
                sub_params |= set(pid for pid in mmeta.sub_params.values() if pid is not None)
        return sub_params
    
    def get_garbage_modules(self) -> set[ModuleID]:
        """
        Retrieve module IDs that are unreachable from the root module.

        Returns:
            Set[ModuleID]: A set of module IDs that are considered garbage.
        """
        sub_modules: set[ModuleID] = self.get_sub_modules(ROOT_MODULE_ID)
        return set(self.meta.keys()) - sub_modules
    
    def get_garbage_params(self) -> set[ParamID]:
        """
        Retrieve parameter IDs that are unreachable from the root module.

        Returns:
            Set[ParamID]: A set of parameter IDs that are considered garbage.
        """
        sub_params: set[ParamID] = self.get_sub_params(ROOT_MODULE_ID)
        return set(self.config.keys()) - sub_params
    
    def remove_modules(self, modules: set[ModuleID]) -> None:
        """
        Remove metadata of specified modules.

        Args:
            modules (Set[ModuleID]): A set of specified module IDs to remove.
        """
        for mid in modules: del self.meta[mid]
        
    def remove_params(self, params: set[ParamID]) -> None:
        """
        Remove metadata of specified parameters.

        Args:
            params (Set[ParamID]): A set of specified parameter IDs to remove.
        """
        for pid in params: del self.config[pid]
        
        
class ParticleDataImp(ParticleData["VersionDataImp", "ParticleDataImp"]):
    """
    Implementation of ParticleData.

    This class stores references to all parameters of corresponding particles and is accessed when particles undergo particle operations. 
    Instances are stored in `ParticleKernel.data`.

    Args:
        modules (ModuleDict): 
            Contains references to all modules within particles.
        params (ParamDict): 
            Contains references to all parameters within particles.

    Attributes:
        modules (ModuleDict): 
            Contains references to all modules within particles.
        params (ParamDict): 
            Contains references to all parameters within particles.
    """
    
    __slots__ = "modules", "params"
    
    modules: ModuleDict
    params: ParamDict
    
    def __init__(self, modules: ModuleDict, params: ParamDict) -> None:
        """
        Initialize ParticleDataImp with modules and params.

        Args:
            modules (ModuleDict): 
                Contains references to all modules within particles.
            params (ParamDict): 
                Contains references to all parameters within particles.
        """
        self.modules = modules
        self.params = params
        
    def __repr__(self) -> str:
        return f"ParticleDataImp(modules={self.modules}, params={self.params})"
        
    def init_kernel(self, kernel: ParticleKernel) -> None:
        for mod in self.modules.values():
            if isinstance(mod, NytoModuleBase):
                mod._particle_kernel = kernel
                
    def copy(self, vdata: VersionDataImp) -> ParticleDataImp:
        return self.create(vdata, self.params)
    
    def add_modules(self, modules: ModuleDict) -> None:
        """
        Add modules to the current ParticleDataImp instance.

        The keys in the passed `modules` should not already exist. 
        Offsets can be applied to all keys in `modules` if needed.

        Args:
            modules (ModuleDict): The modules to be added.
        """
        assert len(set(self.modules.keys()) & set(modules.keys())) == 0
        self.modules.update(modules)
        
    def add_params(self, params: ParamDict) -> None:
        """
        Add params to the current ParticleDataImp instance.

        The keys in the passed `params` should not already exist. 
        Offsets can be applied to all keys in `params` if needed.

        Args:
            params (ParamDict): The params to be added.
        """
        assert len(set(self.params.keys()) & set(params.keys())) == 0
        self.params.update(params)
        
    def remove_modules(self, modules: set[ModuleID]) -> None:
        """
        Remove modules from the current ParticleDataImp instance.

        Args:
            modules (set[ModuleID]): The set of ModuleIDs to be removed.
        """
        for mid in modules: del self.modules[mid]
        
    def remove_params(self, params: set[ParamID]) -> None:
        """
        Remove params from the current ParticleDataImp instance.

        Args:
            params (set[ParamID]): The set of ParamIDs to be removed.
        """
        for pid in params: del self.params[pid]
        
    def assign(self, params: ParamDict) -> None:
        """
        Replace current params with the given params.

        The keys in the passed `params` must match the current params.

        Args:
            params (ParamDict): The new params to be assigned.
        """
        assert params.keys() == self.params.keys()
        for lp, rp in zip(self.params.values(), params.values()):
            if rp is None or lp is None: continue
            lp.data = rp.data
        
    def create(self, vdata: VersionDataImp, params: ParamDict) -> ParticleDataImp:
        """
        Create a new ParticleDataImp with the same structure but different params.

        Args:
            vdata (VersionDataImp): 
                Used to guide the establishment of references between modules and params.
            params (ParamDict): 
                Params for the new particle.

        Returns:
            ParticleDataImp: The newly created ParticleDataImp instance.
        """
        modules_copy = copy_modules(self.modules)
        make_modules_ref(modules_copy, params, vdata.meta, set(modules_copy.keys()))
        return ParticleDataImp(modules_copy, params)
