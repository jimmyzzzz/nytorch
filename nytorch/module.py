from __future__ import annotations
from collections import OrderedDict
from collections.abc import Iterable
from .base import NytoModuleBase, ParticleDataImp, ParticleKernalImp, VersionDataImp
from .kernal import Particle, ParticleData, ParticleUpdater, Product, VersionData, VersionUpdater
from .mtype import ConfigDict, MetaDict, Module, ModuleDict, ModuleID, ModuleMeta, ParamConfig, ParamDict, ParamID, ParamType, ROOT_MODULE_ID
from .version_updater import AddModuleVersionUpdater, AddParamVersionUpdater, DelBufferVersionUpdater, DelModuleVersionUpdater, DelParamVersionUpdater, SetModuleNoneVersionUpdater, SetParamNoneVersionUpdater, RegisterBufferVersionUpdater
from torch import nn
from typing import Any, Callable, Generic, Optional, TypeVar
from typing_extensions import Self
import torch
import warnings


Tmodule = TypeVar("Tmodule", bound="NytoModule")


def unary_lambda_wrapper(fn:  Callable[[ParamType], torch.Tensor]) -> Callable[[ParamType, ParamConfig], ParamType]:
    """
    Wraps a unary function for parameter operations.

    Args:
        fn (Callable[[ParamType], torch.Tensor]): Unary operation function.

    Returns:
        Callable[[ParamType, ParamConfig], ParamType]: Wrapped unary function.
    """
    return (lambda p, pconf: 
            ParamType(fn(p), requires_grad=p.requires_grad)
            if pconf.operational else
            ParamType(p.clone(), requires_grad=p.requires_grad)
            if pconf.clone else p)


def binary_lambda_wrapper(fn:  Callable[[ParamType, ParamType], torch.Tensor]) -> Callable[[ParamType, ParamType, ParamConfig], ParamType]:
    """
    Wraps a binary function for parameter operations.

    Args:
        fn (Callable[[ParamType, ParamType], torch.Tensor]): Binary operation function.

    Returns:
        Callable[[ParamType, ParamType, ParamConfig], ParamType]: Wrapped binary function.
    """
    return (lambda lp, rp, pconf:
            ParamType(fn(lp, rp), requires_grad=lp.requires_grad)
            if pconf.operational else
            ParamType(lp.clone(), requires_grad=lp.requires_grad)
            if pconf.clone else lp)


def module_to_product(module: Tmodule) -> ParamProduct:
    """
    Transform a NytoModule instance into a ParamProduct instance.

    Args:
        module (Tmodule): The NytoModule instance to transform.

    Returns:
        ParamProduct: The corresponding ParamProduct instance.
    
    Raises:
        AssertionError: If module._particle_kernal is None.
    """
    assert module._particle_kernal is not None
    pdata: ParticleDataImp = module._particle_kernal.data
    return ParamProduct(module._particle_kernal, module._module_id, pdata.params)


def product_to_module(product: ParamProduct) -> Module:
    """
    Transform a ParamProduct instance back into a NytoModule instance.

    Args:
        product (ParamProduct): The ParamProduct instance to transform.

    Returns:
        Module: The corresponding NytoModule instance.
    """
    new_data: ParticleDataImp = product.pdata.create(product.vdata, product.params)
    new_particle: ParticleKernalImp = product.kernal.create(new_data)
    return new_particle.data.modules[product.module_id]


def product_assign_to_module(product: ParamProduct, module: Tmodule) -> None:
    """
    Assign parameters from a ParamProduct instance to a NytoModule instance.

    Args:
        product (ParamProduct): 
            The ParamProduct instance containing parameters to assign.
        module (Tmodule): 
            The NytoModule instance to assign parameters to.
    
    Raises:
        AssertionError: If product.kernal is not module._version_kernal or if module._particle_kernal is None.
    """
    assert product.kernal.version is module._version_kernal
    assert module._particle_kernal is not None
    pdata: ParticleDataImp = module._particle_kernal.data
    return pdata.assign(product.params)


class ParamProduct(Product[Tmodule]):
    """
    Implementation of Product for NytoModule instances.
    
    Implements particle operations and transformation to NytoModule.

    Args:
        kernal (ParticleKernalImp): 
            Reference to the particle kernel.
        module_id (ModuleID): 
            Specifies which module to return when transforming to NytoModule.
        params (ParamDict): 
            Model parameters for particle operations.
            
    Attributes:
        kernal (ParticleKernalImp): 
            Reference to the particle kernel.
        module_id (ModuleID): 
            Specifies which module to return when transforming to NytoModule.
        params (ParamDict): 
            Model parameters for particle operations.
    """
    
    __slots__ = "kernal", "module_id", "params"
    
    kernal: ParticleKernalImp
    module_id: ModuleID
    params: ParamDict

    def __init__(self, kernal: ParticleKernalImp, module_id: ModuleID, params: ParamDict) -> None:
        """
        Initialize ParamProduct with kernal, module_id, and params.

        Args:
            kernal (ParticleKernalImp): 
                Reference to the particle kernel.
            module_id (ModuleID): 
                Specifies which module to return when transforming to NytoModule.
            params (ParamDict): 
                Model parameters for particle operations.
        """

        self.kernal = kernal
        self.module_id = module_id
        self.params = params

    def __repr__(self) -> str:
        return f"ParamProduct({self.params})"

    @property
    def vdata(self) -> VersionDataImp:
        """Version data of the particle."""
        vdata: VersionDataImp = self.kernal.version.data
        return vdata

    @property
    def pdata(self) -> ParticleDataImp:
        """Parameter data of the particle."""
        pdata: ParticleDataImp = self.kernal.data
        return pdata

    def particle(self) -> Tmodule:  # type: ignore
        """Transforms into the corresponding NytoModule subclass."""
        module: Module = product_to_module(self)
        assert isinstance(module, NytoModule)
        return module  # type: ignore

    def module(self) -> Tmodule:
        """Transforms into the corresponding NytoModule subclass."""
        return self.particle()

    def unary_operator(self, fn: Callable[[ParamType, ParamConfig], ParamType]) -> ParamProduct:
        r"""
        Perform a unary operation on the parameters.
        
        Args:
            fn (Callable[[ParamType, ParamConfig], ParamType]): Unary operation logic.

        Returns:
            ParamProduct: New ParamProduct instance after the unary operation.
        
        .. note::
            
            In writing the function ``fn``, 
            gradient calculation does not need to be disabled
            because ``torch.no_grad`` is used within the ``unary_operator()`` method to disable gradient calculation.
        
        Example::
        
            class Net(NytoModule):
                def __init__(self):
                    super().__init__()
                    self.my_param = nn.Parameter(torch.Tensor([0., 1., 2.]))

            net = Net()
            product = net.product()
            new_product = product.unary_operator(lambda param, conf: nn.Parameter(param+10.))
            new_net = new_product.module()
            
            >>> new_net.my_param
            Parameter containing:
            tensor([10., 11., 12.], requires_grad=True)
        """

        with torch.no_grad():
            new_params: ParamDict = OrderedDict(
                (pid, fn(p, pconf)) for (pid, p), pconf in zip(
                    self.params.items(),
                    self.vdata.config.values()))
            return ParamProduct(self.kernal, self.module_id, new_params)

    def binary_operator(self,
                        other: ParamProduct,
                        fn: Callable[[ParamType, ParamType, ParamConfig], ParamType]) -> ParamProduct:
        r"""
        
        Perform a binary operation between two ParamProduct instances.

        Args:
            other (ParamProduct): 
                Another ParamProduct instance involved in the binary operation.
            fn (Callable[[ParamType, ParamType, ParamConfig], ParamType]): 
                Binary operation logic.

        Returns:
            ParamProduct: New ParamProduct instance after the binary operation.

        Raises:
            AssertionError: If other.kernal.version is not self.kernal.version.
        
        .. note::
            
            In writing the function ``fn``, 
            gradient calculation does not need to be disabled
            because ``torch.no_grad`` is used within the ``unary_operator()`` method to disable gradient calculation.
            
        .. note::
            
            The source of ``other`` must belong to the same species as ``self``,
            which can be checked as follows::
            
                assert other.kernal.version is self.kernal.version
        
        Example::
        
            class Net(NytoModule):
                def __init__(self, my_tensor):
                    super().__init__()
                    self.my_param = nn.Parameter(my_tensor)

            net1 = Net(torch.Tensor([0., 1., 2.]))
            net2 = net1.clone_from(Net(torch.Tensor([5., 4., 3.])))
            product1 = net1.product()
            product2 = net2.product()
            
            fn = lambda param1, param2, conf: nn.Parameter(param1-param2)
            new_product = product1.binary_operator(product2, fn)
            new_net = new_product.module()

            >>> new_net.my_param
            Parameter containing:
            tensor([-5., -3., -1.], requires_grad=True)
        """
        assert isinstance(other, ParamProduct)
        assert self.kernal.version is other.kernal.version
        with torch.no_grad():
            new_params: ParamDict = OrderedDict(
                (pid, fn(lp, rp, pconf)) for (pid, lp), rp, pconf in zip(
                    self.params.items(),
                    other.params.values(),
                    self.vdata.config.values()))
            return ParamProduct(self.kernal, self.module_id, new_params)

    def __neg__(self) -> ParamProduct:
        return self.unary_operator(unary_lambda_wrapper(lambda p: -p))

    def __pow__(self, power) -> ParamProduct:
        assert not isinstance(power, NytoModule)
        if isinstance(power, ParamProduct):
            return self.binary_operator(power, binary_lambda_wrapper(lambda lp, rp: lp**rp))
        return self.unary_operator(unary_lambda_wrapper(lambda p: p**power))

    def __rpow__(self, base) -> ParamProduct:
        assert not isinstance(base, NytoModule)
        return self.unary_operator(unary_lambda_wrapper(lambda p: base**p))

    def __add__(self, other) -> ParamProduct:
        assert not isinstance(other, NytoModule)
        if isinstance(other, ParamProduct):
            return self.binary_operator(other, binary_lambda_wrapper(lambda lp, rp: lp+rp))
        return self.unary_operator(unary_lambda_wrapper(lambda p: p+other))

    def __sub__(self, other) -> ParamProduct:
        assert not isinstance(other, NytoModule)
        if isinstance(other, ParamProduct):
            return self.binary_operator(other, binary_lambda_wrapper(lambda lp, rp: lp-rp))
        return self.unary_operator(unary_lambda_wrapper(lambda p: p-other))

    def __rsub__(self, other) -> ParamProduct:
        assert not isinstance(other, NytoModule)
        return self.unary_operator(unary_lambda_wrapper(lambda p: other-p))

    def __mul__(self, other) -> ParamProduct:
        assert not isinstance(other, NytoModule)
        if isinstance(other, ParamProduct):
            return self.binary_operator(other, binary_lambda_wrapper(lambda lp, rp: lp*rp))
        return self.unary_operator(unary_lambda_wrapper(lambda p: p*other))

    def __truediv__(self, other) -> ParamProduct:
        assert not isinstance(other, NytoModule)
        if isinstance(other, ParamProduct):
            return self.binary_operator(other, binary_lambda_wrapper(lambda lp, rp: lp/rp))
        return self.unary_operator(unary_lambda_wrapper(lambda p: p/other))

    def __rtruediv__(self, other) -> ParamProduct:
        assert not isinstance(other, NytoModule)
        return self.unary_operator(unary_lambda_wrapper(lambda p: other/p))

    def clone(self) -> ParamProduct:
        return self.unary_operator(unary_lambda_wrapper(lambda p: p.clone()))

    def randn(self) -> ParamProduct:
        return self.unary_operator(unary_lambda_wrapper(lambda p: torch.randn_like(p)))

    def rand(self) -> ParamProduct:
        return self.unary_operator(unary_lambda_wrapper(lambda p: torch.rand_like(p)))


def _get_module_id(vdata: VersionDataImp, module_id: ParamID, name: str) -> ModuleID:
    """
    Retrieve the module ID from version data by name.

    Args:
        vdata (VersionDataImp): 
            The version data.
        module_id (ParamID): 
            The ID of the module.
        name (str): 
            The name of the attribute.

    Returns:
        ModuleID: The ID of the module with the given name.

    Raises:
        ValueError: If the attribute name is None.
    """
    mid: Optional[ModuleID] = vdata.meta[module_id].sub_modules[name]
    if mid is None:
        raise ValueError(f"attar name {name} is None")
    return mid

def _get_param_id(vdata: VersionDataImp, module_id: ParamID, name: str) -> ParamID:
    """
    Retrieve the parameter ID from version data by name.

    Args:
        vdata (VersionDataImp): 
            The version data.
        module_id (ParamID): 
            The ID of the module.
        name (str): 
            The name of the attribute.

    Returns:
        ParamID: The ID of the parameter with the given name.

    Raises:
        ValueError: If the attribute name is None.
    """
    pid: Optional[ParamID] =  vdata.meta[module_id].sub_params[name]
    if pid is None:
        raise ValueError(f"attar name {name} is None")
    return pid

def _apply_module_config(vdata: VersionDataImp, 
                         module_id: ModuleID, 
                         fn: Callable[[ParamID, ParamConfig], None]) -> None:
    """
    Apply configuration to modules based on version data.

    Args:
        vdata (VersionDataImp): 
            The version data.
        module_id (ModuleID): 
            The ID of the module.
        fn (Callable[[ParamID, ParamConfig], None]): 
            The function to apply.

    Example:
        class Net(NytoModule):
            def __init__(self):
                super().__init__()
                
        net = Net()
        _apply_module_config(version_data, net._module_id, lambda pid, pconf: net.set_parameter(pid, pconf))
    """
    sub_params: set[ParamID] = vdata.get_sub_params(module_id)
    for pid, pconf in vdata.config.items():
        if pid in sub_params:
            fn(pid, pconf)


def _apply_param_config(vdata: VersionDataImp, 
                        param_id: ParamID,
                        fn: Callable[[ParamID, ParamConfig], None]) -> None:
    """
    Apply configuration to parameters based on version data.

    Args:
        vdata (VersionDataImp): 
            The version data.
        param_id (ParamID): 
            The ID of the parameter.
        fn (Callable[[ParamID, ParamConfig], None]): 
            The function to apply.
    """
    fn(param_id, vdata.config[param_id])


class NytoModule(NytoModuleBase, Particle[ParamProduct]):
    r"""Implementation of NytoModuleBase, compatible with torch.nn.Module functionality.

    Features:
        * Particle operations.
        * Transformation to ParamProduct.
        * Version management.
    
    Example1: Particle Operations::
    
        class Net(NytoModule):
            def __init__(self):
                super().__init__()
                self.my_param = nn.Parameter(torch.Tensor([0., 1., 2.]))
                
        net = Net()
        new_net = 2 * net
        
        >>> new_net.my_param
        Parameter containing:
        tensor([0., 2., 4.], requires_grad=True)
    
    Example2: Transform to ParamProduct::
    
        class Net(NytoModule):
            def __init__(self):
                super().__init__()
                self.my_param = nn.Parameter(torch.Tensor([0., 1., 2.]))
                
        net = Net()
        product = net.product()
        assert isinstance(product, ParamProduct)
    
    Example3: Version Management::
        
        class Net(NytoModule):
            def __init__(self):
                super().__init__()
                self.my_param = nn.Parameter(torch.Tensor([0., 1., 2.]))
                
        net = Net()
        new_net = 2 * net
        new_net.my_param2 = nn.Parameter(torch.Tensor([3., 4.]))
        net.touch()
        
        >>> net.my_param2
        Parameter containing:
        tensor([3., 4.], requires_grad=True)
    """
    
    def __setattr__(self: Tmodule, name: str, value: Any) -> None:
        if isinstance(value, NytoModuleBase):
            if self._particle_kernal is value._particle_kernal and value._module_id == ROOT_MODULE_ID:
                warnings.warn("circular references to the root module")
            if self._particle_kernal is not value._particle_kernal and value._module_id != ROOT_MODULE_ID:
                warnings.warn("modules from different particles refer to the same module")

        if isinstance(value, ParamType):
            return self.register_parameter(name, value)
        if isinstance(value, Module):
            return self.add_module(name, value)

        # value is None
        if name in self.__dict__.get('_parameters', {}):
            return self.register_parameter(name, value)
        if name in self.__dict__.get('_modules', {}):
            return self.add_module(name, value)
        if name in self.__dict__.get('_buffers', {}):
            return self.register_buffer(name, value)

        return super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in self._modules:
            self._version_del_module(name)
        elif name in self._parameters:
            self._version_del_param(name)
        elif name in self._buffers:
            self._version_del_buffer(name)
        else:
            super().__delattr__(name)

    def _version_add_module(self, attr_name: str, module: Module) -> None:
        r"""
        Update version: add new module.

        Args:
            attr_name (str): 
                Name of the module attribute.
            module (Module): 
                Module to add.
        
        Example::
        
            class Net(NytoModule):
                def __init__(self):
                    super().__init__()
                    
            net = Net()
            net._version_add_module("my_module", nn.Linear(2, 3))
            
            >>> net.my_module
            Linear(in_features=2, out_features=3, bias=True)
        """
        assert isinstance(module, Module)
        assert self._particle_kernal is not None
        self._particle_kernal.version_update(
            AddModuleVersionUpdater(self._module_id, attr_name, module))

    def _version_add_param(self, attr_name: str, param: ParamType) -> None:
        r"""
        Update version: add new parameter.
        
        Args:
            attr_name (str): 
                Name of the parameter attribute.
            module (Module): 
                Parameter to add.
        
        Example::
        
            class Net(NytoModule):
                def __init__(self):
                    super().__init__()
                    
            net = Net()
            net._version_add_param("my_param", torch.nn.Parameter(torch.randn(2, 2)))
            
            >>> net.my_param
            Parameter containing:
            tensor([[-0.4265,  1.4951],
                    [-1.1099, -0.9068]], requires_grad=True)
        """
        assert isinstance(param, ParamType)
        assert self._particle_kernal is not None
        self._particle_kernal.version_update(
            AddParamVersionUpdater(self._module_id, attr_name, param))

    def _version_register_buffer(self, attr_name: str, value: Optional[torch.Tensor], persistent: bool) -> None:
        r"""
        Update version: register new buffer.
        
        Args:
            attr_name (str): 
                Name of the buffer attribute.
            value (Optional[torch.Tensor]): 
                Buffer to register.
            persistent (bool): 
                Whether the buffer is part of this module's ``state_dict``.
        
        Example::
        
            class Net(NytoModule):
                def __init__(self):
                    super().__init__()
                    
            net = Net()
            net._version_register_buffer("my_buffer", torch.randn(2, 2))
            
            >>> net.my_buffer
            tensor([[-0.4265,  1.4951],
                    [-1.1099, -0.9068]])
        """
        if value is not None:
            assert isinstance(value, torch.Tensor)
        assert self._particle_kernal is not None
        self._particle_kernal.version_update(
            RegisterBufferVersionUpdater(self._module_id, attr_name, value, persistent))

    def _version_del_module(self, attr_name: str) -> None:
        r"""Update version: delete module.
        
        Args:
            attr_name (str): Name of the module attribute.
        
        Example::
        
            class Net(NytoModule):
                def __init__(self):
                    super().__init__()
                    self.my_module = nn.Linear(2, 3)
                    
            net = Net()
            net._version_del_module("my_module")
            
            >>> hasattr(net.my_module)
            False
        """
        assert isinstance(attr_name, str)
        assert self._particle_kernal is not None
        self._particle_kernal.version_update(
            DelModuleVersionUpdater(self._module_id, attr_name))

    def _version_del_param(self, attr_name: str) -> None:
        r"""Update version: delete parameter.
        
        Args:
            attr_name (str): Name of the parameter attribute.
        
        Example::
        
            class Net(NytoModule):
                def __init__(self):
                    super().__init__()
                    self.my_param = torch.nn.Parameter(torch.randn(2, 2))
                    
            net = Net()
            net._version_del_param("my_param")
            
            >>> hasattr(net.my_param)
            False
        """
        assert isinstance(attr_name, str)
        assert self._particle_kernal is not None
        self._particle_kernal.version_update(
            DelParamVersionUpdater(self._module_id, attr_name))

    def _version_del_buffer(self, attr_name: str) -> None:
        r"""Update version: delete buffer.
        
        Args:
            attr_name (str): Name of the buffer attribute.
            
        Example::
        
            class Net(NytoModule):
                def __init__(self):
                    super().__init__()
                    self.my_buffer = torch.randn(2, 2)
                    
            net = Net()
            net._version_del_buffer("my_buffer")
            
            >>> hasattr(net.my_buffer)
            False
        """
        assert isinstance(attr_name, str)
        assert self._particle_kernal is not None
        self._particle_kernal.version_update(
            DelBufferVersionUpdater(self._module_id, attr_name))

    def _version_set_none_module(self, attr_name: str) -> None:
        r"""Update version: set module to None.
        
        Args:
            attr_name (str): Name of the module attribute.
        
        Example::
        
            class Net(NytoModule):
                def __init__(self):
                    super().__init__()
                    self.my_module = nn.Linear(2, 3)
                    
            net = Net()
            net._version_set_none_module("my_module")
            
            >>> net.my_module is None
            True
        """
        assert isinstance(attr_name, str)
        assert self._particle_kernal is not None
        self._particle_kernal.version_update(
            SetModuleNoneVersionUpdater(self._module_id, attr_name))

    def _version_set_none_param(self, attr_name: str) -> None:
        r"""Update version: set parameter to None.
        
        Args:
            attr_name (str): Name of the parameter attribute.
        
        Example::
        
            class Net(NytoModule):
                def __init__(self):
                    super().__init__()
                    self.my_param = torch.nn.Parameter(torch.randn(2, 2))
                    
            net = Net()
            net._version_set_none_param("my_param")
            
            >>> net.my_param is None
            True
        """
        assert isinstance(attr_name, str)
        assert self._particle_kernal is not None
        self._particle_kernal.version_update(
            SetParamNoneVersionUpdater(self._module_id, attr_name))

    def clone_from(self: Tmodule, source: Module) -> Tmodule:
        r"""Clone another particle from a different species into the current species.
        
        Args:
            attr_name (str): Particle from another species.
        
        Example::
        
            class Net(NytoModule):
                def __init__(self):
                    super().__init__()
                    self.my_param = torch.nn.Parameter(torch.randn(2, 2))
            
            net1 = Net()
            net2 = Net()
            net3 = net1.clone_from(net2)
            
            >>> net1._version_kernal is net2._version_kernal
            False
            
            >>> net1._version_kernal is net3._version_kernal
            True
            
            >>> net1.my_param is net3.my_param
            False
        """
        self_clone = self.clone()
        self_clone.load_state_dict(source.state_dict())
        return self_clone

    def detach(self: Tmodule) -> Tmodule:
        r"""Copy the parameters of the current particle to a new species.
        
        Example::
        
            class Net(NytoModule):
                def __init__(self):
                    super().__init__()
                    self.my_param = torch.nn.Parameter(torch.randn(2, 2))
            
            net1 = Net()
            net2 = net1.detach()
            
            >>> net1._version_kernal is net2._version_kernal
            False 
            
            >>> net1.my_param is net2.my_param
            True
        """
        assert self._particle_kernal is not None
        self.touch()
        pdata: ParticleDataImp = self._particle_kernal.detach().data
        module: Module = pdata.modules[self._module_id]
        assert isinstance(module, type(self))
        return module

    def touch(self) -> Self:
        r"""Update the particle to the latest version.
        
        Example::
        
            class Net(NytoModule):
                def __init__(self):
                    super().__init__()

            net1 = Net()
            net2 = net1.clone()
            net1.my_param = torch.nn.Parameter(torch.randn(2, 2))

            >>> hasattr(net2, "my_param")
            False
            
            >>> net2.touch()
            >>> hasattr(net2, "my_param")
            True
        """
        
        assert self._particle_kernal is not None
        if not self._version_kernal.is_newest:
            self._particle_kernal.particle_update()
        return self

    def product(self: Tmodule) -> ParamProduct:
        """
        Transform into a ParamProduct to optimize particle operations.

        Returns:
            ParamProduct: The ParamProduct instance representing the module.
        """
        return module_to_product(self.touch())

    def product_(self, product: ParamProduct) -> Self:
        """
        Import the parameters of a ParamProduct into the current module.

        Args:
            product (ParamProduct): The ParamProduct instance containing parameters to import.

        Returns:
            Self: The current module after importing the parameters.
        """
        product_assign_to_module(product, self)
        return self

    def apply_param_config(self, fn: Callable[[ParamID, ParamConfig], None], name: Optional[str]=None) -> None:
        r"""Manually modify the ParamConfig instance.
        
        Args:
            fn (Callable[[ParamID, ParamConfig], None]):
                Function that takes a param ID and its corresponding ParamConfig instance, and modifies the configuration.
            name (str): 
                Name of the attribute to modify the configuration for, default is to modify the configuration for all attributes.
        
        Example::
        
            class Net(NytoModule):
                def __init__(self):
                    super().__init__()
                    self.my_module = torch.nn.Linear(3, 2)
                    self.my_param = torch.nn.Parameter(torch.randn(2, 2))
                    
            def set_operational_false(pid: ParamID, conf: ParamConfig):
                conf.operational = False
                
            net = Net()
            net.apply_param_config(set_operational_false, "my_module")
        """
        vdata: VersionDataImp = self._version_kernal.data
        if name is None:
            return  _apply_module_config(vdata, self._module_id, fn)
        if self._modules.get(name, None) is not None:
            return _apply_module_config(vdata, 
                                        _get_module_id(vdata, self._module_id, name), 
                                        fn)
        if self._parameters.get(name, None) is not None:
            return _apply_param_config(vdata, 
                                       _get_param_id(vdata, self._module_id, name), 
                                       fn)
        raise ValueError(f"attar {name} is None or not exists")

    def set_param_config(self,
                         operational: Optional[bool]=None, 
                         clone: Optional[bool]=None,
                         name: Optional[str]=None) -> None:
        r"""Modify the default configuration of the ParamConfig instance.
        
         Args:
            operational (Optional[bool]):
                If True, the parameter is included in particle operations.
            clone (Optional[bool]): 
                If False, the parameter is cloned if not included in particle operations.
            name (Optional[str]): 
                Name of the attribute to modify the configuration for, default is to modify the configuration for all attributes.

        Example::
        
            class Net(NytoModule):
                def __init__(self):
                    super().__init__()
                    self.my_module = torch.nn.Linear(3, 2)
                    self.my_param = torch.nn.Parameter(torch.randn(2, 2))
            
            net = Net()
            net.set_param_config(operational=False, name="my_module")
        """
        
        def set_param_config_fn(pid: ParamID, conf: ParamConfig) -> None:
            if operational is not None:
                conf.operational = operational
            if clone is not None:
                conf.clone = clone
        return self.apply_param_config(set_param_config_fn, name)

    def get_param_id(self, target_param: ParamType) -> ParamID:
        r"""Find the ID of the specified parameter.
        
        Args:
            target_param (ParamType):
                The specified parameter.
        
        Example::
            
            class Net(NytoModule):
                def __init__(self):
                    super().__init__()
                    self.my_module = torch.nn.Linear(3, 2)
                    self.my_param = torch.nn.Parameter(torch.randn(2, 2))

            net = Net()
            
            >>> net.get_param_id(net.my_param)
            2
        """
        assert self._particle_kernal is not None
        for pid, param in self._particle_kernal.data.params.items():
            if param is target_param:
                return pid
        raise ValueError(f"can not found param: {target_param}")

    def register_parameter(self, name: str, param: Optional[ParamType]) -> None:
        if param is None:
            self._version_set_none_param(name)
        else:
            self._version_add_param(name, param)

    def add_module(self, name: str, module: Optional[Module]) -> None:
        if module is None:
            self._version_set_none_module(name)
        else:
            self._version_add_module(name, module)

    def register_buffer(self, name: str, tensor: Optional[torch.Tensor], persistent: bool=True) -> None:
        self._version_register_buffer(name, tensor, persistent)
