from __future__ import annotations
from nytorch.base import NytoModuleBase, ParticleKernelImp
from nytorch.kernel import Particle, Product
from nytorch.module import NytoModule, ParamProduct, Tmodule
from nytorch.mtype import ModuleID, ParamConfig, ParamDict, ParamType, ROOT_MODULE_ID
import torch.nn as nn
from typing import Callable, Generic
from typing_extensions import Self


class PMProduct(Product['ParticleModule']):
    r"""Decorator for ParamProduct.

    Implements particle operations and transforms into ParticleModule.

    Args:
        kernel (ParticleKernelImp): 
            Particle kernel instance.
        module_id (ModuleID): 
            ID of the module.
        params (ParamDict): 
            Parameters.
            
    Attributes:
        product (ParamProduct): Instance of ParamProduct.
    """
    
    __slots__ = ("product",)
    
    @classmethod
    def from_ParamProduct(cls, product: ParamProduct) -> PMProduct:
        r"""Wrap a ParamProduct instance into a PMProduct instance.

        Args:
            product (ParamProduct): Wrapped ParamProduct instance.

        Returns:
            PMProduct: Wrapped PMProduct instance.
        """
        return PMProduct(product.kernel, 
                         product.module_id, 
                         product.params)
    
    def __init__(self, kernel: ParticleKernelImp, module_id: ModuleID, params: ParamDict) -> None:
        r"""
        Not recommended to create manually, please use ParticleModule.product for generation.

        Args:
            kernel (ParticleKernelImp): 
                Particle kernel instance.
            module_id (ModuleID): 
                ID of the module.
            params (ParamDict): 
                Parameters.
        
        Example::
        
            class Net(NytoModule):
                def __init__(self):
                    super().__init__()
                    self.my_module = torch.nn.Linear(3, 2)
                    self.my_param = torch.nn.Parameter(torch.randn(2, 2))

            net = ParticleModule(Net())
            product = net.product()
        """
        self.product : ParamProduct = ParamProduct(kernel, 
                                                   module_id,
                                                   params)
    
    def particle(self) -> ParticleModule:
        """Transform into ParticleModule."""
        return ParticleModule(self.product.module())
    
    def module(self) -> ParticleModule:
        """Transform into ParticleModule."""
        return self.particle()
    
    def unary_operator(self, fn: Callable[[ParamType, ParamConfig], ParamType]) -> PMProduct:
        r"""Custom unary operation.
        
        Args:
            fn (Callable[[ParamType, ParamConfig], ParamType]): Unary operation logic.

        Returns:
            PMProduct: Resultant PMProduct instance after applying unary operation.
        
        .. note::
            
            In writing the function ``fn``, 
            gradient calculation does not need to be disabled
            because ``torch.no_grad`` is used within the ``unary_operator()`` method to disable gradient calculation.
        
        Example::
        
            class Net(NytoModule):
                def __init__(self):
                    super().__init__()
                    self.my_param = nn.Parameter(torch.Tensor([0., 1., 2.]))

            net = ParticleModule(Net())
            product = net.product()
            new_product = product.unary_operator(lambda param, conf: nn.Parameter(param+10.))
            new_net = new_product.module()
            
            >>> new_net.root_module.my_param
            Parameter containing:
            tensor([10., 11., 12.], requires_grad=True)
        """
        return type(self).from_ParamProduct(self.product.unary_operator(fn))
        
    def binary_operator(self,
                        other: PMProduct,
                        fn: Callable[[ParamType, ParamType, ParamConfig], ParamType]) -> PMProduct:
        r"""Custom binary operation.
        
        Args:
            other (PMProduct): 
                Another ParamProduct instance participating in the binary operation.
            fn (Callable[[ParamType, ParamType, ParamConfig], ParamType]): 
                Binary operation logic.

        Returns:
            PMProduct: Resultant PMProduct instance after applying binary operation.
        
        .. note::
            
            In writing the function ``fn``, 
            gradient calculation does not need to be disabled
            because ``torch.no_grad`` is used within the ``unary_operator()`` method to disable gradient calculation.
            
        .. note::
            
            The source of ``other`` must belong to the same species as the source of ``self``,
            which can be checked as follows::
            
                assert other.product.kernel.version is self.product.kernel.version
        
        Example::
        
            class Net(NytoModule):
                def __init__(self, my_tensor):
                    super().__init__()
                    self.my_param = nn.Parameter(my_tensor)

            net1 = ParticleModule(Net(torch.Tensor([0., 1., 2.])))
            net2 = ParticleModule(Net(torch.Tensor([5., 4., 3.])))
            net2 = net1.clone_from(net2)
            product1 = net1.product()
            product2 = net2.product()
            
            fn = lambda param1, param2, conf: nn.Parameter(param1-param2)
            new_product = product1.binary_operator(product2, fn)
            new_net = new_product.module()

            >>> new_net.root_module.my_param
            Parameter containing:
            tensor([-5., -3., -1.], requires_grad=True)
        """
        assert isinstance(other, PMProduct)
        return type(self).from_ParamProduct(self.product.binary_operator(other.product, 
                                                                         fn))
    
    def __neg__(self) -> PMProduct:
        return type(self).from_ParamProduct(-self.product)
    
    def __pow__(self, power) -> PMProduct:
        if isinstance(power, PMProduct):
            return type(self).from_ParamProduct(self.product**power.product)
        return type(self).from_ParamProduct(self.product**power)
    
    def __rpow__(self, base) -> PMProduct:
        if isinstance(base, PMProduct):
            type(self).from_ParamProduct(base.product**self.product)
        return type(self).from_ParamProduct(base**self.product)
    
    def __add__(self, other) -> PMProduct:
        if isinstance(other, PMProduct):
            return type(self).from_ParamProduct(self.product+other.product)
        return type(self).from_ParamProduct(self.product+other)
    
    def __sub__(self, other) -> PMProduct:
        if isinstance(other, PMProduct):
            return type(self).from_ParamProduct(self.product-other.product)
        return type(self).from_ParamProduct(self.product-other)
    
    def __rsub__(self, other) -> PMProduct:
        if isinstance(other, PMProduct):
            return type(self).from_ParamProduct(other.product-self.product)
        return type(self).from_ParamProduct(other-self.product)
    
    def __mul__(self, other) -> PMProduct:
        if isinstance(other, PMProduct):
            return type(self).from_ParamProduct(self.product*other.product)
        return type(self).from_ParamProduct(self.product*other)
    
    def __truediv__(self, other) -> PMProduct:
        if isinstance(other, PMProduct):
            return type(self).from_ParamProduct(self.product/other.product)
        return type(self).from_ParamProduct(self.product/other)
    
    def __rtruediv__(self, other) -> PMProduct:
        if isinstance(other, PMProduct):
            return type(self).from_ParamProduct(other.product/self.product)
        return type(self).from_ParamProduct(other/self.product)
    
    def clone(self) -> PMProduct:
        return type(self).from_ParamProduct(self.product.clone())
    
    def randn(self) -> PMProduct:
        return type(self).from_ParamProduct(self.product.randn())
    
    def rand(self) -> PMProduct:
        return type(self).from_ParamProduct(self.product.rand())


class ParticleModule(nn.Module, Particle[PMProduct], Generic[Tmodule]):
    r"""Decorator for NytoModule.
    
    This class wraps a NytoModule to handle particle operations and transformations, 
    addressing the issue of circular references by allowing the clearing and restoring 
    of references to the particle kernel.

    Features:
        * Implements particle operations and transforms to PMProduct.
        * Facilitates clearing and restoring the module's reference to the particle kernel.

    .. note::

        Clearing the module's reference to the particle kernel can eliminate circular references,
        reducing memory pressure.

    Args:
        root_module (Tmodule): The NytoModule instance to be wrapped.

    Attributes:
        particle_kernel (ParticleKernelImp): 
            Reference to the particle kernel for restoring the module's reference.
        root_module (Tmodule): 
            The root NytoModule being wrapped.
    """
    
    __slots__ = "particle_kernel", "root_module"
    
    particle_kernel: ParticleKernelImp
    root_module: Tmodule
    
    def __init__(self, root_module: Tmodule) -> None:
        """
        Initialize ParticleModule with a root module.

        Args:
            root_module (Tmodule): The NytoModule instance to be wrapped.
        """
        assert isinstance(root_module, NytoModule)
        assert root_module._module_id == ROOT_MODULE_ID
        assert root_module._particle_kernel is not None
        
        super().__init__()
        self.particle_kernel = root_module._particle_kernel
        self.root_module = root_module
        self.clear_kernel_ref()
    
    def clear_kernel_ref(self) -> None:
        """
        Clears the module's reference to the particle kernel.

        This method is used to eliminate circular references and reduce memory usage.
        """
        for submodule in self.root_module.modules():
            if isinstance(submodule, NytoModuleBase):
                submodule._particle_kernel = None
                
    def restore_kernel_ref(self) -> None:
        """
        Restores the module's reference to the particle kernel.

        This method is used to re-establish references that were cleared to reduce memory usage.
        """
        for submodule in self.root_module.modules():
            if isinstance(submodule, NytoModuleBase):
                submodule._particle_kernel = self.particle_kernel
                
    def forward(self, *args, **kwargs):
        """
        Forward pass through the root module.

        Args:
            *args: 
                Positional arguments for the root module's forward method.
            **kwargs: 
                Keyword arguments for the root module's forward method.

        Returns:
            Tensor: The result of the root module's forward method.
        """
        return self.root_module(*args, **kwargs)
    
    def product(self) -> PMProduct:
        """
        Transform the module into a PMProduct.

        Returns:
            PMProduct: A PMProduct instance representing the module.
        """
        return PMProduct(self.particle_kernel, 
                         ROOT_MODULE_ID, 
                         self.particle_kernel.data.params)
    
    def product_(self, product: PMProduct) -> Self:
        """
        Import parameters from a PMProduct instance.

        Args:
            product (PMProduct): The PMProduct instance to import from.

        Returns:
            ParticleModule: The ParticleModule instance with imported parameters.
        """
        self.restore_kernel_ref()
        self.root_module.product_(product.product)
        self.clear_kernel_ref()
        return self
    
    def clone_from(self, source: ParticleModule) -> ParticleModule:
        """
        Clone the particle from another ParticleModule instance.

        Args:
            source (ParticleModule): The ParticleModule instance to clone from.

        Returns:
            ParticleModule: The cloned ParticleModule instance.
        """
        assert isinstance(source, ParticleModule)
        self_clone = self.clone()
        self_clone.load_state_dict(source.state_dict())
        return self_clone


class GetNytoModule(Generic[Tmodule]):
    r"""Context manager for automatic restoration and cleanup of kernel reference.

    Args:
        particle_module(ParticleModule[Tmodule]): The ParticleModule to be processed.

    Attributes:
        _module(ParticleModule[Tmodule]): The ParticleModule being processed.

    To directly assign attributes to an instance of a NytoModule subclass wrapped by ParticleModule, 
    call `restore_kernel_ref` first to restore the kernel reference, 
    and then call `clear_kernel_ref` after the assignment operation::

        model = ParticleModule(MyNytoModule())
        model.restore_kernel_ref()
        model.root_module.data_embed = nn.Embedding(30, 8)
        model.clear_kernel_ref()

    Alternatively, use `GetNytoModule` to simplify::

        model = ParticleModule(MyNytoModule())
        with GetNytoModule(model) as my_nyto_module:
            my_nyto_module.data_embed = nn.Embedding(30, 8)
    """
    
    __slots__ = "_module"
    
    _module: ParticleModule[Tmodule]
    
    def __init__(self, particle_module: ParticleModule[Tmodule]):
        assert isinstance(particle_module, ParticleModule)
        self._module = particle_module
    
    def __enter__(self) -> Tmodule:
        self._module.restore_kernel_ref()
        return self._module.root_module

    def __exit__(self, exc_type, exc_value, traceback_obj):
        self._module.clear_kernel_ref()