
from __future__ import annotations
from typing import Generic, Optional, Type, TypeVar
from typing_extensions import Self
import abc


Tvdata = TypeVar("Tvdata", bound="VersionData")
Tpdata = TypeVar("Tpdata", bound="ParticleData")
Tproduct = TypeVar("Tproduct", bound="Product")
Tparticle = TypeVar("Tparticle", bound="Particle")


class VersionData(abc.ABC, Generic[Tvdata, Tpdata]):
    r"""
    Abstract base class for storing metadata of particles.

    This class is responsible for storing the metadata of particles, specifically the metadata of the species 
    to which the particle belongs. It is essential for particle operations and will be stored in `VersionKernal.data`.

    Once an instance is created, its content remains unchanged. When new metadata is required, a copy is created 
    and modifications are applied to the new instance.

    A corresponding subclass of `ParticleData` should be defined when instantiating this class, as shown below:

    .. code-block:: python

        class VersionDataImp(VersionData[VersionDataImp, ParticleDataImp]):
            ...

        class ParticleDataImp(ParticleData[VersionDataImp, ParticleDataImp]):
            ...
    """
    
    @abc.abstractmethod
    def init_kernal(self, kernal: 'VersionKernal') -> None:
        """
        Initializes the version kernal before creating a new version.

        This method ensures that the `VersionKernal` is properly initialized and ready to store this instance.

        Args:
            kernal (VersionKernal): The `VersionKernal` instance that will store this instance.
        """
        return 
    
    @abc.abstractmethod
    def init_particle(self, pdata: Tpdata) -> None:
        """
        Ensures the keys of `pdata` are in order before the particle enters the new version.

        This method verifies and organizes the keys of the particle data before it is added to the new version.

        Args:
            pdata (Tpdata): The `ParticleData` subclass instance of the particle entering the new version.
        """
        return  
    
    @abc.abstractmethod
    def copy(self) -> Tvdata:
        """
        Creates a deep copy of the instance.

        This method returns a deep copy of the current instance, ensuring that all nested data structures are 
        also copied.

        Returns:
            Tvdata: A deep copy of the instance.
        """
        return NotImplemented
    
    
class ParticleData(abc.ABC, Generic[Tvdata, Tpdata]):
    r"""
    Abstract base class for storing parameters of particles.

    This class stores references to all parameters of the corresponding particle. It is essential for particle 
    operations and will be stored in `ParticleKernal.data`.

    A corresponding subclass of `VersionData` should be defined when instantiating this class, as shown below:

    .. code-block:: python

        class VersionDataImp(VersionData[VersionDataImp, ParticleDataImp]):
            ...

        class ParticleDataImp(ParticleData[VersionDataImp, ParticleDataImp]):
            ...
    """

    @abc.abstractmethod
    def init_kernal(self, kernal: ParticleKernal) -> None:
        """
        Initializes the particle kernal before creating a new particle.

        This method ensures that the `ParticleKernal` is properly initialized and ready to store this instance.

        Args:
            kernal (ParticleKernal): The `ParticleKernal` instance that will store this instance.
        """
        return  
    
    @abc.abstractmethod
    def copy(self, vdata: Tvdata) -> Tpdata:
        """
        Creates a shallow copy of the current particle data instance.

        This method returns a new instance that shares references to the same parameters as the current instance.

        Args:
            vdata (Tvdata): The corresponding `VersionData` instance.

        Returns:
            Tpdata: A shallow copy of the particle data instance.
        """
        return NotImplemented
    

class VersionUpdater(abc.ABC, Generic[Tvdata, Tpdata]):
    r"""
    Abstract base class for tools that update VersionKernal instances.

    This class provides a template for implementing tools that handle updates
    of VersionKernal instances. A corresponding subclass of ParticleUpdater should be implemented
    to manage the particle updates.

    The following methods are called in sequence when updating a VersionKernal instance:
    
        1. `set_version_data`
        2. `set_particle_data`
        3. `run`

    Subclasses must implement these methods to define the update process.
    """
    
    @abc.abstractmethod
    def set_version_data(self, vdata: Tvdata) -> Self:
        """
        Sets the version data required before executing the update.

        Args:
            vdata (Tvdata): An instance of a VersionData subclass that stores the current version's particle metadata.
        
        Returns:
            Self: The updated instance of the class.
        """
        return NotImplemented
    
    @abc.abstractmethod
    def set_particle_data(self, pdata: Tpdata) -> Self:
        """
        Sets the particle data required before executing the update.

        Args:
            pdata (Tpdata): An instance of a ParticleData subclass that stores the current particle parameters.
        
        Returns:
            Self: The updated instance of the class.
        """
        return NotImplemented
    
    @abc.abstractmethod
    def run(self, vdata: Tvdata) -> tuple[Tvdata, 'ParticleUpdater']:
        """
        Executes the update process.

        This method generates an instance of the next version's VersionData subclass and
        the current version's ParticleUpdater.

        Args:
            vdata (Tvdata): An instance of a VersionData subclass that stores the current version's particle metadata.
        
        Returns:
            Tuple[Tvdata, ParticleUpdater]: A tuple containing the next version's VersionData instance and the current version's ParticleUpdater.
        """
        return NotImplemented
    
    
class ParticleUpdater(abc.ABC, Generic[Tvdata, Tpdata]):
    r"""
    Abstract base class for tools that update ParticleKernal instances.

    This class provides a template for implementing tools that handle direct operations on particles
    when updating to the next version. The `run` method defines these operations.

    The following methods are called in sequence when updating a ParticleKernal instance:
    
        1. `set_version_data`
        2. `set_next_version_data`
        3. `set_particle_data`
        4. `run`

    Methods 1, 2, and 3 are called once during initialization, while method 4 is called
    once for each particle that needs to be updated to the new version.

    Subclasses must implement these methods to define the update process.
    """

    @abc.abstractmethod
    def set_version_data(self, vdata: Tvdata) -> 'ParticleUpdater[Tvdata, Tpdata]':
        """
        Sets the current version data required before performing the update.

        Args:
            vdata (Tvdata): An instance of a VersionData subclass that stores the current version's particle metadata.

        Returns:
            ParticleUpdater[Tvdata, Tpdata]: The instance of the class with updated version data.
        """
        return NotImplemented
    
    @abc.abstractmethod
    def set_next_version_data(self, vdata: Tvdata) -> 'ParticleUpdater[Tvdata, Tpdata]':
        """
        Sets the next version data required before performing the update.

        Args:
            vdata (Tvdata): An instance of a VersionData subclass that stores the next version's particle metadata.

        Returns:
            ParticleUpdater[Tvdata, Tpdata]: The instance of the class with updated next version data.
        """
        return NotImplemented
    
    @abc.abstractmethod
    def set_particle_data(self, pdata: Tpdata) -> 'ParticleUpdater[Tvdata, Tpdata]':
        """
        Sets the particle data required before performing the update.

        Args:
            pdata (Tpdata): An instance of a ParticleData subclass that stores the current particle parameters.

        Returns:
            ParticleUpdater[Tvdata, Tpdata]: The instance of the class with updated particle data.
        """
        return NotImplemented
    
    @abc.abstractmethod
    def run(self, pdata: Tpdata) -> None:
        """
        Executes the particle update process.

        This method retrieves the data of the particle to be updated and performs the necessary operations
        to update the particle to the new version.

        Args:
            pdata (Tpdata): An instance of a ParticleData subclass that stores the current particle parameters.
        """
        return
    
    
class VersionKernal(Generic[Tvdata, Tpdata]):
    r"""
    Core class for managing versions of particle metadata.

    Each `VersionKernel` instance represents a version or state of particle metadata. When the metadata state
    of a particle changes, a new `VersionKernel` instance is created and linked via `next_version`.
    The `particle_updater` facilitates updates to particles within the current version.

    Args:
        data (Tvdata): 
            The VersionData subclass instance storing particle metadata, used to assist particle operations.

    Attributes:
        next_version (Optional[VersionKernal[Tvdata, Tpdata]]): 
            Points to the next version.
        particle_updater (Optional[ParticleUpdater[Tvdata, Tpdata]]): 
            Tool for updating particles within the current version.
        data (Tvdata): 
            The VersionData subclass instance storing particle metadata, used to assist particle operations.
    """
    
    __slots__ = "next_version", "particle_updater", "data"

    next_version: Optional[VersionKernal[Tvdata, Tpdata]]
    particle_updater: Optional[ParticleUpdater[Tvdata, Tpdata]]
    data: Tvdata
    
    def __init__(self, data: Tvdata) -> None:
        """
        Initializes a `VersionKernel` instance with given data.

        Args:
            data (Tvdata): The VersionData subclass instance storing particle metadata, used to assist particle operations.
        """
        self.next_version = None
        self.particle_updater = None
        self.data = data
        self.data.init_kernal(self)
    
    def __repr__(self) -> str:
        return f"VersionKernal({self.data})"
    
    @property
    def is_newest(self) -> bool:
        """
        Checks if the current version is the newest version.

        Returns:
            bool: True if the current version is the newest, False otherwise.
        """
        return self.next_version is None
    
    def version_update(self, version_updater: VersionUpdater[Tvdata, Tpdata], particle_data: Tpdata) -> None:
        """
        Generates a new version.

        Args:
            version_updater (VersionUpdater[Tvdata, Tpdata]): 
                The VersionUpdater subclass instance that generates `next_version` and `particle_updater`.
            particle_data (Tpdata): 
                The ParticleData subclass instance of the particle initiating this operation.
        
        Raises:
            AssertionError: If `particle_updater` or `next_version` is None.
        """
        next_data, self.particle_updater = (version_updater.set_version_data(self.data)
                                                           .set_particle_data(particle_data)
                                                           .run(self.data))
        self.next_version = VersionKernal(next_data)
        (self.particle_updater.set_version_data(self.data)
                              .set_next_version_data(next_data)
                              .set_particle_data(particle_data))
        
    def particle_update(self, particle_data: Tpdata) -> None:
        """
        Updates the particle data to the new version using the particle updater.

        Args:
            particle_data (Tpdata): The instance of ParticleData subclass to be updated.
        
        Raises:
            AssertionError: If `particle_updater` or `next_version` is None.
        """
        assert self.particle_updater is not None
        assert self.next_version is not None
        self.particle_updater.run(particle_data)
        self.next_version.data.init_particle(particle_data)
    
    def del_next_version(self) -> None:
        """
        Removes the new version.

        This method is called after updating the version if an error occurs when updating particles,
        indicating that there is a problem with the new version, and the new version will be removed.
        """
        self.next_version = None
        self.particle_updater = None
    
    def copy(self) -> VersionKernal:
        """
        Creates a shallow copy of the current version.

        Returns:
            VersionKernel: A shallow copy of the current VersionKernel.
        """
        return VersionKernal(self.data.copy())
        

class ParticleKernal(Generic[Tvdata, Tpdata]):
    r"""
    Core class for managing particles.

    Each `ParticleKernal` instance represents a particle, storing the particle's specific parameters.
    The particle's metadata is stored in the version instance pointed to by `version`.

    Args:
        version (VersionKernal[Tvdata, Tpdata]):
            Points to the corresponding version.
        data (Tpdata):
            The ParticleData subclass instance storing particle parameters, used to assist particle operations.

    Attributes:
        version (VersionKernal[Tvdata, Tpdata]): 
            Points to the corresponding version.
        data (Tpdata): 
            The ParticleData subclass instance storing particle parameters, used to assist particle operations.
    """
    
    __slots__ = "version", "data"

    version: VersionKernal[Tvdata, Tpdata]
    data: Tpdata
    
    def __init__(self, version: VersionKernal[Tvdata, Tpdata], data: Tpdata) -> None:
        """
        Initializes a `ParticleKernel` instance with the given version and data.

        Args:
            version (VersionKernel[Tvdata, Tpdata]): 
                Points to the corresponding version.
            data (Tpdata): 
                The ParticleData subclass instance storing particle parameters.
        """
        self.version = version
        self.data = data
        self.data.init_kernal(self)
        
    def __repr__(self) -> str:
        return f"ParticleKernal({self.data})"
        
    def create(self, data: Tpdata) -> ParticleKernal[Tvdata, Tpdata]:
        """
        Creates a ParticleKernel belonging to the same species.

        Args:
            data (Tpdata): The ParticleData subclass instance storing particle parameters.

        Returns:
            ParticleKernel[Tvdata, Tpdata]: A new ParticleKernel instance.
        """
        return ParticleKernal(self.version, data)
    
    def version_update(self, version_updater: VersionUpdater[Tvdata, Tpdata]) -> None:
        """
        Creates a new version and updates the particle.

        This method ensures that if updating the version succeeds but updating the particle fails,
        the version update is rolled back.

        Args:
            version_updater (VersionUpdater[Tvdata, Tpdata]): The VersionUpdater subclass instance used to update the version.
        """
        if not self.version.is_newest:
            self.particle_update()
        self.version.version_update(version_updater, self.data)
        
        # Roll back the version update if particle update fails.
        try:
            self.particle_update()
        except Exception as error:
            self.version.del_next_version()
            raise error
        
    def particle_update(self) -> None:
        """
        Updates the particle to the newest version.

        This method ensures the particle is updated through all intermediate versions to the latest version.
        """
        while not self.version.is_newest:
            assert self.version.next_version is not None
            self.version.particle_update(self.data)
            self.version = self.version.next_version
            
    def detach(self) -> ParticleKernal[Tvdata, Tpdata]:
        """
        Detaches the current particle kernel instance to a new species.

        Creates a deep copy of the version kernel and a shallow copy of the current particle kernel,
        then associates the copied version kernel with the new particle kernel instance.

        Returns:
            ParticleKernel[Tvdata, Tpdata]: The detached ParticleKernel instance.
        """
        return ParticleKernal(self.version.copy(),
                              self.data.copy(self.version.data))


class Product(abc.ABC, Generic[Tparticle]):
    r"""Abstract base class to be used in conjunction with a corresponding Particle subclass.

    Subclasses of this class must implement particle operations and transformation methods.
    The corresponding Particle subclass should also be defined to allow transformation between
    the two using the ``product`` and ``particle`` methods.

    Example usage:
    
    .. code-block:: python
    
        class ProductImp(Product[ParticleImp]):
            ...
            
        class ParticleImp(Particle[ProductImp]):
            ...
            
        particle: ParticleImp = ParticleImp()
        product: ProductImp = particle.product()
        new_particle: ParticleImp = product.particle()
    """

    def __pos__(self: Tproduct) -> Tproduct:
        """
        Unary positive operation.

        Returns:
            Tproduct: A clone of the current product.
        """
        return self.clone()
    
    @abc.abstractmethod
    def __neg__(self: Tproduct) -> Tproduct:
        """
        Unary negation operation.

        Returns:
            Tproduct: A new product representing the negation of the current product.
        """
        return NotImplemented
    
    @abc.abstractmethod
    def __pow__(self: Tproduct, power) -> Tproduct:
        """
        Power operation.

        Args:
            power: The exponent.

        Returns:
            Tproduct: A new product representing the current product raised to the given power.
        """
        return NotImplemented
    
    @abc.abstractmethod
    def __rpow__(self: Tproduct, base) -> Tproduct:
        """
        Reverse power operation.

        Args:
            base: The base.

        Returns:
            Tproduct: A new product representing the base raised to the power of the current product.
        """
        return NotImplemented
    
    @abc.abstractmethod
    def __add__(self: Tproduct, other) -> Tproduct:
        """
        Addition operation.

        Args:
            other: The other product to add.

        Returns:
            Tproduct: A new product representing the sum of the current product and the other product.
        """
        return NotImplemented
    
    def __radd__(self: Tproduct, other) -> Tproduct:
        """
        Reverse addition operation.

        Args:
            other: The other product to add.

        Returns:
            Tproduct: A new product representing the sum of the other product and the current product.
        """
        return self.__add__(other)
    
    @abc.abstractmethod
    def __sub__(self: Tproduct, other) -> Tproduct:
        """
        Subtraction operation.

        Args:
            other: The other product to subtract.

        Returns:
            Tproduct: A new product representing the difference between the current product and the other product.
        """
        return NotImplemented
    
    @abc.abstractmethod
    def __rsub__(self: Tproduct, other) -> Tproduct:
        """
        Reverse subtraction operation.

        Args:
            other: The other product to subtract.

        Returns:
            Tproduct: A new product representing the difference between the other product and the current product.
        """
        return NotImplemented
    
    @abc.abstractmethod
    def __mul__(self: Tproduct, other) -> Tproduct:
        """
        Multiplication operation.

        Args:
            other: The other product to multiply.

        Returns:
            Tproduct: A new product representing the product of the current product and the other product.
        """
        return NotImplemented
    
    def __rmul__(self: Tproduct, other) -> Tproduct:
        """
        Reverse multiplication operation.

        Args:
            other: The other product to multiply.

        Returns:
            Tproduct: A new product representing the product of the other product and the current product.
        """
        return self.__mul__(other)
    
    @abc.abstractmethod
    def __truediv__(self: Tproduct, other) -> Tproduct:
        """
        True division operation.

        Args:
            other: The other product to divide by.

        Returns:
            Tproduct: A new product representing the division of the current product by the other product.
        """
        return NotImplemented
    
    @abc.abstractmethod
    def __rtruediv__(self: Tproduct, other) -> Tproduct:
        """
        Reverse true division operation.

        Args:
            other: The other product to divide by.

        Returns:
            Tproduct: A new product representing the division of the other product by the current product.
        """
        return NotImplemented
    
    @abc.abstractmethod
    def particle(self) -> Tparticle:
        """
        Transforms the current particle into its corresponding Product instance.

        Returns:
            Tparticle: The corresponding Particle instance.
        """
        return NotImplemented
    
    @abc.abstractmethod
    def clone(self: Tproduct) -> Tproduct:
        """
        Clone the current product.

        Returns:
            Tproduct: A new product that is a clone of the current product.
        """
        return NotImplemented
    
    @abc.abstractmethod
    def randn(self: Tproduct) -> Tproduct:
        """
        Generate a random product with normally distributed values.

        Returns:
            Tproduct: A new product with randomly generated values.
        """
        return NotImplemented
    
    @abc.abstractmethod
    def rand(self: Tproduct) -> Tproduct:
        """
        Generate a random product with uniformly distributed values.

        Returns:
            Tproduct: A new product with randomly generated values.
        """
        return NotImplemented
    

class Particle(abc.ABC, Generic[Tproduct]):
    r"""
    Abstract base class to be used in conjunction with a corresponding Product subclass.
    
    Subclasses of this class must implement transformation methods (``product`` and ``product_``),
    which automatically perform particle operations through the corresponding Product subclass.
    The corresponding Product subclass should also be defined to allow transformation between
    the two using the ``product`` and ``particle`` methods.

    Example usage:
    
    .. code-block:: python
    
        class ProductImp(Product[ParticleImp]):
            ...
            
        class ParticleImp(Particle[ProductImp]):
            ...
            
        particle: ParticleImp = ParticleImp()
        product: ProductImp = particle.product()
        new_particle: ParticleImp = product.particle()
    """
    
    def __pos__(self: Tparticle) -> Tparticle:
        """
        Unary positive operation.

        Returns:
            Tparticle: A clone of the current particle.
        """
        return self.clone()
    
    def __neg__(self: Tparticle) -> Tparticle:
        """
        Unary negation operation.

        Returns:
            Tparticle: A new particle representing the negation of the current particle.
        """
        return (-self.product()).particle()
    
    def __pow__(self: Tparticle, power) -> Tparticle:
        """
        Power operation.

        Args:
            power: The exponent.

        Returns:
            Tparticle: A new particle representing the current particle raised to the given power.

        Raises:
            AssertionError: If power is an instance of Product.
        """
        assert not isinstance(power, Product)
        if isinstance(power, type(self)):
            return (self.product()**power.product()).particle()
        return (self.product()**power).particle()
    
    def __rpow__(self: Tparticle, base) -> Tparticle:
        """
        Reverse power operation.

        Args:
            base: The base.

        Returns:
            Tparticle: A new particle representing the base raised to the power of the current particle.

        Raises:
            AssertionError: If base is an instance of Product.
        """
        assert not isinstance(base, Product)
        return (base**self.product()).particle()
    
    def __ipow__(self, power) -> Self:
        """
        In-place power operation.

        Args:
            power: The exponent.

        Returns:
            Self: The current particle after raising to the given power.

        Raises:
            AssertionError: If power is an instance of Product.
        """
        assert not isinstance(power, Product)
        if isinstance(power, type(self)):
            return self.product_(self.product()**power.product())
        return self.product_(self.product()**power)
    
    def __add__(self: Tparticle, other) -> Tparticle:
        """
        Addition operation.

        Args:
            other: The other particle to add.

        Returns:
            Tparticle: A new particle representing the sum of the current particle and the other particle.

        Raises:
            AssertionError: If other is an instance of Product.
        """
        assert not isinstance(other, Product)
        if isinstance(other, type(self)):
            return (self.product()+other.product()).particle()
        return (self.product()+other).particle()
    
    def __radd__(self: Tparticle, other) -> Tparticle:
        """
        Reverse addition operation.

        Args:
            other: The other particle to add.

        Returns:
            Tparticle: A new particle representing the sum of the other particle and the current particle.
        """
        return self.__add__(other)
    
    def __iadd__(self, other) -> Self:
        """
        In-place addition operation.

        Args:
            other: The other particle to add.

        Returns:
            Self: The current particle after addition.

        Raises:
            AssertionError: If other is an instance of Product.
        """
        assert not isinstance(other, Product)
        if isinstance(other, type(self)):
            return self.product_(self.product()+other.product())
        return self.product_(self.product()+other)
    
    def __sub__(self: Tparticle, other) -> Tparticle:
        """
        Subtraction operation.

        Args:
            other: The other particle to subtract.

        Returns:
            Tparticle: A new particle representing the difference between the current particle and the other particle.

        Raises:
            AssertionError: If other is an instance of Product.
        """
        assert not isinstance(other, Product)
        if isinstance(other, type(self)):
            return (self.product()-other.product()).particle()
        return (self.product()-other).particle()
    
    def __rsub__(self: Tparticle, other) -> Tparticle:
        """
        Reverse subtraction operation.

        Args:
            other: The other particle to subtract.

        Returns:
            Tparticle: A new particle representing the difference between the other particle and the current particle.

        Raises:
            AssertionError: If other is an instance of Product.
        """
        assert not isinstance(other, Product)
        return (other-self.product()).particle()
    
    def __isub__(self, other) -> Self:
        """
        In-place subtraction operation.

        Args:
            other: The other particle to subtract.

        Returns:
            Self: The current particle after subtraction.

        Raises:
            AssertionError: If other is an instance of Product.
        """
        assert not isinstance(other, Product)
        if isinstance(other, type(self)):
            return self.product_(self.product()-other.product())
        return self.product_(self.product()-other)
    
    def __mul__(self: Tparticle, other) -> Tparticle:
        """
        Multiplication operation.

        Args:
            other: The other particle to multiply.

        Returns:
            Tparticle: A new particle representing the product of the current particle and the other particle.

        Raises:
            AssertionError: If other is an instance of Product.
        """
        assert not isinstance(other, Product)
        if isinstance(other, type(self)):
            return (self.product()*other.product()).particle()
        return (self.product()*other).particle()
    
    def __rmul__(self: Tparticle, other) -> Tparticle:
        """
        Reverse multiplication operation.

        Args:
            other: The other particle to multiply.

        Returns:
            Tparticle: A new particle representing the product of the other particle and the current particle.
        """
        return self.__mul__(other)
    
    def __imul__(self, other) -> Self:
        """
        In-place multiplication operation.

        Args:
            other: The other particle to multiply.

        Returns:
            Self: The current particle after multiplication.

        Raises:
            AssertionError: If other is an instance of Product.
        """
        assert not isinstance(other, Product)
        if isinstance(other, type(self)):
            return self.product_(self.product()*other.product())
        return self.product_(other*self.product())
    
    def __truediv__(self: Tparticle, other) -> Tparticle:
        """
        True division operation.

        Args:
            other: The other particle to divide by.

        Returns:
            Tparticle: A new particle representing the division of the current particle by the other particle.

        Raises:
            AssertionError: If other is an instance of Product.
        """
        assert not isinstance(other, Product)
        if isinstance(other, type(self)):
            return (self.product()/other.product()).particle()
        return (self.product()/other).particle()
    
    def __rtruediv__(self: Tparticle, other) -> Tparticle:
        """
        Reverse true division operation.

        Args:
            other: The other particle to divide by.

        Returns:
            Tparticle: A new particle representing the division of the other particle by the current particle.

        Raises:
            AssertionError: If other is an instance of Product.
        """
        assert not isinstance(other, Product)
        return (other/self.product()).particle()
    
    def __itruediv__(self, other) -> Self:
        """
        In-place true division operation.

        Args:
            other: The other particle to divide by.

        Returns:
            Self: The current particle after division.

        Raises:
            AssertionError: If other is an instance of Product.
        """
        assert not isinstance(other, Product)
        if isinstance(other, type(self)):
            return self.product_(self.product()/other.product())
        return self.product_(self.product()/other)

    @abc.abstractmethod
    def product(self) -> Tproduct:
        """
        Transforms the current particle into its corresponding Product instance.

        Returns:
            Tproduct: The corresponding Product instance.
        """
        return NotImplemented
    
    @abc.abstractmethod
    def product_(self, product: Tproduct) -> Self:
        """
        Transforms the current particle into a new particle instance based on the given Product.

        Args:
            product (Tproduct): The corresponding Product instance containing parameters to import.

        Returns:
            Self: The current particle instance after parameter importation.
        """
        return NotImplemented
    
    def clone(self: Tparticle) -> Tparticle:
        """
        Clone the current particle.

        Returns:
            Tparticle: A new particle that is a clone of the current particle.
        """
        return self.product().clone().particle()
    
    def randn(self: Tparticle) -> Tparticle:
        """
        Generate a random particle with normally distributed values.

        Returns:
            Tparticle: A new particle with randomly generated values.
        """
        return self.product().randn().particle()
    
    def rand(self: Tparticle) -> Tparticle:
        """
        Generate a random particle with uniformly distributed values.

        Returns:
            Tparticle: A new particle with randomly generated values.
        """
        return self.product().rand().particle()
