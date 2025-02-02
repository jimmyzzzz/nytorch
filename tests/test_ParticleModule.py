from nytorch import NytoModule, ParticleModule
from nytorch.particle_module import PMProduct
import torch
import torch.nn as nn
import unittest


class MySubModule(NytoModule):
    def __init__(self, w2):
        super().__init__()
        self.param2 = nn.Parameter(torch.Tensor([w2]))


class MyModule(NytoModule):
    def __init__(self, w1, w2):
        super().__init__()
        self.param1 = nn.Parameter(torch.Tensor([w1]))
        self.sub_module = MySubModule(w2)

    @property
    def param2(self):
        return self.sub_module.param2


class TestParticleModuleClearRef(unittest.TestCase):
    def test_init_clear_ref(self):
        nyto_module = MyModule(1., 2.)
        particle_kernel = nyto_module._particle_kernel
        particle_module = ParticleModule(nyto_module)

        self.assertIsNone(nyto_module._particle_kernel)
        self.assertIsNone(nyto_module.sub_module._particle_kernel)

    def test_restore_and_clear_ref(self):
        nyto_module = MyModule(1., 2.)
        particle_kernel = nyto_module._particle_kernel
        particle_module = ParticleModule(nyto_module)

        particle_module.restore_kernel_ref()
        self.assertIs(nyto_module._particle_kernel, particle_kernel)
        self.assertIs(nyto_module.sub_module._particle_kernel, particle_kernel)

        particle_module.clear_kernel_ref()
        self.assertIsNone(nyto_module._particle_kernel)
        self.assertIsNone(nyto_module.sub_module._particle_kernel)


class TestParticleModuleCloneFrom(unittest.TestCase):
    def test_clone_from1(self):
        module1 = ParticleModule(MyModule(1., 2.))
        module2 = ParticleModule(MyModule(3., 4.))
        
        with self.assertRaises(Exception):
            module1 + module2
            module1 - module2
            module1 * module2
            module1 / module2
            module1 += module2
            module1 -= module2
            module1 *= module2
            module1 /= module2

    def test_clone_from2(self):
        module1 = ParticleModule(MyModule(1., 2.))
        module2 = ParticleModule(MyModule(3., 4.))

        module3 = module1.clone_from(module2)
        self.assertIsNot(module2.root_module.param1, module3.root_module.param1)
        self.assertIsNot(module2.root_module.param2, module3.root_module.param2)
        self.assertTrue(torch.equal(module2.root_module.param1, module3.root_module.param1))
        self.assertTrue(torch.equal(module2.root_module.param2, module3.root_module.param2))


class TestParticleModuleTransform(unittest.TestCase):
    def test_transform(self):
        module = ParticleModule(MyModule(1., 2.))
        product = module.product()
        new_module = product.module()

        self.assertIsInstance(product, PMProduct)
        self.assertIsInstance(new_module, ParticleModule)
        self.assertTrue(torch.equal(module.root_module.param1, new_module.root_module.param1))
        self.assertTrue(torch.equal(module.root_module.param2, new_module.root_module.param2))


class TestParticleModuleOperation(unittest.TestCase):
    def test_neg(self):
        module1 = ParticleModule(MyModule(1., 2.))
        module2 = -module1
        self.assertIsNot(module1.root_module.param1, module2.root_module.param1)
        self.assertIsNot(module1.root_module.param2, module2.root_module.param2)
        self.assertTrue(torch.equal(module2.root_module.param1, torch.Tensor([-1.])))
        self.assertTrue(torch.equal(module2.root_module.param2, torch.Tensor([-2.])))
        
    def test_pos(self):
        module1 = ParticleModule(MyModule(1., 2.))
        module2 = +module1
        self.assertIsNot(module1.root_module.param1, module2.root_module.param1)
        self.assertIsNot(module1.root_module.param2, module2.root_module.param2)
        self.assertTrue(torch.equal(module2.root_module.param1, torch.Tensor([1.])))
        self.assertTrue(torch.equal(module2.root_module.param2, torch.Tensor([2.])))

    def test_pow1(self):
        module1 = ParticleModule(MyModule(1., 2.))
        module2 = module1 ** 2
        self.assertIsNot(module1.root_module.param1, module2.root_module.param1)
        self.assertIsNot(module1.root_module.param2, module2.root_module.param2)
        self.assertTrue(torch.equal(module2.root_module.param1, torch.Tensor([1.**2])))
        self.assertTrue(torch.equal(module2.root_module.param2, torch.Tensor([2.**2])))

    def test_pow2(self):
        module1 = ParticleModule(MyModule(1., 2.))
        module2 = 2 ** module1
        self.assertIsNot(module1.root_module.param1, module2.root_module.param1)
        self.assertIsNot(module1.root_module.param2, module2.root_module.param2)
        self.assertTrue(torch.equal(module2.root_module.param1, torch.Tensor([2 ** 1.])))
        self.assertTrue(torch.equal(module2.root_module.param2, torch.Tensor([2 ** 2.])))
        
    def test_pow3(self):
        module = ParticleModule(MyModule(1., 2.))
        module_param1 = module.root_module.param1
        module_param2 = module.root_module.param2
        module **= 2
        self.assertIs(module.root_module.param1, module_param1)
        self.assertIs(module.root_module.param2, module_param2)
        self.assertTrue(torch.equal(module.root_module.param1, torch.Tensor([1.**2])))
        self.assertTrue(torch.equal(module.root_module.param2, torch.Tensor([2.**2])))

    def test_pow4(self):
        module1 = ParticleModule(MyModule(1., 2.))
        module2 = module1.clone_from(ParticleModule(MyModule(3., 4.)))
        module3 = module1 ** module2
        self.assertIsNot(module1.root_module.param1, module3.root_module.param1)
        self.assertIsNot(module1.root_module.param2, module3.root_module.param2)
        self.assertIsNot(module2.root_module.param1, module3.root_module.param1)
        self.assertIsNot(module2.root_module.param2, module3.root_module.param2)
        self.assertTrue(torch.equal(module3.root_module.param1, torch.Tensor([1.**3.])))
        self.assertTrue(torch.equal(module3.root_module.param2, torch.Tensor([2.**4.])))

    def test_pow5(self):
        module1 = ParticleModule(MyModule(1., 2.))
        module2 = module1.clone_from(ParticleModule(MyModule(3., 4.)))
        module2_param1 = module2.root_module.param1
        module2_param2 = module2.root_module.param2
        module2 **= module1
        self.assertIs(module2.root_module.param1, module2_param1)
        self.assertIs(module2.root_module.param2, module2_param2)
        self.assertTrue(torch.equal(module2.root_module.param1, torch.Tensor([3.**1.])))
        self.assertTrue(torch.equal(module2.root_module.param2, torch.Tensor([4.**2.])))
    
    def test_add1(self):
        module1 = ParticleModule(MyModule(1., 2.))
        module2 = module1 + 10
        self.assertIsNot(module1.root_module.param1, module2.root_module.param1)
        self.assertIsNot(module1.root_module.param2, module2.root_module.param2)
        self.assertTrue(torch.equal(module2.root_module.param1, torch.Tensor([11.])))
        self.assertTrue(torch.equal(module2.root_module.param2, torch.Tensor([12.])))

    def test_add2(self):
        module1 = ParticleModule(MyModule(1., 2.))
        module2 = 10 + module1
        self.assertIsNot(module1.root_module.param1, module2.root_module.param1)
        self.assertIsNot(module1.root_module.param2, module2.root_module.param2)
        self.assertTrue(torch.equal(module2.root_module.param1, torch.Tensor([11.])))
        self.assertTrue(torch.equal(module2.root_module.param2, torch.Tensor([12.])))
        
    def test_add3(self):
        module = ParticleModule(MyModule(1., 2.))
        module_param1 = module.root_module.param1
        module_param2 = module.root_module.param2
        module += 10
        self.assertIs(module.root_module.param1, module_param1)
        self.assertIs(module.root_module.param2, module_param2)
        self.assertTrue(torch.equal(module.root_module.param1, torch.Tensor([11.])))
        self.assertTrue(torch.equal(module.root_module.param2, torch.Tensor([12.])))

    def test_add4(self):
        module1 = ParticleModule(MyModule(1., 2.))
        module2 = module1.clone_from(ParticleModule(MyModule(3., 4.)))
        module3 = module1 + module2
        self.assertIsNot(module1.root_module.param1, module3.root_module.param1)
        self.assertIsNot(module1.root_module.param2, module3.root_module.param2)
        self.assertIsNot(module2.root_module.param1, module3.root_module.param1)
        self.assertIsNot(module2.root_module.param2, module3.root_module.param2)
        self.assertTrue(torch.equal(module3.root_module.param1, torch.Tensor([1.+3.])))
        self.assertTrue(torch.equal(module3.root_module.param2, torch.Tensor([2.+4.])))

    def test_add5(self):
        module1 = ParticleModule(MyModule(1., 2.))
        module2 = module1.clone_from(ParticleModule(MyModule(3., 4.)))
        module2_param1 = module2.root_module.param1
        module2_param2 = module2.root_module.param2
        module2 += module1
        self.assertIs(module2.root_module.param1, module2_param1)
        self.assertIs(module2.root_module.param2, module2_param2)
        self.assertTrue(torch.equal(module2.root_module.param1, torch.Tensor([3.+1.])))
        self.assertTrue(torch.equal(module2.root_module.param2, torch.Tensor([4.+2.])))

    def test_sub1(self):
        module1 = ParticleModule(MyModule(1., 2.))
        module2 = module1 - 10
        self.assertIsNot(module1.root_module.param1, module2.root_module.param1)
        self.assertIsNot(module1.root_module.param2, module2.root_module.param2)
        self.assertTrue(torch.equal(module2.root_module.param1, torch.Tensor([1. - 10])))
        self.assertTrue(torch.equal(module2.root_module.param2, torch.Tensor([2. - 10])))

    def test_sub2(self):
        module1 = ParticleModule(MyModule(1., 2.))
        module2 = 10 - module1
        self.assertIsNot(module1.root_module.param1, module2.root_module.param1)
        self.assertIsNot(module1.root_module.param2, module2.root_module.param2)
        self.assertTrue(torch.equal(module2.root_module.param1, torch.Tensor([10 - 1.])))
        self.assertTrue(torch.equal(module2.root_module.param2, torch.Tensor([10 - 2.])))
        
    def test_sub3(self):
        module = ParticleModule(MyModule(1., 2.))
        module_param1 = module.root_module.param1
        module_param2 = module.root_module.param2
        module -= 10
        self.assertIs(module.root_module.param1, module_param1)
        self.assertIs(module.root_module.param2, module_param2)
        self.assertTrue(torch.equal(module.root_module.param1, torch.Tensor([1. - 10])))
        self.assertTrue(torch.equal(module.root_module.param2, torch.Tensor([2. - 10])))

    def test_sub4(self):
        module1 = ParticleModule(MyModule(1., 2.))
        module2 = module1.clone_from(ParticleModule(MyModule(3., 4.)))
        module3 = module1 - module2
        self.assertIsNot(module1.root_module.param1, module3.root_module.param1)
        self.assertIsNot(module1.root_module.param2, module3.root_module.param2)
        self.assertIsNot(module2.root_module.param1, module3.root_module.param1)
        self.assertIsNot(module2.root_module.param2, module3.root_module.param2)
        self.assertTrue(torch.equal(module3.root_module.param1, torch.Tensor([1.-3.])))
        self.assertTrue(torch.equal(module3.root_module.param2, torch.Tensor([2.-4.])))

    def test_sub5(self):
        module1 = ParticleModule(MyModule(1., 2.))
        module2 = module1.clone_from(ParticleModule(MyModule(3., 4.)))
        module2_param1 = module2.root_module.param1
        module2_param2 = module2.root_module.param2
        module2 -= module1
        self.assertIs(module2.root_module.param1, module2_param1)
        self.assertIs(module2.root_module.param2, module2_param2)
        self.assertTrue(torch.equal(module2.root_module.param1, torch.Tensor([3.-1.])))
        self.assertTrue(torch.equal(module2.root_module.param2, torch.Tensor([4.-2.])))

    def test_mul1(self):
        module1 = ParticleModule(MyModule(1., 2.))
        module2 = module1 * 10
        self.assertIsNot(module1.root_module.param1, module2.root_module.param1)
        self.assertIsNot(module1.root_module.param2, module2.root_module.param2)
        self.assertTrue(torch.equal(module2.root_module.param1, torch.Tensor([1. * 10])))
        self.assertTrue(torch.equal(module2.root_module.param2, torch.Tensor([2. * 10])))

    def test_mul2(self):
        module1 = ParticleModule(MyModule(1., 2.))
        module2 = 10 * module1
        self.assertIsNot(module1.root_module.param1, module2.root_module.param1)
        self.assertIsNot(module1.root_module.param2, module2.root_module.param2)
        self.assertTrue(torch.equal(module2.root_module.param1, torch.Tensor([10 * 1.])))
        self.assertTrue(torch.equal(module2.root_module.param2, torch.Tensor([10 * 2.])))
        
    def test_mul3(self):
        module = ParticleModule(MyModule(1., 2.))
        module_param1 = module.root_module.param1
        module_param2 = module.root_module.param2
        module *= 10
        self.assertIs(module.root_module.param1, module_param1)
        self.assertIs(module.root_module.param2, module_param2)
        self.assertTrue(torch.equal(module.root_module.param1, torch.Tensor([1. * 10])))
        self.assertTrue(torch.equal(module.root_module.param2, torch.Tensor([2. * 10])))

    def test_mul4(self):
        module1 = ParticleModule(MyModule(1., 2.))
        module2 = module1.clone_from(ParticleModule(MyModule(3., 4.)))
        module3 = module1 * module2
        self.assertIsNot(module1.root_module.param1, module3.root_module.param1)
        self.assertIsNot(module1.root_module.param2, module3.root_module.param2)
        self.assertIsNot(module2.root_module.param1, module3.root_module.param1)
        self.assertIsNot(module2.root_module.param2, module3.root_module.param2)
        self.assertTrue(torch.equal(module3.root_module.param1, torch.Tensor([1.*3.])))
        self.assertTrue(torch.equal(module3.root_module.param2, torch.Tensor([2.*4.])))

    def test_mul5(self):
        module1 = ParticleModule(MyModule(1., 2.))
        module2 = module1.clone_from(ParticleModule(MyModule(3., 4.)))
        module2_param1 = module2.root_module.param1
        module2_param2 = module2.root_module.param2
        module2 *= module1
        self.assertIs(module2.root_module.param1, module2_param1)
        self.assertIs(module2.root_module.param2, module2_param2)
        self.assertTrue(torch.equal(module2.root_module.param1, torch.Tensor([3.*1.])))
        self.assertTrue(torch.equal(module2.root_module.param2, torch.Tensor([4.*2.])))

    def test_truediv1(self):
        module1 = ParticleModule(MyModule(1., 2.))
        module2 = module1 / 10
        self.assertIsNot(module1.root_module.param1, module2.root_module.param1)
        self.assertIsNot(module1.root_module.param2, module2.root_module.param2)
        self.assertTrue(torch.equal(module2.root_module.param1, torch.Tensor([1. / 10])))
        self.assertTrue(torch.equal(module2.root_module.param2, torch.Tensor([2. / 10])))

    def test_truediv2(self):
        module1 = ParticleModule(MyModule(1., 2.))
        module2 = 10 / module1
        self.assertIsNot(module1.root_module.param1, module2.root_module.param1)
        self.assertIsNot(module1.root_module.param2, module2.root_module.param2)
        self.assertTrue(torch.equal(module2.root_module.param1, torch.Tensor([10 / 1.])))
        self.assertTrue(torch.equal(module2.root_module.param2, torch.Tensor([10 / 2.])))
        
    def test_truediv3(self):
        module = ParticleModule(MyModule(1., 2.))
        module_param1 = module.root_module.param1
        module_param2 = module.root_module.param2
        module /= 10
        self.assertIs(module.root_module.param1, module_param1)
        self.assertIs(module.root_module.param2, module_param2)
        self.assertTrue(torch.equal(module.root_module.param1, torch.Tensor([1. / 10])))
        self.assertTrue(torch.equal(module.root_module.param2, torch.Tensor([2. / 10])))

    def test_truediv4(self):
        module1 = ParticleModule(MyModule(1., 2.))
        module2 = module1.clone_from(ParticleModule(MyModule(3., 4.)))
        module3 = module1 / module2
        self.assertIsNot(module1.root_module.param1, module3.root_module.param1)
        self.assertIsNot(module1.root_module.param2, module3.root_module.param2)
        self.assertIsNot(module2.root_module.param1, module3.root_module.param1)
        self.assertIsNot(module2.root_module.param2, module3.root_module.param2)
        self.assertTrue(torch.equal(module3.root_module.param1, torch.Tensor([1./3.])))
        self.assertTrue(torch.equal(module3.root_module.param2, torch.Tensor([2./4.])))

    def test_truediv5(self):
        module1 = ParticleModule(MyModule(1., 2.))
        module2 = module1.clone_from(ParticleModule(MyModule(3., 4.)))
        module2_param1 = module2.root_module.param1
        module2_param2 = module2.root_module.param2
        module2 /= module1
        self.assertIs(module2.root_module.param1, module2_param1)
        self.assertIs(module2.root_module.param2, module2_param2)
        self.assertTrue(torch.equal(module2.root_module.param1, torch.Tensor([3./1.])))
        self.assertTrue(torch.equal(module2.root_module.param2, torch.Tensor([4./2.])))

    def test_clone(self):
        module1 = ParticleModule(MyModule(1., 2.))
        module2 = module1.clone()
        self.assertIsNot(module1.root_module.param1, module2.root_module.param1)
        self.assertIsNot(module1.root_module.param2, module2.root_module.param2)
        self.assertTrue(torch.equal(module2.root_module.param1, torch.Tensor([1.])))
        self.assertTrue(torch.equal(module2.root_module.param2, torch.Tensor([2.])))

    def test_rand(self):
        module1 = ParticleModule(MyModule(1., 2.))
        module2 = module1.rand()
        self.assertIsNot(module1.root_module.param1, module2.root_module.param1)
        self.assertIsNot(module1.root_module.param2, module2.root_module.param2)
        self.assertEqual(module2.root_module.param1.shape, torch.Size([1]))
        self.assertEqual(module2.root_module.param2.shape, torch.Size([1]))
        self.assertTrue((module2.root_module.param1 >= 0).all())
        self.assertTrue((module2.root_module.param1 <= 1).all())
        self.assertTrue((module2.root_module.param2 >= 0).all())
        self.assertTrue((module2.root_module.param2 <= 1).all())

    def test_randn(self):
        module1 = ParticleModule(MyModule(1., 2.))
        module2 = module1.randn()
        self.assertIsNot(module1.root_module.param1, module2.root_module.param1)
        self.assertIsNot(module1.root_module.param2, module2.root_module.param2)
        self.assertEqual(module2.root_module.param1.shape, torch.Size([1]))
        self.assertEqual(module2.root_module.param2.shape, torch.Size([1]))
