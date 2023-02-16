#!/usr/bin/env python

import unittest
from pyscf import neo, lib


class KnownValues(unittest.TestCase):
    def test_grad_cdft(self):
        mol = neo.M(atom='''H 0 0 0; F 0 0 0.94''', basis='ccpvdz',
                    quantum_nuc=[0])
        mf = neo.CDFT(mol)
        mf.scf()
        grad = mf.Gradients().kernel()
        self.assertAlmostEqual(grad[0,-1], 0.005128004854961732, 6)

    def test_grad_cdft2(self):
        mol = neo.M(atom='''H 0 0 0; F 0 0 0.94''', basis='ccpvdz',
                    quantum_nuc=[0,1])
        mf = neo.CDFT(mol)
        mf.scf()
        grad = mf.Gradients().kernel()
        self.assertAlmostEqual(grad[0,-1], 0.00429936220126681, 6)

    def test_grad_fd(self):
        mol = neo.M(atom='H 0 0 0; C 0 0 1.0754; N 0 0 2.2223',
                    basis='ccpvdz', quantum_nuc=[0])
        mf = neo.CDFT(mol)
        mf.run()
        de = mf.nuc_grad_method().kernel()

        mfs = mf.as_scanner()
        e1 = mfs('H 0 0 -0.001; C 0 0 1.0754; N 0 0 2.2223')
        e2 = mfs('H 0 0  0.001; C 0 0 1.0754; N 0 0 2.2223')

        self.assertAlmostEqual(de[0,2], (e2-e1)/0.002*lib.param.BOHR, 5)


if __name__ == "__main__":
    print("Full Tests for neo.ks")
    unittest.main()
