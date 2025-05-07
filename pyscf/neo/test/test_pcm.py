#!/usr/bin/env python

import unittest
import numpy
from pyscf import neo, lib


class KnownValues(unittest.TestCase):
    def test_grad_fd(self):
        mol = neo.M(atom='H 0 0 0; C 0 0 1.0754; N 0 0 2.2223',
                    basis='ccpvdz', quantum_nuc=[0])
        mf = neo.CDFT(mol, xc='b3lypg').PCM()
        mf.run()
        de = mf.nuc_grad_method().kernel()

        mfs = mf.as_scanner()
        e1 = mfs('H 0 0 -0.001; C 0 0 1.0754; N 0 0 2.2223')
        e2 = mfs('H 0 0  0.001; C 0 0 1.0754; N 0 0 2.2223')

        self.assertAlmostEqual(de[0,2], (e2-e1)/0.002*lib.param.BOHR, 5)

    def test_hess_fd(self):
        mol = neo.M(atom='H 0 0 0; C 0 0 1.0754; N 0 0 2.2223',
                    basis='ccpvdz', quantum_nuc=[0])
        mf = neo.CDFT(mol, xc='b3lypg').PCM()
        mf.run()
        de = mf.Hessian().kernel()

        mf_grad = mf.nuc_grad_method()
        mf_grad.grid_response = True
        mfs = mf_grad.as_scanner()
        _, g1 = mfs('H 0 0 -0.004; C 0 0 1.0754; N 0 0 2.2223')
        _, g2 = mfs('H 0 0 -0.003; C 0 0 1.0754; N 0 0 2.2223')
        _, g3 = mfs('H 0 0 -0.002; C 0 0 1.0754; N 0 0 2.2223')
        _, g4 = mfs('H 0 0 -0.001; C 0 0 1.0754; N 0 0 2.2223')
        _, g5 = mfs('H 0 0  0.001; C 0 0 1.0754; N 0 0 2.2223')
        _, g6 = mfs('H 0 0  0.002; C 0 0 1.0754; N 0 0 2.2223')
        _, g7 = mfs('H 0 0  0.003; C 0 0 1.0754; N 0 0 2.2223')
        _, g8 = mfs('H 0 0  0.004; C 0 0 1.0754; N 0 0 2.2223')

        fd = 1/280 * g1[0,2] + -4/105 * g2[0,2] + 1/5 * g3[0,2] + -4/5 * g4[0,2] \
             + 4/5 * g5[0,2] + -1/5 * g6[0,2] + 4/105 * g7[0,2] - 1/280 * g8[0,2]
        self.assertAlmostEqual(de[0,0,2,2], fd/0.001*lib.param.BOHR, 4) # can't use 5

    def test_hess_unrestricted(self):
        mol = neo.M(atom='H 0 0 0; C 0 0 1.0754; N 0 0 2.2223',
                    basis='ccpvdz', quantum_nuc=[0])
        mf = neo.CDFT(mol, xc='b3lypg').PCM()
        mf.run()
        de = mf.Hessian().kernel()
        mf2 = neo.CDFT(mol, xc='b3lypg', unrestricted=True).PCM()
        mf2.run()
        de2 = mf2.Hessian().kernel()

        self.assertAlmostEqual(numpy.max(numpy.abs(de-de2)), 0, 5)

if __name__ == "__main__":
    print("Full Tests for neo PCM")
    unittest.main()
