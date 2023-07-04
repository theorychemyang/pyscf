#!/usr/bin/env python

import unittest
from pyscf import neo, lib


class KnownValues(unittest.TestCase):
    def test_grad_cdft(self):
        mol = neo.M(atom='''H 0 0 0; F 0 0 0.94''', basis='ccpvdz',
                    quantum_nuc=[0])
        mf = neo.CDFT(mol)
        mf.mf_elec.xc = 'b3lyp5'
        mf.scf()
        grad = mf.Gradients().kernel()
        self.assertAlmostEqual(grad[0,-1], 0.0051328678351677814, 6)

    def test_grad_cdft2(self):
        mol = neo.M(atom='''H 0 0 0; F 0 0 0.94''', basis='ccpvdz',
                    quantum_nuc=[0,1])
        mf = neo.CDFT(mol)
        mf.mf_elec.xc = 'b3lyp5'
        mf.scf()
        grad = mf.Gradients().kernel()
        self.assertAlmostEqual(grad[0,-1], 0.004304132955144091, 6)

    def test_grad_fd(self):
        mol = neo.M(atom='H 0 0 0; C 0 0 1.0754; N 0 0 2.2223',
                    basis='ccpvdz', quantum_nuc=[0])
        mf = neo.CDFT(mol)
        mf.mf_elec.xc = 'b3lyp5'
        mf.run()
        de = mf.nuc_grad_method().kernel()

        mfs = mf.as_scanner()
        e1 = mfs('H 0 0 -0.001; C 0 0 1.0754; N 0 0 2.2223')
        e2 = mfs('H 0 0  0.001; C 0 0 1.0754; N 0 0 2.2223')

        self.assertAlmostEqual(de[0,2], (e2-e1)/0.002*lib.param.BOHR, 5)

    def test_epc_grad(self):
        mol = neo.M(atom='''H 0 0 0; F 0 0 0.94''', basis='def2-tzvppd',
                    quantum_nuc=[0])
        mf = neo.CDFT(mol, epc='17-2')
        mf.mf_elec.xc = 'b3lyp5'
        mf.mf_elec.grids.atom_grid = (99,590)
        mf.run()
        de = mf.nuc_grad_method().kernel()
        mfs = mf.as_scanner()
        e1 = mfs('H 0 0 -0.0004; F 0 0 0.94')
        e2 = mfs('H 0 0 -0.0003; F 0 0 0.94')
        e3 = mfs('H 0 0 -0.0002; F 0 0 0.94')
        e4 = mfs('H 0 0 -0.0001; F 0 0 0.94')
        e5 = mfs('H 0 0 0.0001; F 0 0 0.94')
        e6 = mfs('H 0 0 0.0002; F 0 0 0.94')
        e7 = mfs('H 0 0 0.0003; F 0 0 0.94')
        e8 = mfs('H 0 0 0.0004; F 0 0 0.94')

        fd = 1/280 * e1 + -4/105 * e2 + 1/5 * e3 + -4/5 * e4 \
             + 4/5 * e5 + -1/5 * e6 + 4/105 * e7 - 1/280 * e8
        self.assertAlmostEqual(de[0,2], fd/0.0001*lib.param.BOHR, 5)


if __name__ == "__main__":
    print("Full Tests for neo.ks")
    unittest.main()
