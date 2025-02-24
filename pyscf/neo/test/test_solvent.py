#!/usr/bin/env python

import unittest
from pyscf import neo, lib


class KnownValues(unittest.TestCase):
    def test_energy(self):
        mol = neo.M(atom='H 0 0 0; C 0 0 1.0754; N 0 0 2.2223', basis='ccpvdz', quantum_nuc=[0])
        mf = neo.CDFT(mol, xc='b3lyp5')
        mf = mf.ddCOSMO()
        e = mf.scf()
        self.assertAlmostEqual(e, -93.3452280000717, 8)

    def test_grad_finite_diff(self):

        mol = neo.M(atom='H 0 0 0; C 0 0 1.0754; N 0 0 2.2223', basis='ccpvdz', quantum_nuc=[0])
        mf = neo.CDFT(mol, xc='b3lyp5')
        mf = mf.ddCOSMO()
        mf.with_solvent.lmax = 10
        mf.with_solvent.lebedev_order = 29
        mf.run()
        de = mf.nuc_grad_method().kernel()

        mol1 = neo.M(atom='H 0 0 -0.0001; C 0 0 1.0754; N 0 0 2.2223', basis='ccpvdz', quantum_nuc=[0])
        mf1 = neo.CDFT(mol1, xc='b3lyp5')
        mf1 = mf1.ddCOSMO()
        mf1.with_solvent.lmax = 10
        mf1.with_solvent.lebedev_order = 29
        e1 = mf1.scf()

        mol2 = neo.M(atom='H 0 0 0.0001; C 0 0 1.0754; N 0 0 2.2223', basis='ccpvdz', quantum_nuc=[0])
        mf2 = neo.CDFT(mol2, xc='b3lyp5')
        mf2 = mf2.ddCOSMO()
        mf2.with_solvent.lmax = 10
        mf2.with_solvent.lebedev_order = 29
        e2 = mf2.scf()

        self.assertAlmostEqual(de[0,2], (e2-e1)/0.0002*lib.param.BOHR, 5)


if __name__ == "__main__":
    print("Full Tests for neo.solvent")
    unittest.main()
