#!/usr/bin/env python

import numpy
import unittest
from pyscf import neo

class KnownValues(unittest.TestCase):
    def test_scf(self):
        mol = neo.Mole()
        mol.build(atom='''H 0 0 0; C 0 0 1.064; N 0 0 2.220''', basis='ccpvdz', quantum_nuc=[0])
        mf = neo.HF(mol)
        energy = mf.scf()
        self.assertAlmostEqual(mol.mass[1], 12.011, 4)
        self.assertAlmostEqual(mol.mass[2], 14.007, 4)
        self.assertAlmostEqual(energy, -92.8437063565785, 8)

    def test_scf2(self):
        mol = neo.Mole()
        mol.build(atom='''H 0 0 0; C 0 0 1.064; N 0 0 2.220''', basis='ccpvdz', quantum_nuc=[0,1])
        mf = neo.HF(mol)
        energy = mf.scf()
        self.assertAlmostEqual(mol.mass[1], 11.996708520544875, 4)
        self.assertAlmostEqual(mol.mass[2], 14.007, 4)
        self.assertAlmostEqual(energy, -92.30126987320989, 8)

    def test_scf3(self):
        mol = neo.Mole()
        mol.build(atom='''H 0 0 0; C 0 0 1.064; N 0 0 2.220''', basis='ccpvdz', quantum_nuc=[0,1,2])
        mf = neo.HF(mol)
        energy = mf.scf()
        self.assertAlmostEqual(mol.mass[1], 11.996708520544875, 4)
        self.assertAlmostEqual(mol.mass[2], 13.999233940635687, 4)
        self.assertAlmostEqual(energy, -91.60852522077765, 8)

    def test_scf_hcore_guess(self):
        mol = neo.Mole()
        mol.build(atom='''H 0 0 0; C 0 0 1.064; N 0 0 2.220''', basis='ccpvdz', quantum_nuc=[0])
        mf = neo.HF(mol)
        mf.init_guess = 'hcore'
        energy = mf.scf()
        self.assertAlmostEqual(mol.mass[1], 12.011, 4)
        self.assertAlmostEqual(mol.mass[2], 14.007, 4)
        self.assertAlmostEqual(energy, -92.8437063565785, 8)

if __name__ == "__main__":
    print("Full Tests for neo.hf")
    unittest.main()
