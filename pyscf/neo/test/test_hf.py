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
        self.assertAlmostEqual(energy, -92.8437163895288, 8)

    def test_scf2(self):
        mol = neo.Mole()
        mol.build(atom='''H 0 0 0; C 0 0 1.064; N 0 0 2.220''', basis='ccpvdz', quantum_nuc=[0,1])
        mf = neo.HF(mol)
        energy = mf.scf()
        self.assertAlmostEqual(energy, -92.3015949385622, 8)

    def test_scf3(self):
        mol = neo.Mole()
        mol.build(atom='''H 0 0 0; C 0 0 1.064; N 0 0 2.220''', basis='ccpvdz', quantum_nuc=[0,1,2])
        mf = neo.HF(mol)
        energy = mf.scf()
        self.assertAlmostEqual(energy, -91.6090379843231, 8)


if __name__ == "__main__":
    print("Full Tests for neo.hf")
    unittest.main()
