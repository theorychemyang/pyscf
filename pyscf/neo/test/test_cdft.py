#!/usr/bin/env python

import numpy
import unittest
from pyscf import neo

mol = neo.Mole()
mol.build(atom='''H 0 0 0; C 0 0 1.064; N 0 0 2.220''', basis='ccpvdz', quantum_nuc=[0])

class KnownValues(unittest.TestCase):
    def test_scf_noepc(self):
        mf = neo.CDFT(mol, epc=None)
        self.assertAlmostEqual(mf.scf(), -93.3384022881535, 6)
        self.assertAlmostEqual(mf.f[0][-1], -4.03031884e-02, 5)

    def test_scf_epc17_1(self):
        mf = neo.CDFT(mol, epc='17-1')
        self.assertAlmostEqual(mf.scf(), -93.3960495642599, 5)

    def test_scf_epc17_2(self):
        mf = neo.CDFT(mol, epc='17-2')
        self.assertAlmostEqual(mf.scf(), -93.3661447224845, 6)


if __name__ == "__main__":
    print("Full Tests for neo.cdft")
    unittest.main()
