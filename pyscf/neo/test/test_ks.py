#!/usr/bin/env python

import unittest
from pyscf import neo

mol = neo.Mole()
mol.build(atom='''H 0 0 0; C 0 0 1.064; N 0 0 2.220''', basis='ccpvdz', quantum_nuc=[0])

class KnownValues(unittest.TestCase):
    def test_scf_noepc(self):
        mf = neo.KS(mol, epc=None)
        self.assertAlmostEqual(mf.scf(), -93.3393561862119, 8)

    def test_scf_epc17_1(self):
        mf = neo.KS(mol, epc='17-1')
        mf.max_cycle = 2000
        self.assertAlmostEqual(mf.scf(), -93.3963856345767, 5)

    def test_scf_epc17_2(self):
        mf = neo.KS(mol, epc='17-2')
        mf.max_cycle = 1000
        self.assertAlmostEqual(mf.scf(), -93.3670499235643, 5)


if __name__ == "__main__":
    print("Full Tests for neo.ks")
    unittest.main()
