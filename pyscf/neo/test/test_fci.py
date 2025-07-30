#!/usr/bin/env python

import numpy
import unittest
from pyscf import neo

def setUpModule():
    global mol
    mol = neo.M(atom='''H 0 0 0; H+ 0 0 0.75''', basis='ccpvtz', nuc_basis='pb4f1')

def tearDownModule():
    global mol
    del mol

class KnownValues(unittest.TestCase):
    def test_cneo_fci(self):
        mf = neo.CDFT(mol, xc='HF', unrestricted=True, epc=None)
        mf.conv_tol_grad = 1e-7
        mf.scf()
        solver = neo.FCI(mf)
        e, _, f = solver.kernel()
        self.assertAlmostEqual(e, -1.1341661019492149, 6)
        if f.size == 6:
            self.assertAlmostEqual(f[2], -0.01128137, 5)
            self.assertAlmostEqual(f[5], 0.01043684, 5)
        else:
            self.assertAlmostEqual(f[0], -0.01128137, 5)
            self.assertAlmostEqual(f[1], 0.01043684, 5)

if __name__ == "__main__":
    print("Full Tests for neo.fci")
    unittest.main()
