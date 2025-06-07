#!/usr/bin/env python

import numpy
import unittest
from pyscf import neo

class KnownValues(unittest.TestCase):
    def test_neo(self):
        # quantum nucleus
        mol_neo = neo.M(atom='''H 0 0 0''', basis='6-31G',
                       charge=-1, spin=0,
                       positron_charge=0, positron_spin=1,
                       quantum_nuc=['H'], nuc_basis='pb4d')
        mf = neo.HF(mol_neo)
        mf.conv_tol_grad = 1e-8
        self.assertAlmostEqual(mf.scf(), -0.5164169985056389, 9)

    def test_rhf_uhf(self):
        # proton + 2 * e- + 2 * e+, possible to be restricted
        mol1 = neo.M(atom='''H 0 0 0''', basis='6-31G',
                     charge=-1, spin=0,
                     positron_charge=-1, positron_spin=0,
                     quantum_nuc=['H'])
        mf1 = neo.HF(mol1, unrestricted=False)
        mf1.conv_tol_grad = 1e-8
        mf2 = neo.HF(mol1, unrestricted=True)
        mf2.conv_tol_grad = 1e-8
        self.assertAlmostEqual(mf1.scf(), mf2.scf(), 9)


if __name__ == "__main__":
    print("Full Tests for positron")
    unittest.main()
