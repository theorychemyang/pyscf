#!/usr/bin/env python

import numpy
import unittest
from pyscf import scf
from pyscf import neo
from pyscf.neo import qc

class KnownValues(unittest.TestCase):
    def test_qc(self):
        mol = neo.M(atom='H 0 0 0; H 0.74 0 0', basis='STO-3G', 
                    quantum_nuc = [0,1], nuc_basis = '1s1p', cart=True, spin=0)
        mf = neo.HF(mol)
        mf.scf()
        qc_mf = qc.FCI(mf)
        e, c, n, s2 = qc_mf.kernel()
        self.assertAlmostEqual(e[0], -1.057564106860389, 10)
        self.assertAlmostEqual(n[0],  4.00000, 8)
        self.assertAlmostEqual(s2[0], 0.0000000, 8)

    def test_qc1(self):
        mol = neo.M(atom='H 0 0 0; H 0.74 0 0', basis='STO-3G', 
                    quantum_nuc = [1], nuc_basis = '2s1p', cart=True, spin=0)
        mf = neo.HF(mol)
        mf.scf()
        qc_mf = qc.CFCI(mf)
        e, c, n, s2 = qc_mf.kernel()
        self.assertAlmostEqual(e, -1.096596826382109, 10)
        self.assertAlmostEqual(n,  3.00000, 8)
        self.assertAlmostEqual(s2, 0.0000000, 8)

    def test_qc2(self):
        mol = neo.M(atom='H 0 0 0; H 0.74 0 0', basis='STO-3G', 
                    quantum_nuc = [0,1], nuc_basis = '1s1p', cart=True, spin=0)
        mf = neo.HF(mol)
        mf.scf()
        qc_mf = qc.UCC(mf)
        e, c, n, s2 = qc_mf.kernel()
        self.assertAlmostEqual(e, -1.057505919249053, 10)
        self.assertAlmostEqual(n,  4.00000, 8)
        self.assertAlmostEqual(s2, 0.0000000, 8)

    def test_qc3(self):
        mol = neo.M(atom='H 0 0 0; H 0.74 0 0', basis='STO-3G', 
                    quantum_nuc = [1], nuc_basis = '2s1p', cart=True, spin=0)
        mf = neo.CDFT(mol)
        mf.mf_elec.xc = 'HF'
        mf.scf()
        qc_mf = qc.CUCC(mf)
        e, c, n, s2 = qc_mf.kernel()
        self.assertAlmostEqual(e, -1.096545337919290, 8)
        self.assertAlmostEqual(n,  3.00000, 8)
        self.assertAlmostEqual(s2, 0.0000000, 8)

if __name__ == "__main__":
    print("Full Tests for neo.qc")
    unittest.main()
