#!/usr/bin/env python

import numpy
import scipy
import unittest
from pyscf import scf
from pyscf import neo
from pyscf.neo import qc

class KnownValues(unittest.TestCase):
    def test_qc(self):
        mol = neo.M(atom='H 0 0 0; H 0.74 0 0', basis='STO-3G', 
                    quantum_nuc = [0,1], nuc_basis = '1s1p', spin=0)
        mf = neo.HF(mol)
        mf.scf()
        qc_mf = qc.FCI(mf)
        e, c, n, s2 = qc_mf.kernel()
        self.assertAlmostEqual(e[0], -1.057677072326751, 10)
        self.assertAlmostEqual(n[0],  4.00000, 8)
        self.assertAlmostEqual(s2[0], 0.0000000, 8)

    def test_qc1(self):
        mol = neo.M(atom='H 0 0 0; H 0.74 0 0', basis='STO-3G', 
                    quantum_nuc = [1], nuc_basis = '2s1p', spin=0)
        mf = neo.HF(mol)
        mf.scf()
        qc_mf = qc.CFCI(mf)
        e, c, n, s2 = qc_mf.kernel()
        self.assertAlmostEqual(e, -1.096593699077799, 10)
        self.assertAlmostEqual(n,  3.00000, 8)
        self.assertAlmostEqual(s2, 0.0000000, 8)

    def test_qc2(self):
        mol = neo.M(atom='H 0 0 0; H 0.74 0 0', basis='STO-3G', 
                    quantum_nuc = [0,1], nuc_basis = '1s1p', spin=0)
        mf = neo.HF(mol)
        mf.scf()
        qc_mf = qc.UCC(mf)
        e, c, n, s2 = qc_mf.kernel()
        self.assertAlmostEqual(e, -1.057617041570153, 9)
        self.assertAlmostEqual(n,  4.00000, 8)
        self.assertAlmostEqual(s2, 0.0000000, 8)

    def test_qc3(self):
        mol = neo.M(atom='H 0 0 0; H 0.74 0 0', basis='STO-3G', 
                    quantum_nuc = [1], nuc_basis = '2s1p', spin=0)
        mf = neo.CDFT(mol, xc='HF')
        mf.scf()
        qc_mf = qc.CUCC(mf)
        e, c, n, s2 = qc_mf.kernel()
        self.assertAlmostEqual(e, -1.096541541344102, 8)
        self.assertAlmostEqual(n,  3.00000, 8)
        self.assertAlmostEqual(s2, 0.0000000, 8)

    def test_qc4(self):
        mol = neo.M(atom='H 0 0 0; H+ 0.57 0 0', basis='STO-6G', 
                    quantum_nuc = [0,1], nuc_basis = '2s1p', spin=0)
        mf = neo.CDFT(mol, xc='HF')
        mf.scf()
        qc_mf = qc.CFCI(mf)
        e, c, n, s2 = qc_mf.kernel()

        rdm1_e = qc_mf.make_rdm1_e()
        rdm1_n = qc_mf.make_rdm1_n()

        eigvals_e = scipy.linalg.eigh(rdm1_e, eigvals_only=True)
        svn_1rdm_e = -sum(e*numpy.log(e) for e in eigvals_e if e>1e-15)

        svn_1rdm_n = []
        for i in range(len(mf.mf_nuc)):
            eigvals_n = scipy.linalg.eigh(rdm1_n[i], eigvals_only=True)
            svn_1rdm_n.append(-sum(e*numpy.log(e) for e in eigvals_n if e>1e-15))

        s_e = qc_mf.entropy(0)
        s_n1 = qc_mf.entropy(1)
        s_n2 = qc_mf.entropy(2)
        s_en1 = qc_mf.entropy([0,1])
        s_en2 = qc_mf.entropy([0,2])
        s_n1n2 = qc_mf.entropy([1,2])

        self.assertAlmostEqual(e, -1.037664417123998, 10)
        self.assertAlmostEqual(n,  4.0000, 8)
        self.assertAlmostEqual(s2, 0.0000000, 8)
        self.assertAlmostEqual(svn_1rdm_e, 0.08957917799469316, 10)
        self.assertAlmostEqual(svn_1rdm_n[0], 0.3586027811164264, 10)
        self.assertAlmostEqual(svn_1rdm_n[1], 0.3572930391859235, 10)
        self.assertAlmostEqual(s_e, 0.021693342011370367, 10)
        self.assertAlmostEqual(s_n1, 0.35860278111642574, 10)
        self.assertAlmostEqual(s_n2, 0.3572930391859235, 10)
        self.assertAlmostEqual(s_en1, 0.35729303918592326, 10)
        self.assertAlmostEqual(s_en2, 0.3586027811164263, 10)
        self.assertAlmostEqual(s_n1n2, 0.02169334201137005, 10)

if __name__ == "__main__":
    print("Full Tests for neo.qc")
    unittest.main()
