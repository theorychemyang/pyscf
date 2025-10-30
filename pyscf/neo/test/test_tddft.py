#!/usr/bin/env python

import unittest
from pyscf import neo
from pyscf.neo import tddft, tddft_slow

def setUpModule():
    global mol
    mol = neo.M(atom='''H 0 0 0; F 0 0 0.9''', basis='ccpvdz',
                quantum_nuc=[0])

def tearDownModule():
    global mol
    del mol

class KnownValues(unittest.TestCase):
    def test_tddft_rhf(self):
        mf = neo.HF(mol)
        mf.run()

        td_mf1 = tddft_slow.TDDirect(mf)
        e1 = td_mf1.kernel()[0]

        td_mf2 = tddft.TDDFT(mf)
        e2 = td_mf2.kernel()[0]

        self.assertAlmostEqual(abs((e2[:len(e1)] - e1) * 27.2114).max(), 0, 5)

    def test_tddft_rks(self):
        mf = neo.KS(mol, xc='b3lyp5')
        mf.run()

        td_mf1 = tddft_slow.TDDirect(mf)
        e1 = td_mf1.kernel()[0]

        td_mf2 = tddft.TDDFT(mf)
        e2 = td_mf2.kernel()[0]

        self.assertAlmostEqual(abs((e2[:len(e1)] - e1) * 27.2114).max(), 0, 5)

    def test_tddft_epc_17_1(self):
        mf = neo.KS(mol, xc='b3lyp5', epc='17-1')
        mf.run()

        td_mf1 = tddft_slow.TDDirect(mf)
        e1 = td_mf1.kernel()[0]

        td_mf2 = tddft.TDDFT(mf)
        e2 = td_mf2.kernel()[0]

        self.assertAlmostEqual(abs((e2[:len(e1)] - e1) * 27.2114).max(), 0, 5)

    def test_tddft_epc_17_2(self):
        mf = neo.KS(mol, xc='b3lyp5', epc='17-2')
        mf.run()

        td_mf1 = tddft_slow.TDDirect(mf)
        e1 = td_mf1.kernel()[0]

        td_mf2 = tddft.TDDFT(mf)
        e2 = td_mf2.kernel()[0]

        self.assertAlmostEqual(abs((e2[:len(e1)] - e1) * 27.2114).max(), 0, 5)

    def test_tddft_uhf(self):
        mf = neo.HF(mol, unrestricted=True)
        mf.run()

        td_mf1 = tddft_slow.TDDirect(mf)
        e1 = td_mf1.kernel()[0]

        td_mf2 = tddft.TDDFT(mf)
        td_mf2.nstates = 5
        e2 = td_mf2.kernel()[0]

        self.assertAlmostEqual(abs((e2[:len(e1)] - e1) * 27.2114).max(), 0, 5)

    def test_tddft_uks(self):
        mf = neo.KS(mol, xc='b3lyp5', unrestricted=True)
        mf.run()

        td_mf1 = tddft_slow.TDDirect(mf)
        e1 = td_mf1.kernel()[0]

        td_mf2 = tddft.TDDFT(mf)
        e2 = td_mf2.kernel()[0]

        self.assertAlmostEqual(abs((e2[:len(e1)] - e1) * 27.2114).max(), 0, 5)


if __name__ == "__main__":
    print("Full Tests for neo.tddft (NEO-TDDFT Davidson)")
    unittest.main()
