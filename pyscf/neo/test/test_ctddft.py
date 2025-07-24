#!/usr/bin/env python

import unittest
from pyscf import neo

def setUpModule():
    global mol
    mol = neo.M(atom='''H 0 0 0; F 0 0 0.9''', basis='ccpvdz',
                quantum_nuc=[0])

def tearDownModule():
    global mol
    del mol

class KnownValues(unittest.TestCase):
    def test_tddft_rhf(self):
        mf = neo.CDFT(mol, xc='hf')
        mf.run()

        td_mf1 = mf.TDDirect()
        e1 = td_mf1.kernel()[0]

        td_mf2 = mf.TDDFT()
        e2 = td_mf2.kernel()[0]

        self.assertAlmostEqual(abs((e2[:len(e1)] - e1) * 27.2114).max(), 0, 5)

    def test_tddft_rks(self):
        mf = neo.CDFT(mol, xc='b3lyp5')
        mf.run()

        td_mf1 = mf.TDDirect()
        e1 = td_mf1.kernel()[0]

        td_mf2 = mf.TDDFT()
        e2 = td_mf2.kernel()[0]

        self.assertAlmostEqual(abs((e2[:len(e1)] - e1) * 27.2114).max(), 0, 5)

    def test_tddft_uhf(self):
        mf = neo.CDFT(mol, xc='hf', unrestricted=True)
        mf.run()

        td_mf1 = mf.TDDirect()
        e1 = td_mf1.kernel()[0]

        td_mf2 = mf.TDDFT()
        e2 = td_mf2.kernel()[0]

        self.assertAlmostEqual(abs((e2[:len(e1)] - e1) * 27.2114).max(), 0, 5)

    def test_tddft_uks(self):
        mf = neo.CDFT(mol, xc='b3lyp5', unrestricted=True)
        mf.run()

        td_mf1 = mf.TDDirect()
        e1 = td_mf1.kernel()[0]

        td_mf2 = mf.TDDFT()
        e2 = td_mf2.kernel()[0]

        self.assertAlmostEqual(abs((e2[:len(e1)] - e1) * 27.2114).max(), 0, 5)


if __name__ == "__main__":
    print("Full Tests for neo.ctddft (Frozen nuclear orbital CNEO-TDDFT Davidson)")
    unittest.main()
