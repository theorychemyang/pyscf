#!/usr/bin/env python

import unittest
from pyscf import neo

def setUpModule():
    global mol
    mol = neo.M(atom='''H 0 0 0; C 0 0 1.064; N 0 0 2.220''', basis='ccpvdz',
                quantum_nuc=[0])

def tearDownModule():
    global mol
    del mol

class KnownValues(unittest.TestCase):
    def test_scf_noepc(self):
        mol.direct_vee = False
        mf = neo.KS(mol, xc='b3lyp5', epc=None)
        self.assertAlmostEqual(mf.scf(), -93.3393561862047, 8)

    def test_scf_epc17_1(self):
        mol.direct_vee = False
        mf = neo.KS(mol, xc='b3lyp5', epc='17-1')
        mf.max_cycle = 300
        self.assertAlmostEqual(mf.scf(), -93.3963855884953, 6)

    def test_scf_epc17_2(self):
        mol.direct_vee = False
        mf = neo.KS(mol, xc='b3lyp5', epc='17-2')
        self.assertAlmostEqual(mf.scf(), -93.3670499232414, 6)

    def test_scf_epc17_2_dvee(self):
        mol.direct_vee = True
        mf = neo.KS(mol, xc='b3lyp5', epc='17-2')
        self.assertAlmostEqual(mf.scf(), -93.3670499232414, 6)

    def test_scf_epc17_2_UKS(self):
        mol.direct_vee = False
        mf = neo.KS(mol, xc='b3lyp5', epc='17-2', unrestricted=True)
        self.assertAlmostEqual(mf.scf(), -93.3670499232414, 6)

    def test_scf_epc17_2_dvee_UKS(self):
        mol.direct_vee = True
        mf = neo.KS(mol, xc='b3lyp5', epc='17-2', unrestricted=True)
        self.assertAlmostEqual(mf.scf(), -93.3670499232414, 6)

    def test_direct_scf_energy(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 0.9', basis='sto-3g',
                    quantum_nuc=[0], verbose=0)
        mf = neo.KS(mol, xc='b3lyp5', epc=None)
        mf.direct_scf = True
        e_direct = mf.kernel()

        mf = neo.KS(mol, xc='b3lyp5', epc=None)
        mf.direct_scf = False
        for comp in mf.components.values():
            comp.direct_scf = False
        e_incore = mf.kernel()
        self.assertAlmostEqual(e_direct, e_incore, 8)


if __name__ == "__main__":
    print("Full Tests for neo.ks")
    unittest.main()
