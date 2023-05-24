#!/usr/bin/env python

import numpy
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
        mf = neo.CDFT(mol, epc=None)
        self.assertAlmostEqual(mf.scf(), -93.33840228714486, 6)
        self.assertAlmostEqual(mf.f[0][-1], -0.040303732060570516, 5)

    def test_scf_epc17_1(self):
        mf = neo.CDFT(mol, epc='17-1')
        self.assertAlmostEqual(mf.scf(), -93.39604973627863, 5)

    def test_scf_epc17_2(self):
        mf = neo.CDFT(mol, epc='17-2')
        self.assertAlmostEqual(mf.scf(), -93.36614467776664, 6)

    def test_scf_epc18_1(self):
        mf = neo.CDFT(mol, epc='18-1')
        self.assertAlmostEqual(mf.scf(), -93.38492562345472, 5)

    def test_scf_epc18_2(self):
        mf = neo.CDFT(mol, epc='18-2')
        self.assertAlmostEqual(mf.scf(), -93.36401432623929, 6)


if __name__ == "__main__":
    print("Full Tests for neo.cdft")
    unittest.main()
