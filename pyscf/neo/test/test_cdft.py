#!/usr/bin/env python

import numpy
import unittest
from pyscf import neo

def setUpModule():
    global mol, mol_symm
    mol = neo.M(atom='''H 0 0 0; C 0 0 1.064; N 0 0 2.220''', basis='ccpvdz',
                quantum_nuc=[0])
    mol_symm = neo.M(atom='''H 0 0 0; C 1.064 0 0; N 2.220 0 0''', basis='ccpvdz',
                     quantum_nuc=[0], symmetry=True)

def tearDownModule():
    global mol, mol_symm
    del mol, mol_symm

class KnownValues(unittest.TestCase):
    def test_scf_noepc(self):
        mf = neo.CDFT(mol, xc='b3lyp5', epc=None)
        mf.conv_tol = 1e-11
        mf.conv_tol_grad = 1e-6
        self.assertAlmostEqual(mf.scf(), -93.33840228460394, 8)
        self.assertAlmostEqual(mf.f[0][-1], -0.04030159716283275, 6)
        self.assertAlmostEqual(mf.dip_moment()[-1], -2.8571852181865407, 5)
        mf_symm = neo.CDFT(mol_symm, xc='b3lyp5', epc=None)
        mf_symm.conv_tol = 1e-11
        mf_symm.conv_tol_grad = 1e-6
        self.assertAlmostEqual(mf_symm.scf(), -93.33840228460409, 8)
        self.assertAlmostEqual(mf_symm.f[0][0], -0.040301597162828474, 6)

    def test_scf_epc17_1(self):
        mf = neo.CDFT(mol, xc='b3lyp5', epc='17-1')
        self.assertAlmostEqual(mf.scf(), -93.39604973627863, 5)

    def test_scf_epc17_2(self):
        mf = neo.CDFT(mol, xc='b3lyp5', epc='17-2')
        self.assertAlmostEqual(mf.scf(), -93.36614467776664, 6)

    def test_scf_epc18_1(self):
        mf = neo.CDFT(mol, xc='b3lyp5', epc='18-1')
        mf.max_cycle = 80
        self.assertAlmostEqual(mf.scf(), -93.38492562345472, 5)

    def test_scf_epc18_2(self):
        mf = neo.CDFT(mol, xc='b3lyp5', epc='18-2')
        self.assertAlmostEqual(mf.scf(), -93.36401432623929, 6)

    def test_isotope(self):
        mol_D_atom = neo.M(atom='H+ 0 0 0', spin=1)
        mf = neo.CDFT(mol_D_atom, xc='b3lyp5')
        self.assertAlmostEqual(mf.scf(), -0.432707430519338, 6)


if __name__ == "__main__":
    print("Full Tests for neo.cdft")
    unittest.main()
