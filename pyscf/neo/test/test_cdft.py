#!/usr/bin/env python

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
    def test_scf_symmetry(self):
        for epc in (None, '17-2'):
            with self.subTest(epc=epc):
                energies = []
                for symmetry in (False, True):
                    mol = neo.M(atom='H 0 0 0; C 0 0 1.064; N 0 0 2.220',
                                basis='sto3g', nuc_basis='pb4d', quantum_nuc=[0],
                                symmetry=symmetry, verbose=0)
                    mf = neo.CDFT(mol, xc='pbe', epc=epc)
                    mf.conv_tol = 1e-10
                    energies.append(mf.scf())
                    self.assertTrue(mf.converged)
                    self.assertAlmostEqual(mf.mo_occ['n0'].sum(), 1.)
                self.assertAlmostEqual(energies[0], energies[1], 7)
    def test_diis_type(self):
        def run(diis_type):
            mf = neo.CDFT(mol, xc='b3lyp5', epc=None,
                          diis_type=diis_type)
            mf.conv_tol = 1e-11
            mf.conv_tol_grad = 1e-6
            energy = mf.kernel()
            self.assertTrue(mf.converged)
            return mf, energy
        reference, reference_energy = run(3)
        mf, energy = run(4)
        self.assertAlmostEqual(energy, reference_energy, 7)
        self.assertAlmostEqual(mf.f[0][-1], reference.f[0][-1], 5)

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
