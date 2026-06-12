#!/usr/bin/env python

import numpy
import unittest
from pyscf import neo

class KnownValues(unittest.TestCase):
    def test_scf(self):
        mol = neo.Mole()
        mol.build(atom='''H 0 0 0; C 0 0 1.064; N 0 0 2.220''', basis='ccpvdz', quantum_nuc=[0])
        mf = neo.HF(mol)
        energy = mf.scf()
        self.assertAlmostEqual(mol.mass[1], 12.011, 4)
        self.assertAlmostEqual(mol.mass[2], 14.007, 4)
        self.assertAlmostEqual(energy, -92.8437063565785, 8)

    def test_scf2(self):
        mol = neo.Mole()
        mol.build(atom='''H 0 0 0; C 0 0 1.064; N 0 0 2.220''', basis='ccpvdz', quantum_nuc=[0,1])
        mf = neo.HF(mol)
        energy = mf.scf()
        self.assertAlmostEqual(mol.mass[1], 11.996708520544875, 4)
        self.assertAlmostEqual(mol.mass[2], 14.007, 4)
        self.assertAlmostEqual(energy, -92.30126987320989, 8)

    def test_scf3(self):
        mol = neo.Mole()
        mol.build(atom='''H 0 0 0; C 0 0 1.064; N 0 0 2.220''', basis='ccpvdz', quantum_nuc=[0,1,2])
        mf = neo.HF(mol)
        energy = mf.scf()
        self.assertAlmostEqual(mol.mass[1], 11.996708520544875, 4)
        self.assertAlmostEqual(mol.mass[2], 13.999233940635687, 4)
        self.assertAlmostEqual(energy, -91.60852522077765, 8)

    def test_scf_hcore_guess(self):
        mol = neo.Mole()
        mol.build(atom='''H 0 0 0; C 0 0 1.064; N 0 0 2.220''', basis='ccpvdz', quantum_nuc=[0])
        mf = neo.HF(mol)
        mf.init_guess = 'hcore'
        energy = mf.scf()
        self.assertAlmostEqual(mol.mass[1], 12.011, 4)
        self.assertAlmostEqual(mol.mass[2], 14.007, 4)
        self.assertAlmostEqual(energy, -92.8437063565785, 8)

    def test_direct_scf_energy(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 0.9', basis='sto-3g',
                    quantum_nuc=[0], verbose=0)
        mf = neo.HF(mol)
        mf.direct_scf = True
        e_direct = mf.kernel()

        mf = neo.HF(mol)
        mf.direct_scf = False
        for comp in mf.components.values():
            comp.direct_scf = False
        e_incore = mf.kernel()
        self.assertAlmostEqual(e_direct, e_incore, 8)

    def test_component_fock_has_inter_type_coulomb(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 1', basis='sto-3g',
                    quantum_nuc=[0], verbose=0)
        mf = neo.HF(mol)
        mf.scf()
        dm = mf.make_rdm1()
        vhf = mf.get_veff(mf.mol, dm)
        fock = mf.get_fock(mf.get_hcore(), mf.get_ovlp(), vhf, dm)
        fock_e = mf.components['e'].get_fock(dm=dm['e'])
        self.assertAlmostEqual(numpy.linalg.norm(fock['e'] - fock_e), 0, 12)

    def test_component_fock_without_inter_type_cache_raises(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 1', basis='sto-3g',
                    quantum_nuc=[0], verbose=0)
        mf = neo.HF(mol)
        dm = mf.get_init_guess()
        with self.assertRaises(RuntimeError):
            mf.components['e'].get_fock(dm=dm['e'])

if __name__ == "__main__":
    print("Full Tests for neo.hf")
    unittest.main()
