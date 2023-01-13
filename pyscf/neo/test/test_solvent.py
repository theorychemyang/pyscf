#!/usr/bin/env python

import unittest
from pyscf import neo, lib
from pyscf.neo.solvent import ddcosmo_for_neo



class KnownValues(unittest.TestCase):
    def test_energy(self):
        mol = neo.Mole()
        mol.build(atom='H 0 0 0; C 0 0 1.0754; N 0 0 2.2223', basis='ccpvdz', quantum_nuc=[0])
        mf = neo.CDFT(mol)
        mf.scf(cycle=1) 
        mf = ddcosmo_for_neo(mf)
        e = mf.scf()
        self.assertAlmostEqual(e, -93.34534437591918, 8)

    def test_grad_finite_diff(self):
        mol = neo.Mole()
        mol.build(atom='H 0 0 0; C 0 0 1.0754; N 0 0 2.2223', basis='ccpvdz', quantum_nuc=[0])
        mf = neo.CDFT(mol)
        mf.run(cycle=1) # TODO: remove this
        mf = ddcosmo_for_neo(mf)
        mf.run()
        de = mf.nuc_grad_method().kernel()

        mol1 = neo.Mole()
        mol1.build(atom='H 0 0 -0.001; C 0 0 1.0754; N 0 0 2.2223', basis='ccpvdz', quantum_nuc=[0])
        mf1 = neo.CDFT(mol1)
        mf1.run(cycle=1)
        mf1 = ddcosmo_for_neo(mf1)
        e1 = mf1.scf()

        mol2 = neo.Mole()
        mol2.build(atom='H 0 0 0.001; C 0 0 1.0754; N 0 0 2.2223', basis='ccpvdz', quantum_nuc=[0])
        mf2 = neo.CDFT(mol2)
        mf2.run(cycle=1) 
        mf2 = ddcosmo_for_neo(mf2)
        e2 = mf2.scf()

        self.assertAlmostEqual(de[0,2], (e2-e1)/0.002*lib.param.BOHR, 5)


if __name__ == "__main__":
    print("Full Tests for neo.solvent")
    unittest.main()
