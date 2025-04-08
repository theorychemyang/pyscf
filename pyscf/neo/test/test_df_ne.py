#!/usr/bin/env python

import unittest
from pyscf import neo, lib

class KnownValues(unittest.TestCase):
    def test_scf(self):
        mol = neo.M(atom='''H 0 0 0; F 0 0 1''', basis='aug-ccpvdz', quantum_nuc=[0])
        mf = neo.CDFT(mol, xc='b3lypg').density_fit(auxbasis='aug-cc-pvdz-jkfit', df_ne=True)
        self.assertAlmostEqual(mf.scf(), -100.41889510781353, 6)

    def test_scf_multi_proton(self):
        mol = neo.M(atom='''H 0 0 0; H 0 0 1''', basis='aug-ccpvdz', quantum_nuc=['H'])
        mf = neo.CDFT(mol, xc='b3lypg').density_fit(auxbasis='aug-cc-pvdz-jkfit', df_ne=True)
        self.assertAlmostEqual(mf.scf(), -1.078857621403627, 6)

    def test_grad(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 1', basis='aug-ccpvdz')
        mf = neo.CDFT(mol, xc='b3lypg').density_fit(auxbasis='aug-cc-pvdz-jkfit', df_ne=True)
        mf.scf()
        de = mf.Gradients().kernel()

        e_scanner = mf.as_scanner()
        e1 = e_scanner('H 0 0 -0.001; F 0 0 1')
        e2 = e_scanner('H 0 0  0.001; F 0 0 1')

        self.assertAlmostEqual(de[0,2], (e2-e1)/0.002*lib.param.BOHR, 5)

    def test_grad_full_q(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 1', basis='aug-ccpvdz', quantum_nuc=[0,1])
        mf = neo.CDFT(mol, xc='b3lypg').density_fit(auxbasis='aug-cc-pvdz-jkfit', df_ne=True)
        mf.scf()
        de = mf.Gradients().kernel()

        e_scanner = mf.as_scanner()
        e1 = e_scanner('H 0 0 0; F 0 0 0.999')
        e2 = e_scanner('H 0 0 0; F 0 0 1.001')

        self.assertAlmostEqual(de[1,2], (e2-e1)/0.002*lib.param.BOHR, 5)

    def test_scanner(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 0.94', basis='aug-ccpvdz')
        mf = neo.CDFT(mol, xc='b3lypg').density_fit(auxbasis='aug-cc-pvdz-jkfit', df_ne=True)
        grad_scanner = mf.nuc_grad_method().as_scanner()

        mol2 = neo.M(atom='H 0 0 0; F 0 0 1.1', basis='aug-ccpvdz')
        mf2 = neo.CDFT(mol2, xc='b3lypg').density_fit(auxbasis='aug-cc-pvdz-jkfit', df_ne=True)
        mf2.scf()
        grad2 = mf2.Gradients().grad()
        _, grad = grad_scanner(mol2)
        self.assertTrue(abs(grad-grad2).max() < 1e-6)

        mol2 = neo.M(atom='H 0 0 0; F 0 0 1.2', basis='aug-ccpvdz')
        mf2 = neo.CDFT(mol2, xc='b3lypg').density_fit(auxbasis='aug-cc-pvdz-jkfit', df_ne=True)
        mf2.scf()
        grad2 = mf2.Gradients().grad()
        _, grad = grad_scanner(mol2)
        self.assertTrue(abs(grad-grad2).max() < 1e-6)

if __name__ == "__main__":
    print("Full Tests for ee and ne density-fitting in CDFT")
    unittest.main()
