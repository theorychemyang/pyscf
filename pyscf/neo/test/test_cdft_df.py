#!/usr/bin/env python

import unittest
from pyscf import neo, lib

class KnownValues(unittest.TestCase):
    def test_scf(self):
        mol = neo.M(atom='''H 0 0 0; F 0 0 1''', basis='aug-ccpvdz',
                    quantum_nuc=[0])
        mf = neo.CDFT(mol, xc='b3lyp').density_fit(auxbasis='aug-ccpvdz-ri', df_ne=True)
        self.assertAlmostEqual(mf.scf(), -100.42101829298, 6)

    def test_scf_multi_proton(self):
        mol = neo.M(atom='''H 0 0 0; H 0 0 1''', basis='aug-ccpvdz',
                    quantum_nuc=['H'])
        mf = neo.CDFT(mol, xc='b3lyp').density_fit(auxbasis='aug-ccpvdz-ri', df_ne=True)
        self.assertAlmostEqual(mf.scf(), -1.07318416263312, 6)

    def test_grad(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 1', basis='aug-ccpvdz')
        mf = neo.CDFT(mol, xc='b3lyp').density_fit(auxbasis='aug-ccpvdz-ri', df_ne=True)
        mf.scf()
        de = mf.Gradients().kernel()

        mol = neo.M(atom='H 0 0 -0.001; F 0 0 1', basis='aug-ccpvdz')
        mf = neo.CDFT(mol, xc='b3lyp').density_fit(auxbasis='aug-ccpvdz-ri', df_ne=True)
        e1 = mf.scf()

        mol = neo.M(atom='H 0 0 0.001; F 0 0 1', basis='aug-ccpvdz')
        mf = neo.CDFT(mol, xc='b3lyp').density_fit(auxbasis='aug-ccpvdz-ri', df_ne=True)
        e2 = mf.scf()

        self.assertAlmostEqual(de[0,2], (e2-e1)/0.002*lib.param.BOHR, 5)

if __name__ == "__main__":
    print("Full Tests for ee and ne density-fitting in CDFT")
    unittest.main()
