#!/usr/bin/env python

import unittest
from pyscf import neo, lib, scf


class KnownValues(unittest.TestCase):
    def test_grad_fd(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 0.9', basis='ccpvdz', quantum_nuc=[0])
        mf = neo.CDFT(mol)
        mf.run()

        hess = mf.Hessian() 
        hess.kernel()
        de = neo.hessian.dipole_grad(hess)

        mol1 = neo.M(atom='H 0 0 -0.001; F 0 0 0.9', basis='ccpvdz', quantum_nuc=[0])
        mf1 = neo.CDFT(mol1)
        mf1.scf()

        mol2 = neo.M(atom='H 0 0 0.001; F 0 0 0.9', basis='ccpvdz', quantum_nuc=[0])
        mf2 = neo.CDFT(mol2)
        mf2.scf()

        self.assertAlmostEqual(de[0,-1,-1], (scf.hf.dip_moment(mol2, mf2.dm_elec, unit='au')[-1] - scf.hf.dip_moment(mol1, mf1.dm_elec, unit='au')[-1])/0.002*lib.param.BOHR, 5)

if __name__ == "__main__":
    print("Full Tests for gradients of molecular dipole moments")
    unittest.main()