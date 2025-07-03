#!/usr/bin/env python

import unittest
from pyscf import neo, lib


class KnownValues(unittest.TestCase):
    def test_tdgrad_rhf(self):
        mol = neo.M(atom='''H 0 0 0; F 0 0 0.9''', basis='ccpvdz',
                    quantum_nuc=[0])
        mf = neo.CDFT(mol, xc='b3lyp5')
        mf.scf()
        td_mf = mf.TDBase()
        td_mf.kernel()
        td_mfs = td_mf.as_scanner()
        de = td_mf.Gradients().kernel()

        e1 = td_mfs('H 0 0 -0.004; F 0 0 0.9')
        e2 = td_mfs('H 0 0 -0.003; F 0 0 0.9')
        e3 = td_mfs('H 0 0 -0.002; F 0 0 0.9')
        e4 = td_mfs('H 0 0 -0.001; F 0 0 0.9')
        _ = td_mfs('H 0 0 -0.000; F 0 0 0.9')
        e5 = td_mfs('H 0 0  0.001; F 0 0 0.9')
        e6 = td_mfs('H 0 0  0.002; F 0 0 0.9')
        e7 = td_mfs('H 0 0  0.003; F 0 0 0.9')
        e8 = td_mfs('H 0 0  0.004; F 0 0 0.9')

        fd = 1/280 * e1 + -4/105 * e2 + 1/5 * e3 + -4/5 * e4 \
             + 4/5 * e5 + -1/5 * e6 + 4/105 * e7 - 1/280 * e8
        self.assertAlmostEqual(de[0,2], fd[0]/0.001*lib.param.BOHR, 5)

    def test_tdgrad_uhf(self):
        mol = neo.M(atom='''H 0 0 0; F 0 0 0.9''', basis='ccpvdz',
                    quantum_nuc=[0])
        mf = neo.CDFT(mol, xc='b3lyp5', unrestricted=True)
        mf.scf()
        td_mf = mf.TDBase()
        td_mf.kernel(nstates=5)
        de = td_mf.Gradients().kernel(state=3)
        self.assertAlmostEqual(de[0,2], 0.26122615997035065, 6)

if __name__ == "__main__":
    print("Full Tests for neo.tdgrad")
    unittest.main()