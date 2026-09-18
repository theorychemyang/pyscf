#!/usr/bin/env python

import unittest
from pyscf import neo, lib


class KnownValues(unittest.TestCase):
    def test_tdgrad_rhf(self):
        mol = neo.M(atom='''H 0 0 0; F 0 0 0.9''', basis='ccpvdz',
                    quantum_nuc=[0])
        mf = neo.CDFT(mol, xc='hf')
        mf.conv_tol = 1e-11
        mf.conv_tol_grad = 1e-7
        mf.scf()
        td_mf = mf.TDDFT()
        td_mf.kernel()
        td_mfs = td_mf.as_scanner()
        de = td_mf.Gradients().kernel()

        e1 = td_mfs('H 0 0 -0.004; F 0 0 0.9')
        e2 = td_mfs('H 0 0 -0.003; F 0 0 0.9')
        e3 = td_mfs('H 0 0 -0.002; F 0 0 0.9')
        e4 = td_mfs('H 0 0 -0.001; F 0 0 0.9')
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
        mf = neo.CDFT(mol, xc='hf', unrestricted=True)
        mf.scf()
        td_mf = mf.TDDFT()
        td_mf.kernel(nstates=5)
        de = td_mf.Gradients().kernel(state=3)
        self.assertAlmostEqual(de[0,2], 0.28214115897987635, 6)

    def test_tdgrad_rks(self):
        mol = neo.M(atom='''H 0 0 0; F 0 0 0.9''', basis='ccpvdz',
                    quantum_nuc=[0])
        mf = neo.CDFT(mol, xc='b3lyp5')
        mf.conv_tol = 1e-11
        mf.conv_tol_grad = 1e-7
        mf.scf()
        td_mf = mf.TDDFT()
        td_mf.kernel()
        td_mfs = td_mf.as_scanner()
        de = td_mf.Gradients().kernel()

        e1 = td_mfs('H 0 0 -0.004; F 0 0 0.9')
        e2 = td_mfs('H 0 0 -0.003; F 0 0 0.9')
        e3 = td_mfs('H 0 0 -0.002; F 0 0 0.9')
        e4 = td_mfs('H 0 0 -0.001; F 0 0 0.9')
        e5 = td_mfs('H 0 0  0.001; F 0 0 0.9')
        e6 = td_mfs('H 0 0  0.002; F 0 0 0.9')
        e7 = td_mfs('H 0 0  0.003; F 0 0 0.9')
        e8 = td_mfs('H 0 0  0.004; F 0 0 0.9')

        fd = 1/280 * e1 + -4/105 * e2 + 1/5 * e3 + -4/5 * e4 \
             + 4/5 * e5 + -1/5 * e6 + 4/105 * e7 - 1/280 * e8
        self.assertAlmostEqual(de[0,2], fd[0]/0.001*lib.param.BOHR, 5)

    def test_tdgrad_uks(self):
        mol = neo.M(atom='''H 0 0 0; F 0 0 0.9''', basis='ccpvdz',
                    quantum_nuc=[0])
        mf = neo.CDFT(mol, xc='b3lyp5', unrestricted=True)
        mf.scf()
        td_mf = mf.TDDFT()
        td_mf.kernel(nstates=5)
        de = td_mf.Gradients().kernel(state=3)
        self.assertAlmostEqual(de[0,2], 0.26122615997035065, 6)

    def test_tdgrad_multiple_proton_rks(self):
        mol = neo.M(atom='''O     0.0000   0.0000   0.0000;
                            H     0.7574   0.5868   0.0000;
                            H    -0.7574   0.5868   0.0000''',
                    basis='ccpvdz')
        mf = neo.CDFT(mol, xc='b3lyp5')
        mf.conv_tol = 1e-11
        mf.conv_tol_grad = 1e-7
        mf.scf()
        td_mf = mf.TDDFT()

        td_mf.kernel()
        td_mfs = td_mf.as_scanner()
        de = td_mf.Gradients().kernel()

        e1 = td_mfs('O 0 0 0; H 0.7534 0.5868 0; H -0.7574 0.5868 0')
        e2 = td_mfs('O 0 0 0; H 0.7544 0.5868 0; H -0.7574 0.5868 0')
        e3 = td_mfs('O 0 0 0; H 0.7554 0.5868 0; H -0.7574 0.5868 0')
        e4 = td_mfs('O 0 0 0; H 0.7564 0.5868 0; H -0.7574 0.5868 0')
        e5 = td_mfs('O 0 0 0; H 0.7584 0.5868 0; H -0.7574 0.5868 0')
        e6 = td_mfs('O 0 0 0; H 0.7594 0.5868 0; H -0.7574 0.5868 0')
        e7 = td_mfs('O 0 0 0; H 0.7604 0.5868 0; H -0.7574 0.5868 0')
        e8 = td_mfs('O 0 0 0; H 0.7614 0.5868 0; H -0.7574 0.5868 0')

        fd = 1/280 * e1 + -4/105 * e2 + 1/5 * e3 + -4/5 * e4 \
             + 4/5 * e5 + -1/5 * e6 + 4/105 * e7 - 1/280 * e8
        self.assertAlmostEqual(de[1,0], fd[0]/0.001*lib.param.BOHR, 5)

    def test_tdgrad_multiple_proton_uks(self):
        mol = neo.M(atom='''O     0.0000   0.0000   0.0000;
                            H     0.7574   0.5868   0.0000;
                            H    -0.7574   0.5868   0.0000''',
                    basis='ccpvdz')
        mf = neo.CDFT(mol, xc='b3lyp5', unrestricted=True)
        mf.scf()
        td_mf = mf.TDDFT()
        td_mf.kernel(nstates=5)
        de = td_mf.Gradients().kernel(state=2)
        self.assertAlmostEqual(de[1,0], -0.0902848007, 6)

if __name__ == "__main__":
    print("Full Tests for neo.tdgrad")
    unittest.main()
