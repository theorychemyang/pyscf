#!/usr/bin/env python

import unittest
from pyscf import neo

class KnownValues(unittest.TestCase):
    def test_scf_epc17_2(self):
        mol = neo.M(atom='''H 0 0 0; C 0 0 1.064; N 0 0 2.220''', basis='ccpvdz',
                    quantum_nuc=[0])
        mf = neo.KS(mol, xc='b3lyp5', epc='17-2').density_fit(auxbasis='cc-pVTZ-JKFIT')
        self.assertAlmostEqual(mf.scf(), -93.3670499232414, 4)

    def test_grad_cdft(self):
        mol = neo.M(atom='''H 0 0 0; F 0 0 0.94''', basis='ccpvdz',
                    quantum_nuc=[0])
        mf = neo.CDFT(mol, xc='b3lyp5').density_fit(auxbasis='cc-pVTZ-JKFIT')
        mf.scf()
        grad = mf.Gradients().kernel()
        self.assertAlmostEqual(grad[0,-1], 0.0051328678351677814, 4)

    def test_hess_H2O(self):
        mol = neo.Mole()
        mol.build(atom='''H -8.51391085e-01 -4.92895828e-01 -3.82461113e-16;
                          H  6.79000285e-01 -7.11874586e-01 -9.84713973e-16;
                          O  6.51955650e-04  4.57954140e-03 -1.81537015e-15''',
                  basis='ccpvdz', quantum_nuc=[0,1])
        mf = neo.CDFT(mol, xc='b3lyp5').density_fit(auxbasis='cc-pVTZ-JKFIT')
        mf.scf()

        hess = neo.Hessian(mf)
        h = hess.kernel()
        results = hess.harmonic_analysis(mol, h)

        self.assertAlmostEqual(results['freq_wavenumber'][-1], 3713.686, 0)
        self.assertAlmostEqual(results['freq_wavenumber'][-2], 3609.801, 0)
        self.assertAlmostEqual(results['freq_wavenumber'][-3], 1572.818, 0)


if __name__ == "__main__":
    print("Full Tests for electronic DF in NEO")
    unittest.main()
