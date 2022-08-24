#!/usr/bin/env python

import numpy
import unittest
from pyscf import neo

class KnownValues(unittest.TestCase):
    def test_hess_HF(self):
        mol = neo.Mole()
        mol.build(atom = 'H 0 0 0; F 0 0 0.945', basis = 'ccpvdz', quantum_nuc=[0])
        mf = neo.CDFT(mol)
        mf.mf_elec.xc = 'b3lyp'
        mf.scf()

        hess = neo.Hessian(mf)
        h = hess.kernel()
        results = hess.harmonic_analysis(mol, h)

        self.assertAlmostEqual(results['freq_au'][0], 0.75277497, 5)

    def test_hess_H2O(self):
        mol = neo.Mole()
        mol.build(atom = 'H -8.51391085e-01 -4.92895828e-01 -3.82461113e-16; H 6.79000285e-01 -7.11874586e-01 -9.84713973e-16; O 6.51955650e-04  4.57954140e-03 -1.81537015e-15', basis = 'ccpvdz', quantum_nuc=[0, 1])
        mf = neo.CDFT(mol)
        mf.mf_elec.xc='b3lyp'
        mf.scf()

        hess = neo.Hessian(mf)
        h = hess.kernel()
        results = hess.harmonic_analysis(mol, h)

        self.assertAlmostEqual(results['freq_au'][0], 0.30596696, 5)
        self.assertAlmostEqual(results['freq_au'][1], 0.70222886, 5)
        self.assertAlmostEqual(results['freq_au'][2], 0.72243825, 5)


if __name__ == "__main__":
    print("Full Tests for neo.hess")
    unittest.main()
