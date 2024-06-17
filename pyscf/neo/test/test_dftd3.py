#!/usr/bin/env python

import numpy
import unittest
from pyscf import neo

class KnownValues(unittest.TestCase):
    def test_hess_H2O(self):
        mol = neo.M(atom='''H -8.51391085e-01 -4.92895828e-01 -3.82461113e-16;
                            H  6.79000285e-01 -7.11874586e-01 -9.84713973e-16;
                            O  6.51955650e-04  4.57954140e-03 -1.81537015e-15''',
                    basis='ccpvdz', quantum_nuc=[0,1])
        mf = neo.CDFT(mol)
        mf.mf_elec.xc = 'b3lyp5'
        mf.disp = 'd3bj'
        self.assertAlmostEqual(mf.scf(), -76.30826622431475, 6)
        self.assertAlmostEqual(mf.Gradients().grad()[0,0], 0.00043510510599942265, 5)

        hess = neo.Hessian(mf)
        h = hess.kernel()
        results = hess.harmonic_analysis(mol, h)

        self.assertAlmostEqual(results['freq_wavenumber'][-1], 3713.606, 2)
        self.assertAlmostEqual(results['freq_wavenumber'][-2], 3609.742, 2)
        self.assertAlmostEqual(results['freq_wavenumber'][-3], 1572.843, 2)

if __name__ == "__main__":
    print("Full Tests for dftd")
    unittest.main()
