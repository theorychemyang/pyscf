#!/usr/bin/env python

import numpy
import unittest
from pyscf import neo

class KnownValues(unittest.TestCase):
    def test_hess_HF(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 0.945', basis='ccpvdz', quantum_nuc=[0])
        mf = neo.CDFT(mol, xc='b3lyp5')
        mf.scf()

        hess = mf.Hessian()
        h = hess.kernel()
        results = hess.harmonic_analysis(mol, h)

        self.assertAlmostEqual(results['freq_wavenumber'][-1], 3869.636, 1)

    def test_hess_H2O(self):
        mol = neo.M(atom='''H -8.51391085e-01 -4.92895828e-01 -3.82461113e-16;
                            H  6.79000285e-01 -7.11874586e-01 -9.84713973e-16;
                            O  6.51955650e-04  4.57954140e-03 -1.81537015e-15''',
                    basis='ccpvdz', quantum_nuc=[0,1])
        mf = neo.CDFT(mol, xc='b3lyp5')
        mf.scf()

        hess = mf.Hessian()
        h = hess.kernel()
        results = hess.harmonic_analysis(mol, h)

        self.assertAlmostEqual(results['freq_wavenumber'][-1], 3713.686, 1)
        self.assertAlmostEqual(results['freq_wavenumber'][-2], 3609.801, 1)
        self.assertAlmostEqual(results['freq_wavenumber'][-3], 1572.818, 1)

    def test_hess_HF_full_q(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 0.945', basis='ccpvdz', quantum_nuc=[0,1])
        mf = neo.CDFT(mol, xc='b3lyp5')
        mf.scf()

        hess = mf.Hessian()
        h = hess.kernel()
        results = hess.harmonic_analysis(mol, h)

        self.assertAlmostEqual(results['freq_wavenumber'][-1], 3873.004, 1)

    def test_hess_H2O_full_q(self):
        mol = neo.M(atom='''H -8.51391085e-01 -4.92895828e-01 -3.82461113e-16;
                            H  6.79000285e-01 -7.11874586e-01 -9.84713973e-16;
                            O  6.51955650e-04  4.57954140e-03 -1.81537015e-15''',
                    basis='ccpvdz', quantum_nuc=[0,1,2])
        mf = neo.CDFT(mol, xc='b3lyp5')
        mf.scf()

        hess = mf.Hessian()
        h = hess.kernel()
        results = hess.harmonic_analysis(mol, h)

        self.assertAlmostEqual(results['freq_wavenumber'][-1], 3720.785, 1)
        self.assertAlmostEqual(results['freq_wavenumber'][-2], 3619.616, 1)
        self.assertAlmostEqual(results['freq_wavenumber'][-3], 1572.570, 1)

    def test_hess_HF_uks(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 0.945', basis='ccpvdz', quantum_nuc=[0])
        mf = neo.CDFT(mol, xc='b3lyp5', unrestricted=True)
        mf.scf()

        hess = mf.Hessian()
        h = hess.kernel()
        results = hess.harmonic_analysis(mol, h)

        self.assertAlmostEqual(results['freq_wavenumber'][-1], 3869.636, 1)

    def test_hess_H2O_uks(self):
        mol = neo.M(atom='''H -8.51391085e-01 -4.92895828e-01 -3.82461113e-16;
                            H  6.79000285e-01 -7.11874586e-01 -9.84713973e-16;
                            O  6.51955650e-04  4.57954140e-03 -1.81537015e-15''',
                    basis='ccpvdz', quantum_nuc=[0,1])
        mf = neo.CDFT(mol, xc='b3lyp5', unrestricted=True)
        mf.scf()

        hess = mf.Hessian()
        h = hess.kernel()
        results = hess.harmonic_analysis(mol, h)

        self.assertAlmostEqual(results['freq_wavenumber'][-1], 3713.686, 1)
        self.assertAlmostEqual(results['freq_wavenumber'][-2], 3609.801, 1)
        self.assertAlmostEqual(results['freq_wavenumber'][-3], 1572.818, 1)

    def test_hess_HF_full_q_uks(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 0.945', basis='ccpvdz', quantum_nuc=[0,1])
        mf = neo.CDFT(mol, xc='b3lyp5', unrestricted=True)
        mf.scf()

        hess = mf.Hessian()
        h = hess.kernel()
        results = hess.harmonic_analysis(mol, h)

        self.assertAlmostEqual(results['freq_wavenumber'][-1], 3873.004, 1)

    def test_hess_H2O_full_q_uks(self):
        mol = neo.M(atom='''H -8.51391085e-01 -4.92895828e-01 -3.82461113e-16;
                            H  6.79000285e-01 -7.11874586e-01 -9.84713973e-16;
                            O  6.51955650e-04  4.57954140e-03 -1.81537015e-15''',
                    basis='ccpvdz', quantum_nuc=[0,1,2])
        mf = neo.CDFT(mol, xc='b3lyp5', unrestricted=True)
        mf.scf()

        hess = mf.Hessian()
        h = hess.kernel()
        results = hess.harmonic_analysis(mol, h)

        self.assertAlmostEqual(results['freq_wavenumber'][-1], 3720.785, 1)
        self.assertAlmostEqual(results['freq_wavenumber'][-2], 3619.616, 1)
        self.assertAlmostEqual(results['freq_wavenumber'][-3], 1572.570, 1)

    def test_hess_CH3_full_q_uks(self):
        mol = neo.M(atom='''C  0.0   0.00000   0.000000
                            H  0.0   0.00000   1.076238
                            H  0.0   0.93205  -0.538119
                            H  0.0  -0.93205  -0.538119''',
                    spin=1, basis='ccpvdz', quantum_nuc=[0,1,2,3])
        mf = neo.CDFT(mol, xc='b3lyp5') # spin != 0 will be unrestricted anyway
        mf.scf()

        hess = mf.Hessian()
        h = hess.kernel()
        results = hess.harmonic_analysis(mol, h)

        self.assertAlmostEqual(results['freq_wavenumber'][-1], 3500.978, 1)
        self.assertAlmostEqual(results['freq_wavenumber'][-3], 3295.659, 1)
        self.assertAlmostEqual(results['freq_wavenumber'][-4], 1242.998, 1)

if __name__ == "__main__":
    print("Full Tests for neo.hess")
    unittest.main()
