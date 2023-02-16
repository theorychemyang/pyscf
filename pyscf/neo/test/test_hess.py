#!/usr/bin/env python

import numpy
import unittest
from pyscf import neo

class KnownValues(unittest.TestCase):
    def test_hess_HF(self):
        mol = neo.Mole()
        mol.build(atom='H 0 0 0; F 0 0 0.945', basis='ccpvdz', quantum_nuc=[0])
        mf = neo.CDFT(mol)
        mf.mf_elec.xc = 'b3lyp'
        mf.scf()

        hess = neo.Hessian(mf)
        h = hess.kernel()
        results = hess.harmonic_analysis(mol, h)

        self.assertAlmostEqual(results['freq_au'][0], 0.7525775753724816, 5)

    def test_hess_H2O(self):
        mol = neo.Mole()
        mol.build(atom='''H -8.51391085e-01 -4.92895828e-01 -3.82461113e-16;
                          H  6.79000285e-01 -7.11874586e-01 -9.84713973e-16;
                          O  6.51955650e-04  4.57954140e-03 -1.81537015e-15''',
                  basis='ccpvdz', quantum_nuc=[0,1])
        mf = neo.CDFT(mol)
        mf.mf_elec.xc = 'b3lyp'
        mf.scf()

        hess = neo.Hessian(mf)
        h = hess.kernel()
        results = hess.harmonic_analysis(mol, h)

        self.assertAlmostEqual(results['freq_au'][0], 0.3058957084327294, 5)
        self.assertAlmostEqual(results['freq_au'][1], 0.7020421408894397, 5)
        self.assertAlmostEqual(results['freq_au'][2], 0.722252776562671, 5)

    def test_hess_HF_full_q(self):
        mol = neo.Mole()
        mol.build(atom='H 0 0 0; F 0 0 0.945', basis='ccpvdz', quantum_nuc=[0,1])
        mf = neo.CDFT(mol)
        mf.mf_elec.xc = 'b3lyp'
        mf.scf()

        hess = neo.Hessian(mf)
        h = hess.kernel()
        results = hess.harmonic_analysis(mol, h)

        self.assertAlmostEqual(results['freq_au'][0], 0.7532278378619939, 5)

    def test_hess_H2O_full_q(self):
        mol = neo.Mole()
        mol.build(atom='''H -8.51391085e-01 -4.92895828e-01 -3.82461113e-16;
                          H  6.79000285e-01 -7.11874586e-01 -9.84713973e-16;
                          O  6.51955650e-04  4.57954140e-03 -1.81537015e-15''',
                  basis='ccpvdz', quantum_nuc=[0,1,2])
        mf = neo.CDFT(mol)
        mf.mf_elec.xc = 'b3lyp'
        mf.scf()

        hess = neo.Hessian(mf)
        h = hess.kernel()
        results = hess.harmonic_analysis(mol, h)

        self.assertAlmostEqual(results['freq_au'][0], 0.3058413821370073, 5)
        self.assertAlmostEqual(results['freq_au'][1], 0.7039435940342046, 5)
        self.assertAlmostEqual(results['freq_au'][2], 0.7236193177369298, 5)

    def test_hess_HF_uks(self):
        mol = neo.Mole()
        mol.build(atom='H 0 0 0; F 0 0 0.945', basis='ccpvdz', quantum_nuc=[0])
        mf = neo.CDFT(mol, unrestricted=True)
        mf.mf_elec.xc = 'b3lyp'
        mf.scf()

        hess = neo.Hessian(mf)
        h = hess.kernel()
        results = hess.harmonic_analysis(mol, h)

        self.assertAlmostEqual(results['freq_au'][0], 0.7525775753724816, 5)

    def test_hess_H2O_uks(self):
        mol = neo.Mole()
        mol.build(atom='''H -8.51391085e-01 -4.92895828e-01 -3.82461113e-16;
                          H  6.79000285e-01 -7.11874586e-01 -9.84713973e-16;
                          O  6.51955650e-04  4.57954140e-03 -1.81537015e-15''',
                  basis='ccpvdz', quantum_nuc=[0,1])
        mf = neo.CDFT(mol, unrestricted=True)
        mf.mf_elec.xc = 'b3lyp'
        mf.scf()

        hess = neo.Hessian(mf)
        h = hess.kernel()
        results = hess.harmonic_analysis(mol, h)

        self.assertAlmostEqual(results['freq_au'][0], 0.3058957084327294, 5)
        self.assertAlmostEqual(results['freq_au'][1], 0.7020421408894397, 5)
        self.assertAlmostEqual(results['freq_au'][2], 0.722252776562671, 5)

    def test_hess_HF_full_q_uks(self):
        mol = neo.Mole()
        mol.build(atom='H 0 0 0; F 0 0 0.945', basis='ccpvdz', quantum_nuc=[0,1])
        mf = neo.CDFT(mol, unrestricted=True)
        mf.mf_elec.xc = 'b3lyp'
        mf.scf()

        hess = neo.Hessian(mf)
        h = hess.kernel()
        results = hess.harmonic_analysis(mol, h)

        self.assertAlmostEqual(results['freq_au'][0], 0.7532278378619939, 5)

    def test_hess_H2O_full_q_uks(self):
        mol = neo.Mole()
        mol.build(atom='''H -8.51391085e-01 -4.92895828e-01 -3.82461113e-16;
                          H  6.79000285e-01 -7.11874586e-01 -9.84713973e-16;
                          O  6.51955650e-04  4.57954140e-03 -1.81537015e-15''',
                  basis='ccpvdz', quantum_nuc=[0,1,2])
        mf = neo.CDFT(mol, unrestricted=True)
        mf.mf_elec.xc = 'b3lyp'
        mf.scf()

        hess = neo.Hessian(mf)
        h = hess.kernel()
        results = hess.harmonic_analysis(mol, h)

        self.assertAlmostEqual(results['freq_au'][0], 0.3058413821370073, 5)
        self.assertAlmostEqual(results['freq_au'][1], 0.7039435940342046, 5)
        self.assertAlmostEqual(results['freq_au'][2], 0.7236193177369298, 5)

    def test_hess_CH3_full_q_uks(self):
        mol = neo.Mole()
        mol.build(atom='''C  0.0   0.00000   0.000000
                          H  0.0   0.00000   1.076238
                          H  0.0   0.93205  -0.538119
                          H  0.0  -0.93205  -0.538119''',
                  spin=1, basis='ccpvdz', quantum_nuc=[0,1,2,3])
        mf = neo.CDFT(mol) # spin != 0 will be unrestricted anyway
        mf.mf_elec.xc = 'b3lyp'
        mf.scf()

        hess = neo.Hessian(mf)
        h = hess.kernel()
        results = hess.harmonic_analysis(mol, h)

        self.assertAlmostEqual(results['freq_au'][-1], 0.68084772, 5)
        self.assertAlmostEqual(results['freq_au'][-3], 0.64094077, 5)
        self.assertAlmostEqual(results['freq_au'][-4], 0.2417382, 5)

if __name__ == "__main__":
    print("Full Tests for neo.hess")
    unittest.main()
