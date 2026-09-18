#!/usr/bin/env python

import unittest
from pyscf import neo, scf

class KnownValues(unittest.TestCase):
    def test_scanner_spin(self):
        mol = neo.M(atom='H 0 0 0; Li 0 0 1.6', basis='sto-3g',
                    nuc_basis='pb4d', quantum_nuc=[0])
        mol2 = neo.M(atom='H 0 0 0; Li 0 0 1.6', basis='sto-3g',
                     nuc_basis='pb4d', quantum_nuc=[0], charge=1, spin=1)
        for df_ne in (None, False, True):
            with self.subTest(df_ne=df_ne):
                mf = neo.CDFT(mol, xc='PBE0')
                if df_ne is not None:
                    mf = mf.density_fit(auxbasis='weigend', df_ne=df_ne)
                mf.conv_tol = 1e-10
                scanner = mf.nuc_grad_method().as_scanner()
                scanner(mol)
                for mol_test, unrestricted in ((mol2, False), (mol, False), (mol, True)):
                    scanner.base.unrestricted = unrestricted
                    mf_ref = neo.CDFT(mol_test, xc='PBE0', unrestricted=unrestricted)
                    if df_ne is not None:
                        mf_ref = mf_ref.density_fit(auxbasis='weigend', df_ne=df_ne)
                    mf_ref.conv_tol = 1e-10
                    e, grad = scanner(mol_test)
                    self.assertAlmostEqual(e, mf_ref.scf(), 8)
                    self.assertTrue(abs(grad-mf_ref.Gradients().kernel()).max() < 1e-6)
                    self.assertEqual(isinstance(scanner.base.components['e'], scf.uhf.UHF),
                                     unrestricted or mol_test.spin != 0)

    def test_scanner_different_mol(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 0.9', basis='sto-3g',
                    nuc_basis='pb4d', quantum_nuc=[0])
        mf = neo.CDFT(mol, xc='LDA,VWN').density_fit(auxbasis='weigend',
                                                     df_ne=False)
        mf.conv_tol = 1e-10
        grad_scanner = mf.nuc_grad_method().as_scanner()
        grad_scanner(mol)
        mf_e = grad_scanner.base.components['e']

        mol2 = neo.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
                     basis='sto-3g', nuc_basis='pb4d', quantum_nuc=[1,2])
        for mol_test in (mol2, mol):
            mf_ref = neo.CDFT(mol_test, xc='LDA,VWN').density_fit(
                auxbasis='weigend', df_ne=False)
            mf_ref.conv_tol = 1e-10
            e_ref = mf_ref.scf()
            grad_ref = mf_ref.Gradients().grad()
            e, grad = grad_scanner(mol_test)
            self.assertIs(grad_scanner.base.components['e'], mf_e)
            self.assertAlmostEqual(e, e_ref, 9)
            self.assertTrue(abs(grad-grad_ref).max() < 1e-8)

    def test_hf_direct_scf(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 1', basis='sto-3g',
                    quantum_nuc=[0])
        es = []
        for direct_scf in (False, True):
            mf = neo.HF(mol).density_fit(auxbasis='weigend')
            mf.direct_scf = mf.components['e'].direct_scf = direct_scf
            es.append(mf.scf())
        self.assertAlmostEqual(es[0], es[1], 8)

    def test_scf_epc17_2(self):
        mol = neo.M(atom='''H 0 0 0; C 0 0 1.064; N 0 0 2.220''', basis='ccpvdz',
                    quantum_nuc=[0])
        mf = neo.KS(mol, xc='b3lyp5', epc='17-2').density_fit(auxbasis='cc-pVTZ-JKFIT')
        self.assertAlmostEqual(mf.scf(), -93.3670499232414, 4)

    def test_scf_rsh(self):
        mol = neo.M(atom='''H 0 0 0; C 0 0 1.064; N 0 0 2.220''', basis='ccpvdz',
                    quantum_nuc=[0])
        mf = neo.KS(mol, xc='camb3lyp').density_fit(auxbasis='cc-pVTZ-JKFIT')
        self.assertAlmostEqual(mf.scf(), -93.34261097393602, 4)

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

    def test_hess_df_ne(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 1', basis='sto-3g',
                    quantum_nuc=[0])
        mf = neo.HF(mol).density_fit(auxbasis='weigend', df_ne=True)
        mf.scf()

        with self.assertRaises(NotImplementedError):
            mf.Hessian()
        with self.assertRaises(NotImplementedError):
            neo.Hessian(mf)

    def test_grad_ctddft(self):
        mol = neo.M(atom='''H 0 0 0; F 0 0 0.94''', basis='ccpvdz',
                    quantum_nuc=[0])
        for xc in ['lda', 'b3lyp5', 'camb3lyp']:
            mf = neo.CDFT(mol, xc=xc)
            mf.scf()
            gref = mf.TDDFT().Gradients().kernel()

            mf = neo.CDFT(mol, xc=xc).density_fit(auxbasis='cc-pVTZ-JKFIT')
            mf.scf()
            grad = mf.TDDFT().Gradients().kernel()
            self.assertAlmostEqual(abs(gref-grad).max(), 0, 4)

if __name__ == "__main__":
    print("Full Tests for electronic DF in NEO")
    unittest.main()
