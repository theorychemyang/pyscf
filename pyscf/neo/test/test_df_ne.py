#!/usr/bin/env python

import unittest
import numpy
from pyscf import neo, lib
from pyscf.neo import df

class KnownValues(unittest.TestCase):
    def test_hf_direct_scf(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 1', basis='sto-3g',
                    quantum_nuc=[0])
        es = []
        for direct_scf in (False, True):
            mf = neo.HF(mol).density_fit(auxbasis='weigend', df_ne=True,
                                         ee_only_dfj=True)
            mf.direct_scf = mf.components['e'].direct_scf = direct_scf
            es.append(mf.scf())
        self.assertAlmostEqual(es[0], es[1], 8)

    def test_custom_with_df(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 1', basis='sto-3g',
                    quantum_nuc=[0])
        with_df = df.DF(mol, 'weigend')
        mf = neo.HF(mol).density_fit(with_df=with_df, df_ne=True)
        self.assertIs(mf.with_df, with_df)
        for t, comp in mf.components.items():
            self.assertEqual(with_df._charges[t], comp.charge)
            self.assertEqual(with_df._unrestricted[t], False)
        self.assertAlmostEqual(mf.scf(), -98.52306824045718, 8)
        with self.assertRaises(TypeError):
            neo.HF(mol).density_fit(with_df=object(), df_ne=True)

    def test_df_j_on_the_fly(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 1', basis='sto-3g',
                    quantum_nuc=[0], verbose=0)
        for scheme in ('electron', 'global'):
            mf = neo.HF(mol).density_fit(auxbasis='weigend', df_ne=True,
                                         df_ne_scheme=scheme)
            dm = mf.get_init_guess()

            vj = mf.with_df.get_j(dm)
            self.assertIsNone(mf.with_df._cderi)

            mf_ref = neo.HF(mol).density_fit(auxbasis='weigend', df_ne=True,
                                             df_ne_scheme=scheme)
            mf_ref.with_df.build()
            vj_ref, vk_ref = mf_ref.with_df.get_jk(dm, with_k=False)
            self.assertIsNotNone(mf_ref.with_df._cderi)
            self.assertTrue(numpy.allclose(vk_ref, 0))

            for t in dm:
                self.assertAlmostEqual(abs(vj[t] - vj_ref[t]).max(), 0, 9)
                self.assertAlmostEqual(abs(vj[t].vint - vj_ref[t].vint).max(), 0, 9)

    def test_df_j_outcore_loop(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 1', basis='sto-3g',
                    quantum_nuc=[0], verbose=0)
        mf_ref = neo.HF(mol).density_fit(auxbasis='weigend', df_ne=True,
                                         df_ne_scheme='electron')
        dm = mf_ref.get_init_guess()
        mf_ref.with_df.build()
        vj_ref, vk_ref = mf_ref.with_df.get_jk(dm, with_k=False)

        mf = neo.HF(mol).density_fit(auxbasis='weigend', df_ne=True,
                                     df_ne_scheme='electron')
        mf.with_df.max_memory = 1e-4
        mf.with_df.build()
        self.assertIsNotNone(mf.with_df._cderi)
        self.assertFalse(isinstance(mf.with_df._cderi, dict))
        vj, vk = mf.with_df.get_jk(dm, with_k=False)
        self.assertTrue(numpy.allclose(vk, 0))

        for t in dm:
            self.assertAlmostEqual(abs(vj[t] - vj_ref[t]).max(), 0, 9)
            self.assertAlmostEqual(abs(vj[t].vint - vj_ref[t].vint).max(), 0, 9)

    def test_component_fock_has_df_ne_coulomb(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 1', basis='sto-3g',
                    quantum_nuc=[0], verbose=0)
        mf = neo.HF(mol).density_fit(auxbasis='weigend', df_ne=True)
        mf.scf()
        dm = mf.make_rdm1()
        vhf = mf.get_veff(mf.mol, dm)
        fock = mf.get_fock(mf.get_hcore(), mf.get_ovlp(), vhf, dm)
        fock_e = mf.components['e'].get_fock(dm=dm['e'])
        self.assertAlmostEqual(numpy.linalg.norm(fock['e'] - fock_e), 0, 12)

    def test_component_fock_after_undo_df_without_cache_raises(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 1', basis='sto-3g',
                    quantum_nuc=[0], verbose=0)
        mf = neo.HF(mol).density_fit(auxbasis='weigend', df_ne=True)
        mf.scf()
        dm = mf.make_rdm1()
        mf.get_veff(mf.mol, dm)
        mf = mf.undo_df()
        dm = mf.make_rdm1()
        with self.assertRaises(RuntimeError):
            mf.components['e'].get_fock(dm=dm['e'])

    def test_cdft_direct_scf(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 1', basis='sto-3g',
                    quantum_nuc=[0])
        es = []
        for direct_scf in (False, True):
            mf = neo.CDFT(mol, xc='LDA,VWN').density_fit(auxbasis='weigend',
                                                         df_ne=True)
            mf.direct_scf = mf.components['e'].direct_scf = direct_scf
            es.append(mf.scf())
        self.assertAlmostEqual(es[0], es[1], 8)

    def test_cdft_epc_direct_scf(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 1', basis='sto-3g',
                    quantum_nuc=[0])
        es = []
        for direct_scf in (False, True):
            mf = neo.CDFT(mol, xc='b3lyp5', epc='17-2').density_fit(
                auxbasis='weigend', df_ne=True)
            mf.direct_scf = mf.components['e'].direct_scf = direct_scf
            es.append(mf.scf())
        self.assertAlmostEqual(es[0], es[1], 8)

    def test_scf(self):
        mol = neo.M(atom='''H 0 0 0; F 0 0 1''', basis='aug-ccpvdz', quantum_nuc=[0])
        mf = neo.CDFT(mol, xc='b3lypg').density_fit(auxbasis='aug-cc-pvdz-jkfit', df_ne=True)
        self.assertAlmostEqual(mf.scf(), -100.4195436624414, 6)

    def test_scf_rsh(self):
        mol = neo.M(atom='''H 0 0 0; C 0 0 1.064; N 0 0 2.220''',
                    basis='ccpvdz', quantum_nuc=[0])
        mf = neo.KS(mol, xc='camb3lyp').density_fit(auxbasis='cc-pVTZ-JKFIT',
                                                    df_ne=True)
        self.assertAlmostEqual(mf.scf(), -93.34259794767802, 4)

    def test_scf_multi_proton(self):
        mol = neo.M(atom='''H 0 0 0; H 0 0 1''', basis='aug-ccpvdz', quantum_nuc=['H'])
        mf = neo.CDFT(mol, xc='b3lypg').density_fit(auxbasis='aug-cc-pvdz-jkfit', df_ne=True)
        self.assertAlmostEqual(mf.scf(), -1.0790336414836041, 6)

    def test_global_df_ne_metric(self):
        mol = neo.M(atom='H 0 0 0; H 0 0 1', basis='sto-3g',
                    quantum_nuc=[0, 1], verbose=0)
        mf_ee_df = neo.CDFT(mol, xc='LDA,VWN').density_fit(
            auxbasis='weigend', df_ne=False)
        mf_ne_df = neo.CDFT(mol, xc='LDA,VWN').density_fit(
            auxbasis='weigend', df_ne=True, df_ne_scheme='electron')
        mf_ne_df_global = neo.CDFT(mol, xc='LDA,VWN').density_fit(
            auxbasis='weigend', df_ne=True)

        e_ee_df = mf_ee_df.kernel()
        err_electron = abs(mf_ne_df.kernel() - e_ee_df)
        err_global = abs(mf_ne_df_global.kernel() - e_ee_df)
        self.assertLess(err_global, err_electron)

    def test_nuc_auxbasis_name(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 1', basis='sto-3g',
                    quantum_nuc=[0], verbose=0)
        aux_pb4d = df._make_nuc_auxmol(mol.components['n0'], 'pb4d')
        aux_weigend = df._make_nuc_auxmol(mol.components['n0'], 'weigend')
        aux_aug_etb = df._make_nuc_auxmol(mol.components['n0'], 'aug_etb')
        aux_aug_etb_dense = df._make_nuc_auxmol(mol.components['n0'], 'aug_etb',
                                                nuc_auxbasis_beta=1.7)
        self.assertEqual(aux_pb4d.nao, mol.components['n0'].nao)
        self.assertNotEqual(aux_pb4d.nao, aux_weigend.nao)
        self.assertNotEqual(aux_pb4d.nao, aux_aug_etb.nao)
        self.assertGreater(aux_aug_etb_dense.nao, aux_aug_etb.nao)

    def test_grad(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 1', basis='aug-ccpvdz')
        mf = neo.CDFT(mol, xc='b3lypg').density_fit(
            auxbasis='aug-cc-pvdz-jkfit', df_ne=True, df_ne_scheme='global')
        mf.scf()
        de = mf.Gradients().kernel()

        e_scanner = mf.as_scanner()
        e1 = e_scanner('H 0 0 -0.001; F 0 0 1')
        e2 = e_scanner('H 0 0  0.001; F 0 0 1')

        self.assertAlmostEqual(de[0,2], (e2-e1)/0.002*lib.param.BOHR, 5)

    def test_hf_grad(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 1', basis='sto-3g',
                    quantum_nuc=[0])
        mf = neo.HF(mol).density_fit(auxbasis='weigend', df_ne=True,
                                     df_ne_scheme='global')
        mf.conv_tol = 1e-10
        mf.scf()
        de = mf.Gradients().kernel()

        e_scanner = mf.as_scanner()
        e1 = e_scanner('H 0 0 -0.001; F 0 0 1')
        e2 = e_scanner('H 0 0  0.001; F 0 0 1')

        self.assertAlmostEqual(de[0,2], (e2-e1)/0.002*lib.param.BOHR, 5)

    def test_grad_atmlst(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 1', basis='sto-3g',
                    quantum_nuc=[0], verbose=0)
        mf = neo.HF(mol).density_fit(auxbasis='weigend', df_ne=True)
        mf.conv_tol = 1e-10
        mf.scf()

        grad = mf.Gradients()
        de = grad.kernel()
        de0 = grad.kernel(atmlst=[0])
        de1 = grad.kernel(atmlst=[1])

        self.assertAlmostEqual(abs(de0 - de[[0]]).max(), 0, 12)
        self.assertAlmostEqual(abs(de1 - de[[1]]).max(), 0, 12)

    def test_grad_full_q(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 1', basis='aug-ccpvdz', quantum_nuc=[0,1])
        mf = neo.CDFT(mol, xc='b3lypg').density_fit(
            auxbasis='aug-cc-pvdz-jkfit', df_ne=True, df_ne_scheme='global')
        mf.scf()
        de = mf.Gradients().kernel()

        e_scanner = mf.as_scanner()
        e1 = e_scanner('H 0 0 0; F 0 0 0.999')
        e2 = e_scanner('H 0 0 0; F 0 0 1.001')

        self.assertAlmostEqual(de[1,2], (e2-e1)/0.002*lib.param.BOHR, 5)

    def test_scanner(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 0.94', basis='aug-ccpvdz')
        mf = neo.CDFT(mol, xc='b3lypg').density_fit(auxbasis='aug-cc-pvdz-jkfit', df_ne=True)
        grad_scanner = mf.nuc_grad_method().as_scanner()

        mol2 = neo.M(atom='H 0 0 0; F 0 0 1.1', basis='aug-ccpvdz')
        mf2 = neo.CDFT(mol2, xc='b3lypg').density_fit(auxbasis='aug-cc-pvdz-jkfit', df_ne=True)
        mf2.scf()
        grad2 = mf2.Gradients().grad()
        _, grad = grad_scanner(mol2)
        self.assertTrue(abs(grad-grad2).max() < 1e-6)

        mol2 = neo.M(atom='H 0 0 0; F 0 0 1.2', basis='aug-ccpvdz')
        mf2 = neo.CDFT(mol2, xc='b3lypg').density_fit(auxbasis='aug-cc-pvdz-jkfit', df_ne=True)
        mf2.scf()
        grad2 = mf2.Gradients().grad()
        _, grad = grad_scanner(mol2)
        self.assertTrue(abs(grad-grad2).max() < 1e-6)

if __name__ == "__main__":
    print("Full Tests for ee and ne density-fitting in CDFT")
    unittest.main()
