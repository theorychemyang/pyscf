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
        self.assertAlmostEqual(mf.scf(), -98.52301447921684, 8)
        with self.assertRaises(TypeError):
            neo.HF(mol).density_fit(with_df=object(), df_ne=True)

    def test_df_j_on_the_fly(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 1', basis='sto-3g',
                    quantum_nuc=[0], verbose=0)
        for scheme in ('electron', 'global'):
            mf = neo.HF(mol).density_fit(auxbasis='weigend', df_ne=True,
                                         df_ne_scheme=scheme,
                                         df_ne_component_vint=True)
            dm = mf.get_init_guess()

            mf.with_df.max_memory = 0
            vj = mf.with_df.get_j(dm)
            self.assertIsNone(mf.with_df._cderi)

            mf_ref = neo.HF(mol).density_fit(auxbasis='weigend', df_ne=True,
                                             df_ne_scheme=scheme,
                                             df_ne_component_vint=True)
            mf_ref.with_df.build()
            vj_ref = mf_ref.with_df.get_jk(dm, with_k=False)[0]
            self.assertIsNotNone(mf_ref.with_df._cderi)

            for t in dm:
                self.assertAlmostEqual(abs(vj[t] - vj_ref[t]).max(), 0, 9)
                self.assertAlmostEqual(abs(vj[t].vint - vj_ref[t].vint).max(), 0, 9)

    def test_df_j_outcore_loop(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 1', basis='sto-3g',
                    quantum_nuc=[0], verbose=0)
        mf_ref = neo.HF(mol).density_fit(auxbasis='weigend', df_ne=True,
                                         df_ne_scheme='electron',
                                         df_ne_component_vint=True)
        dm = mf_ref.get_init_guess()
        mf_ref.with_df.build()
        vj_ref, vk_ref = mf_ref.with_df.get_jk(dm, with_k=False)

        mf = neo.HF(mol).density_fit(auxbasis='weigend', df_ne=True,
                                     df_ne_scheme='electron',
                                     df_ne_component_vint=True)
        mf.with_df.max_memory = 0
        mf.with_df.build()
        self.assertIsNotNone(mf.with_df._cderi)
        self.assertFalse(isinstance(mf.with_df._cderi, dict))
        vj = mf.with_df.get_jk(dm, with_k=False)[0]

        for t in dm:
            self.assertAlmostEqual(abs(vj[t] - vj_ref[t]).max(), 0, 9)
            self.assertAlmostEqual(abs(vj[t].vint - vj_ref[t].vint).max(), 0, 9)

    def test_global_df_route_matrix(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 1', basis='sto-3g',
                    quantum_nuc=[0], verbose=0)
        global_routes = ('incore', 'outcore', 'on_the_fly')
        k_routes = ('incore', 'outcore', 'non_df')
        orders = ('global_first', 'k_first')

        def make_mf(k_route):
            return neo.HF(mol).density_fit(auxbasis='weigend', df_ne=True,
                                           df_ne_scheme='global',
                                           df_ne_component_vint=True,
                                           ee_only_dfj=(k_route == 'non_df'))

        dm_ref = make_mf('incore').get_init_guess()
        mf_j_ref = make_mf('incore')
        mf_j_ref.with_df.build()
        vj_ref = mf_j_ref.with_df.get_j(dm_ref)
        vk_ref = {}
        for k_route in k_routes:
            mf_k_ref = make_mf(k_route)
            if k_route == 'incore':
                mf_k_ref.components['e'].with_df.max_memory = 2000
            elif k_route == 'outcore':
                mf_k_ref.components['e'].with_df.max_memory = 0
            vk_ref[k_route] = mf_k_ref.components['e'].get_k(
                mol.components['e'], dm_ref['e'])

        def run_global(mf, dm, global_route, k_route):
            build_e_cderi = k_route != 'non_df'
            if global_route == 'on_the_fly':
                mf.with_df.max_memory = 0
                mf.with_df.get_j(dm)
            else:
                mf.with_df.max_memory = 2000 if global_route == 'incore' else 0
                with lib.temporary_env(mf.with_df, _build_e_cderi=build_e_cderi):
                    next(mf.with_df.loop())

        def run_k(mf, dm, k_route):
            if k_route == 'incore':
                mf.components['e'].with_df.max_memory = 2000
            elif k_route == 'outcore':
                mf.components['e'].with_df.max_memory = 0
            return mf.components['e'].get_k(mol.components['e'], dm['e'])

        for global_route in global_routes:
            for k_route in k_routes:
                for order in orders:
                    with self.subTest(global_route=global_route,
                                      k_route=k_route, order=order):
                        mf = make_mf(k_route)
                        dm = {t: numpy.array(dm_ref[t], copy=True)
                              for t in dm_ref}
                        cderi_e_before_global = None
                        if order == 'global_first':
                            run_global(mf, dm, global_route, k_route)
                            cderi_e_before_k = mf.components['e'].with_df._cderi
                            vk = run_k(mf, dm, k_route)
                            if global_route != 'on_the_fly' or k_route == 'non_df':
                                self.assertIs(mf.components['e'].with_df._cderi,
                                              cderi_e_before_k)
                        else:
                            vk = run_k(mf, dm, k_route)
                            cderi_e_before_global = mf.components['e'].with_df._cderi
                            run_global(mf, dm, global_route, k_route)

                        cderi = mf.with_df._cderi
                        cderi_e = mf.components['e'].with_df._cderi
                        if global_route == 'incore':
                            self.assertTrue(isinstance(cderi, dict))
                        elif global_route == 'outcore':
                            self.assertIsNotNone(cderi)
                            self.assertFalse(isinstance(cderi, dict))
                        else:
                            self.assertIsNone(cderi)

                        if k_route == 'non_df':
                            self.assertIsNone(cderi_e)
                        elif order == 'global_first' and global_route in ('incore', 'outcore'):
                            self.assertEqual(isinstance(cderi_e, numpy.ndarray),
                                             global_route == 'incore')
                        elif order == 'k_first' and global_route == 'incore':
                            self.assertTrue(isinstance(cderi_e, numpy.ndarray))
                        elif order == 'k_first' and global_route == 'outcore':
                            self.assertIs(cderi_e, cderi_e_before_global)
                            self.assertEqual(isinstance(cderi_e, numpy.ndarray),
                                             k_route == 'incore')
                        elif global_route == 'on_the_fly':
                            self.assertEqual(isinstance(cderi_e, numpy.ndarray),
                                             k_route == 'incore')

                        vj = mf.with_df.get_j(dm)
                        for t in dm:
                            self.assertAlmostEqual(abs(vj[t] - vj_ref[t]).max(),
                                                   0, 9)
                            self.assertAlmostEqual(abs(vj[t].vint - vj_ref[t].vint).max(),
                                                   0, 9)
                        self.assertAlmostEqual(abs(vk - vk_ref[k_route]).max(),
                                               0, 12)

    def test_global_df_no_accidental_e_cderi_for_dft(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 1', basis='sto-3g',
                    quantum_nuc=[0], verbose=0)
        with self.subTest(xc='LDA,VWN'):
            mf = neo.CDFT(mol, xc='LDA,VWN').density_fit(
                auxbasis='weigend', df_ne=True, df_ne_scheme='global')
            dm = mf.get_init_guess()
            vhf = mf.get_veff(mf.mol, dm)
            self.assertIsNone(vhf['e'].vk)
            self.assertIsNone(mf.with_df._cderi)
            self.assertIsNone(mf.components['e'].with_df._cderi)

        with self.subTest(xc='b3lypg'):
            mf = neo.CDFT(mol, xc='b3lypg').density_fit(
                auxbasis='weigend', df_ne=True, df_ne_scheme='global')
            dm = mf.get_init_guess()
            vhf = mf.get_veff(mf.mol, dm)
            self.assertIsNotNone(vhf['e'].vk)
            self.assertIsNone(mf.with_df._cderi)
            self.assertIsNotNone(mf.components['e'].with_df._cderi)

    def test_electron_scheme_ee_only_dfj_uses_component_k(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 1', basis='sto-3g',
                    quantum_nuc=[0], verbose=0)
        mf = neo.HF(mol).density_fit(auxbasis='weigend', df_ne=True,
                                     df_ne_scheme='electron',
                                     ee_only_dfj=True)
        dm = mf.get_init_guess()
        called = []
        get_k = mf.components['e'].get_k
        def get_k_hook(*args, **kwargs):
            called.append(True)
            return get_k(*args, **kwargs)
        mf.components['e'].get_k = get_k_hook
        mf.get_veff(mf.mol, dm)
        self.assertTrue(called)

    def test_component_fock_has_df_ne_coulomb(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 1', basis='sto-3g',
                    quantum_nuc=[0], verbose=0)
        mf = neo.HF(mol).density_fit(auxbasis='weigend', df_ne=True,
                                     df_ne_component_vint=True)
        mf.scf()
        dm = mf.make_rdm1()
        vhf = mf.get_veff(mf.mol, dm)
        fock = mf.get_fock(mf.get_hcore(), mf.get_ovlp(), vhf, dm)
        fock_e = mf.components['e'].get_fock(dm=dm['e'])
        self.assertAlmostEqual(numpy.linalg.norm(fock['e'] - fock_e), 0, 12)

    def test_component_fock_global_j_on_the_fly(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 1', basis='sto-3g',
                    quantum_nuc=[0], verbose=0)
        mf = neo.CDFT(mol, xc='LDA,VWN').density_fit(
            auxbasis='weigend', df_ne=True, df_ne_scheme='global',
            df_ne_component_vint=True)
        mf.with_df.max_memory = 0
        dm = mf.get_init_guess()
        vhf = mf.get_veff(mf.mol, dm)
        fock = mf.get_fock(mf.get_hcore(), mf.get_ovlp(), vhf, dm)
        fock_e = mf.components['e'].get_fock(dm=dm['e'])
        self.assertIsNone(mf.with_df._cderi)
        self.assertAlmostEqual(numpy.linalg.norm(fock['e'] - fock_e), 0, 12)

    def test_component_fock_without_df_ne_cache_raises(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 1', basis='sto-3g',
                    quantum_nuc=[0], verbose=0)
        mf = neo.HF(mol).density_fit(auxbasis='weigend', df_ne=True)
        mf.scf()
        dm = mf.make_rdm1()
        mf.get_veff(mf.mol, dm)
        with self.assertRaises(RuntimeError):
            mf.components['e'].get_fock(dm=dm['e'])

    def test_component_fock_after_undo_df_with_cache_raises(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 1', basis='sto-3g',
                    quantum_nuc=[0], verbose=0)
        mf = neo.HF(mol).density_fit(auxbasis='weigend', df_ne=True,
                                     df_ne_component_vint=True)
        mf.scf()
        dm = mf.make_rdm1()
        mf.get_veff(mf.mol, dm)
        mf.components['e'].get_fock(dm=dm['e'])
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

    def test_cdft_xc_change_after_density_fit(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 1', basis='sto-3g',
                    quantum_nuc=[0], verbose=0)
        mf = neo.CDFT(mol, xc='LDA,VWN').density_fit(auxbasis='weigend',
                                                     df_ne=True)
        mf.xc = mf.components['e'].xc = 'b3lypg'
        dm = mf.get_init_guess()
        self.assertIsNone(mf.with_df._cderi)
        self.assertIsNone(mf.components['e'].with_df._cderi)
        vhf = mf.get_veff(mf.mol, dm)
        self.assertIsNotNone(vhf['e'].vk)
        self.assertIsNotNone(mf.components['e'].with_df._cderi)
        self.assertGreater(mf.with_df.auxmol.nao,
                           mf.components['e'].with_df.auxmol.nao)

        mf = neo.CDFT(mol, xc='b3lypg').density_fit(auxbasis='weigend',
                                                    df_ne=True)
        mf.xc = mf.components['e'].xc = 'LDA,VWN'
        dm = mf.get_init_guess()
        vhf = mf.get_veff(mf.mol, dm)
        self.assertIsNone(vhf['e'].vk)
        self.assertIsNone(mf.components['e'].with_df._cderi)

    def test_cdft_epc_direct_scf(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 1', basis='sto-3g',
                    quantum_nuc=[0])
        es = []
        for direct_scf in (False, True):
            mf = neo.CDFT(mol, xc='b3lyp5', epc='17-2').density_fit(auxbasis='weigend',
                                                                     df_ne=True)
            mf.direct_scf = mf.components['e'].direct_scf = direct_scf
            es.append(mf.scf())
        self.assertAlmostEqual(es[0], es[1], 8)

    def test_ks_epc(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 1', basis='sto-3g',
                    quantum_nuc=[0])
        mf = neo.KS(mol, xc='LDA,VWN', epc='17-2').density_fit(
            auxbasis='weigend', df_ne=True)
        self.assertAlmostEqual(mf.scf(), -98.24173535806187, 6)

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
        mf_ee_df = neo.CDFT(mol, xc='LDA,VWN').density_fit(auxbasis='weigend',
                                                           df_ne=False)
        mf_ne_df = neo.CDFT(mol, xc='LDA,VWN').density_fit(auxbasis='weigend',
                                                           df_ne=True,
                                                           df_ne_scheme='electron')
        mf_ne_df_global = neo.CDFT(mol, xc='LDA,VWN').density_fit(auxbasis='weigend',
                                                                  df_ne=True)

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
        mf = neo.CDFT(mol, xc='b3lypg').density_fit(auxbasis='aug-cc-pvdz-jkfit',
                                                    df_ne=True,
                                                    df_ne_scheme='global')
        mf.scf()
        de = mf.Gradients().kernel()

        e_scanner = mf.as_scanner()
        e1 = e_scanner('H 0 0 -0.001; F 0 0 1')
        e2 = e_scanner('H 0 0  0.001; F 0 0 1')

        self.assertAlmostEqual(de[0,2], (e2-e1)/0.002*lib.param.BOHR, 5)

    def test_pure_dft_grad(self):
        mol = neo.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
                    basis='6-31g', nuc_basis='pb4d', quantum_nuc=[1])
        mf = neo.CDFT(mol, xc='PBE').density_fit(auxbasis='weigend',
                                                 df_ne=True)
        mf.components['e'].grids.atom_grid = (99, 590)
        mf.conv_tol = 1e-11
        mf.scf()
        de = mf.Gradients().kernel()

        e_scanner = mf.as_scanner()
        e1 = e_scanner('O 0 0 0; H 0 -0.758 0.587; H 0 0.757 0.587')
        e2 = e_scanner('O 0 0 0; H 0 -0.756 0.587; H 0 0.757 0.587')

        self.assertAlmostEqual(de[1,1], (e2-e1)/0.002*lib.param.BOHR, 5)

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
        mf = neo.CDFT(mol, xc='b3lypg').density_fit(auxbasis='aug-cc-pvdz-jkfit',
                                                    df_ne=True,
                                                    df_ne_scheme='global')
        mf.scf()
        de = mf.Gradients().kernel()

        e_scanner = mf.as_scanner()
        e1 = e_scanner('H 0 0 0; F 0 0 0.999')
        e2 = e_scanner('H 0 0 0; F 0 0 1.001')

        self.assertAlmostEqual(de[1,2], (e2-e1)/0.002*lib.param.BOHR, 5)

    def test_df_nn_grad(self):
        mol = neo.M(atom='H 0 0 0; H 0 0 1', basis='aug-ccpvdz',
                    nuc_basis='pb4d', quantum_nuc=[0,1])
        mf = neo.CDFT(mol, xc='LDA,VWN').density_fit(auxbasis='weigend',
                                                     df_ne=True, df_nn=True)
        mf.conv_tol = 1e-11
        mf.scf()
        de = mf.Gradients().kernel()

        e_scanner = mf.as_scanner()
        e1 = e_scanner('H 0 0 0; H 0 0 0.999')
        e2 = e_scanner('H 0 0 0; H 0 0 1.001')

        self.assertAlmostEqual(de[1,2], (e2-e1)/0.002*lib.param.BOHR, 5)

    def test_scanner(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 0.94', basis='aug-ccpvdz')
        mf = neo.CDFT(mol, xc='b3lypg').density_fit(auxbasis='aug-cc-pvdz-jkfit', df_ne=True)
        grad_scanner = mf.nuc_grad_method().as_scanner()
        grad_scanner(mol)

        mol2 = neo.M(atom='H 0 0 0; F 0 0 1.1', basis='aug-ccpvdz')
        mf2 = neo.CDFT(mol2, xc='b3lypg').density_fit(auxbasis='aug-cc-pvdz-jkfit', df_ne=True)
        e2 = mf2.scf()
        grad2 = mf2.Gradients().grad()
        e, grad = grad_scanner(mol2)
        self.assertAlmostEqual(e, e2, 9)
        self.assertTrue(abs(grad-grad2).max() < 1e-6)

        mol2 = neo.M(atom='H 0 0 0; F 0 0 1.2', basis='aug-ccpvdz')
        mf2 = neo.CDFT(mol2, xc='b3lypg').density_fit(auxbasis='aug-cc-pvdz-jkfit', df_ne=True)
        e2 = mf2.scf()
        grad2 = mf2.Gradients().grad()
        e, grad = grad_scanner(mol2)
        self.assertAlmostEqual(e, e2, 9)
        self.assertTrue(abs(grad-grad2).max() < 1e-6)

    def test_df_nn_scanner(self):
        mol = neo.M(atom='H 0 0 0; H 0 0 0.94', basis='aug-ccpvdz',
                    nuc_basis='pb4d', quantum_nuc=[0,1])
        mf = neo.CDFT(mol, xc='PBE0').density_fit(auxbasis='weigend',
                                                  df_ne=True, df_nn=True)
        grad_scanner = mf.nuc_grad_method().as_scanner()
        grad_scanner(mol)

        mol2 = neo.M(atom='H 0 0 0; H 0 0 1.1', basis='aug-ccpvdz',
                     nuc_basis='pb4d', quantum_nuc=[0,1])
        mf2 = neo.CDFT(mol2, xc='PBE0').density_fit(auxbasis='weigend',
                                                    df_ne=True, df_nn=True)
        e2 = mf2.scf()
        grad2 = mf2.Gradients().grad()
        e, grad = grad_scanner(mol2)
        self.assertAlmostEqual(e, e2, 9)
        self.assertTrue(abs(grad-grad2).max() < 1e-6)

if __name__ == "__main__":
    print("Full Tests for ee and ne density-fitting in CDFT")
    unittest.main()
