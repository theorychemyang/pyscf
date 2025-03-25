
import warnings
import numpy
from pyscf import lib, scf, neo
from pyscf.neo import hf, ks
from pyscf.df.df_jk import _DFHF
from pyscf.lib import logger

def dot_cderi_dm(eri1, eri2, dm1):
    dms1 = numpy.asarray(dm1)
    dm_shape1 = dms1.shape
    nao1 = dm_shape1[-1]
    dms1 = dms1.reshape(-1,nao1,nao1)

    idx = numpy.arange(nao1)
    dmtril = lib.pack_tril(dms1 + dms1.conj().transpose(0,2,1))
    dmtril[:,idx*(idx+1)//2+idx] *= .5
    vj = dmtril.dot(eri1.T).dot(eri2)
    dm2_len = int((numpy.sqrt(8*eri2.shape[1]+1)-1)/2)
    return lib.unpack_tril(vj, 1).reshape((dm2_len, dm2_len))

def _build_cderi(mf_e, mf_n, cart, max_memory, verbose=0):
    auxmol_e = mf_e.with_df.auxmol
    if auxmol_e is None:
        from pyscf.df.addons import make_auxmol
        auxmol_e = make_auxmol(mf_e.mol, mf_e.with_df.auxbasis)
    naux_e = auxmol_e.nao_nr()
    mol_n = mf_n.mol
    nao_n = mol_n.nao_nr()
    nao_n_pair = nao_n*(nao_n+1)//2

    max_memory = max_memory - lib.current_memory()[0]
    if cart:
        int3c = 'int3c2e_cart'
        int2c = 'int2c2e_cart'
    else:
        int3c = 'int3c2e_sph'
        int2c = 'int2c2e_sph'
    if nao_n_pair*naux_e*8/1e6 < .9*max_memory:
        from pyscf.df.incore import cholesky_eri
        return cholesky_eri(mol_n, auxmol=auxmol_e,
                            int3c=int3c, int2c=int2c,
                            max_memory=max_memory, verbose=verbose)
    else:
        raise NotImplementedError('outcore df_ne not implemented')

def density_fit(mf, auxbasis=None, ee_only_dfj=False, df_ne=False):
    assert isinstance(mf, neo.HF)
    if not isinstance(mf.components['e'], _DFHF):
        mf.components['e'] = mf.components['e'].density_fit(auxbasis=auxbasis, only_dfj=ee_only_dfj)
    if isinstance(mf,_DFNEO):
        return mf
    dfmf = _DFNEO(mf, auxbasis, ee_only_dfj, df_ne)
    return lib.set_class(dfmf, (_DFNEO, mf.__class__))

class DFInteractionCoulomb(hf.InteractionCoulomb):
    def __init__(self, *args, df_ne=False, auxbasis=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.df_ne = df_ne
        self.auxbasis = auxbasis
        self._cderi = None
    def get_vint(self, dm):
        if not self.df_ne:
            return super().get_vint(dm)
        assert isinstance(dm, dict)
        assert self.mf1_type in dm or self.mf2_type in dm
        # Spin-insensitive interactions: sum over spin in dm first
        dm1 = dm.get(self.mf1_type)
        dm2 = dm.get(self.mf2_type)
        # Get total densities if the dm has two spin channels
        if dm1 is not None and self.mf1_unrestricted:
            assert dm1.ndim > 2 and dm1.shape[0] == 2
            dm1 = dm1[0] + dm1[1]
        if dm2 is not None and self.mf2_unrestricted:
            assert dm2.ndim > 2 and dm2.shape[0] == 2
            dm2 = dm2[0] + dm2[1]
        mol1 = self.mf1.mol
        mol2 = self.mf2.mol
        mol = mol1.super_mol
        assert mol == mol2.super_mol
        vj = {}
        if self.df_ne and ((self.mf1_type.startswith('n') and self.mf2_type.startswith('e')) or
                           ((self.mf1_type.startswith('e') and self.mf2_type.startswith('n')))):
            if self.mf1_type == 'e':
                mf_e = self.mf1
                dm_e = dm1
                mf_n = self.mf2
                dm_n = dm2
                t_n = self.mf2_type
            else:
                mf_e = self.mf2
                dm_e = dm2
                mf_n = self.mf1
                dm_n = dm1
                t_n = self.mf1_type
            assert isinstance(mf_e, _DFHF)
            if self._cderi is None:
                from pyscf.df.incore import cholesky_eri
                self._cderi = _build_cderi(mf_e, mf_n, mol.cart, self.max_memory, mol.verbose)
            if dm_e is not None:
                vj[t_n] = dot_cderi_dm(mf_e._cderi, self._cderi, dm_e)
            if dm_n is not None:
                vj['e'] = dot_cderi_dm(self._cderi, mf_e._cderi, dm_n)
            charge_product = self.mf1.charge * self.mf2.charge
            if self.mf1_type in vj:
                vj[self.mf1_type] *= charge_product
            if self.mf2_type in vj:
                vj[self.mf2_type] *= charge_product

        else:
            if (not mol.direct_vee and
                (self._eri is not None or mol.incore_anyway or self._is_mem_enough())):
                if self._eri is None:
                    if mol.verbose >= logger.DEBUG:
                        cput0 = (logger.process_clock(), logger.perf_counter())
                    self._eri = hf._build_eri(mol1, mol2, mol.cart)
                    if mol.verbose >= logger.DEBUG and self._eri is not None:
                        logger.timer(mol,
                                    f'Incore ERI between {self.mf1_type} and {self.mf2_type}',
                                    *cput0)
                        logger.debug(mol, f'    Memory usage: {self._eri.nbytes/1024**2:.3f} MB')
                if dm2 is not None:
                    vj[self.mf1_type] = hf.dot_eri_dm(self._eri, dm2,
                                                nao_v=mol1.nao, eri_dot_dm=True)
                if dm1 is not None:
                    vj[self.mf2_type] = hf.dot_eri_dm(self._eri, dm1,
                                                nao_v=mol2.nao, eri_dot_dm=False)
            else:
                if not mol.direct_vee:
                    warnings.warn(f'Direct Vee is used for {self.mf1_type}-{self.mf2_type} ERIs, '
                                +'might be slow. '
                                +f'PYSCF_MAX_MEMORY is set to {mol.max_memory} MB, '
                                +f'required memory: {mol1.nao**2*mol2.nao**2*2/1e6=:.2f} MB')
                if dm1 is not None and dm2 is not None:
                    vj[self.mf1_type], vj[self.mf2_type] = \
                            scf.jk.get_jk((mol1, mol1, mol2, mol2),
                                        (dm2, dm1),
                                        scripts=('ijkl,lk->ij', 'ijkl,ji->kl'),
                                        intor='int2e', aosym='s4')
                elif dm1 is not None:
                    vj[self.mf2_type] = \
                            scf.jk.get_jk((mol1, mol1, mol2, mol2),
                                        dm1,
                                        scripts='ijkl,ji->kl',
                                        intor='int2e', aosym='s4')
                else:
                    vj[self.mf1_type] = \
                            scf.jk.get_jk((mol1, mol1, mol2, mol2),
                                        dm2,
                                        scripts='ijkl,lk->ij',
                                        intor='int2e', aosym='s4')
            charge_product = self.mf1.charge * self.mf2.charge
            if self.mf1_type in vj:
                vj[self.mf1_type] *= charge_product
            if self.mf2_type in vj:
                vj[self.mf2_type] *= charge_product
        return vj

class DFInteractionCorrelation(DFInteractionCoulomb, ks.InteractionCorrelation):
    def __init__(self, *args, df_ne=False, auxbasis=None, epc=None, **kwargs):
        super().__init__(*args, df_ne=df_ne, auxbasis=auxbasis, **kwargs)
        self.epc = epc
        self.grids = None
        self._elec_grids_hash = None
        self._skip_epc = False

    def get_vint(self, dm, *args, no_epc=False, **kwargs):
        '''Copied from neo.ks'''
        from pyscf.neo.ks import (_hash_grids, eval_ao, eval_epc, eval_rho,
                                  _dot_ao_ao, _dot_ao_ao_sparse, _scale_ao,
                                  precompute_epc_electron, NBINS, BLKSIZE)
        import copy
        vj = super().get_vint(dm, *args, **kwargs)
        # For nuclear initial guess, use Coulomb only
        if no_epc or \
                not (self.mf1_type in dm and self.mf2_type in dm and self._need_epc()):
            return vj

        if self.mf1_type == 'e':
            mf_e, dm_e = self.mf1, dm[self.mf1_type]
            if self.mf1_unrestricted:
                assert dm_e.ndim > 2 and dm_e.shape[0] == 2
                dm_e = dm_e[0] + dm_e[1]
            mf_n, dm_n = self.mf2, dm[self.mf2_type]
            n_type = self.mf2_type
        else:
            mf_e, dm_e = self.mf2, dm[self.mf2_type]
            if self.mf2_unrestricted:
                assert dm_e.ndim > 2 and dm_e.shape[0] == 2
                dm_e = dm_e[0] + dm_e[1]
            mf_n, dm_n = self.mf1, dm[self.mf1_type]
            n_type = self.mf1_type

        ni = mf_e._numint
        mol_e = mf_e.mol
        mol_n = mf_n.mol
        nao_e = mol_e.nao
        nao_n = mol_n.nao
        ao_loc_e = mol_e.ao_loc_nr()
        ao_loc_n = mol_n.ao_loc_nr()

        grids_e = mf_e.grids
        grids_changed = (self._elec_grids_hash != _hash_grids(grids_e))
        if grids_changed:
            self._skip_epc = False
        if self._skip_epc:
            return vj

        if self.grids is None or grids_changed:
            if grids_e.coords is None:
                grids_e.build(with_non0tab=True)
            self._elec_grids_hash = _hash_grids(grids_e)
            # Screen grids based on nuclear basis functions
            non0tab_n = ni.make_mask(mol_n, grids_e.coords)
            blk_index = numpy.where(numpy.any(non0tab_n > 0, axis=1))[0]

            # Skip if no nuclear basis functions
            if len(blk_index) == 0:
                self._skip_epc = True
                return vj

            # Update grid coordinates and weights
            starts = blk_index[:, None] * BLKSIZE + numpy.arange(BLKSIZE)
            mask = starts < len(grids_e.coords)
            valid_indices = starts[mask]
            self.grids = copy.copy(grids_e)
            self.grids.coords = grids_e.coords[valid_indices]
            self.grids.weights = grids_e.weights[valid_indices]
            self.grids.non0tab = ni.make_mask(mol_e, self.grids.coords)
            self.grids.screen_index = self.grids.non0tab

        grids = self.grids

        exc_sum = 0
        vxc_e = numpy.zeros((nao_e, nao_e))
        vxc_n = numpy.zeros((nao_n, nao_n))

        cutoff = grids.cutoff * 1e2
        nbins = NBINS * 2 - int(NBINS * numpy.log(cutoff) / numpy.log(grids.cutoff))
        pair_mask_e = mol_e.get_overlap_cond() < -numpy.log(ni.cutoff)

        non0tab_n = ni.make_mask(mol_n, grids.coords)

        p1 = 0
        for ao_e, mask_e, weight, coords in ni.block_loop(mol_e, grids, nao_e):
            p0, p1 = p1, p1 + weight.size
            mask_n = non0tab_n[p0//BLKSIZE:p1//BLKSIZE+1]

            rho_e = eval_rho(mol_e, ao_e, dm_e, mask_e)
            rho_e[rho_e < 0] = 0  # Ensure non-negative density
            common = precompute_epc_electron(self.epc, rho_e)

            ao_n = eval_ao(mol_n, coords, non0tab=mask_n)
            rho_n = eval_rho(mol_n, ao_n, dm_n)
            rho_n[rho_n < 0] = 0  # Ensure non-negative density

            exc, vxc_n_grid, vxc_e_grid = eval_epc(common, rho_n)

            den = rho_n * weight
            exc_sum += numpy.dot(den, exc)

            # x0.5 for vmat + vmat.T
            aow = _scale_ao(ao_n, 0.5 * weight * vxc_n_grid)
            vxc_n += _dot_ao_ao(mol_n, ao_n, aow, mask_n,
                                (0, mol_n.nbas), ao_loc_n)
            _dot_ao_ao_sparse(ao_e, ao_e, 0.5 * weight * vxc_e_grid,
                              nbins, mask_e, pair_mask_e, ao_loc_e, 1, vxc_e)

        vxc_n = vxc_n + vxc_n.conj().T
        vxc_e = vxc_e + vxc_e.conj().T

        vxc = {}
        vxc['e'] = lib.tag_array(vj['e'] + vxc_e, exc=exc_sum, vj=vj['e'])
        vxc[n_type] = lib.tag_array(vj[n_type] + vxc_n, exc=0, vj=vj[n_type])
        return vxc



class _DFNEO:

    _keys = {'ee_only_dfj', 'df_ne'}

    def __init__(self, mf, auxbasis=None, ee_only_dfj=False, df_ne=False):
        self.__dict__.update(mf.__dict__)
        self.ee_only_dfj = ee_only_dfj
        self.df_ne = df_ne
        self.auxbasis = auxbasis
        if isinstance(mf, neo.KS):
            self.interactions = hf.generate_interactions(self.components, DFInteractionCorrelation,
                                                         self.max_memory, df_ne=self.df_ne,
                                                         auxbasis=self.auxbasis, epc=mf.epc)
        else:
            self.interactions = hf.generate_interactions(self.components, DFInteractionCoulomb,
                                                        self.max_memory, df_ne=self.df_ne,
                                                        auxbasis=self.auxbasis)
