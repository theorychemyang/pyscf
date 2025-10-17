#!/usr/bin/env python

'''
Density-fitting for interaction Coulomb in (C)NEO
'''

import tempfile
import numpy
import scipy
from pyscf import gto, lib, scf, df, neo
from pyscf.df import df_jk
from pyscf.neo import hf, ks
from pyscf.lib import logger
from pyscf.df.incore import MAX_MEMORY, LINEAR_DEP_THR, aux_e2, _eig_decompose


def cholesky_eri(mol, auxbasis='weigend+etb', auxmol=None,
                 int3c='int3c2e', aosym='s2ij', int2c='int2c2e', comp=1,
                 max_memory=MAX_MEMORY, decompose_j2c='cd',
                 lindep=LINEAR_DEP_THR, verbose=0, fauxe2=aux_e2, low=None):
    '''
    Modified from pyscf.df.incore, add low as additional return
    Returns:
        2D array of (naux,nao*(nao+1)/2) in C-contiguous
        Lower triangular matrix from decompostion
    '''
    from pyscf.df.outcore import _guess_shell_ranges
    assert (comp == 1)
    t0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mol, verbose)
    if auxmol is None:
        auxmol = df.addons.make_auxmol(mol, auxbasis)

    if low is None:
        j2c = auxmol.intor(int2c, hermi=1)
        if decompose_j2c == 'eig':
            low = _eig_decompose(mol, j2c, lindep)
        else:
            try:
                low = scipy.linalg.cholesky(j2c, lower=True)
                decompose_j2c = 'cd'
            except scipy.linalg.LinAlgError:
                low = _eig_decompose(mol, j2c, lindep)
                decompose_j2c = 'eig'
        j2c = None

    naux, naoaux = low.shape
    log.debug('size of aux basis %d', naux)
    log.timer_debug1('2c2e', *t0)

    int3c = gto.moleintor.ascint3(mol._add_suffix(int3c))
    atm, bas, env = gto.mole.conc_env(mol._atm, mol._bas, mol._env,
                                      auxmol._atm, auxmol._bas, auxmol._env)
    ao_loc = gto.moleintor.make_loc(bas, int3c)
    nao = int(ao_loc[mol.nbas])

    if aosym == 's1':
        nao_pair = nao * nao
    else:
        nao_pair = nao * (nao+1) // 2

    cderi = numpy.empty((naux, nao_pair))

    max_words = max_memory*.98e6/8 - low.size - cderi.size
    # Divide by 3 because scipy.linalg.solve may create a temporary copy for
    # ints and return another copy for results
    buflen = min(max(int(max_words/naoaux/comp/3), 8), nao_pair)
    shranges = _guess_shell_ranges(mol, buflen, aosym)
    log.debug1('shranges = %s', shranges)

    cintopt = gto.moleintor.make_cintopt(atm, bas, env, int3c)
    bufs1 = numpy.empty((comp*max([x[2] for x in shranges]),naoaux))
    bufs2 = numpy.empty_like(bufs1)

    p1 = 0
    for istep, sh_range in enumerate(shranges):
        log.debug('int3c2e [%d/%d], AO [%d:%d], nrow = %d',
                  istep+1, len(shranges), *sh_range)
        bstart, bend, nrow = sh_range
        shls_slice = (bstart, bend, 0, mol.nbas, mol.nbas, mol.nbas+auxmol.nbas)
        ints = gto.moleintor.getints3c(int3c, atm, bas, env, shls_slice, comp,
                                       aosym, ao_loc, cintopt, out=bufs1)

        if ints.ndim == 3 and ints.flags.f_contiguous:
            ints = lib.transpose(ints.T, axes=(0,2,1), out=bufs2).reshape(naoaux,-1)
            bufs1, bufs2 = bufs2, bufs1
        else:
            ints = ints.reshape((-1,naoaux)).T

        p0, p1 = p1, p1 + nrow
        if decompose_j2c == 'cd':
            if ints.flags.c_contiguous:
                trsm, = scipy.linalg.get_blas_funcs(('trsm',), (low, ints))
                dat = trsm(1.0, low, ints.T, lower=True, trans_a = 1,
                           side = 1, overwrite_b=True).T
            else:
                dat = scipy.linalg.solve_triangular(low, ints, lower=True,
                                                    overwrite_b=True,
                                                    check_finite=False)
            if dat.flags.f_contiguous:
                dat = lib.transpose(dat.T, out=bufs2)
            cderi[:,p0:p1] = dat
        else:
            dat = numpy.ndarray((naux, ints.shape[1]), buffer=bufs2)
            cderi[:,p0:p1] = lib.dot(low, ints, c=dat)
        dat = ints = None

    log.timer('cholesky_eri', *t0)
    return cderi, low

def dot_cderi_dm(eri1, eri2, dm1, dm2_len):
    dms1 = numpy.asarray(dm1)
    dm_shape1 = dms1.shape
    nao1 = dm_shape1[-1]
    dms1 = dms1.reshape(-1,nao1,nao1)

    idx = numpy.arange(nao1)
    dmtril = lib.pack_tril(dms1 + dms1.conj().transpose(0,2,1))
    dmtril[:,idx*(idx+1)//2+idx] *= .5
    vj = dmtril.dot(eri1.T).dot(eri2)
    if len(dm_shape1) == 2:
        return lib.unpack_tril(vj, 1).reshape((dm2_len, dm2_len))
    # To support multiple dm1
    return lib.unpack_tril(vj, 1).reshape((-1, dm2_len, dm2_len))

def _build_cderi(mf_e, mf_n, cart, max_memory, verbose=0):
    auxmol_e = mf_e.with_df.auxmol
    low = mf_e.with_df._low

    if auxmol_e is None:
        auxmol_e = df.addons.make_auxmol(mf_e.mol, mf_e.with_df.auxbasis)
        mf_e.with_df.auxmol = auxmol_e

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
        return cholesky_eri(mol_n, auxmol=auxmol_e,
                            int3c=int3c, int2c=int2c,
                            max_memory=max_memory, verbose=verbose,
                            low=low)
    else:
        raise NotImplementedError('outcore df_ne not implemented')

def density_fit_e(mf, auxbasis=None, with_df=None, only_dfj=False):
    '''Modified from pyscf.df.df_jk to use DFE class instead of df.DF as with_df'''
    assert (isinstance(mf, scf.hf.SCF))

    if with_df is None:
        with_df = DFE(mf.mol)
        with_df.max_memory = mf.max_memory
        with_df.stdout = mf.stdout
        with_df.verbose = mf.verbose
        with_df.auxbasis = auxbasis

    if isinstance(mf, df_jk._DFHF):
        if mf.with_df is None:
            mf.with_df = with_df
        elif getattr(mf.with_df, 'auxbasis', None) != auxbasis:
            #logger.warn(mf, 'DF might have been initialized twice.')
            mf = mf.copy()
            mf.with_df = with_df
            mf.only_dfj = only_dfj
        return mf

    dfmf = df_jk._DFHF(mf, with_df, only_dfj)
    return lib.set_class(dfmf, (df_jk._DFHF, mf.__class__))

def density_fit(mf, auxbasis=None, ee_only_dfj=False, df_ne=False):
    assert isinstance(mf, neo.HF)

    if not isinstance(mf.components['e'], df_jk._DFHF):
        mf.components['e'] = density_fit_e(mf.components['e'],
                                           auxbasis=auxbasis,
                                           only_dfj=ee_only_dfj)

    if isinstance(mf, _DFNEO):
        return mf

    dfmf = _DFNEO(mf, auxbasis, ee_only_dfj, df_ne)
    if df_ne:
        name = _DFNEO.__name_mixin__ + '-EE&NE-' + mf.__class__.__name__
    else:
        name = _DFNEO.__name_mixin__ + '-EE-' + mf.__class__.__name__
    return lib.set_class(dfmf, (_DFNEO, mf.__class__), name)

class DFInteractionCoulomb(hf.InteractionCoulomb):
    def __init__(self, *args, df_ne=False, auxbasis=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.df_ne = df_ne
        self.auxbasis = auxbasis
        self._cderi = None
        self._low = None

    def get_vint(self, dm):
        if not self.df_ne or not \
           ((self.mf1_type.startswith('n') and self.mf2_type.startswith('e')) or
            (self.mf1_type.startswith('e') and self.mf2_type.startswith('n'))):
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

        assert isinstance(mf_e, df_jk._DFHF)

        if self._cderi is None:
            self._cderi, self._low = _build_cderi(mf_e, mf_n, mol.cart,
                                                  self.max_memory, mol.verbose)
            if mf_e.with_df._low is None:
                mf_e.with_df._low = self._low

        if dm_e is not None:
            vj[t_n] = dot_cderi_dm(mf_e._cderi, self._cderi, dm_e, mf_n.mol.nao_nr())

        if dm_n is not None:
            vj['e'] = dot_cderi_dm(self._cderi, mf_e._cderi, dm_n, mf_e.mol.nao_nr())

        charge_product = self.mf1.charge * self.mf2.charge

        if self.mf1_type in vj:
            vj[self.mf1_type] *= charge_product

        if self.mf2_type in vj:
            vj[self.mf2_type] *= charge_product

        return vj

class DFInteractionCorrelation(ks.InteractionCorrelation, DFInteractionCoulomb):
    def __init__(self, *args, df_ne=False, auxbasis=None, epc=None, **kwargs):
        super().__init__(*args, epc=epc, **kwargs)
        DFInteractionCoulomb.__init__(self, *args, df_ne=df_ne,
                                      auxbasis=auxbasis, **kwargs)

class DFE(df.DF):
    '''Modfied from pyscf.df to also hold int2c2e deocomposition low'''
    def __init__(self, mol, auxbasis=None):
        super().__init__(mol, auxbasis)
        self._low = None # chelosky decomposition of int2c2e

    def build(self):
        t0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)

        self.check_sanity()
        self.dump_flags()
        if self._cderi is not None and self.auxmol is None:
            log.info('Skip DF.build(). Tensor _cderi will be used.')
            return self

        mol = self.mol
        auxmol = self.auxmol = df.addons.make_auxmol(self.mol, self.auxbasis)
        nao = mol.nao_nr()
        naux = auxmol.nao_nr()
        nao_pair = nao*(nao+1)//2

        is_custom_storage = isinstance(self._cderi_to_save, str)
        max_memory = self.max_memory - lib.current_memory()[0]
        int3c = mol._add_suffix('int3c2e')
        int2c = mol._add_suffix('int2c2e')

        if (nao_pair*naux*8/1e6 < .9*max_memory and not is_custom_storage):
            self._cderi, self._low = cholesky_eri(
                mol, int3c=int3c, int2c=int2c,
                auxmol=auxmol, max_memory=max_memory,
                verbose=log, low=self._low) # NOTE: only difference from df.DF
        else:
            if self._cderi_to_save is None:
                self._cderi_to_save = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
            cderi = self._cderi_to_save

            if is_custom_storage:
                cderi_name = cderi
            else:
                cderi_name = cderi.name
            if self._cderi is None:
                log.info('_cderi_to_save = %s', cderi_name)
            else:
                # If cderi needs to be saved in
                log.warn('Value of _cderi is ignored. DF integrals will be '
                         'saved in file %s .', cderi_name)

            if self._compatible_format:
                df.outcore.cholesky_eri(
                    mol, cderi, dataname=self._dataname,
                    int3c=int3c, int2c=int2c, auxmol=auxmol,
                    max_memory=max_memory, verbose=log)
            else:
                # Store DF tensor in blocks. This is to reduce the
                # initialization overhead
                df.outcore.cholesky_eri_b(
                    mol, cderi, dataname=self._dataname,
                    int3c=int3c, int2c=int2c, auxmol=auxmol,
                    max_memory=max_memory, verbose=log)

            self._cderi = cderi
            log.timer_debug1('Generate density fitting integrals', *t0)
        return self

class _DFNEO:
    __name_mixin__ = 'DF'

    _keys = {'ee_only_dfj', 'df_ne', 'auxbasis'}

    def __init__(self, mf, auxbasis=None, ee_only_dfj=False, df_ne=False):
        self.__dict__.update(mf.__dict__)
        self.ee_only_dfj = ee_only_dfj
        self.df_ne = df_ne

        self.auxbasis = auxbasis

        if isinstance(mf, neo.KS):
            self.interactions = hf.generate_interactions(
                self.components, DFInteractionCorrelation,
                self.max_memory, df_ne=self.df_ne,
                auxbasis=self.auxbasis, epc=mf.epc)
        else:
            self.interactions = hf.generate_interactions(
                self.components, DFInteractionCoulomb,
                self.max_memory, df_ne=self.df_ne,
                auxbasis=self.auxbasis)

    def nuc_grad_method(self):
        import pyscf.neo.df_grad
        return pyscf.neo.df_grad.Gradients(self)

    Gradients = lib.alias(nuc_grad_method, alias_name='Gradients')

    def reset(self, mol=None):
        '''Reset mol and clean up relevant attributes for scanner mode'''
        super().reset(mol)
        # Need this because components and interactions can be
        # completely destroyed in super().reset
        if not isinstance(self.components['e'], df_jk._DFHF):
            self.components['e'] = density_fit_e(self.components['e'],
                                                 auxbasis=self.auxbasis,
                                                 only_dfj=self.ee_only_dfj)
            self.interactions.clear()
            if isinstance(self, neo.KS):
                self.interactions.update(hf.generate_interactions(
                    self.components, DFInteractionCorrelation,
                    self.max_memory, df_ne=self.df_ne,
                    auxbasis=self.auxbasis, epc=self.epc))
            else:
                self.interactions.update(hf.generate_interactions(
                    self.components, DFInteractionCoulomb,
                    self.max_memory, df_ne=self.df_ne,
                    auxbasis=self.auxbasis))
        if self.components['e'].with_df is not None:
            self.components['e'].with_df._low = None
        for t, comp in self.interactions.items():
            comp._cderi = None
            comp._low = None
        return self
