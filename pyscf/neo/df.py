import contextlib
import ctypes
import os
import copy as cp
import re
import numpy
import scipy.linalg
import h5py
from pyscf import lib, gto, scf, neo
from pyscf.neo import mole as neo_mole
from pyscf.lib import logger
from pyscf.lib.exceptions import BasisNotFoundError
from pyscf.df import addons, df, df_jk
from pyscf.df.incore import aux_e2, _eig_decompose
from pyscf.ao2mo.outcore import _load_from_h5g
from pyscf.ao2mo import _ao2mo
from pyscf import __config__


MAX_MEMORY = getattr(__config__, 'df_outcore_max_memory', 2000)  # 2GB
# LINEAR_DEP_THR cannot be below 1e-7,
# see qchem default setting in https://manual.q-chem.com/5.4/sec_Basis_Customization.html
LINEAR_DEP_THR = getattr(__config__, 'df_df_DF_lindep', 1e-7)

def cholesky_eri_incore(mol, auxbasis='weigend+etb', auxmol=None,
                        int3c='int3c2e', aosym='s2ij', int2c='int2c2e', comp=1,
                        max_memory=MAX_MEMORY, decompose_j2c='cd',
                        lindep=LINEAR_DEP_THR, verbose=0, fauxe2=aux_e2):
    from pyscf.df.outcore import _guess_shell_ranges
    assert (comp == 1)
    t0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mol, verbose)
    if auxmol is None:
        auxmol = addons.make_auxmol(mol.components['e'], auxbasis)

    if not mol.cart and auxmol.cart:
        raise NotImplementedError('Interface for int3c2e_ssc')
    elif mol.cart and not auxmol.cart:
        raise RuntimeError('Cartesian orbitals for mol and spherical orbitals for auxmol not supported')

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

    max_words = max_memory*.98e6/8 - low.size
    cderi = {}
    for t, mol_ in mol.components.items():
        int3c = gto.moleintor.ascint3(mol_._add_suffix(int3c))
        atm, bas, env = gto.mole.conc_env(mol_._atm, mol_._bas, mol_._env,
                                          auxmol._atm, auxmol._bas, auxmol._env)
        ao_loc = gto.moleintor.make_loc(bas, int3c)
        nao = int(ao_loc[mol_.nbas])

        if aosym == 's1':
            nao_pair = nao * nao
        else:
            nao_pair = nao * (nao+1) // 2

        cderi[t] = numpy.empty((naux, nao_pair))

        max_words -= cderi[t].size
        # Divide by 3 because scipy.linalg.solve may create a temporary copy for
        # ints and return another copy for results
        buflen = min(max(int(max_words/naoaux/comp/3), 8), nao_pair)
        shranges = _guess_shell_ranges(mol_, buflen, aosym)
        log.debug1('shranges = %s', shranges)

        cintopt = gto.moleintor.make_cintopt(atm, bas, env, int3c)
        bufs1 = numpy.empty((comp*max([x[2] for x in shranges]),naoaux))
        bufs2 = numpy.empty_like(bufs1)

        p1 = 0
        for istep, sh_range in enumerate(shranges):
            log.debug('int3c2e for %s [%d/%d], AO [%d:%d], nrow = %d',
                      t, istep+1, len(shranges), *sh_range)
            bstart, bend, nrow = sh_range
            shls_slice = (bstart, bend, 0, mol_.nbas, mol_.nbas, mol_.nbas+auxmol.nbas)
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
                    dat = trsm(1.0, low, ints.T, lower=True, trans_a = 1, side = 1, overwrite_b=True).T
                else:
                    dat = scipy.linalg.solve_triangular(low, ints, lower=True,
                                                       overwrite_b=True, check_finite=False)
                if dat.flags.f_contiguous:
                    dat = lib.transpose(dat.T, out=bufs2)
                cderi[t][:,p0:p1] = dat
            else:
                dat = numpy.ndarray((naux, ints.shape[1]), buffer=bufs2)
                cderi[t][:,p0:p1] = lib.dot(low, ints, c=dat)
            dat = ints = None

    log.timer('cholesky_eri', *t0)
    return cderi

def _combine_auxbasis(mol, auxmols):
    '''Build a component-unified auxmol with parent atom indexing.'''
    fake_mol = mol.components['e'].copy(deep=False)
    basis = {}
    atoms = []
    for ia, atom in enumerate(fake_mol._atom):
        symb = fake_mol.atom_symbol(ia)
        label = f'{symb}{ia}'
        atoms.append((label, atom[1]))
        basis[label] = []
        for auxmol in auxmols:
            if auxmol.natm != mol.natm:
                raise RuntimeError('DF auxiliary molecule is inconsistent with NEO molecule')
            aux_symb = auxmol.atom_symbol(ia)
            basis[label].extend(cp.deepcopy(auxmol._basis.get(aux_symb, ())))
    fake_mol._atom = atoms
    return addons.make_auxmol(fake_mol, basis)

def _make_single_nuc_mol(mol_n):
    ia = mol_n.atom_index
    label = mol_n.atom_symbol(ia)
    fake_mol = gto.Mole()
    fake_mol.build(atom=[(label, mol_n.atom_coord(ia))],
                   basis={label: mol_n._basis[label]},
                   unit='Bohr', charge=gto.charge(label), spin=0,
                   dump_input=False, parse_arg=False, verbose=0)
    return fake_mol

def _make_nuc_aug_etb(fake_mol, beta):
    with lib.temporary_env(addons, USE_VERSION_26_AUXBASIS=False):
        return addons.aug_etb(fake_mol, beta=beta)

def _make_nuc_auxbasis(mol_n, nuc_auxbasis, nuc_auxbasis_beta=2.0):
    ia = mol_n.atom_index
    label = mol_n.atom_symbol(ia)

    fake_mol = _make_single_nuc_mol(mol_n)
    if nuc_auxbasis is None or nuc_auxbasis == 'aug_etb':
        return _make_nuc_aug_etb(fake_mol, nuc_auxbasis_beta)
    if nuc_auxbasis == 'autoaux':
        return addons.autoaux(fake_mol)
    if nuc_auxbasis == 'autoabs':
        raise NotImplementedError('autoabs requires a named orbital basis; '
                                  'NEO nuclear bases are formatted primitives')

    if isinstance(nuc_auxbasis, str) and re.fullmatch(r'\d+s\d+p\d+d(\d+f)?',
                                                      nuc_auxbasis):
        return {label: neo_mole.make_even_tempered_nuclear_basis(
            mol_n.super_mol, ia, nuc_auxbasis, alpha_scale=2.0)}

    if isinstance(nuc_auxbasis, str):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull), \
                 contextlib.redirect_stderr(devnull):
                try:
                    auxmol = addons.make_auxmol(fake_mol, nuc_auxbasis)
                    return {label: auxmol._basis[label]}
                except BasisNotFoundError:
                    pass
        return {label: neo_mole.make_nuclear_basis(mol_n.super_mol, ia,
                                                   nuc_auxbasis)}
    if isinstance(nuc_auxbasis, (list, tuple)):
        return {label: nuc_auxbasis}
    if isinstance(nuc_auxbasis, dict):
        return nuc_auxbasis
    raise TypeError('nuc_auxbasis must be None, a string, a basis list, '
                    'or a basis dictionary')

def _make_nuc_auxmol(mol_n, nuc_auxbasis=None, nuc_auxbasis_beta=2.0):
    auxbasis = _make_nuc_auxbasis(mol_n, nuc_auxbasis, nuc_auxbasis_beta)
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stderr(devnull):
            return addons.make_auxmol(mol_n, auxbasis)

def cholesky_eri_b_outcore(mol, erifile, auxbasis='weigend+etb', dataname='j3c',
                           int3c='int3c2e', aosym='s2ij', int2c='int2c2e', comp=1,
                           max_memory=MAX_MEMORY, auxmol=None, decompose_j2c='CD',
                           lindep=LINEAR_DEP_THR, verbose=logger.NOTE):
    from pyscf.df.outcore import _guess_shell_ranges, _create_h5file
    assert (aosym in ('s1', 's2ij'))
    log = logger.new_logger(mol, verbose)
    time0 = (logger.process_clock(), logger.perf_counter())

    if auxmol is None:
        auxmol = addons.make_auxmol(mol.components['e'], auxbasis)

    if not mol.cart and auxmol.cart:
        raise NotImplementedError('Interface for int3c2e_ssc')
    elif mol.cart and not auxmol.cart:
        raise RuntimeError('Cartesian orbitals for mol and spherical orbitals for auxmol not supported')

    j2c = auxmol.intor(int2c, hermi=1)
    log.debug('size of aux basis %d', j2c.shape[0])
    time1 = log.timer('2c2e', *time0)
    decompose_j2c = decompose_j2c.upper()
    if decompose_j2c != 'CD':
        low = _eig_decompose(mol, j2c, lindep)
    else:
        try:
            low = scipy.linalg.cholesky(j2c, lower=True)
            decompose_j2c = 'CD'
        except scipy.linalg.LinAlgError:
            low = _eig_decompose(mol, j2c, lindep)
            decompose_j2c = 'ED'
    j2c = None
    naoaux, naux = low.shape
    time1 = log.timer('Cholesky 2c2e', *time1)

    def transform(b):
        if b.ndim == 3 and b.flags.f_contiguous:
            b = lib.transpose(b.T, axes=(0,2,1)).reshape(naoaux,-1)
        else:
            b = b.reshape((-1,naoaux)).T
        if decompose_j2c != 'CD':
            return lib.dot(low, b)

        if b.flags.c_contiguous:
            trsm, = scipy.linalg.get_blas_funcs(('trsm',), (low, b))
            return trsm(1.0, low, b.T, lower=True, trans_a = 1, side = 1,
                     overwrite_b=True).T
        else:
            return scipy.linalg.solve_triangular(low, b, lower=True,
                                             overwrite_b=True, check_finite=False)

    for t, mol_ in mol.components.items():
        int3c = gto.moleintor.ascint3(mol_._add_suffix(int3c))
        atm, bas, env = gto.mole.conc_env(mol_._atm, mol_._bas, mol_._env,
                                          auxmol._atm, auxmol._bas, auxmol._env)
        ao_loc = gto.moleintor.make_loc(bas, int3c)
        nao = int(ao_loc[mol_.nbas])
        naoaux = int(ao_loc[-1] - nao)
        if aosym == 's1':
            nao_pair = nao * nao
            buflen = min(max(int(max_memory*.24e6/8/naoaux/comp), 1), nao_pair)
            shranges = _guess_shell_ranges(mol_, buflen, 's1')
        else:
            nao_pair = nao * (nao+1) // 2
            buflen = min(max(int(max_memory*.24e6/8/naoaux/comp), 1), nao_pair)
            shranges = _guess_shell_ranges(mol_, buflen, 's2ij')
        log.debug('erifile %.8g MB, IO buf size %.8g MB',
                  naoaux*nao_pair*8/1e6, comp*buflen*naoaux*8/1e6)
        log.debug1('shranges = %s', shranges)
        # TODO: Libcint-3.14 and newer version support to compute int3c2e without
        # the opt for the 3rd index.
        #if '3c2e' in int3c:
        #    cintopt = gto.moleintor.make_cintopt(atm, mol._bas, env, int3c)
        #else:
        #    cintopt = gto.moleintor.make_cintopt(atm, bas, env, int3c)
        cintopt = gto.moleintor.make_cintopt(atm, bas, env, int3c)
        bufs1 = numpy.empty((comp*max([x[2] for x in shranges]),naoaux))
        bufs2 = numpy.empty_like(bufs1)

        def process(sh_range):
            nonlocal bufs1, bufs2
            bufs2, bufs1 = bufs1, bufs2
            bstart, bend, nrow = sh_range
            shls_slice = (bstart, bend, 0, mol_.nbas, mol_.nbas, mol_.nbas+auxmol.nbas)
            ints = gto.moleintor.getints3c(int3c, atm, bas, env, shls_slice, comp,
                                           aosym, ao_loc, cintopt, out=bufs1)
            if comp == 1:
                dat = transform(ints)
            else:
                dat = [transform(x) for x in ints]
            return dat

        feri = _create_h5file(erifile, f'{dataname}/{t}')

        for istep, dat in enumerate(lib.map_with_prefetch(process, shranges)):
            sh_range = shranges[istep]
            label = f'{dataname}/{t}/{istep}'
            if comp == 1:
                feri[label] = dat
            else:
                shape = (len(dat),) + dat[0].shape
                fdat = feri.create_dataset(label, shape, dat[0].dtype.char)
                for i, b in enumerate(dat):
                    fdat[i] = b
            dat = None
            log.debug('int3c2e for %s [%d/%d], AO [%d:%d], nrow = %d',
                      t, istep+1, len(shranges), *sh_range)
            time1 = log.timer('gen CD eri for %s [%d/%d]' % (t,istep+1,len(shranges)), *time1)
        bufs1 = None
        bufs2 = None
    feri.flush()
    feri.close()
    return erifile

class DF(df.DF):
    '''build all e-e and e-n cderi'''
    _keys = df.DF._keys.union(['df_ne_scheme', 'nuc_auxbasis',
                               'nuc_auxbasis_beta'])

    def __init__(self, mol, auxbasis=None, df_ne_scheme='global',
                 nuc_auxbasis=None, nuc_auxbasis_beta=2.0):
        super().__init__(mol, auxbasis)
        self._cderi_names = list(self.mol.components.keys())
        self._charges = {}
        self._unrestricted = {}
        self.df_ne_scheme = df_ne_scheme
        self.nuc_auxbasis = nuc_auxbasis
        self.nuc_auxbasis_beta = nuc_auxbasis_beta

    def _check_df_ne_scheme(self):
        if self.df_ne_scheme not in ('electron', 'global'):
            raise ValueError(f'Unsupported df_ne_scheme {self.df_ne_scheme}')

    def make_auxmol(self):
        auxmol = addons.make_auxmol(self.mol.components['e'], self.auxbasis)
        if self.df_ne_scheme != 'global':
            return auxmol

        # Global DF-NE metric.
        #
        # The motivation is related to the multicomponent Cholesky-decomposition
        # idea in J. Chem. Theory Comput. 2023, 19, 6255-6262
        # (DOI: 10.1021/acs.jctc.3c00686), but this implementation is not a
        # literal reproduction of that paper.  They use a larger metric for the
        # electron-proton interaction in an indistinguishable-proton NEO-DFT
        # setting, and their algorithm is formulated as CD rather than PySCF's
        # RI-style auxiliary-basis density fitting.
        #
        # For distinguishable NEO nuclei there are several natural RI choices:
        #   1. Use the conventional e-e auxiliary metric for e-e, and separate
        #      pair-specific auxiliary metrics for each e-n pair.
        #   2. Use the conventional e-e metric for e-e, and one global mixed
        #      e/n auxiliary metric for all e-n pairs.  This is closest in
        #      spirit to the JCTC paper, aside from RI vs CD and distinguishable
        #      nuclei.
        #   3. Use one global mixed e/n auxiliary metric for both e-e and e-n.
        #
        # This code chooses option 3.  The main reason is efficiency, not a
        # claim that option 3 is universally the most accurate RI objective:
        # all components share one auxiliary metric and one set of transformed
        # three-center tensors, so the SCF J build can reuse the same D*L
        # contractions for e-e and e-n.  Option 2 still uses one e-n metric and
        # one e-e metric, so it loses some sharing.  Option 1 is more expensive
        # for multiple quantum nuclei because each pair-specific e-n metric
        # requires its own density projection and back transformation.  These
        # tradeoffs may be revisited if accuracy or robustness cases justify
        # the extra cost.
        auxmols = [auxmol]
        for t, mol_n in self.mol.components.items():
            if t == 'e':
                continue
            auxmols.append(_make_nuc_auxmol(mol_n, self.nuc_auxbasis,
                                            self.nuc_auxbasis_beta))
        return _combine_auxbasis(self.mol, auxmols)

    __getstate__, __setstate__ = lib.generate_pickle_methods(
            excludes=('_cderi_to_save', '_cderi', '_cderi_names', '_vjopt', '_rsh_df'),
            reset_state=True)

    def build(self):
        t0 = (logger.process_clock(), logger.perf_counter())
        log = logger.Logger(self.stdout, self.verbose)

        self._check_df_ne_scheme()
        self.check_sanity()
        self.dump_flags()
        if self._cderi is not None and self.auxmol is None:
            log.info('Skip DF.build(). Tensor _cderi will be used.')
            return self

        mol = self.mol
        auxmol = self.auxmol = self.make_auxmol()
        naux = auxmol.nao_nr()
        nao_pair = 0
        for t, comp in mol.components.items():
            nao = comp.nao_nr()
            nao_pair += nao*(nao+1)//2

        is_custom_storage = isinstance(self._cderi_to_save, str)
        max_memory = self.max_memory - lib.current_memory()[0]
        int3c = mol._add_suffix('int3c2e')
        int2c = mol._add_suffix('int2c2e')
        if (nao_pair*naux*8/1e6 < .9*max_memory and not is_custom_storage):
            self._cderi = cholesky_eri_incore(mol, int3c=int3c, int2c=int2c,
                                              auxmol=auxmol,
                                              max_memory=max_memory, verbose=log)
        else:
            log.warn(f'Low memory: {max_memory=}. Outcore DF integrals.')
            if self._cderi_to_save is None:
                self._cderi_to_save = lib.NamedTemporaryFile(dir=lib.param.TMPDIR)
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
                raise NotImplementedError
            else:
                # Store DF tensor in blocks. This is to reduce the
                # initialization overhead
                cholesky_eri_b_outcore(mol, cderi, dataname=self._dataname,
                                       int3c=int3c, int2c=int2c, auxmol=auxmol,
                                       max_memory=max_memory, verbose=log)
            self._cderi = cderi
            log.timer_debug1('Generate density fitting integrals', *t0)
        return self

    def reset(self, mol=None):
        '''Reset mol and clean up relevant attributes for scanner mode'''
        if mol is not None:
            self.mol = mol
            self.auxmol = None
        self._cderi = None
        self._cderi_names = list(self.mol.components.keys())
        self._vjopt = None
        self._rsh_df = {}
        return self

    def loop(self, blksize=None):
        if self._cderi is None:
            self.build()
        if blksize is None:
            blksize = self.blockdim

        names = self._cderi_names

        if isinstance(self._cderi, dict):
            cderi = self._cderi
            naoaux = cderi[names[0]].shape[0]

            for name in names:
                v = cderi.get(name, None)
                if v is None:
                    raise RuntimeError(f'Missing incore cderi entry {name}')
                if v.shape[0] != naoaux:
                    raise RuntimeError(f'Inconsistent naoaux: cderi[{name}].shape[0]={v.shape[0]} != {naoaux}')

            for b0, b1 in self.prange(0, naoaux, blksize):
                out = {}
                for name in names:
                    out[name] = numpy.asarray(cderi[name][b0:b1], order='C')
                yield out
        else:
            with addons.load(self._cderi, None) as root:
                loaders = []
                naoaux = None
                for name in names:
                    path = f'{self._dataname}/{name}'
                    if path not in root:
                        raise RuntimeError(f'Cannot find outcore entry "{path}" in cderi container')

                    feri = root[path]
                    if isinstance(feri, h5py.Group):
                        # starting from pyscf-1.7, DF tensor may be stored in
                        # block format
                        naoaux_ = feri['0'].shape[0]
                        def load(aux_slice, feri=feri):
                            b0, b1 = aux_slice
                            return _load_from_h5g(feri, b0, b1)
                    else:
                        naoaux_ = feri.shape[0]
                        def load(aux_slice, feri=feri):
                            b0, b1 = aux_slice
                            return numpy.asarray(feri[b0:b1])

                    if naoaux is None:
                        naoaux = naoaux_
                    elif naoaux_ != naoaux:
                        raise RuntimeError(f'Inconsistent naoaux: entry "{path}" has {naoaux_}, expected {naoaux}')

                    loaders.append((name, load))

                def load_all(aux_slice):
                    out = {}
                    for name, fn in loaders:
                        out[name] = fn(aux_slice)
                    return out

                for dat in lib.map_with_prefetch(load_all, self.prange(0, naoaux, blksize)):
                    yield dat
                    dat = None

    def get_naoaux(self):
        # determine naoaux with self._cderi, because DF object may be used as CD
        # object when self._cderi is provided.
        if self._cderi is None:
            self.build()
        names = self._cderi_names
        if isinstance(self._cderi, dict):
            return self._cderi[names[0]].shape[0]
        else:
            with addons.load(self._cderi, f'{self._dataname}/{names[0]}') as feri:
                if isinstance(feri, h5py.Group):
                    return feri['0'].shape[0]
                else:
                    return feri.shape[0]

    def get_jk(self, dm, hermi=1, with_j=True, with_k=True,
               direct_scf_tol=getattr(__config__, 'scf_hf_SCF_direct_scf_tol', 1e-13),
               omega=None):
        if omega is not None:
            raise ValueError('RSH-DF is supposed to be handled by elec component.')

        return get_jk(self, dm, hermi, with_j, with_k, direct_scf_tol)

    def get_j(self, dm, hermi=1, omega=None):
        return self.get_jk(dm, hermi, with_k=False, omega=omega)[0]

    def get_eri(self):
        raise NotImplementedError

    def ao2mo(self, mo_coeffs,
              compact=getattr(__config__, 'df_df_DF_ao2mo_compact', True)):
        raise NotImplementedError

    def range_coulomb(self, omega):
        raise ValueError

def get_jk(dfobj, dm, hermi=0, with_j=True, with_k=True, direct_scf_tol=1e-13):
    '''vj returned is already combined for alpha and beta spins'''
    assert (with_j or with_k)
    if (not with_k and not dfobj.mol.incore_anyway and
        # 3-center integral tensor is not initialized
        dfobj._cderi is None):
        return get_j(dfobj, dm, hermi, direct_scf_tol), None

    t0 = t1 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(dfobj.stdout, dfobj.verbose)
    fmmm = _ao2mo.libao2mo.AO2MOmmm_bra_nr_s2
    fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv
    ftrans = _ao2mo.libao2mo.AO2MOtranse2_nr_s2
    null = lib.c_null_ptr()

    dms = {}
    dm_shape = {}
    dm_e = None # 'e' total density for unrestricted case
    dm_shape_e = None # 'e' total density matrix shape
    vj = {}
    vj_inter_e = None
    nao_e = None
    nao_max = 0
    for t, dm_ in dm.items():
        dms[t] = numpy.asarray(dm_)
        dm_shape[t] = dms[t].shape
        nao = dm_shape[t][-1]
        if t == 'e':
            nao_e = nao
        elif nao > nao_max:
            nao_max = nao
        if t == 'e' and dfobj._unrestricted[t]:
            # TODO: what if dm[0] + dm[1] is passed to get_jk with with_k=False?
            assert dms['e'].shape[0] == 2
            # get total density for vj
            dm_e = dms['e'][0] + dms['e'][1]
            dm_shape_e = dm_e.shape
            dm_e = dm_e.reshape(-1,nao,nao)
        dms[t] = dms[t].reshape(-1,nao,nao)
        if t == 'e' and not dfobj._unrestricted[t]:
            # restricted, simply a mapping. TODO: ROHF?
            dm_e = dms['e']
            dm_shape_e = dm_shape['e']
        vj[t] = 0
    if with_j:
        vj_inter_e = 0
    assert nao_max > 0 # max of nuc nao
    vk = numpy.zeros_like(dms['e'])
    nset = dms['e'].shape[0]
    if dfobj._unrestricted['e']:
        expected_nset = nset // 2
    else:
        expected_nset = nset
    # make sure nset is consistent across all dm
    for t, dm_ in dm.items():
        if t == 'e':
            continue
        if not t.startswith('n'):
            raise NotImplementedError
        if dms[t].shape[0] != expected_nset:
            raise ValueError(f'Density matrix of {t} has nset={dms[t].shape[0]}, expected {expected_nset}.')

    if numpy.iscomplexobj(dms['e']):
        if with_j:
            assert numpy.iscomplexobj(dm_e)
            vj['e'] = numpy.zeros_like(dm_e)
            vj_inter_e = numpy.zeros_like(dm_e)
            for t, dm_ in dms.items():
                if t == 'e':
                    continue
                # in this case, nuc dm and vj should also be complex
                vj[t] = numpy.zeros_like(dm_, dtype=dm_e.dtype)

        max_memory = dfobj.max_memory - lib.current_memory()[0]
        total_nao_square = nao_e**2
        if with_j:
            for t, dm_shape_ in dm_shape.items():
                if t == 'e':
                    continue
                total_nao_square += dm_shape_[-1]**2
        blksize = max(4, int(min(dfobj.blockdim, max_memory*.22e6/8/total_nao_square)))
        nao = nao_e
        buf = numpy.empty((blksize,nao,nao))
        buf1 = numpy.empty((nao,blksize,nao))
        buf_n = numpy.empty((blksize,nao_max,nao_max))
        for eri1 in dfobj.loop(blksize):
            eri1_e = eri1['e']
            naux, nao_pair = eri1_e.shape
            eri1_e = lib.unpack_tril(eri1_e, out=buf)
            if with_j:
                eri_e_dm_real = numpy.einsum('pij,nji->pn', eri1_e, dm_e.real)
                eri_e_dm_imag = numpy.einsum('pij,nji->pn', eri1_e, dm_e.imag)
                eri_n_dm_real = 0
                eri_n_dm_imag = 0
                for t, dm_ in dms.items():
                    if t == 'e':
                        continue
                    eri1_n = eri1[t]
                    eri1_n = lib.unpack_tril(eri1_n, out=buf_n)
                    vj_t_real = numpy.einsum('pn,pij->nij', eri_e_dm_real, eri1_n) * dfobj._charges[t]
                    vj_t_imag = numpy.einsum('pn,pij->nij', eri_e_dm_imag, eri1_n) * dfobj._charges[t]
                    vj[t].real += vj_t_real
                    vj[t].imag += vj_t_imag
                    eri_n_dm_real += numpy.einsum('pij,nji->pn', eri1_n, dm_.real) * dfobj._charges[t]
                    eri_n_dm_imag += numpy.einsum('pij,nji->pn', eri1_n, dm_.imag) * dfobj._charges[t]
                vj['e'].real += numpy.einsum('pn,pij->nij', eri_e_dm_real + eri_n_dm_real, eri1_e)
                vj['e'].imag += numpy.einsum('pn,pij->nij', eri_e_dm_imag + eri_n_dm_imag, eri1_e)
                vj_inter_e.real += numpy.einsum('pn,pij->nij', eri_n_dm_real, eri1_e)
                vj_inter_e.imag += numpy.einsum('pn,pij->nij', eri_n_dm_imag, eri1_e)
            buf2 = numpy.ndarray((nao_e,naux,nao_e), buffer=buf1)
            for k in range(nset):
                buf2[:] = lib.einsum('pij,jk->ipk', eri1_e, dms['e'][k].real)
                vk[k].real += lib.einsum('ipk,pkj->ij', buf2, eri1_e)
                buf2[:] = lib.einsum('pij,jk->ipk', eri1_e, dms['e'][k].imag)
                vk[k].imag += lib.einsum('ipk,pkj->ij', buf2, eri1_e)
            t1 = log.timer_debug1('jk', *t1)
        if with_j:
            vj['e'] = vj['e'].reshape(dm_shape_e)
            vj_inter_e = vj_inter_e.reshape(dm_shape_e)
            vj['e'] = lib.tag_array(vj['e'], vint=vj_inter_e)
            for t, dm_shape_ in dm_shape.items():
                if t == 'e':
                    continue
                vj[t] = vj[t].reshape(dm_shape_)
                vj[t] = lib.tag_array(vj[t], vint=vj[t])
        if with_k: vk = vk.reshape(dm_shape['e'])
        logger.timer(dfobj, 'df vj and vk', *t0)
        return vj, vk

    for t, dm_ in dms.items():
        assert not numpy.iscomplexobj(dm_)

    if with_j:
        dmtril = {}
        for t, dm_ in dms.items():
            nao = dm_shape[t][-1]
            idx = numpy.arange(nao)
            if t == 'e':
                dmtril[t] = lib.pack_tril(dm_e + dm_e.conj().transpose(0,2,1))
            else:
                dmtril[t] = lib.pack_tril(dm_ + dm_.conj().transpose(0,2,1))
            dmtril[t][:,idx*(idx+1)//2+idx] *= .5

    if not with_k:
        for eri1 in dfobj.loop():
            # uses numpy.matmul
            eri1_e = eri1['e']
            dm_eri_e = dmtril['e'].dot(eri1_e.T)
            dm_eri_n = 0
            for t, dm_ in dmtril.items():
                if t == 'e':
                    continue
                eri1_n = eri1[t]
                dm_eri_n += dm_.dot(eri1_n.T) * dfobj._charges[t]
                vj_t = dm_eri_e.dot(eri1_n) * dfobj._charges[t]
                vj[t] += vj_t
            vj['e'] += (dm_eri_e + dm_eri_n).dot(eri1_e)
            vj_inter_e += dm_eri_n.dot(eri1_e)

    elif getattr(dm['e'], 'mo_coeff', None) is not None:
        #TODO: test whether dm.mo_coeff matching dm
        mo_coeff = numpy.asarray(dm['e'].mo_coeff, order='F')
        mo_occ   = numpy.asarray(dm['e'].mo_occ)
        nmo = mo_occ.shape[-1]
        nao = nao_e
        mo_coeff = mo_coeff.reshape(-1,nao,nmo)
        mo_occ   = mo_occ.reshape(-1,nmo)
        if mo_occ.shape[0] * 2 == nset: # handle ROHF DM
            mo_coeff = numpy.vstack((mo_coeff, mo_coeff))
            mo_occa = numpy.array(mo_occ> 0, dtype=numpy.double)
            mo_occb = numpy.array(mo_occ==2, dtype=numpy.double)
            assert (mo_occa.sum() + mo_occb.sum() == mo_occ.sum())
            mo_occ = numpy.vstack((mo_occa, mo_occb))

        orbo = []
        for k in range(nset):
            c = numpy.einsum('pi,i->pi', mo_coeff[k][:,mo_occ[k]>0],
                             numpy.sqrt(mo_occ[k][mo_occ[k]>0]))
            orbo.append(numpy.asarray(c, order='F'))

        max_memory = dfobj.max_memory - lib.current_memory()[0]
        total_nao_square = nao_e**2
        if with_j:
            for t, dm_shape_ in dm_shape.items():
                if t == 'e':
                    continue
                total_nao_square += dm_shape_[-1]**2
        blksize = max(4, int(min(dfobj.blockdim, max_memory*.3e6/8/total_nao_square)))
        buf = numpy.empty((blksize*nao,nao))
        for eri1 in dfobj.loop(blksize):
            eri1_e = eri1['e']
            naux, nao_pair = eri1_e.shape
            assert (nao_pair == nao*(nao+1)//2)
            if with_j:
                # uses numpy.matmul
                dm_eri_e = dmtril['e'].dot(eri1_e.T)
                dm_eri_n = 0
                for t, dm_ in dmtril.items():
                    if t == 'e':
                        continue
                    eri1_n = eri1[t]
                    dm_eri_n += dm_.dot(eri1_n.T) * dfobj._charges[t]
                    vj_t = dm_eri_e.dot(eri1_n) * dfobj._charges[t]
                    vj[t] += vj_t
                vj['e'] += (dm_eri_e + dm_eri_n).dot(eri1_e)
                vj_inter_e += dm_eri_n.dot(eri1_e)

            for k in range(nset):
                nocc = orbo[k].shape[1]
                if nocc > 0:
                    buf1 = buf[:naux*nocc]
                    fdrv(ftrans, fmmm,
                         buf1.ctypes.data_as(ctypes.c_void_p),
                         eri1_e.ctypes.data_as(ctypes.c_void_p),
                         orbo[k].ctypes.data_as(ctypes.c_void_p),
                         ctypes.c_int(naux), ctypes.c_int(nao),
                         (ctypes.c_int*4)(0, nocc, 0, nao),
                         null, ctypes.c_int(0))
                    vk[k] += lib.dot(buf1.T, buf1)
            t1 = log.timer_debug1('jk', *t1)
    else:
        nao = nao_e
        #:vk = numpy.einsum('pij,jk->pki', cderi, dm)
        #:vk = numpy.einsum('pki,pkj->ij', cderi, vk)
        rargs = (ctypes.c_int(nao), (ctypes.c_int*4)(0, nao, 0, nao),
                 null, ctypes.c_int(0))
        dms['e'] = [numpy.asarray(x, order='F') for x in dms['e']]
        max_memory = dfobj.max_memory - lib.current_memory()[0]
        total_nao_square = nao_e**2
        if with_j:
            for t, dm_shape_ in dm_shape.items():
                if t == 'e':
                    continue
                total_nao_square += dm_shape_[-1]**2
        blksize = max(4, int(min(dfobj.blockdim, max_memory*.22e6/8/total_nao_square)))
        buf = numpy.empty((2,blksize,nao,nao))
        for eri1 in dfobj.loop(blksize):
            eri1_e = eri1['e']
            naux, nao_pair = eri1_e.shape
            assert (nao_pair == nao*(nao+1)//2)
            if with_j:
                # uses numpy.matmul
                dm_eri_e = dmtril['e'].dot(eri1_e.T)
                dm_eri_n = 0
                for t, dm_ in dmtril.items():
                    if t == 'e':
                        continue
                    eri1_n = eri1[t]
                    dm_eri_n += dm_.dot(eri1_n.T) * dfobj._charges[t]
                    vj_t = dm_eri_e.dot(eri1_n) * dfobj._charges[t]
                    vj[t] += vj_t
                vj['e'] += (dm_eri_e + dm_eri_n).dot(eri1_e)
                vj_inter_e += dm_eri_n.dot(eri1_e)

            for k in range(nset):
                buf1 = buf[0,:naux]
                fdrv(ftrans, fmmm,
                     buf1.ctypes.data_as(ctypes.c_void_p),
                     eri1_e.ctypes.data_as(ctypes.c_void_p),
                     dms['e'][k].ctypes.data_as(ctypes.c_void_p),
                     ctypes.c_int(naux), *rargs)

                buf2 = lib.unpack_tril(eri1_e, out=buf[1])
                vk[k] += lib.dot(buf1.reshape(-1,nao).T, buf2.reshape(-1,nao))
            t1 = log.timer_debug1('jk', *t1)

    if with_j:
        vj['e'] = lib.unpack_tril(vj['e'], 1).reshape(dm_shape_e)
        vj_inter_e = lib.unpack_tril(vj_inter_e, 1).reshape(dm_shape_e)
        vj['e'] = lib.tag_array(vj['e'], vint=vj_inter_e)
        for t, dm_shape_ in dm_shape.items():
            if t == 'e':
                continue
            vj[t] = lib.unpack_tril(vj[t], 1).reshape(dm_shape_)
            vj[t] = lib.tag_array(vj[t], vint=vj[t])
    if with_k: vk = vk.reshape(dm_shape['e'])
    logger.timer(dfobj, 'df vj and vk', *t0)
    return vj, vk

def get_j(dfobj, dm, hermi=0, direct_scf_tol=1e-13):
    '''vj returned is already combined for alpha and beta spins'''
    from pyscf.scf import _vhf
    from pyscf.scf import jk
    t0 = t1 = (logger.process_clock(), logger.perf_counter())

    mol = dfobj.mol
    if dfobj._vjopt is None:
        opt = {}
        if dfobj.auxmol is None:
            dfobj.auxmol = dfobj.make_auxmol()
        auxmol = dfobj.auxmol

        j2c = auxmol.intor('int2c2e', hermi=1)
        j2c_diag = numpy.sqrt(abs(j2c.diagonal()))
        aux_loc = auxmol.ao_loc
        aux_q_cond = [j2c_diag[i0:i1].max()
                      for i0, i1 in zip(aux_loc[:-1], aux_loc[1:])]

        try:
            j2c = scipy.linalg.cho_factor(j2c, lower=True)
            j2c_type = 'cd'
        except scipy.linalg.LinAlgError:
            j2c_type = 'regular'

        # jk.get_jk function supports 4-index integrals. Use bas_placeholder
        # (l=0, nctr=1, 1 function) to hold the last index.
        bas_placeholder = numpy.array([0, 0, 1, 1, 0, 0, 0, 0],
                                      dtype=numpy.int32)

        for t, mol_ in mol.components.items():
            opt[t] = _vhf._VHFOpt(mol_, 'int3c2e', 'CVHFnr3c2e_schwarz_cond',
                                  dmcondname='CVHFnr_dm_cond',
                                  direct_scf_tol=direct_scf_tol)

            # q_cond part 1: the regular int2e (ij|ij) for mol's basis
            opt[t].init_cvhf_direct(mol_, 'int2e', 'CVHFnr_int2e_q_cond')

            # Update q_cond to include the 2e-integrals (auxmol|auxmol)
            q_cond = numpy.hstack((opt[t].q_cond.ravel(), aux_q_cond))
            opt[t].q_cond = q_cond

            opt[t].j2c = j2c
            opt[t].j2c_type = j2c_type

            fakemol = mol_ + auxmol
            fakemol._bas = numpy.vstack((fakemol._bas, bas_placeholder))
            opt[t].fakemol = fakemol

        dfobj._vjopt = opt
        t1 = logger.timer_debug1(dfobj, 'df-vj init_direct_scf', *t1)

    jaux_n = 0
    opt = dfobj._vjopt
    n_dm = None
    dm_shape = {}
    for t, mol_ in mol.components.items():
        opt_ = opt[t]
        fakemol = opt_.fakemol
        dm_ = numpy.asarray(dm[t], order='C')
        assert dm_.dtype == numpy.float64
        # when 'e' is unrestricted, should use total density instead of spin density
        # TODO: what if dm[0] + dm[1] is passed to get_j?
        if dfobj._unrestricted[t]:
            assert dm_.shape[0] == 2
            dm_ = dm_[0] + dm_[1]
        dm_shape[t] = dm_.shape
        nao = dm_shape[t][-1]
        dm_ = dm_.reshape(-1,nao,nao)
        if not (t == 'e' or t.startswith('n')):
            raise NotImplementedError
        # make sure n_dm is consistent across all dm
        if n_dm is None:
            n_dm = dm_.shape[0]
        elif n_dm != dm_.shape[0]:
            raise ValueError(f'Density matrix of {t} has n_dm={dm_.shape[0]}, expected {n_dm}.')

        # First compute the density in auxiliary basis
        # j3c = fauxe2(mol, auxmol)
        # jaux = numpy.einsum('ijk,ji->k', j3c, dm)
        # rho = numpy.linalg.solve(auxmol.intor('int2c2e'), jaux)
        nbas = mol_.nbas
        nbas1 = mol_.nbas + dfobj.auxmol.nbas
        shls_slice = (0, nbas, 0, nbas, nbas, nbas1, nbas1, nbas1+1)
        with lib.temporary_env(opt_, prescreen='CVHFnr3c2e_vj_pass1_prescreen'):
            jaux = jk.get_jk(fakemol, dm_, ['ijkl,ji->kl']*n_dm, 'int3c2e',
                             aosym='s2ij', hermi=0, shls_slice=shls_slice,
                             vhfopt=opt_)
        # remove the index corresponding to bas_placeholder
        jaux = numpy.array(jaux)[:,:,0]
        if t == 'e':
            jaux_e = jaux * dfobj._charges[t]
        else:
            jaux_n += jaux * dfobj._charges[t]
    t1 = logger.timer_debug1(dfobj, 'df-vj pass 1', *t1)

    if opt['e'].j2c_type == 'cd':
        rho_e = scipy.linalg.cho_solve(opt['e'].j2c, jaux_e.T)
        rho_n = scipy.linalg.cho_solve(opt['e'].j2c, jaux_n.T)
    else:
        rho_e = scipy.linalg.solve(opt['e'].j2c, jaux_e.T)
        rho_n = scipy.linalg.solve(opt['e'].j2c, jaux_n.T)
    # transform rho to shape (:,1,naux), to adapt to 3c2e integrals (ij|k)
    rho_e = rho_e.T[:,numpy.newaxis,:]
    rho_n = rho_n.T[:,numpy.newaxis,:]
    t1 = logger.timer_debug1(dfobj, 'df-vj solve ', *t1)

    vj = {}
    vj_inter_e = None
    # CVHFnr3c2e_vj_pass2_prescreen requires custom dm_cond
    aux_loc = dfobj.auxmol.ao_loc
    dm_cond_e = numpy.array([abs(rho_e[:,:,i0:i1]).max()
                             for i0, i1 in zip(aux_loc[:-1], aux_loc[1:])])
    dm_cond_n = numpy.array([abs(rho_n[:,:,i0:i1]).max()
                             for i0, i1 in zip(aux_loc[:-1], aux_loc[1:])])
    for t, mol_ in mol.components.items():
        opt_ = opt[t]
        fakemol = opt_.fakemol
        # Next compute the Coulomb matrix
        # j3c = fauxe2(mol, auxmol)
        # vj = numpy.einsum('ijk,k->ij', j3c, rho)
        # temporarily set "_dmcondname=None" to skip the call to set_dm method.
        nbas = mol_.nbas
        nbas1 = mol_.nbas + dfobj.auxmol.nbas
        shls_slice = (0, nbas, 0, nbas, nbas, nbas1, nbas1, nbas1+1)
        with lib.temporary_env(opt_, prescreen='CVHFnr3c2e_vj_pass2_prescreen',
                               _dmcondname=None):
            if t == 'e':
                opt_.dm_cond = dm_cond_e
                vj_e = jk.get_jk(fakemol, rho_e, ['ijkl,lk->ij']*n_dm, 'int3c2e',
                                 aosym='s2ij', hermi=1, shls_slice=shls_slice,
                                 vhfopt=opt_)
                opt_.dm_cond = dm_cond_n
                vj_inter_e = jk.get_jk(fakemol, rho_n, ['ijkl,lk->ij']*n_dm, 'int3c2e',
                                       aosym='s2ij', hermi=1, shls_slice=shls_slice,
                                       vhfopt=opt_)
                vj[t] = numpy.asarray(vj_e) + numpy.asarray(vj_inter_e)
            else:
                opt_.dm_cond = dm_cond_e
                vj[t] = jk.get_jk(fakemol, rho_e, ['ijkl,lk->ij']*n_dm, 'int3c2e',
                                  aosym='s2ij', hermi=1, shls_slice=shls_slice,
                                  vhfopt=opt_)
        vj[t] = numpy.asarray(vj[t]).reshape(dm_shape[t]) * dfobj._charges[t]
        if t == 'e':
            vj_inter_e = numpy.asarray(vj_inter_e).reshape(dm_shape[t]) * dfobj._charges[t]
            vj[t] = lib.tag_array(vj[t], vint=vj_inter_e)
        else:
            vj[t] = lib.tag_array(vj[t], vint=vj[t])

    t1 = logger.timer_debug1(dfobj, 'df-vj pass 2', *t1)
    logger.timer(dfobj, 'df-vj', *t0)
    return vj

def density_fit(mf, auxbasis=None, with_df=None, ee_only_dfj=False,
                df_ne=False, df_ne_scheme='global', nuc_auxbasis=None,
                nuc_auxbasis_beta=2.0):
    '''Apply density fitting to NEO SCF objects.

    If ``df_ne`` is false, only the electronic e-e Coulomb build is density
    fitted and ``with_df`` is the normal :class:`pyscf.df.DF` object.  If
    ``df_ne`` is true, ``with_df`` is :class:`pyscf.neo.df.DF` and the DF
    tensor also covers electron-nuclear Coulomb interactions.

    ``df_ne_scheme='electron'`` uses the electronic auxiliary basis for the
    e-n fit.  It is kept for comparison and backward compatibility, but it can
    have large e-n fitting errors because the electronic auxiliary basis is not
    designed for compact nuclear densities.

    ``df_ne_scheme='global'`` is the default.  It builds one mixed auxiliary
    metric for the electronic and nuclear auxiliary functions and uses the same
    transformed tensor for e-e and e-n Coulomb builds.  The default nuclear
    auxiliary basis is generated by PySCF's ``aug_etb`` recipe with the
    exponent-sum range, which targets AO-product densities rather than AO
    functions.

    ``nuc_auxbasis`` controls only the nuclear auxiliary functions in the
    global scheme.  Named nuclear bases such as ``'pb4d'`` can be used, but
    they are generally not recommended as fitting bases because they were
    designed for nuclear orbitals instead of nuclear density products.
    Explicit even-tempered strings such as ``'8s8p8d'`` are also accepted; for
    these manual nuclear auxiliary bases the starting exponent is doubled
    relative to the NEO AO basis generator to match the equal-exponent product
    scale.  ``nuc_auxbasis_beta`` controls the spacing of the default
    ``aug_etb`` nuclear auxiliary basis.
    '''
    assert isinstance(mf, neo.HF)
    assert 'e' in mf.components
    assert isinstance(mf.components['e'], scf.hf.SCF)
    if 'p' in mf.components:
        raise NotImplementedError

    if df_ne:
        logger.warn(mf, 'NEO density fitting for electron-nuclear Coulomb '
                    'interactions is an experimental feature. Features and '
                    'APIs may be changed in the future.')
        if df_ne_scheme == 'electron':
            logger.warn(mf, 'df_ne_scheme="electron" uses the electronic '
                        'auxiliary basis for electron-nuclear fitting and can '
                        'have large electron-nuclear fitting errors.')

    if with_df is None and df_ne:
        mol = mf.mol
        mol_e = mol.components['e']
        mf_e = mf.components['e']
        if auxbasis is None and isinstance(mol_e.basis, str):
            if isinstance(mf_e, scf.hf.KohnShamDFT):
                xc = mf_e.xc
            else:
                xc = 'HF'
            if xc == 'LDA,VWN':
                # This is likely the default xc setting of a KS instance.
                # Postpone the auxbasis assignment to with_df.build().
                auxbasis = None
            else:
                auxbasis = addons.predefined_auxbasis(mol_e, mol_e.basis, xc)
        # e-e and e-n with_df
        with_df = DF(mol, auxbasis, df_ne_scheme=df_ne_scheme,
                     nuc_auxbasis=nuc_auxbasis,
                     nuc_auxbasis_beta=nuc_auxbasis_beta)

    if with_df is not None and df_ne:
        if not isinstance(with_df, DF):
            raise TypeError('with_df must be neo.df.DF when df_ne=True')
        if with_df.mol is not mf.mol:
            if with_df._cderi is not None:
                raise ValueError('A built with_df object cannot be reused for a different NEO mol')
            with_df.reset(mf.mol)
        with_df.df_ne_scheme = df_ne_scheme
        with_df.nuc_auxbasis = nuc_auxbasis
        with_df.nuc_auxbasis_beta = nuc_auxbasis_beta
        with_df.max_memory = mf.max_memory
        with_df.stdout = mf.stdout
        with_df.verbose = mf.verbose
        with_df._charges.clear()
        with_df._unrestricted.clear()
        for t, mf_ in mf.components.items():
            with_df._charges[t] = mf_.charge
            if isinstance(mf_, scf.rohf.ROHF):
                raise NotImplementedError
            with_df._unrestricted[t] = isinstance(mf_, scf.uhf.UHF)

    if with_df is not None and not df_ne:
        assert isinstance(with_df, df.DF) and not isinstance(with_df, DF)
        if isinstance(mf.components['e'], df_jk._DFHF):
            # if it is already a DF object, do not overwrite, but update
            mf = mf.copy()
            mf.components['e'].with_df = with_df
            mf.components['e'].only_dfj = ee_only_dfj
            return mf

    if isinstance(mf, _DFNEO):
        # if it is already a DF object, do not overwrite, but update
        mf = mf.copy()
        mf.with_df = with_df
        mf.ee_only_dfj = ee_only_dfj
        mf.df_ne = df_ne

    # NOTE: RSH is handled by with_df in elec component
    _charge = mf.components['e'].charge
    _mass = mf.components['e'].mass
    _is_nucleus = mf.components['e'].is_nucleus
    _nuc_occ_state = mf.components['e'].nuc_occ_state
    base = mf.components['e'].undo_component()
    # with_df is None or with_df is DF class in this file, need to rebuild elec DF
    if isinstance(base, df_jk._DFHF):
        base = base.undo_df()
    if with_df is not None:
        auxbasis = with_df.auxbasis
    mf.components['e'] = neo.hf.general_scf(df_jk.density_fit(base,
                                                              auxbasis=auxbasis,
                                                              with_df=None,
                                                              only_dfj=ee_only_dfj),
                                            charge=_charge, mass=_mass,
                                            is_nucleus=_is_nucleus,
                                            nuc_occ_state=_nuc_occ_state)
    if isinstance(mf, neo.KS):
        mf.interactions = neo.hf.generate_interactions(mf.components,
                                                       neo.ks.InteractionCorrelation,
                                                       mf.max_memory,
                                                       mf.direct_scf_tol,
                                                       epc=mf.epc)
    else:
        mf.interactions = neo.hf.generate_interactions(mf.components,
                                                       neo.hf.InteractionCoulomb,
                                                       mf.max_memory,
                                                       mf.direct_scf_tol)

    if isinstance(mf, _DFNEO):
        return mf

    dfmf = _DFNEO(mf, with_df, ee_only_dfj, df_ne)
    if df_ne:
        name = _DFNEO.__name_mixin__ + '-EE&NE-' + mf.__class__.__name__
    else:
        name = _DFNEO.__name_mixin__ + '-EE-' + mf.__class__.__name__
    return lib.set_class(dfmf, (_DFNEO, mf.__class__), name)

class _DFNEO:
    __name_mixin__ = 'DF'

    _keys = {'with_df', 'ee_only_dfj', 'df_ne'}

    def __init__(self, mf, df=None, ee_only_dfj=None, df_ne=None):
        self.__dict__.update(mf.__dict__)
        self._eri = None
        self.with_df = df
        self.ee_only_dfj = ee_only_dfj
        self.df_ne = df_ne
        # Unless DF is used only for J matrix, disable direct_scf for K build.
        # It is more efficient to construct K matrix with MO coefficients than
        # the incremental method in direct_scf.
        self.direct_scf = self.components['e'].direct_scf = ee_only_dfj

    def undo_df(self):
        '''Remove the DFNEO Mixin'''
        obj = lib.view(self, lib.drop_class(self.__class__, _DFNEO))
        obj.components = {}
        for t, comp in self.components.items():
            if t == 'e':
                # also undo_df for the elec component
                base = comp.undo_component().undo_df()
            else:
                base = comp.undo_component()
            obj.components[t] = neo.hf.general_scf(base.copy(),
                                                   charge=comp.charge,
                                                   mass=comp.mass,
                                                   is_nucleus=comp.is_nucleus,
                                                   nuc_occ_state=comp.nuc_occ_state)
        if isinstance(obj, neo.ks.KS):
            obj.interactions = neo.hf.generate_interactions(obj.components,
                                                            neo.ks.InteractionCorrelation,
                                                            obj.max_memory,
                                                            obj.direct_scf_tol,
                                                            epc=obj.epc)
            if isinstance(obj.components['e'], scf.hf.KohnShamDFT):
                obj._numint = obj.components['e']._numint
            else:
                obj._numint = None
        else:
            obj.interactions = neo.hf.generate_interactions(obj.components,
                                                            neo.hf.InteractionCoulomb,
                                                            obj.max_memory,
                                                            obj.direct_scf_tol)
        if hasattr(self, 'f') and self.f is not None:
            obj.f = numpy.array(self.f, copy=True)
        del obj.with_df, obj.ee_only_dfj, obj.df_ne
        return obj

    def reset(self, mol=None):
        if self.with_df is not None:
            self.with_df.reset(mol)
        return super().reset(mol)

    def _get_nn_vint_full_delta(self, dm, dm_last=0, vhf_last=0):
        '''Build n-n inter-type Coulomb potential as full and delta pieces.'''
        incremental_j = (
            isinstance(dm_last, dict) and isinstance(vhf_last, dict) and
            all(t in vhf_last and hasattr(vhf_last[t], 'vint_inc')
                for t in self.components))
        nn_vint_full = {}
        nn_vint_delta = {}
        for t in self.components:
            nn_vint_full[t] = 0
            nn_vint_delta[t] = 0
        if incremental_j:
            ddm = {}
            for t, dm_ in dm.items():
                dm_ = numpy.asarray(dm_)
                dm_last_ = numpy.asarray(dm_last[t])
                assert dm_last_.ndim == 0 or dm_last_.ndim == dm_.ndim
                ddm[t] = dm_ - dm_last_
        for t_pair, interaction in self.interactions.items():
            if 'e' in t_pair:
                continue
            if interaction._is_direct_vint():
                v = interaction.get_vint(ddm if incremental_j else dm,
                                         coulomb_only=True)
                target = nn_vint_delta
            else:
                v = interaction.get_vint(dm, coulomb_only=True)
                target = nn_vint_full
            for t in (interaction.mf1_type, interaction.mf2_type):
                target[t] += v[t]
        return nn_vint_full, nn_vint_delta

    def _attach_global_elec_df(self):
        # The global DF-NE tensor replaces the electronic component DF tensor
        # when component-level electronic DF code is called, e.g. gradients.
        if self.with_df is not None and self.with_df.df_ne_scheme == 'global':
            if self.with_df._cderi is None:
                self.with_df.build()
            mf_e = self.components['e']
            mf_e.with_df.auxmol = self.with_df.auxmol
            mf_e.with_df._cderi = self.with_df._cderi['e']

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if not self.with_df or not self.df_ne:
            return super().get_veff(mol, dm, dm_last, vhf_last, hermi)
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()

        mol_e = mol.components['e']
        mf_e = self.components['e']
        self._attach_global_elec_df()
        if isinstance(mf_e, scf.rohf.ROHF) or isinstance(mf_e, scf.ghf.GHF):
            raise NotImplementedError
        e_unrestricted = False
        if isinstance(mf_e, scf.uhf.UHF):
            e_unrestricted = True
        if e_unrestricted:
            # numpy.asarray will remove tag_array, and dm.mo_coeff is crucial for exchange
            # so use numpy.asarray only when necessary
            if not isinstance(dm['e'], numpy.ndarray):
                dm['e'] = numpy.asarray(dm['e'])
            if dm['e'].ndim == 2:  # RHF DM
                logger.warn(mf_e, 'Incompatible dm dimension. Treat dm as RHF density matrix.')
                dm['e'] = numpy.repeat(dm['e'][None]*.5, 2, axis=0)

        if not isinstance(mf_e, scf.hf.KohnShamDFT):
            with_dfk = not self.ee_only_dfj
            # Initialize vint_inc with a full-density J build, then update it
            # with density differences in later cycles.
            incremental_j = (self.direct_scf and
                             isinstance(dm_last, dict) and isinstance(vhf_last, dict) and
                             all(t in vhf_last and hasattr(vhf_last[t], 'vint_inc') for t in dm))
            nn_vint_full, nn_vint_delta = self._get_nn_vint_full_delta(
                dm, dm_last, vhf_last)
            include_last_vint_delta = (
                incremental_j or
                any(isinstance(nn_vint_delta[t], numpy.ndarray) for t in dm))
            vint_full, vint_delta = neo.hf._init_vint_full_delta(
                dm, vhf_last, include_last_vint_delta)
            if incremental_j:
                _dm = {}
                for t, dm_ in dm.items():
                    dm_ = numpy.asarray(dm_)
                    dm_last_ = numpy.asarray(dm_last[t])
                    assert dm_last_.ndim == 0 or dm_last_.ndim == dm_.ndim
                    _dm[t] = dm_ - dm_last_
            else:
                _dm = dm
            if with_dfk:
                vj, vk = self.with_df.get_jk(_dm, hermi)
            else:
                vj = self.with_df.get_j(_dm, hermi)
                vk = mf_e.get_k(mol_e, _dm['e'], hermi)
            # vj.vint is the e-n part of DF-J.  If _dm is the density used for
            # incremental updates, this contribution is cached in vint_inc;
            # otherwise it is treated as a full-density contribution.
            neo.hf._accumulate_vint(vint_full, vint_delta, vj, dm,
                                    self.direct_scf, attr='vint')
            # n-n contributions were already separated by their own integral
            # builders, independent of the electronic DF-J choice.
            for t in dm:
                vint_full[t] += nn_vint_full[t]
                vint_delta[t] += nn_vint_delta[t]
            vint = neo.hf._tag_vint_full_delta(vint_full, vint_delta, dm)
            vhf = {'e': vj['e']}
            if incremental_j:
                vhf['e'] += numpy.asarray(vhf_last['e'])
            if e_unrestricted:
                vhf['e'] = vhf['e'] - vk
            else:
                vhf['e'] = vhf['e'] - vk * .5
            for t in dm:
                if t == 'e':
                    vhf[t] = lib.tag_array(vhf[t], vint=vint[t],
                                           vint_inc=vint_delta[t])
                else:
                    vhf[t] = lib.tag_array(vint[t], vint=vint[t],
                                           vint_inc=vint_delta[t])
                self.components[t]._vint = vint[t]
        else:
            mf_e.initialize_grids(mol_e, dm['e'])

            t0 = (logger.process_clock(), logger.perf_counter())

            if e_unrestricted:
                ground_state = (dm['e'].ndim == 3 and dm['e'].shape[0] == 2)
            else:
                ground_state = (isinstance(dm['e'], numpy.ndarray) and dm['e'].ndim == 2)

            ni = mf_e._numint
            if hermi == 2:  # because rho = 0
                if e_unrestricted:
                    n = (0,0)
                else:
                    n = 0
                exc, vxc = 0, 0
            else:
                max_memory = mf_e.max_memory - lib.current_memory()[0]
                if e_unrestricted:
                    n, exc, vxc = ni.nr_uks(mol_e, mf_e.grids, mf_e.xc,
                                            dm['e'], max_memory=max_memory)
                else:
                    n, exc, vxc = ni.nr_rks(mol_e, mf_e.grids, mf_e.xc,
                                            dm['e'], max_memory=max_memory)
                logger.debug(mf_e, 'nelec by numeric integration = %s', n)
                if mf_e.do_nlc():
                    if ni.libxc.is_nlc(mf_e.xc):
                        xc = mf_e.xc
                    else:
                        assert ni.libxc.is_nlc(mf_e.nlc)
                        xc = mf_e.nlc
                    if e_unrestricted:
                        n, enlc, vnlc = ni.nr_nlc_vxc(mol_e, mf_e.nlcgrids, xc, dm['e'][0]+dm['e'][1],
                                                      max_memory=max_memory)
                    else:
                        n, enlc, vnlc = ni.nr_nlc_vxc(mol_e, mf_e.nlcgrids, xc, dm['e'],
                                                      max_memory=max_memory)
                    exc += enlc
                    vxc += vnlc
                    logger.debug(mf_e, 'nelec with nlc grids = %s', n)
                t0 = logger.timer(mf_e, 'vxc', *t0)

            # Initialize vint_inc with a full-density J build, then update it
            # with density differences in later cycles.  XC always uses the
            # current density.
            incremental_jk = (self.direct_scf and
                              isinstance(dm_last, dict) and isinstance(vhf_last, dict) and
                              all(t in vhf_last and hasattr(vhf_last[t], 'vj') and
                                  hasattr(vhf_last[t], 'vint_inc') for t in dm))
            nn_vint_full, nn_vint_delta = self._get_nn_vint_full_delta(
                dm, dm_last, vhf_last)
            if incremental_jk:
                _dm = {}
                for t, dm_ in dm.items():
                    dm_ = numpy.asarray(dm_)
                    dm_last_ = numpy.asarray(dm_last[t])
                    assert dm_last_.ndim == 0 or dm_last_.ndim == dm_.ndim
                    _dm[t] = dm_ - dm_last_
            else:
                _dm = dm
            if not ni.libxc.is_hybrid_xc(mf_e.xc):
                vk = None
                vj = self.with_df.get_j(_dm, hermi)
            else:
                omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf_e.xc, spin=mol_e.spin)
                with_dfk = not self.ee_only_dfj
                if omega == 0:
                    if with_dfk:
                        vj, vk = self.with_df.get_jk(_dm, hermi)
                    else:
                        vj = self.with_df.get_j(_dm, hermi)
                        vk = mf_e.get_k(mol_e, _dm['e'], hermi)
                    vk *= hyb
                elif alpha == 0: # LR=0, only SR exchange
                    vj = self.with_df.get_j(_dm, hermi)
                    vk = mf_e.get_k(mol_e, _dm['e'], hermi, omega=-omega)
                    vk *= hyb
                elif hyb == 0: # SR=0, only LR exchange
                    vj = self.with_df.get_j(_dm, hermi)
                    vk = mf_e.get_k(mol_e, _dm['e'], hermi, omega=omega)
                    vk *= alpha
                else: # SR and LR exchange with different ratios
                    if with_dfk:
                        vj, vk = self.with_df.get_jk(_dm, hermi)
                    else:
                        vj = self.with_df.get_j(_dm, hermi)
                        vk = mf_e.get_k(mol_e, _dm['e'], hermi)
                    vk *= hyb
                    vklr = mf_e.get_k(mol_e, _dm['e'], hermi, omega=omega)
                    vklr *= (alpha - hyb)
                    vk += vklr

                if incremental_jk:
                    vk += vhf_last['e'].vk

                if ground_state:
                    if e_unrestricted:
                        exc -=(numpy.einsum('ij,ji', dm['e'][0], vk[0]).real +
                               numpy.einsum('ij,ji', dm['e'][1], vk[1]).real) * .5
                    else:
                        exc -= numpy.einsum('ij,ji', dm['e'], vk).real * .5 * .5

            include_last_vint_delta = (
                incremental_jk or
                any(isinstance(nn_vint_delta[t], numpy.ndarray)
                    for t in dm))
            vint_full, vint_delta = neo.hf._init_vint_full_delta(
                nn_vint_full, vhf_last, include_last_vint_delta)
            # vj.vint is the e-n part of DF-J.  If _dm is the density used for
            # incremental updates, this contribution is cached in vint_inc;
            # otherwise it is treated as a full-density contribution.
            neo.hf._accumulate_vint(vint_full, vint_delta, vj, nn_vint_full,
                                    self.direct_scf, attr='vint')
            # n-n contributions were already separated by their own integral
            # builders, independent of the electronic DF-J choice.
            for t in nn_vint_full:
                vint_full[t] += nn_vint_full[t]
                vint_delta[t] += nn_vint_delta[t]
            vint = neo.hf._tag_vint_full_delta(vint_full, vint_delta,
                                               nn_vint_full)
            epc = neo.ks._get_epc_vmat(self, dm)
            if incremental_jk:
                for t in vj:
                    vj[t] += vhf_last[t].vj

            vhf = {t: vint[t] + epc[t] for t in nn_vint_full if t != 'e'}
            vhf['e'] = vj['e'] + vxc + epc['e']
            if e_unrestricted:
                if vk is not None:
                    vhf['e'] = vhf['e'] - vk
            else:
                if vk is not None:
                    vhf['e'] = vhf['e'] - vk * .5

            if ground_state:
                if e_unrestricted:
                    ecoul = numpy.einsum('ij,ji', dm['e'][0]+dm['e'][1], vj['e']).real * .5
                else:
                    ecoul = numpy.einsum('ij,ji', dm['e'], vj['e']).real * .5
            else:
                ecoul = None

            if hasattr(epc['e'], 'exc'):
                exc += epc['e'].exc
            vhf['e'] = lib.tag_array(vhf['e'], ecoul=ecoul, exc=exc,
                                     vj=vj['e'], vk=vk, vint=vint['e'],
                                     vint_inc=vint_delta['e'])
            for t in vhf:
                if t != 'e':
                    vhf[t] = lib.tag_array(vhf[t], vj=vj[t], vint=vint[t],
                                           vint_inc=vint_delta[t])
                self.components[t]._vint = numpy.asarray(vint[t] + epc[t])

        return vhf

    @property
    def auxbasis(self):
        if self.with_df is not None:
            return getattr(self.with_df, 'auxbasis', None)
        return self.components['e'].auxbasis

    def nuc_grad_method(self):
        import pyscf.neo.df_grad
        return pyscf.neo.df_grad.Gradients(self)

    Gradients = lib.alias(nuc_grad_method, alias_name='Gradients')

    def Hessian(self):
        raise NotImplementedError


if __name__ == '__main__':
    def run_df_ne_error_case(name, mf_factory, auxbasis, ee_only_dfj=False):
        cases = [
            ('Exact ERI', mf_factory()),
            ('DF ee only',
             mf_factory().density_fit(auxbasis=auxbasis, df_ne=False,
                                      ee_only_dfj=ee_only_dfj)),
            ('DF ee+en e-aux',
             mf_factory().density_fit(auxbasis=auxbasis, df_ne=True,
                                      df_ne_scheme='electron',
                                      ee_only_dfj=ee_only_dfj)),
            ('DF ee+en global aug_etb',
             mf_factory().density_fit(auxbasis=auxbasis, df_ne=True,
                                      df_ne_scheme='global',
                                      ee_only_dfj=ee_only_dfj)),
        ]

        for label, mf in cases:
            mf.conv_tol = 1e-10

        results = [(label, mf.kernel()) for label, mf in cases]
        e_ref = results[0][1]

        print(f'\n{name}')
        print(f'  {"method":25s} {"energy":>18s} {"dE":>12s}')
        for label, energy in results:
            if label == 'Exact ERI':
                print(f'  {label:25s} {energy:18.12f} {"--":>12s}')
            else:
                print(f'  {label:25s} {energy:18.12f} {energy - e_ref:12.4e}')

    run_df_ne_error_case(
        'HF/NEO-HF/def2SVP/weigend, ee_only_dfj=False',
        lambda: neo.HF(neo.M(atom='H 0 0 0; F 0 0 1',
                             basis='def2svp', quantum_nuc=[0], verbose=0)),
        'weigend')

    run_df_ne_error_case(
        'HF/NEO-HF/def2SVP/weigend, ee_only_dfj=True',
        lambda: neo.HF(neo.M(atom='H 0 0 0; F 0 0 1',
                             basis='def2svp', quantum_nuc=[0], verbose=0)),
        'weigend', ee_only_dfj=True)

    h3p_atom = '''H 0.000 0.000 0.000;
                  H 0.000 0.000 0.900;
                  H 0.779 0.000 0.450'''

    run_df_ne_error_case(
        'H3+/CNEO-CAM-B3LYP/cc-pvdz/cc-pvdz-jkfit, ee_only_dfj=False',
        lambda: neo.KS(neo.M(atom=h3p_atom, basis='ccpvdz', charge=1,
                             quantum_nuc=['H'], verbose=0),
                       xc='camb3lyp'),
        'cc-pvdz-jkfit')

    run_df_ne_error_case(
        'H3+/CNEO-CAM-B3LYP/occ-pvdz/cc-pvdz-jkfit, ee_only_dfj=True',
        lambda: neo.KS(neo.M(atom=h3p_atom, basis='ccpvdz', charge=1,
                             quantum_nuc=['H'], verbose=0),
                       xc='camb3lyp'),
        'cc-pvdz-jkfit', ee_only_dfj=True)
