'''Cneo TDDFT with frozen orbital assumption'''

from pyscf.neo import tddft_slow
from pyscf import neo, lib, scf
from pyscf.tdscf import rhf, TDDFT
from pyscf.lib import logger
import numpy

def get_ab(mf):
    if getattr(mf, 'df_ne', False):
        raise NotImplementedError('CNEO-TDDFT with DF-NE')
    if isinstance(mf, neo.KS):
        if mf.epc is not None:
            _a, _b, c = tddft_slow.get_abc(mf)
            a = _a['e']
            b = _b['e']
        else:
            a, b = tddft_slow.get_ab_elec(mf.components['e'])
    else:
        a, b = tddft_slow.get_ab_elec(mf.components['e'])
    if isinstance(a, tuple):
        a = list(a)
        b = list(b)
    return a, b

def _normalize(x1, mo_occ):
    if (mo_occ.ndim == 2):
        nmo = mo_occ[0].size
        nocca = (mo_occ[0]>0).sum()
        noccb = (mo_occ[1]>0).sum()
        nvira = nmo - nocca
        nvirb = nmo - noccb
        xy = []
        for i, z in enumerate(x1):
            x, y = z.reshape(2,-1)
            norm = lib.norm(x)**2 - lib.norm(y)**2
            if norm > 0:
                norm = 1/numpy.sqrt(norm)
                xy.append(((x[:nocca*nvira].reshape(nocca,nvira) * norm,  # X_alpha
                            x[nocca*nvira:].reshape(noccb,nvirb) * norm), # X_beta
                           (y[:nocca*nvira].reshape(nocca,nvira) * norm,  # Y_alpha
                            y[nocca*nvira:].reshape(noccb,nvirb) * norm)))# Y_beta

    else:
        nocc = (mo_occ>0).sum()
        nmo = mo_occ.size
        nvir = nmo - nocc
# 1/sqrt(2) because self.x is for alpha excitation amplitude and 2(X^+*X) = 1
        def norm_xy(z):
            x, y = z.reshape(2,nocc,nvir)
            norm = lib.norm(x)**2 - lib.norm(y)**2
            norm = numpy.sqrt(.5/norm)  # normalize to 0.5 for alpha spin
            return x*norm, y*norm
        xy = [norm_xy(z) for z in x1]

    return xy

class CTDDirect(rhf.TDBase):
    ''' Frozen nuclear orbital CNEO-TDDFT: full matrix diagonalization
    Examples:

    >>> from pyscf import neo
    >>> from pyscf.neo import ctddft
    >>> mol = neo.M(atom='H 0 0 0; C 0 0 1.067; N 0 0 2.213', basis='631g',
                    quantum_nuc = ['H'], nuc_basis = 'pb4d')
    >>> mf = neo.CDFT(mol, xc='hf')
    >>> mf.scf()
    >>> td_mf = ctddft.CTDDirect(mf)
    >>> td_mf.kernel(nstates=5)
    Excitation energies (eV)
    [ 6.82308887  7.68777851  7.68777851 10.05706016 10.05706016]
    '''

    def get_ab(self):
        if not self.singlet:
            raise NotImplementedError('Explicit CNEO triplet response matrix')
        return get_ab(self._scf)

    def get_full(self):
        a, b = self.get_ab()
        if isinstance(a, list):
            a = tddft_slow.aabb2a(a)
            b = tddft_slow.aabb2a(b)
        return tddft_slow.ab2full(a, b)

    def kernel(self, nstates=None):
        cpu0 = (logger.process_clock(), logger.perf_counter())
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates

        log = logger.Logger(self.stdout, self.verbose)
        td_mat = self.get_full()
        w, x1 = tddft_slow.eig_mat(td_mat, nroots=nstates)
        self.converged = [True for i in range(nstates)]
        x1 = x1.T

        self.e = numpy.array(w)
        mo_occ = self._scf.mo_occ['e']
        self.xy = _normalize(x1, mo_occ)

        log.timer('CNEO-TDDFT full matrix diagonalization', *cpu0)
        self._finalize()

        return self.e, self.xy

    def Gradients(self):
        if getattr(self._scf, 'df_ne', False):
            raise NotImplementedError('CNEO-TD gradients with DF-NE')
        if getattr(self._scf.components['e'], 'with_df', None):
            from pyscf.neo import df_tdgrad
            return df_tdgrad.Gradients(self)
        else:
            from pyscf.neo import tdgrad
            return tdgrad.Gradients(self)


class CTDDFT(CTDDirect):
    ''' Frozen nuclear orbital CNEO-TDDFT: Davidson
    Examples:

    >>> from pyscf import neo
    >>> from pyscf.neo import ctddft
    >>> mol = neo.M(atom='H 0 0 0; C 0 0 1.067; N 0 0 2.213', basis='631g',
                    quantum_nuc = ['H'], nuc_basis = 'pb4d')
    >>> mf = neo.CDFT(mol, xc='hf')
    >>> mf.scf()
    >>> td_mf = ctddft.CTDDFT(mf)
    >>> td_mf.kernel(nstates=5)
    Excitation energies (eV)
    [ 6.82308887  7.68777851  7.68777851 10.05706016 10.05706016]
    '''

    def kernel(self, x0=None, nstates=None):
        mf = self._scf
        if isinstance(mf, neo.KS) and mf.epc is not None:
            raise NotImplementedError('epc is not implemented for CNEO-TDDFT davidson')
        if getattr(mf, 'df_ne', False):
            raise NotImplementedError('CNEO-TDDFT with DF-NE')
        mf_elec = mf.components['e']
        if mf_elec.mo_coeff is None or mf_elec.mo_energy is None:
            mf.run()
        td = TDDFT(mf.components['e'], frozen=self.frozen)
        # Forward TD controls while keeping the electronic reference on td.
        td.verbose = self.verbose
        td.stdout = self.stdout
        td.max_memory = self.max_memory
        td.chkfile = self.chkfile
        td.conv_tol = self.conv_tol
        td.nstates = self.nstates
        td.singlet = None if isinstance(td._scf, scf.uhf.UHF) else self.singlet
        td.lindep = self.lindep
        td.level_shift = self.level_shift
        td.max_cycle = self.max_cycle
        td.positive_eig_threshold = self.positive_eig_threshold
        td.deg_eia_thresh = self.deg_eia_thresh
        td.exclude_nlc = self.exclude_nlc
        td.frozen = self.frozen
        td.wfnsym = self.wfnsym
        self.e, self.xy = td.kernel(x0=x0, nstates=nstates)
        self.converged = td.converged
        self.nstates = td.nstates

        return self.e, self.xy

neo.cdft.CDFT.TDDirect = lib.class_as_method(CTDDirect)
neo.cdft.CDFT.TDDFT = lib.class_as_method(CTDDFT)
