'''Cneo TDDFT with frozen orbital assumption'''

from pyscf.neo import tddft_slow
from pyscf import neo, lib, scf
from pyscf.tdscf._lr_eig import eigh as lr_eigh, eig as lr_eig, real_eig
from pyscf.tdscf import rhf, TDDFT
from pyscf.lib import logger
from pyscf import __config__
import numpy

def get_ab(mf):
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
    Excited State energies (eV)
    [ 6.82308887  7.68777851  7.68777851 10.05706016 10.05706016]
    '''

    def get_ab(self):
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

    def nuc_grad_method(self):
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
    Excited State energies (eV)
    [ 6.82308887  7.68777851  7.68777851 10.05706016 10.05706016]
    '''
    max_space = getattr(__config__, 'tdscf_rhf_TDA_max_space', 40)
    conv_tol = getattr(__config__, 'tdscf_rhf_TDA_conv_tol', 1e-6)

    def __init__(self, mf):
        super().__init__(mf)
        self._keys = self._keys.union(['max_space'])

    def gen_vind(self, mf=None):
        if mf is None:
            mf = self._scf
        if isinstance(mf, neo.KS) and mf.epc is not None:
            raise NotImplementedError('epc is not implemented for CNEO-TDDFT davidson')
        return TDDFT(mf.components['e']).gen_vind()

    def get_init_guess(self, mf, nstates=None, wfnsym=None):
        mf_elec = mf.components['e']
        return TDDFT(mf_elec).get_init_guess(mf_elec, nstates, wfnsym)

    def kernel(self, x0=None, nstates=None):
        '''
        Modified from tdscf.rhf/uhf
        '''
        log = logger.new_logger(self)
        cpu0 = (logger.process_clock(), logger.perf_counter())
        self.check_sanity()
        self.dump_flags()
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates

        log = logger.Logger(self.stdout, self.verbose)

        vind, hdiag = self.gen_vind(self._scf)
        precond = self.get_precond(hdiag)
        if x0 is None:
            x0 = self.get_init_guess(self._scf, self.nstates)

        # We only need positive eigenvalues
        def pickeig(w, v, nroots, envs):
            realidx = numpy.where((abs(w.imag) < rhf.REAL_EIG_THRESHOLD) &
                                  (w.real > self.positive_eig_threshold))[0]
            # If the complex eigenvalue has small imaginary part, both the
            # real part and the imaginary part of the eigenvector can
            # approximately be used as the "real" eigen solutions.
            return lib.linalg_helper._eigs_cmplx2real(w, v, realidx,
                                                      real_eigenvectors=True)

        self.converged, w, x1 = \
                lib.davidson_nosym1(vind, x0, precond,
                                    tol=self.conv_tol,
                                    nroots=nstates, lindep=self.lindep,
                                    max_cycle=self.max_cycle,
                                    max_space=self.max_space, pick=pickeig,
                                    verbose=log)

        self.e = numpy.array(w)
        self.xy = _normalize(x1, self._scf.mo_occ['e'])

        log.timer('CNEO-TDDFT Davidson', *cpu0)
        self._finalize()

        return self.e, self.xy

neo.cdft.CDFT.TDDirect = lib.class_as_method(CTDDirect)
neo.cdft.CDFT.TDDFT = lib.class_as_method(CTDDFT)
