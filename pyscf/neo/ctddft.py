'''Cneo TDDFT with frozen orbital assumption'''

from pyscf.neo import tddft_slow
from pyscf import neo, lib
from pyscf.tdscf.common_slow import eig
import numpy

def get_ab(mf):
    if isinstance(mf, neo.KS) and mf.epc is not None:
        a, b = tddft_slow.get_abc(mf)[0:2]
    else:    
        a, b = tddft_slow.get_ab_elec(mf.mf_elec, mf.unrestricted)
    if isinstance(a, tuple):
        a = list(a)
        b = list(b)
    return a, b

    
class CTDBase(object):
    
    def __init__(self, mf, nstates=3, driver='eig'):
        self._scf = mf
        self.mol = mf.mol

        self.stdout = mf.stdout
        self.verbose = mf.verbose
        self.max_memory = mf.max_memory
        self.chkfile = mf.chkfile
        self.nstates = nstates
        self.driver = driver
        self.verbose = mf.verbose
        self.unrestricted = mf.unrestricted
        self.full = None
        self.ab = None
        self.e = None
        self.xy = None

    @property
    def nroots(self):
        return self.nstates
    @nroots.setter
    def nroots(self, x):
        self.nstates = x

    @property
    def e_tot(self):
        '''Excited state energies'''
        return self._scf.e_tot + self.e
    
    def get_ab(self):
        if self.ab is None:
            a, b = get_ab(self._scf)
            self.ab = [a, b]

        return self.ab
    
    def get_full(self):
        if self.full is None:
            a, b = self.get_ab()
            self.full = tddft_slow.full_elec(a, b)

        return self.full

    def kernel(self, nstates=None):
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates
        td_mat = self.get_full()
        w, x1 = eig(td_mat, driver=self.driver, nroots=nstates)
        x1 = x1.T
        # print(x1[0])

        mf_elec = self._scf.mf_elec
        if self.unrestricted:
            nmo = mf_elec.mo_occ[0].size
            nocca = (mf_elec.mo_occ[0]>0).sum()
            noccb = (mf_elec.mo_occ[1]>0).sum()
            nvira = nmo - nocca
            nvirb = nmo - noccb
            e = []
            xy = []
            for i, z in enumerate(x1):
                x, y = z.reshape(2,-1)
                norm = lib.norm(x)**2 - lib.norm(y)**2
                if norm > 0:
                    norm = 1/numpy.sqrt(norm)
                    e.append(w[i])
                    xy.append(((x[:nocca*nvira].reshape(nocca,nvira) * norm,  # X_alpha
                                x[nocca*nvira:].reshape(noccb,nvirb) * norm), # X_beta
                            (y[:nocca*nvira].reshape(nocca,nvira) * norm,  # Y_alpha
                                y[nocca*nvira:].reshape(noccb,nvirb) * norm)))# Y_beta
            self.e = numpy.array(e)
            self.xy = xy

        else:
            nocc = (mf_elec.mo_occ>0).sum()
            nmo = mf_elec.mo_occ.size
            nvir = nmo - nocc
    # 1/sqrt(2) because self.x is for alpha excitation amplitude and 2(X^+*X) = 1
            self.e = w
            def norm_xy(z):
                x, y = z.reshape(2,nocc,nvir)
                norm = lib.norm(x)**2 - lib.norm(y)**2
                norm = numpy.sqrt(.5/norm)  # normalize to 0.5 for alpha spin
                return x*norm, y*norm
            self.xy = [norm_xy(z) for z in x1]

        return self.e, self.xy