from pyscf.tdscf import rhf, uhf
from pyscf import ao2mo
from pyscf import lib
from pyscf.neo.tddft import get_epc_iajb_rhf, get_epc_iajb_uhf
from pyscf.tdscf.common_slow import eig
from pyscf.data import nist
from pyscf.lib import logger
from pyscf import neo
import numpy

def aabb2a(a):
    '''
    a is a list of (a_aaaa, a_aabb, a_bbbb)
    '''
    a_bbaa = a[1].transpose(2,3,0,1)
    a.append(a_bbaa)
    for i, ai in enumerate(a):
        shape0 = ai.shape[0]*ai.shape[1]
        shape1 = ai.shape[2]*ai.shape[3]
        a[i] = a[i].reshape(shape0,shape1)

    return numpy.block([[a[0],a[1]],[a[3],a[2]]])

def ab2full(a,b):
    return numpy.block([[a,b],[-b,-a]])

def c2full(c):
    return numpy.block([[c, c],[-c, -c]])

def get_ab_elec(mf_elec, unrestricted):
    if unrestricted:
        return uhf.get_ab(mf_elec)
    else:
        nocc = mf_elec.mo_coeff[:,mf_elec.mo_occ>0].shape[1]
        nvir = mf_elec.mo_coeff.shape[1]-nocc
        a_e, b_e = rhf.get_ab(mf_elec)
        return a_e.reshape((nocc*nvir,nocc*nvir)),b_e.reshape((nocc*nvir,nocc*nvir))

def get_ab_nuc(mf_nuc):
    mo_occ_p = mf_nuc.mo_occ
    mo_energy_p = mf_nuc.mo_energy
    occidx_p = numpy.where(mo_occ_p==1)[0]
    viridx_p = numpy.where(mo_occ_p==0)[0]
    nocc_p = len(occidx_p)
    nvir_p = len(viridx_p)
    e_ia = lib.direct_sum('a-i->ia', mo_energy_p[viridx_p], mo_energy_p[occidx_p])
    a_p = numpy.diag(e_ia.ravel()).reshape((nocc_p*nvir_p,nocc_p*nvir_p))
    b_p = numpy.zeros_like(a_p)

    return a_p, b_p

def get_c(mf1, mf2, eri, charge):
    nocc_1 = mf1.mo_coeff[:,mf1.mo_occ>0].shape[1]
    nocc_2 = mf2.mo_coeff[:,mf2.mo_occ>0].shape[1]

    co_1= mf1.mo_coeff[:,:nocc_1]
    cv_1= mf1.mo_coeff[:,nocc_1:]

    co_2=mf2.mo_coeff[:,:nocc_2]
    cv_2=mf2.mo_coeff[:,nocc_2:]

    c_12 = charge*ao2mo.incore.general(eri, (co_1, cv_1, co_2, cv_2),compact=False)
    return c_12

def get_c_nn(mf, idx1=0, idx2=1):
    assert(idx1 < idx2)
    mf1 = mf.mf_nuc[idx1]
    mf2 = mf.mf_nuc[idx2]
    eri = mf._eri_nn[idx1][idx2]
    charge = mf.mol.atom_charge(mf1.mol.atom_index) * mf.mol.atom_charge(mf2.mol.atom_index)
    
    return get_c(mf1, mf2, eri, charge)

def get_c_ne_restricted(mf, idx=0):
    mf_elec = mf.mf_elec
    mf_nuc = mf.mf_nuc[idx]
    eri = mf._eri_ne[idx]
    charge = -1 * mf.mol.atom_charge(mf_nuc.mol.atom_index)
    return get_c(mf_nuc, mf_elec, eri, charge)

def get_c_ne_unrestricted(mf, i=0):
    eri = mf._eri_ne[i]
    charge = -1*mf.mol.atom_charge(mf.mf_nuc[i].mol.atom_index)
    mo_coeff = mf.mf_elec.mo_coeff
    mo_occ = mf.mf_elec.mo_occ
    occidx_a = numpy.where(mo_occ[0]==1)[0]
    viridx_a = numpy.where(mo_occ[0]==0)[0]
    occidx_b = numpy.where(mo_occ[1]==1)[0]
    viridx_b = numpy.where(mo_occ[1]==0)[0]
    orbo_a = mo_coeff[0][:,occidx_a]
    orbv_a = mo_coeff[0][:,viridx_a]
    orbo_b = mo_coeff[1][:,occidx_b]
    orbv_b = mo_coeff[1][:,viridx_b]
    p_nocc = mf.mf_nuc[i].mo_coeff[:,mf.mf_nuc[0].mo_occ>0].shape[1]

    co_n=mf.mf_nuc[i].mo_coeff[:,:p_nocc]
    cv_n=mf.mf_nuc[i].mo_coeff[:,p_nocc:]

    c_ne_a = charge*ao2mo.incore.general(eri, (co_n, cv_n, orbo_a, orbv_a),compact=False)
    c_ne_b = charge*ao2mo.incore.general(eri, (co_n, cv_n, orbo_b, orbv_b),compact=False)
    return [c_ne_a, c_ne_b]

def get_c_ne(mf, i):
    if mf.unrestricted:
        return get_c_ne_unrestricted(mf, i)
    else:
        return get_c_ne_restricted(mf,i)
    
def get_abc_no_epc(mf):
    a_e, b_e = get_ab_elec(mf.mf_elec, unrestricted=mf.unrestricted)
    if mf.unrestricted:
        a_e = list(a_e)
        b_e = list(b_e)
    nuc_num = mf.mol.nuc_num
    a_ps = []
    b_ps = []
    c_nes = []
    c_pps = []
    for i in range(nuc_num):
        a_p, b_p = get_ab_nuc(mf.mf_nuc[i])
        a_ps.append(a_p)
        b_ps.append(b_p)
        c_nes.append(get_c_ne(mf, i))
        c_pps.append([None]*nuc_num)
    for i in range(nuc_num):
        for j in range(i+1, nuc_num):
            c_pps[i][j] = get_c_nn(mf, i, j)

    return a_e, b_e, a_ps, b_ps, c_nes, c_pps

def get_epc_iajb(mf):
    if mf.unrestricted:
        iajb_aa, iajb_bb, iajb_ab, iajb_pe_a, iajb_pe_b, iajb_p = get_epc_iajb_uhf(mf, reshape=True)
        iajb_e = [iajb_aa, iajb_ab, iajb_bb]
        iajb_ne = []
        for i in range(mf.mol.nuc_num):
            iajb_ne.append([iajb_pe_a[i], iajb_pe_b[i]])
    else:
        iajb_e, iajb_ne, iajb_p = get_epc_iajb_rhf(mf, reshape=True)
    
    return iajb_e, iajb_ne, iajb_p
    
def add_epc(a, iajb):
    '''
    to accommodate both restrcted and unrestricted
    '''
    if isinstance(a, list):
        for i in range(len(a)):
            a[i] += iajb[i]

    else:
        a += iajb

    return a
    
def get_abc(mf):
    a_e, b_e, a_ns, b_ns, c_nes, c_nns = get_abc_no_epc(mf)
    if isinstance(mf, neo.KS):
        if mf.epc is not None:
            iajb_e, iajb_ne, iajb_n = get_epc_iajb(mf)
            a_e = add_epc(a_e, iajb_e)
            b_e = add_epc(b_e, iajb_e)
            for i in range(mf.mol.nuc_num):
                a_ns[i] += iajb_n[i]
                b_ns[i] += iajb_n[i]
                c_nes[i] = add_epc(c_nes[i], iajb_ne[i])

    return a_e, b_e, a_ns, b_ns, c_nes, c_nns

def full_elec(a_e, b_e):
    if isinstance(a_e, list):
        a_e = aabb2a(a_e)
        b_e = aabb2a(b_e)
    return ab2full(a_e, b_e)

def one_nuc(a_p, b_p):
    return ab2full(a_p, b_p)

def nuc2_to_nuc1(a_ps, b_ps, c_pps, idx, nuc2):
    nuc_num = len(a_ps)
    assert(idx < nuc_num)
    if nuc2 is None or idx == nuc_num-1:
        return one_nuc(a_ps[-1], b_ps[-1])
    
    nuc1_nuc2 = []
    nuc2_nuc1 = []
    for idx2 in range(idx+1,nuc_num):
        c_12 = c_pps[idx][idx2]
        nuc1_nuc2.append(c2full(c_12))
        nuc2_nuc1.append([c2full(c_12.transpose())])

    nuc1_nuc = numpy.block([*nuc1_nuc2])
    nuc_nuc1 = numpy.block([*nuc2_nuc1])
    nuc1_nuc1 = one_nuc(a_ps[idx], b_ps[idx])
    return numpy.block([[nuc1_nuc1, nuc1_nuc],[nuc_nuc1, nuc2]])

def full_nuc(a_ps, b_ps, c_pps):
    nuc_num = len(a_ps)

    nuc2 = None
    for i in reversed(range(nuc_num)):
        nuc1 = nuc2_to_nuc1(a_ps, b_ps, c_pps, i, nuc2)
        nuc2 = nuc1

    return nuc1

def full_ep(c_nes):
    nuc_num = len(c_nes)
    elec_nuc = []
    nuc_elec = []
    for i in range(nuc_num):
        c_ne = c_nes[i]
        if isinstance(c_ne, list):
            c_ne = numpy.hstack((c_ne[0], c_ne[1]))
        else:
            c_ne *= numpy.sqrt(2)
        nuc_elec.append([c2full(c_ne)])
        elec_nuc.append(c2full(c_ne.transpose()))
    
    return numpy.block([*elec_nuc]), numpy.block([*nuc_elec])

def get_td_mat(mf):
    a_e, b_e, a_ns, b_ns, c_nes, c_nns = get_abc(mf)
    elec = full_elec(a_e, b_e)
    nuc = full_nuc(a_ns, b_ns, c_nns)
    elec_nuc, nuc_elec = full_ep(c_nes)
    return numpy.block([[elec, elec_nuc],[nuc_elec, nuc]])

class TDBase(lib.StreamObject):
    '''Full td matrix diagonlization

    Examples::

    >>> from pyscf import neo
    >>> from pyscf.neo import tddft_slow
    >>> mol = neo.M(atom='H 0 0 0; C 0 0 1.067; N 0 0 2.213', basis='631g', 
                    quantum_nuc = ['H'], nuc_basis = 'pb4p', cart=True)
    >>> mf = neo.HF(mol)
    >>> mf.scf()
    >>> td_mf = tddft_slow.TDBase(mf)
    >>> td_mf.kernel(nstates=5)
    Excited State energies (eV)
    [0.69058969 0.69058969 0.78053615 1.33065414 1.97414121]
    '''

    def __init__(self, mf, nstates=3, driver='eig'):
        self.verbose = mf.verbose
        self.stdout = mf.mf_elec.stdout
        self.mol = mf.mol
        self._scf = mf
        self.unrestricted = mf.unrestricted
        self.nstates = nstates
        self.driver = driver
        self.td_mat = None
        self.e = None
        self.x1 = None

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
    
    def get_td_mat(self):
        if self.td_mat is None:
            self.td_mat = get_td_mat(self._scf)
        return self.td_mat
    
    def kernel(self, nstates=None):
        cpu0 = (logger.process_clock(), logger.perf_counter())
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates

        log = logger.Logger(self.stdout, self.verbose)
        
        m = self.get_td_mat()
        w, x1 = eig(m, nroots = nstates, driver = self.driver)
        
        self.e = w
        self.x1 = x1

        log.timer('TDDFT full diagonalization', *cpu0)
        logger.note(self, 'Excited State energies (eV)\n%s', self.e * nist.HARTREE2EV)
        return self.e, self.x1