from pyscf.tdscf import rhf, uhf
from pyscf import ao2mo
from pyscf import lib
from pyscf.neo.tddft import get_epc_iajb_rhf, get_epc_iajb_uhf
from pyscf.data import nist
from pyscf.lib import logger
from pyscf import neo, scf
import numpy

def _normalize(x1, mo_occ, log):
    ''' Normalize the NEO-TDDFT eigenvectors

    <Xe|Xe> - <Ye|Ye> + <Xp|Xp> - <Yp|Yp> = 1

    Args:
        x1: list of 1D array
            Each 1D array is organized as: [Xe, Ye, Xn, Yn]
        mo_occ: dict

    Returns:
        xy: list of dictionaries
            rhf: xy = [{"e":(Xa, Ya), "n": (Xn, Yn)}]
            uhf: xy = [{"e":((Xa, Xb), (Ya, Yb)), "n": (Xn, Yn)}]

    '''
    nocc = {}
    nvir = {}
    nov = {}
    is_unrestricted = False
    is_component_unrestricted = {}

    sorted_keys = sorted(mo_occ.keys())
    for t in sorted_keys:
        mo_occ[t] = numpy.asarray(mo_occ[t])
        if mo_occ[t].ndim > 1: # unrestricted
            assert not t.startswith('n')
            assert mo_occ[t].shape[0] == 2
            is_unrestricted = True
            is_component_unrestricted[t] = True
            nocc[t] = []
            nvir[t] = []
            nov[t] = []
            for i in range(2):
                occidx = mo_occ[t][i] > 0
                viridx = mo_occ[t][i] == 0
                nocc[t].append(numpy.count_nonzero(occidx))
                nvir[t].append(numpy.count_nonzero(viridx))
            nov[t] = nvir[t][0] * nocc[t][0] + nvir[t][1] * nocc[t][1]
        else:
            is_component_unrestricted[t] = False
            occidx = mo_occ[t] > 0
            viridx = mo_occ[t] == 0
            nocc[t] = numpy.count_nonzero(occidx)
            nvir[t] = numpy.count_nonzero(viridx)
            nov[t] = nvir[t] * nocc[t]

    def norm_xy(z1):
        offset = 0
        norm = .0
        zs = {}
        xys = {}
        for t in sorted_keys:
            z = z1[offset: offset+nov[t]*2]
            offset += nov[t] * 2
            x, y = z.reshape(2,-1)
            zs[t] = [x, y]
            norm += lib.norm(x)**2 - lib.norm(y)**2
        if norm < 0:
            log.warn('NEO-TDDFT amplitudes |X| smaller than |Y|')
            norm = abs(norm)
        if is_unrestricted:
            norm = 1/numpy.sqrt(norm)
        else:
            norm = numpy.sqrt(.5/norm)

        for t in sorted_keys:
            x, y = zs[t]
            if is_component_unrestricted[t]:
                xys[t] = ((x[:nocc[t][0]*nvir[t][0]].reshape(nocc[t][0],nvir[t][0]) * norm,  # X_alpha
                           x[nocc[t][0]*nvir[t][0]:].reshape(nocc[t][1],nvir[t][1]) * norm), # X_beta
                          (y[:nocc[t][0]*nvir[t][0]].reshape(nocc[t][0],nvir[t][0]) * norm,  # Y_alpha
                           y[nocc[t][0]*nvir[t][0]:].reshape(nocc[t][1],nvir[t][1]) * norm)) # Y_beta
            else:
                xys[t] = (x.reshape(nocc[t],nvir[t]) * norm,
                          y.reshape(nocc[t],nvir[t]) * norm)

        if not is_unrestricted:
            for t in sorted_keys:
                if t.startswith('n'):
                    xp, yp = xys[t]
                    xys[t] = (xp * numpy.sqrt(2), yp * numpy.sqrt(2))

        return xys

    return [norm_xy(z) for z in x1]

def eig_mat(m, nroots=None, half=True):
    """
    Eigenvalue problem solver.
    Copied from old tdscf.common_slow

    Args:
        m (numpy.ndarray): the matrix to diagonalize;
        nroots (int): the number of roots ot calculate (ignored for `driver` == 'eig');
        half (bool): if True, implies spectrum symmetry and takes only a half of eigenvalues;

    Returns:

    """
    vals, vecs = numpy.linalg.eig(m)
    order = numpy.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    if half:
        vals, vecs = vals[len(vals) // 2:], vecs[:, vecs.shape[1] // 2:]
        vecs = vecs[:, ]
    vals, vecs = vals[:nroots], vecs[:, :nroots]
    return vals, vecs

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

def get_ab_elec(mf_elec):
    if isinstance(mf_elec,scf.uhf.UHF):
        return uhf.get_ab(mf_elec)
    else:
        nocc = mf_elec.mo_coeff[:,mf_elec.mo_occ>0].shape[1]
        nvir = mf_elec.mo_coeff.shape[1]-nocc
        a_e, b_e = rhf.get_ab(mf_elec)
        return a_e.reshape((nocc*nvir,nocc*nvir)), b_e.reshape((nocc*nvir,nocc*nvir))

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

def get_c_int(interaction, coeff_o, coeff_v):
    t1 = interaction.mf1_type
    t2 = interaction.mf2_type
    eri = interaction._eri
    co_1 = coeff_o[t1]
    cv_1 = coeff_v[t1]
    co_2 = coeff_o[t2]
    cv_2 = coeff_v[t2]
    scale = interaction.mf1.charge * interaction.mf2.charge

    if interaction.mf1_unrestricted:
        assert not interaction.mf2_unrestricted
        coeffs = (co_1[0], cv_1[0], co_2, cv_2)
        c_1 = scale*ao2mo.incore.general(eri, coeffs, compact=False)
        coeffs = (co_1[1], cv_1[1], co_2, cv_2)
        c_2 = scale*ao2mo.incore.general(eri, coeffs, compact=False)
        return [c_1, c_2]
    elif interaction.mf2_unrestricted:
        assert not interaction.mf1_unrestricted
        coeffs = (co_1, cv_1, co_2[0], cv_2[0])
        c_1 = scale*ao2mo.incore.general(eri, coeffs, compact=False)
        coeffs = (co_1, cv_1, co_2[1], cv_2[1])
        c_2 = scale*ao2mo.incore.general(eri, coeffs, compact=False)
        return [c_1, c_2]
    else:
        coeffs = (co_1, cv_1, co_2, cv_2)
        c = scale*ao2mo.incore.general(eri, coeffs, compact=False)
        return c

def get_abc_no_epc(mf):
    a = {}
    b = {}
    c = {}
    a_e, b_e = get_ab_elec(mf.components['e'])
    if isinstance(a_e, tuple):
        a_e = list(a_e)
        b_e = list(b_e)
    a['e'] = a_e
    b['e'] = b_e
    for t in mf.components.keys():
        if t.startswith('n'):
            a_p, b_p = get_ab_nuc(mf.components[t])
            a[t] = a_p
            b[t] = b_p

    mo_coeff_o = {}
    mo_coeff_v = {}
    for t, comp in mf.components.items():
        if (t.startswith('e') and isinstance(comp, scf.uhf.UHF)):
            nocca = numpy.count_nonzero((mf.mo_occ[t][0]>0))
            noccb = numpy.count_nonzero((mf.mo_occ[t][1]>0))
            mo_coeff_o[t] = [mf.mo_coeff[t][0][:,:nocca], mf.mo_coeff[t][1][:,:noccb]]
            mo_coeff_v[t] = [mf.mo_coeff[t][0][:,nocca:], mf.mo_coeff[t][1][:,noccb:]]
        else:
            nocc = numpy.count_nonzero((mf.mo_occ[t]>0))
            mo_coeff_o[t] = mf.mo_coeff[t][:,:nocc]
            mo_coeff_v[t] = mf.mo_coeff[t][:,nocc:]
    for t_pair, interaction in mf.interactions.items():
        c[t_pair] = get_c_int(interaction, mo_coeff_o, mo_coeff_v)

    return a, b, c

def get_epc_iajb(mf):
    if isinstance(mf.components['e'], scf.uhf.UHF):
        iajb, iajb_int = get_epc_iajb_uhf(mf, reshape=True)
    else:
        iajb, iajb_int = get_epc_iajb_rhf(mf, reshape=True)

    return iajb, iajb_int

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
    a, b, c = get_abc_no_epc(mf)
    if isinstance(mf, neo.KS):
        if mf.epc is not None:
            iajb, iajb_int = get_epc_iajb(mf)
            for t, comp in iajb.items():
                a[t] = add_epc(a[t], comp)
                b[t] = add_epc(b[t], comp)
            for t_pair, comp in iajb_int.items():
                c[t_pair] = add_epc(c[t_pair], comp)
    return a, b, c

def get_diag_block(a, b):
    '''
    get diagonal blocks [[A, B], [-B, -A]] of the full matrix
    '''
    diag = {}
    for t in a.keys():
        if isinstance(a[t], list):
            _a = aabb2a(a[t])
            _b = aabb2a(b[t])
            diag[t] = ab2full(_a, _b)
        else:
            diag[t] = ab2full(a[t], b[t])
    return diag

def get_cross_block(c):
    '''
    get cross blocks of the full matrix

    cross: [[C, C], [-C, -C]]
    '''
    cross = {}
    for t_pair, comp in c.items():
        t1, t2 = t_pair
        if isinstance(comp, list):
            _c = numpy.vstack((comp[0], comp[1]))
        elif t1.startswith('e'):
            _c = comp * numpy.sqrt(2)
        else:
            _c = comp
        cross[t_pair] = c2full(_c)
        cross[(t2, t1)] = c2full(_c.transpose())
    return cross

def get_full_mat(diag, cross):
    '''
    get full matrix from diagonal and cross blocks
    '''
    tot_size = 0
    offset = {}
    for t in diag.keys():
        offset[t] = tot_size
        tot_size += diag[t].shape[0]

    mat = numpy.zeros((tot_size, tot_size))
    for t in diag.keys():
        i1 = offset[t]
        i2 = i1 + diag[t].shape[0]
        mat[i1:i2, i1:i2] = diag[t]

    for (t1, t2) in cross.keys():
        i1 = offset[t1]
        i2 = i1 + diag[t1].shape[0]
        j1 = offset[t2]
        j2 = j1 + diag[t2].shape[0]
        mat[i1:i2, j1:j2] = cross[(t1, t2)]

    return mat

def get_td_mat(mf):
    a, b, c = get_abc(mf)
    diag = get_diag_block(a, b)
    cross = get_cross_block(c)
    return get_full_mat(diag, cross)


class TDDirect(lib.StreamObject):
    '''Full td matrix diagonlization

    Examples::

    >>> from pyscf import neo
    >>> from pyscf.neo import tddft_slow
    >>> mol = neo.M(atom='H 0 0 0; C 0 0 1.067; N 0 0 2.213', basis='631g',
                    quantum_nuc = ['H'], nuc_basis = 'pb4d')
    >>> mf = neo.HF(mol)
    >>> mf.scf()
    >>> td_mf = tddft_slow.TDDirect(mf)
    >>> td_mf.kernel(nstates=5)
    Excited State energies (eV)
    [0.62060056 0.62060056 0.69023232 1.24762232 1.33973627]
    '''

    def __init__(self, mf, nstates=3):
        self.verbose = mf.verbose
        self.stdout = mf.components['e'].stdout
        self.mol = mf.mol
        self._scf = mf
        self.nstates = nstates
        self.td_mat = None
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
        w, x1 = eig_mat(m, nroots=nstates)

        self.e = w
        self.xy = _normalize(x1.T, self._scf.mo_occ, log)

        log.timer('NEO-TDDFT full diagonalization', *cpu0)
        logger.note(self, 'Excited State energies (eV)\n%s', self.e * nist.HARTREE2EV)
        return self.e, self.xy
