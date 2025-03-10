from pyscf import neo, scf
from pyscf.lib import logger
from pyscf import lib
from pyscf.scf.jk import get_jk
from pyscf.grad import tdrks, tdrhf
from functools import reduce
import numpy

def solve_nos1(fvind, mo_energy, mo_occ, h1, with_f1=False,
               max_cycle=30, tol=1e-9, hermi=False, verbose=logger.WARN,
               level_shift = 0):
    '''
    solver for cneo-tddft gradient z-vector
    modified from neo.cphf.solve_withs1
    '''
    log = logger.new_logger(verbose=verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    occidx = {}
    viridx = {}
    e_i = {}
    e_a = {}
    e_ai = {}
    nocc = {}
    nvir = {}
    hs = {}
    scale = {}
    mo1base = []
    is_component_unrestricted = {}
    nov = {}
    total_mo1 = 0
    total_f1 = 0
    sorted_keys = sorted(mo_occ.keys())

    for t in sorted_keys:
        mo_occ[t] = numpy.asarray(mo_occ[t])
        if mo_occ[t].ndim > 1: # unrestricted
            assert not t.startswith('n')
            assert mo_occ[t].shape[0] == 2
            is_component_unrestricted[t] = True
            raise NotImplementedError('cneo td grad not implemented for unrestricted')
        else:
            is_component_unrestricted[t] = False
            occidx[t] = mo_occ[t] > 0
            viridx[t] = mo_occ[t] == 0
            e_a[t] = mo_energy[t][viridx[t]]
            e_i[t] = mo_energy[t][occidx[t]]
            e_ai[t] = 1. / lib.direct_sum('a-i->ai', e_a[t], e_i[t])
            nvir[t], nocc[t] = e_ai[t].shape
            if with_f1 and t.startswith('n'):
                scale[t] = 2.0
                total_f1 += 3

            hs[t] = h1[t].reshape(-1,nvir[t],nocc[t])
            hs[t] *= -e_ai[t]
            mo1base.append(hs[t].reshape(-1,nvir[t]*nocc[t]))
            nov[t] = nvir[t] * nocc[t]
        total_mo1 += nov[t]

    if with_f1:
        nset = mo1base[0].shape[0]
        for t in sorted_keys:
            if t.startswith('n'):
                mo1base.append(numpy.zeros((nset,3)))
    mo1base = numpy.hstack(mo1base)

    def vind_vo(mo1_and_f1):
        mo1_and_f1 = mo1_and_f1.reshape(-1,total_mo1+total_f1)
        mo1_array = mo1_and_f1[:,:total_mo1]
        mo1 = {}
        offset = 0
        for t in sorted_keys:
            mo1[t] = mo1_array[:,offset:offset+nov[t]]
            if is_component_unrestricted[t]:
                raise NotImplementedError
            offset += nov[t]
        f1 = None
        if with_f1:
            f1_array = mo1_and_f1[:,total_mo1:]
            f1 = {}
            offset = 0
            for t in sorted_keys:
                if t.startswith('n'):
                    f1[t] = f1_array[:,offset:offset+3]
                    offset += 3
        v, r = fvind(mo1, f1=f1)
        for t in v:
            if is_component_unrestricted[t]:
                raise NotImplementedError
            else:
                v[t] = v[t].reshape(-1, nvir[t], nocc[t])
                v[t] *= e_ai[t]
            v[t] = v[t].reshape(-1, nov[t])
        if with_f1 and r is not None:
            for t in r:
                # NOTE: this scale factor is somewhat empirical. The goal is
                # to try to bring the position constraint equation r * mo1 = 0
                # to be of a similar magnitude as compared to the conventional
                # CPHF equations.
                r[t] = r[t] * scale[t] - f1[t]
                r[t] = r[t].reshape(-1,3)
                # NOTE: f1 got subtracted because krylov solver solves (1+a)x=b
            return numpy.hstack([v[k] for k in sorted_keys]
                                + [r[k] for k in sorted_keys if k in r]).ravel()
        return numpy.hstack([v[k] for k in sorted_keys]).ravel()
    
    mo1_and_f1 = lib.krylov(vind_vo, mo1base.ravel(),
                            tol=tol, max_cycle=max_cycle, hermi=hermi, verbose=log)
    mo1_and_f1 = mo1_and_f1.reshape(-1,total_mo1+total_f1)
    mo1_array = mo1_and_f1[:,:total_mo1]
    mo1 = {}
    offset = 0
    for t in sorted_keys:
        mo1[t] = mo1_array[:,offset:offset+nov[t]]
        offset += nov[t]
        if is_component_unrestricted[t]:
            raise NotImplementedError
        else:
            mo1[t] = mo1[t].reshape(-1, nvir[t], nocc[t])
    f1 = None
    if with_f1:
        f1_array = mo1_and_f1[:,total_mo1:]
        f1 = {}
        offset = 0
        for t in sorted_keys:
            if t.startswith('n'):
                f1[t] = f1_array[:,offset:offset+3]
                offset += 3
    log.timer('krylov solver in CNEO-TDDFT', *t0)

    return mo1, None, f1

def position_analysis(z1, int1e_r_vo):

    rfn = {}
    for t in int1e_r_vo.keys():
        rfn[t] = numpy.einsum('xij, ij->x', int1e_r_vo[t], z1[t])

    return rfn

def get_fock_add_cdft(f1n, int1e_r_ao, fac=2.0):
    f_add = {}
    for t in f1n.keys():
        f_add[t] = numpy.einsum('xij, x->ij', int1e_r_ao[t], f1n[t]) * fac

    return f_add

def grad_elec(td_grad, x_y, singlet=True, atmlst=None,
              max_memory=2000, verbose=logger.INFO):
    
    log = logger.new_logger(td_grad, verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    mf = td_grad.base._scf
    if not isinstance(mf,neo.CDFT):
        raise TypeError('td grad is only supported for cneo')
    mol = td_grad.mol
    mf_e = mf.components['e']
    mol_e = mol.components['e']


    if isinstance(mf_e,scf.uhf.UHF):
        raise NotImplementedError
    
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    orbv = {}
    orbo = {}
    nocc = {}
    nvir = {}

    int1e_r_ao = {}
    int1e_r_vo = {}
    for t in mf.components.keys():
        occidx = numpy.where(mo_occ[t] > 0)[0]
        viridx = numpy.where(mo_occ[t] == 0)[0]
        orbo[t] = mo_coeff[t][:,occidx]
        orbv[t] = mo_coeff[t][:,viridx]
        nocc[t] = len(occidx)
        nvir[t] = len(viridx)
        if t.startswith('n'):
            int1e_r_ao[t] = mf.components[t].int1e_r
            int1e_r_vo[t] = numpy.einsum('ja, ki, xjk->xai', orbv[t],orbo[t], int1e_r_ao[t])
    nao_e, nmo_e = mo_coeff['e'].shape

    x, y = x_y
    if x.dtype == 'complex128':
        imag = numpy.max(numpy.abs(x.imag))
        if imag > 1e-8:
            raise NotImplementedError('complex xy is not supported')
    xpy = (x+y).reshape(nocc['e'],nvir['e']).T.real
    xmy = (x-y).reshape(nocc['e'],nvir['e']).T.real

    dvv = numpy.einsum('ai,bi->ab', xpy, xpy) + numpy.einsum('ai,bi->ab', xmy, xmy)
    doo =-numpy.einsum('ai,aj->ij', xpy, xpy) - numpy.einsum('ai,aj->ij', xmy, xmy)
    dmxpy = reduce(numpy.dot, (orbv['e'], xpy, orbo['e'].T))
    dmxmy = reduce(numpy.dot, (orbv['e'], xmy, orbo['e'].T))
    dmzoo = reduce(numpy.dot, (orbo['e'], doo, orbo['e'].T))
    dmzoo+= reduce(numpy.dot, (orbv['e'], dvv, orbv['e'].T))

    if mf.epc is not None:
        raise NotImplementedError('epc is not implemented in analytic td gradients')
    td_grad_e = tdrks.Gradients(mf_e.TDDFT())
    ni = mf_e._numint
    ni.libxc.test_deriv_order(mf_e.xc, 3, raise_error=True)
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf_e.xc, mol_e.spin)
    f1vo, f1oo, vxc1, k1ao = \
            tdrks._contract_xc_kernel(td_grad_e, mf_e.xc, dmxpy,
                                dmzoo, True, True, singlet, max_memory)

    if ni.libxc.is_hybrid_xc(mf_e.xc):
        dm = (dmzoo, dmxpy+dmxpy.T, dmxmy-dmxmy.T)
        vj, vk = mf_e.get_jk(mol, dm, hermi=0)
        vk *= hyb
        if omega != 0:
            vk += mf_e.get_k(mol, dm, hermi=0, omega=omega) * (alpha-hyb)
        veff0doo = vj[0] * 2 - vk[0] + f1oo[0] + k1ao[0] * 2
        wvo_e = reduce(numpy.dot, (orbv['e'].T, veff0doo, orbo['e'])) * 2
        if singlet:
            veff = vj[1] * 2 - vk[1] + f1vo[0] * 2
        else:
            veff = f1vo[0] - vk[1]
        veff0mop = reduce(numpy.dot, (mo_coeff['e'].T, veff, mo_coeff['e']))
        wvo_e -= numpy.einsum('ki,ai->ak', veff0mop[:nocc['e'],:nocc['e']], xpy) * 2
        wvo_e += numpy.einsum('ac,ai->ci', veff0mop[nocc['e']:,nocc['e']:], xpy) * 2
        veff = -vk[2]
        veff0mom = reduce(numpy.dot, (mo_coeff['e'].T, veff, mo_coeff['e']))
        wvo_e -= numpy.einsum('ki,ai->ak', veff0mom[:nocc['e'],:nocc['e']], xmy) * 2
        wvo_e += numpy.einsum('ac,ai->ci', veff0mom[nocc['e']:,nocc['e']:], xmy) * 2
    else:
        vj = mf_e.get_j(mol, (dmzoo, dmxpy+dmxpy.T), hermi=1)
        veff0doo = vj[0] * 2 + f1oo[0] + k1ao[0] * 2
        wvo_e = reduce(numpy.dot, (orbv['e'].T, veff0doo, orbo['e'])) * 2
        if singlet:
            veff = vj[1] * 2 + f1vo[0] * 2
        else:
            veff = f1vo[0]
        veff0mop = reduce(numpy.dot, (mo_coeff['e'].T, veff, mo_coeff['e']))
        wvo_e -= numpy.einsum('ki,ai->ak', veff0mop[:nocc['e'],:nocc['e']], xpy) * 2
        wvo_e += numpy.einsum('ac,ai->ci', veff0mop[nocc['e']:,nocc['e']:], xpy) * 2
        veff0mom = numpy.zeros((nmo_e,nmo_e))

    wvo = {}
    wvo['e'] = wvo_e
    dm = {}
    for t in mf.components:
        dm[t] = None
    dm['e'] = dmzoo
    for t_pair, interaction in mf.interactions.items():
        p1, p2 = t_pair
        if p1.startswith('e'):
            vj_en = interaction.get_vint(dm)
            wvo_n = reduce(numpy.dot, (orbv[p2].T, vj_en[p2], orbo[p2])) * 2
            wvo[p2] = wvo_n

    vresp = mf.gen_response(max_memory=max_memory, hermi=0)
    def fvind(mo1, f1):
        dm = {}
        for t in mo1.keys():
            mo1[t] = mo1[t].reshape(nvir[t], nocc[t])
            dm_t = reduce(numpy.dot, (orbv[t], mo1[t], orbo[t].T))
            dm[t] = dm_t + dm_t.T
        v1ao = vresp(dm)
        v1ao['e'] *= 2
        for t in f1.keys():
            f1[t] = f1[t].ravel()
        f_add = get_fock_add_cdft(f1, int1e_r_ao)

        v1 = {}
        for t in v1ao.keys():
            if t.startswith('n'):
                v1ao[t] += f_add[t]
            v1[t] = reduce(numpy.dot, (orbv[t].T, v1ao[t], orbo[t])).ravel()

        rfn = position_analysis(mo1, int1e_r_vo)

        return v1, rfn
    
    z1, mo_e1, f1 = solve_nos1(fvind, mf.mo_energy, mo_occ, wvo,
                                          with_f1 = True,
                                          max_cycle=td_grad.cphf_max_cycle,
                                          tol=td_grad.cphf_conv_tol,
                                          verbose = verbose)
    for t in z1.keys():
        z1[t] = z1[t].reshape(nvir[t], nocc[t])
    time1 = log.timer('Z-vector using CPHF solver', *time0)

    dm_z = {}
    for t in z1.keys():
        z1ao_t = reduce(numpy.dot, (orbv[t], z1[t], orbo[t].T))
        dm_z[t] = z1ao_t + z1ao_t.T
    veff = vresp(dm_z)
    veff_e = veff['e']
    nocc_e = nocc['e']
    orbo_e = orbo['e']
    mo_energy_e = mf.mo_energy['e']
    z1ao_e = reduce(numpy.dot, (orbv['e'], z1['e'], orbo['e'].T))

    im0_e = numpy.zeros((nmo_e,nmo_e))
    # oo
    im0_e[:nocc_e,:nocc_e] = reduce(numpy.dot, (orbo_e.T, veff0doo+veff_e, orbo_e))
    im0_e[:nocc_e,:nocc_e]+= numpy.einsum('ak,ai->ki', veff0mop[nocc_e:,:nocc_e], xpy)
    im0_e[:nocc_e,:nocc_e]+= numpy.einsum('ak,ai->ki', veff0mom[nocc_e:,:nocc_e], xmy)
    # vv
    im0_e[nocc_e:,nocc_e:] = numpy.einsum('ci,ai->ac', veff0mop[nocc_e:,:nocc_e], xpy)
    im0_e[nocc_e:,nocc_e:]+= numpy.einsum('ci,ai->ac', veff0mom[nocc_e:,:nocc_e], xmy)
    # vo
    im0_e[nocc_e:,:nocc_e] = numpy.einsum('ki,ai->ak', veff0mop[:nocc_e,:nocc_e], xpy)*2
    im0_e[nocc_e:,:nocc_e]+= numpy.einsum('ki,ai->ak', veff0mom[:nocc_e,:nocc_e], xmy)*2

    zeta_e = lib.direct_sum('i+j->ij', mo_energy_e, mo_energy_e) * .5
    zeta_e[nocc_e:,:nocc_e] = mo_energy_e[:nocc_e]
    zeta_e[:nocc_e,nocc_e:] = mo_energy_e[nocc_e:]
    dm1 = numpy.zeros((nmo_e,nmo_e))
    dm1[:nocc_e,:nocc_e] = doo
    dm1[nocc_e:,nocc_e:] = dvv
    dm1[nocc_e:,:nocc_e] = z1['e']
    dm1[:nocc_e,:nocc_e] += numpy.eye(nocc_e)*2 # for ground state
    im0_e = reduce(numpy.dot, (mo_coeff['e'], im0_e+zeta_e*dm1, mo_coeff['e'].T))

    mf_grad = neo.Gradients(mf)
    mf_grad_e = mf_grad.components['e']
    hcore_deriv = {}
    for t, comp in mf_grad.components.items():
        hcore_deriv[t] = comp.hcore_generator()
    
    s1_e = mf_grad_e.get_ovlp(mol_e)

    dmz1doo = z1ao_e + dmzoo
    oo0 = reduce(numpy.dot, (orbo_e, orbo_e.T))

    if ni.libxc.is_hybrid_xc(mf_e.xc):
        dm = (oo0, dmz1doo+dmz1doo.T, dmxpy+dmxpy.T, dmxmy-dmxmy.T)
        vj, vk = td_grad_e.get_jk(mol_e, dm)
        vk *= hyb
        if omega != 0:
            vk += td_grad_e.get_k(mol_e, dm, omega=omega) * (alpha-hyb)
        vj = vj.reshape(-1,3,nao_e,nao_e)
        vk = vk.reshape(-1,3,nao_e,nao_e)
        veff1 = -vk
        if singlet:
            veff1 += vj * 2
        else:
            veff1[:2] += vj[:2] * 2
    else:
        vj = td_grad_e.get_j(mol_e, (oo0, dmz1doo+dmz1doo.T, dmxpy+dmxpy.T))
        vj = vj.reshape(-1,3,nao_e,nao_e)
        veff1 = numpy.zeros((4,3,nao_e,nao_e))
        if singlet:
            veff1[:3] = vj * 2
        else:
            veff1[:2] = vj[:2] * 2

    fxcz1 = tdrks._contract_xc_kernel(td_grad_e, mf_e.xc, z1ao_e, None,
                                False, False, True, max_memory)[0]

    veff1[0] += vxc1[1:]
    veff1[1] +=(f1oo[1:] + fxcz1[1:] + k1ao[1:]*2)*2 # *2 for dmz1doo+dmz1oo.T
    if singlet:
        veff1[2] += f1vo[1:] * 2
    else:
        veff1[2] += f1vo[1:]
    time1 = log.timer('2e AO integral derivatives', *time1)

    dm_gs = mf.make_rdm1()

    if atmlst is None:
        atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    de = numpy.zeros((len(atmlst),3))
    for k, ka in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ka]

        h1ao_e = hcore_deriv['e'](ka)
        h1ao_e[:,p0:p1]   += veff1[0,:,p0:p1]
        h1ao_e[:,:,p0:p1] += veff1[0,:,p0:p1].transpose(0,2,1)

        de[k] -= numpy.einsum('xpq,pq->x', s1_e[:,p0:p1], im0_e[p0:p1])
        de[k] -= numpy.einsum('xqp,pq->x', s1_e[:,p0:p1], im0_e[:,p0:p1])

        de[k] += numpy.einsum('xij,ij->x', veff1[1,:,p0:p1], oo0[p0:p1])
        de[k] += numpy.einsum('xij,ij->x', veff1[2,:,p0:p1], dmxpy[p0:p1,:]) * 2
        de[k] += numpy.einsum('xij,ij->x', veff1[3,:,p0:p1], dmxmy[p0:p1,:]) * 2
        de[k] += numpy.einsum('xji,ij->x', veff1[2,:,p0:p1], dmxpy[:,p0:p1]) * 2
        de[k] -= numpy.einsum('xji,ij->x', veff1[3,:,p0:p1], dmxmy[:,p0:p1]) * 2
        z1ao_e = 0.0

        for t1 in mf.components.keys():
            if t1.startswith('n'):
                z1ao_n = 0.0
                mol_n = mol.components[t1]
                ja = mol_n.atom_index
                charge = -mf.components[t1].charge
                shls_slice = (shl0, shl1) + (0, mol_e.nbas) + (0, mol_n.nbas)*2
                v1en = get_jk((mol_e, mol_e, mol_n, mol_n),
                          (dm_gs[t1], dm_z[t1]), scripts=['ijkl,lk->ij','ijkl,lk->ij'],
                          intor='int2e_ip1', aosym='s2kl', comp=3,
                          shls_slice=shls_slice)
                v1en = [_v * charge for _v in v1en]
                h1ao_e[:,p0:p1] += v1en[0]
                h1ao_e[:,:,p0:p1] += v1en[0].transpose(0,2,1)
            
                z1ao_e += v1en[1]
                h1ao_n = hcore_deriv[t1](ka)

                if ja == ka:
                    # derivative w.r.t. nuclear basis center
                    v1ne = get_jk((mol_n, mol_n, mol_e, mol_e),
                                (dm_gs['e'], dmz1doo), scripts=['ijkl,lk->ij','ijkl,lk->ij'],
                                intor='int2e_ip1', aosym='s2kl', comp=3)
                    v1ne = [_v * charge for _v in v1ne]
                    h1ao_n += v1ne[0] + v1ne[0].transpose(0,2,1)
                    z1ao_n += v1ne[1] * 2.0

                    for t2 in mf.components.keys():
                        if (t2.startswith('n')) and (t2 != t1):
                            mol_n2 = mol.components[t2]
                            v1nn = get_jk((mol_n, mol_n, mol_n2, mol_n2),
                                        (dm_gs[t2], dm_z[t2]), scripts=['ijkl,lk->ij','ijkl,lk->ij'],
                                        intor='int2e_ip1', aosym='s2kl', comp=3)
                            _charge = charge * mf.components[t2].charge
                            v1nn = [_v * _charge for _v in v1nn]
                            h1ao_n += v1nn[0] + v1nn[0].transpose(0,2,1)
                            z1ao_n += v1nn[1]
                if isinstance(h1ao_n, numpy.ndarray):
                    de[k] += numpy.einsum('xij,ij->x', h1ao_n, dm_z[t1]/2+dm_gs[t1])
                if isinstance(z1ao_n, numpy.ndarray):
                    de[k] += numpy.einsum('xij,ij->x', z1ao_n, dm_gs[t1])

        de[k] += numpy.einsum('xij,ij->x', h1ao_e, dmz1doo+dm_gs['e'])
        de[k] += numpy.einsum('xij,ij->x', z1ao_e, dm_gs['e'][p0:p1])

        de[k] += td_grad.extra_force(ka, locals())

    log.timer('CNEO-TDDFT nuclear gradients', *time0)
    return de


def as_scanner(td_grad, state=1):
    '''
    Modified from grad.tdrhf.as_scanner
    '''
    from pyscf import neo
    if isinstance(td_grad, lib.GradScanner):
        return td_grad
    
    if state == 0:
        return td_grad.base._scf.nuc_grad_method().as_scanner()

    logger.info(td_grad, 'Create scanner for %s', td_grad.__class__)

    class CTDSCF_GradScanner(td_grad.__class__, lib.GradScanner):
        def __init__(self, g, state):
            lib.GradScanner.__init__(self, g)
            if state is not None:
                self.state = state
            self._keys = self._keys.union(['e_tot'])
        def __call__(self, mol_or_geom, state=None, **kwargs):
            if isinstance(mol_or_geom, neo.Mole):
                mol = mol_or_geom
            else:
                mol = self.mol.set_geom_(mol_or_geom, inplace=False)
            self.reset(mol)

            if state is None:
                state = self.state
            else:
                self.state = state

            td_scanner = self.base
            td_scanner(mol)
            de = self.kernel(state=state, **kwargs)
            e_tot = self.e_tot[state-1]
            return e_tot, de
        @property
        def converged(self):
            td_scanner = self.base
            return all((td_scanner._scf.converged,
                        td_scanner.converged[self.state]))

    return CTDSCF_GradScanner(td_grad, state)
    

class Gradients(tdrhf.Gradients):
    ''' Analytic gradients for frozen nuclear orbital CNEO-TDDFT

    Examples::

    >>> from pyscf import neo
    >>> from pyscf.neo import ctddft, tdgrad
    >>> mol = neo.M(atom='H 0 0 0; C 0 0 1.067; N 0 0 2.213', basis='631g', 
                    quantum_nuc = ['H'], nuc_basis = 'pb4d')
    >>> mf = neo.CDFT(mol, xc='hf')
    >>> mf.scf()
    >>> td_mf = ctddft.CTDDFT(mf)
    >>> td_mf.kernel(nstates=5)
    >>> td_grad = tdgrad.Gradients(td_mf)
    >>> td_grad.kernel()
    '''

    @lib.with_doc(grad_elec.__doc__)
    def grad_elec(self, xy, singlet, atmlst=None):
        return grad_elec(self, xy, singlet, atmlst, self.max_memory, self.verbose)
    
    def grad_nuc(self, mol=None, atmlst=None):
        mf_grad = self.base._scf.components['e'].nuc_grad_method()
        return mf_grad.grad_nuc(mol, atmlst)
        
    def kernel(self, xy=None, state=None, singlet=True, atmlst=None):
        '''
        Args:
            state : int
                Excited state ID.  state = 1 means the first excited state.
        '''
        if xy is None:
            if state is None:
                state = self.state
            else:
                self.state = state

            if state == 0:
                logger.warn(self, 'state=0 found in the input. '
                            'Gradients of ground state is computed.')
                return neo.Gradients(self.base._scf).kernel(atmlst=atmlst)

            xy = self.base.xy[state-1]

        if singlet is None: singlet = self.base.singlet
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        de = self.grad_elec(xy, singlet, atmlst)
        self.de = de = de + self.grad_nuc(atmlst=atmlst)
        if self.mol.symmetry:
            raise NotImplementedError('Symmetry is not supported')
        self._finalize()
        return self.de
    
    as_scanner = as_scanner