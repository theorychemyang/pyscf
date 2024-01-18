from pyscf import neo
from pyscf.neo import cphf
from pyscf.lib import logger
from pyscf import lib
from pyscf.scf.jk import get_jk
from pyscf.grad import tdrks, tdrhf
from functools import reduce
import numpy
import scipy

def solve_nos1(fvind, mf_e, mf_n, h1e, h1n, with_f1n=False,
               max_cycle=30, tol=1e-9, hermi=False, verbose=logger.WARN):
    
    log = logger.new_logger(verbose=verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    mol = mf_e.mol.super_mol
    total_size = 0 # size of mo1

    mo_energy_e = mf_e.mo_energy
    mo_occ_e = mf_e.mo_occ
    occidx_e = mo_occ_e > 0
    viridx_e = mo_occ_e == 0
    e_a_e = mo_energy_e[viridx_e]
    e_i_e = mo_energy_e[occidx_e]
    e_ai_e = 1. / lib.direct_sum('a-i->ai', e_a_e, e_i_e)
    nvir_e, nocc_e = e_ai_e.shape
    nmo_e = nocc_e + nvir_e
    mo1base_e = h1e * -e_ai_e

    e_size = nvir_e * nocc_e # size of mo1e
    total_size += e_size

    occidx_n = []
    viridx_n = []
    e_a_n = []
    e_i_n = []
    e_ai_n = []
    nvir_n = []
    nocc_n = []
    nmo_n = []
    mo1base_n = []
    n_size = []
    for i in range(len(mf_n)):
        occidx_n.append(mf_n[i].mo_occ > 0)
        viridx_n.append(mf_n[i].mo_occ == 0)
        e_a_n.append(mf_n[i].mo_energy[viridx_n[-1]])
        e_i_n.append(mf_n[i].mo_energy[occidx_n[-1]])
        e_ai_n.append(1. / lib.direct_sum('a-i->ai', e_a_n[-1], e_i_n[-1]))
        tmp1, tmp2 = e_ai_n[-1].shape
        nvir_n.append(tmp1)
        nocc_n.append(tmp2)
        nmo_n.append(tmp1 + tmp2)
        n_size.append(nvir_n[-1] * nocc_n[-1])
        total_size += n_size[-1]
        mo1base_n.append(h1n[i])
        mo1base_n[-1] *= -e_ai_n[-1]

    def vind_vo(mo1s):
        mo1e, mo1n, f1n = cphf.mo1s_disassembly(mo1s, total_size, e_size, n_size,
                                           with_f1n=with_f1n)
        for i in range(len(mo1n)):
            mo1n[i] = mo1n[i].reshape(h1n[i].shape)
        if f1n is not None:
            for i in range(len(f1n)):
                f1n[i] = f1n[i].reshape(-1,3)
        ve, vn, rfn = fvind(mo1e.reshape(h1e.shape), mo1n, f1n=f1n)
        ve = ve.reshape(h1e.shape)
        ve *= e_ai_e
        for i in range(len(vn)):
            vn[i] = vn[i].reshape(h1n[i].shape)
            vn[i] *= e_ai_n[i]
        if rfn is not None:
            for i in range(len(f1n)):
                ia = mf_n[i].mol.atom_index
                charge = mol.atom_charge(ia)
                # NOTE: this 2*charge factor is purely empirical, because equation
                # r * mo1 = 0 is insensitive to the factor mathematically, but
                # the factor will change the numerical solution
                # TODO: find the best factor
                rfn[i] = rfn[i] * 2.0 * charge - f1n[i]
                # note that f got subtracted because krylov solver solves (1+a)x=b
        if with_f1n:
            return numpy.concatenate((ve,
                                      numpy.concatenate(vn, axis=None),
                                      numpy.concatenate(rfn, axis=None)),
                                     axis=None)
        else:
            return numpy.concatenate((ve,
                                      numpy.concatenate(vn, axis=None)),
                                     axis=None)

    if with_f1n:
        mo1base = numpy.concatenate((mo1base_e,
                                     numpy.concatenate(mo1base_n, axis=None),
                                     numpy.zeros(3*len(mf_n))),
                                    axis=None)
    else:
        mo1base = numpy.concatenate((mo1base_e,
                                     numpy.concatenate(mo1base_n, axis=None)),
                                    axis=None)
    mo1s = lib.krylov(vind_vo, mo1base,
                      tol=tol, max_cycle=max_cycle, hermi=hermi, verbose=log)
    mo1e, mo1n, f1n = cphf.mo1s_disassembly(mo1s, total_size, e_size, n_size,
                                       with_f1n=with_f1n)
    mo1e = mo1e.reshape(mo1base_e.shape)
    for i in range(len(mo1n)):
        mo1n[i] = mo1n[i].reshape(h1n[i].shape)
    if f1n is not None:
        for i in range(len(f1n)):
            f1n[i] = f1n[i].reshape(-1,3)
    log.timer('krylov solver in CNEO CPHF', *t0)

    return mo1e, None, mo1n, f1n

def position_analysis(z1n, int1e_r):
    pos_ana = []
    for i in range(len(z1n)):
        pos_ana.append(numpy.einsum('xij, ij->x', int1e_r[i], z1n[i]))

    return numpy.array(pos_ana)


def grad_elec(td_grad, x_y, singlet=True, atmlst=None,
              max_memory=2000, verbose=logger.INFO):
    '''
    Modified from grad.tdrks.grad_elec
    '''
    log = logger.new_logger(td_grad, verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    mf = td_grad.base._scf
    mol = td_grad.mol
    mf_e = mf.mf_elec
    mf_n = mf.mf_nuc
    mol_e = mol.elec
    mol_n = mol.nuc

    if mf.unrestricted:
        raise NotImplementedError('unrestricted neo is not supported')

    mo_coeff_e = mf_e.mo_coeff
    mo_energy_e = mf_e.mo_energy
    mo_occ_e = mf_e.mo_occ
    nao_e, nmo_e = mo_coeff_e.shape
    nocc_e = (mo_occ_e>0).sum()
    nvir_e = nmo_e - nocc_e
    x, y = x_y
    if x.dtype == 'complex128':
        imag = numpy.max(numpy.abs(x.imag))
        if imag > 1e-8:
            raise NotImplementedError('complex xy is not supported')
    xpy = (x+y).reshape(nocc_e,nvir_e).T.real
    xmy = (x-y).reshape(nocc_e,nvir_e).T.real
    orbv_e = mo_coeff_e[:,nocc_e:]
    orbo_e = mo_coeff_e[:,:nocc_e]

    nocc_n = []
    nvir_n = []
    orbv_n = []
    orbo_n = []
    int1e_r_ao = []
    int1e_r_vo = []
    for i in range(len(mf_n)):
        mf_nuc = mf_n[i]
        mo_coeff_n = mf_nuc.mo_coeff
        mo_occ_n = mf_nuc.mo_occ
        occidx_n = numpy.where(mo_occ_n >0)[0]
        viridx_n = numpy.where(mo_occ_n ==0)[0]
        orbv_n.append(mo_coeff_n[:,viridx_n])
        orbo_n.append(mo_coeff_n[:,occidx_n])
        nocc_n.append(len(occidx_n))
        nvir_n.append(len(viridx_n))
        if isinstance(mf, neo.CDFT):
            int1e_r_ao.append(mf_nuc.int1e_r)
            int1e_r_vo.append(numpy.einsum('ja, ki, xjk->xai', orbv_n[-1],orbo_n[-1], int1e_r_ao[-1]))

    dvv = numpy.einsum('ai,bi->ab', xpy, xpy) + numpy.einsum('ai,bi->ab', xmy, xmy)
    doo =-numpy.einsum('ai,aj->ij', xpy, xpy) - numpy.einsum('ai,aj->ij', xmy, xmy)
    dmxpy = reduce(numpy.dot, (orbv_e, xpy, orbo_e.T))
    dmxmy = reduce(numpy.dot, (orbv_e, xmy, orbo_e.T))
    dmzoo = reduce(numpy.dot, (orbo_e, doo, orbo_e.T))
    dmzoo+= reduce(numpy.dot, (orbv_e, dvv, orbv_e.T))

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
        wvo_e = reduce(numpy.dot, (orbv_e.T, veff0doo, orbo_e)) * 2
        if singlet:
            veff = vj[1] * 2 - vk[1] + f1vo[0] * 2
        else:
            veff = f1vo[0] - vk[1]
        veff0mop = reduce(numpy.dot, (mo_coeff_e.T, veff, mo_coeff_e))
        wvo_e -= numpy.einsum('ki,ai->ak', veff0mop[:nocc_e,:nocc_e], xpy) * 2
        wvo_e += numpy.einsum('ac,ai->ci', veff0mop[nocc_e:,nocc_e:], xpy) * 2
        veff = -vk[2]
        veff0mom = reduce(numpy.dot, (mo_coeff_e.T, veff, mo_coeff_e))
        wvo_e -= numpy.einsum('ki,ai->ak', veff0mom[:nocc_e,:nocc_e], xmy) * 2
        wvo_e += numpy.einsum('ac,ai->ci', veff0mom[nocc_e:,nocc_e:], xmy) * 2
    else:
        vj = mf_e.get_j(mol, (dmzoo, dmxpy+dmxpy.T), hermi=1)
        veff0doo = vj[0] * 2 + f1oo[0] + k1ao[0] * 2
        wvo_e = reduce(numpy.dot, (orbv_e.T, veff0doo, orbo_e)) * 2
        if singlet:
            veff = vj[1] * 2 + f1vo[0] * 2
        else:
            veff = f1vo[0]
        veff0mop = reduce(numpy.dot, (mo_coeff_e.T, veff, mo_coeff_e))
        wvo_e -= numpy.einsum('ki,ai->ak', veff0mop[:nocc_e,:nocc_e], xpy) * 2
        wvo_e += numpy.einsum('ac,ai->ci', veff0mop[nocc_e:,nocc_e:], xpy) * 2
        veff0mom = numpy.zeros((nmo_e,nmo_e))

    wvo_n = []
    for i in range(len(mol_n)):
        vj_en = mf.get_j_n_dm_e(i, dmzoo) * 2    # dmzoo: 2T
        wvo = reduce(numpy.dot, (orbv_n[i].T, vj_en, orbo_n[i]))
        wvo_n.append(wvo)

    vresp = mf.gen_response(max_memory=max_memory, hermi=0)
    def fvind(x_e, x_n, f1n=None):
        # x_e is 2z1e; x_n is z1n
        x_e = x_e.reshape(nvir_e, nocc_e)
        dm_e_partial = reduce(numpy.dot, (orbv_e, x_e, orbo_e.T))
        dm_e_symm = dm_e_partial + dm_e_partial.T
        dm_n = []
        for i in range(len(x_n)):
            x_n[i] = x_n[i].reshape(nvir_n[i], nocc_n[i])
            dm_n.append(reduce(numpy.dot, (orbv_n[i], x_n[i], orbo_n[i].T)))

        v1ao_e, v1ao_n = vresp(dm_e_symm, dm_e_partial, dm_n)
        v1ao_e *= 2

        v1e = reduce(numpy.dot, (orbv_e.T, v1ao_e, orbo_e)).ravel()
        v1n = []
        for i in range(len(x_n)):
            if f1n is not None:
                f_add = neo.cdft.get_fock_add_cdft(f1n[i].ravel(), int1e_r_ao[i]) * 2.0
                v1ao_n[i] += f_add
            v1n.append(reduce(numpy.dot, (orbv_n[i].T, v1ao_n[i], orbo_n[i])).ravel())

        rfn = None
        if f1n is not None:
            rfn = position_analysis(x_n, int1e_r_vo)

        return v1e, v1n, rfn
    
    with_f1n = False
    if isinstance(mf, neo.CDFT):
        with_f1n = True
    z1_e, mo_e1_e, z1_n, f1n = solve_nos1(fvind, mf_e, mf_n, wvo_e, wvo_n,
                                          max_cycle=td_grad.cphf_max_cycle,
                                          tol=td_grad.cphf_conv_tol,
                                          with_f1n = with_f1n)
    
    z1_e.reshape(nvir_e, nocc_e)
    for i in range(len(z1_n)):
        z1_n[i] = z1_n[i].reshape(nvir_n[i], nocc_n[i])
    time1 = log.timer('Z-vector using CPHF solver', *time0)

    z1ao_e = reduce(numpy.dot, (orbv_e, z1_e, orbo_e.T))
    dm_e = z1ao_e+z1ao_e.T
    dm_n = []
    for i in range(len(z1_n)):
        z1ao_n = reduce(numpy.dot, (orbv_n[i], z1_n[i], orbo_n[i].T))
        dm_n.append(z1ao_n)
    veff_e, veff_n = vresp(dm_e, dm_e, dm_n)

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
    dm1[nocc_e:,:nocc_e] = z1_e
    dm1[:nocc_e,:nocc_e] += numpy.eye(nocc_e)*2 # for ground state
    im0_e = reduce(numpy.dot, (mo_coeff_e, im0_e+zeta_e*dm1, mo_coeff_e.T))

    mf_grad = neo.Gradients(mf)
    mf_grad_e = mf_grad.g_elec
    
    if td_grad.grid_response is not None:
        mf_grad_e.grid_response = td_grad.grid_response
        vhf = mf_grad_e.get_veff(mol.elec, mf.dm_elec)

    hcore_deriv_e = mf_grad_e.hcore_generator(mol_e)
    s1_e = mf_grad_e.get_ovlp(mol_e)
    hcore_deriv_n = []
    for x in mol_n:
        hcore_deriv_n.append(mf_grad.hcore_generator(x))

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

    if atmlst is None:
        atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    de = numpy.zeros((len(atmlst),3))
    for k, ka in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ka]

        h1ao_e = hcore_deriv_e(ka)
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

        for j in range(mol.nuc_num):
            z1ao_n = 0.0
            ja = mol_n[j].atom_index
            charge = mol.atom_charge(ja)
            shls_slice = (shl0, shl1) + (0, mol.elec.nbas) + (0, mol.nuc[j].nbas)*2
            v1en = get_jk((mol_e, mol_e, mol_n[j], mol_n[j]),
                          (mf.dm_nuc[j], dm_n[j]), scripts=['ijkl,lk->ij','ijkl,lk->ij'],
                          intor='int2e_ip1', aosym='s2kl', comp=3,
                          shls_slice=shls_slice)
            v1en = [v_ * charge for v_ in v1en]
            h1ao_e[:,p0:p1] += v1en[0]
            h1ao_e[:,:,p0:p1] += v1en[0].transpose(0,2,1)
            
            z1ao_e += v1en[1] * 2.0
            # nuclear hcore derivative
            h1ao_n = hcore_deriv_n[j](ka)

            if ja == ka:
                # derivative w.r.t. nuclear basis center
                v1ne = get_jk((mol.nuc[j], mol.nuc[j], mol.elec, mol.elec),
                              (mf.dm_elec, dmz1doo), scripts=['ijkl,lk->ij','ijkl,lk->ij'],
                              intor='int2e_ip1', aosym='s2kl', comp=3)
                v1ne = [v_ * charge for v_ in v1ne]
                h1ao_n += v1ne[0] + v1ne[0].transpose(0,2,1)
                z1ao_n += v1ne[1] * 2.0
                
                for i in range(mol.nuc_num):
                    if i != j:
                        ia = mol.nuc[i].atom_index
                        v1nn = get_jk((mol.nuc[j], mol.nuc[j], mol.nuc[i], mol.nuc[i]),
                                      (mf.dm_nuc[i], dm_n[i]), scripts=['ijkl,lk->ij','ijkl,lk->ij'],
                                      intor='int2e_ip1', aosym='s2kl', comp=3)
                        charge_ = -charge * mol.atom_charge(ia)
                        v1nn = [v_ * charge_ for v_ in v1nn]
                        h1ao_n += v1nn[0] + v1nn[0].transpose(0,2,1)

                        z1ao_n += v1nn[1] * 2.0
            if isinstance(h1ao_n, numpy.ndarray):
                de[k] += numpy.einsum('xij,ij->x', h1ao_n, dm_n[j]+mf.dm_nuc[j]) # dm_nuc: ground state gradients
            if isinstance(z1ao_n, numpy.ndarray):
                de[k] += numpy.einsum('xij,ij->x', z1ao_n, mf.dm_nuc[j])

        de[k] += numpy.einsum('xij,ij->x', h1ao_e, dmz1doo+mf.dm_elec)  # dm_elec: ground state gradients
        de[k] += numpy.einsum('xij,ij->x', z1ao_e, mf.dm_elec[p0:p1])

        de[k] += mf_grad_e.extra_force(ka, locals())

    log.timer('TDHF nuclear gradients', *time0)
    return de


def as_scanner(td_grad, state=1):
    '''
    Copied from grad.tdrhf.as_scanner
    '''
    from pyscf import neo
    if isinstance(td_grad, lib.GradScanner):
        return td_grad

    logger.info(td_grad, 'Create scanner for %s', td_grad.__class__)

    class CTDSCF_GradScanner(td_grad.__class__, lib.GradScanner):
        def __init__(self, g):
            lib.GradScanner.__init__(self, g)
            self._keys = self._keys.union(['e_tot'])
        def __call__(self, mol_or_geom, state=state, **kwargs):
            if isinstance(mol_or_geom, neo.Mole):
                mol = mol_or_geom
            else:
                mol = self.mol.set_geom_(mol_or_geom, inplace=False)
            self.reset(mol)

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

    if state == 0:
        return td_grad.base._scf.nuc_grad_method().as_scanner()
    else:
        return CTDSCF_GradScanner(td_grad)
    

class Gradients(tdrhf.Gradients):

    def __init__(self, td):
        tdrhf.Gradients.__init__(self, td)
        self.grid_response = None
        self._keys = self._keys.union(['grid_response'])

    @lib.with_doc(grad_elec.__doc__)
    def grad_elec(self, xy, singlet, atmlst=None):
        return grad_elec(self, xy, singlet, atmlst, self.max_memory, self.verbose)
    
    def grad_nuc(self, mol=None, atmlst=None):
        mf_grad = self.base._scf.mf_elec.nuc_grad_method()
        return mf_grad.grad_nuc(mol, atmlst)
    
    def optimizer(self, solver='geometric', state=1):
        '''Geometry optimization solver
            Copied from grad.rhf.GradientsMixin.optimizer() (add state)
        '''
        if solver.lower() == 'geometric':
            from pyscf.geomopt import geometric_solver
            return geometric_solver.GeometryOptimizer(self.as_scanner(state=state))
        elif solver.lower() == 'berny':
            from pyscf.geomopt import berny_solver
            return berny_solver.GeometryOptimizer(self.as_scanner(state=state))
        else:
            raise RuntimeError('Unknown geometry optimization solver %s' % solver)
    
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