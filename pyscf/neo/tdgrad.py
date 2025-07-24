from pyscf import neo, scf, gto
from pyscf.tdscf import TDDFT
from pyscf.neo import cphf
from pyscf.lib import logger
from pyscf import lib
from pyscf.scf.jk import get_jk
from pyscf.grad import tdrks, tdrhf, tduks
from functools import reduce
import numpy

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

def grad_elec_rhf(td_grad, x_y, singlet=True, atmlst=None,
              max_memory=2000, verbose=logger.INFO):

    log = logger.new_logger(td_grad, verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    mf = td_grad.base._scf
    if not isinstance(mf,neo.CDFT):
        raise TypeError('td grad is only supported for cneo')
    if mf.epc is not None:
        raise NotImplementedError('epc is not implemented in analytic td gradients')
    mol = td_grad.mol
    mf_e = mf.components['e']
    mol_e = mol.components['e']

    assert not isinstance(mf_e,scf.uhf.UHF)

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
        else:
            x = x.real
            y = y.real
    xpy = (x+y).reshape(nocc['e'],nvir['e']).T.real
    xmy = (x-y).reshape(nocc['e'],nvir['e']).T.real

    dvv = numpy.einsum('ai,bi->ab', xpy, xpy) + numpy.einsum('ai,bi->ab', xmy, xmy)
    doo =-numpy.einsum('ai,aj->ij', xpy, xpy) - numpy.einsum('ai,aj->ij', xmy, xmy)
    dmxpy = reduce(numpy.dot, (orbv['e'], xpy, orbo['e'].T))
    dmxmy = reduce(numpy.dot, (orbv['e'], xmy, orbo['e'].T))
    dmzoo = reduce(numpy.dot, (orbo['e'], doo, orbo['e'].T))
    dmzoo+= reduce(numpy.dot, (orbv['e'], dvv, orbv['e'].T))

    td_grad_e = (TDDFT(mf_e)).Gradients()

    if mf.xc_e.upper() == 'HF': # TDRHF
        vj, vk = mf_e.get_jk(mol, (dmzoo, dmxpy+dmxpy.T, dmxmy-dmxmy.T), hermi=0)
        veff0doo = vj[0] * 2 - vk[0]
        wvo_e = reduce(numpy.dot, (orbv['e'].T, veff0doo, orbo['e'])) * 2
        if singlet:
            veff = vj[1] * 2 - vk[1]
        else:
            veff = -vk[1]
        veff0mop = reduce(numpy.dot, (mo_coeff['e'].T, veff, mo_coeff['e']))
        wvo_e -= numpy.einsum('ki,ai->ak', veff0mop[:nocc['e'],:nocc['e']], xpy) * 2
        wvo_e += numpy.einsum('ac,ai->ci', veff0mop[nocc['e']:,nocc['e']:], xpy) * 2
        veff = -vk[2]
        veff0mom = reduce(numpy.dot, (mo_coeff['e'].T, veff, mo_coeff['e']))
        wvo_e -= numpy.einsum('ki,ai->ak', veff0mom[:nocc['e'],:nocc['e']], xmy) * 2
        wvo_e += numpy.einsum('ac,ai->ci', veff0mom[nocc['e']:,nocc['e']:], xmy) * 2

    else:   #TDRKS
        ni = mf_e._numint
        ni.libxc.test_deriv_order(mf_e.xc, 3, raise_error=True)
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf_e.xc, mol_e.spin)
        f1vo, f1oo, vxc1, k1ao = \
                tdrks._contract_xc_kernel(td_grad_e, mf_e.xc, dmxpy,
                                    dmzoo, True, True, singlet, max_memory)

        if ni.libxc.is_hybrid_xc(mf_e.xc):
            dm = (dmzoo, dmxpy+dmxpy.T, dmxmy-dmxmy.T)
            vj, vk = mf_e.get_jk(mol_e, dm, hermi=0)
            vk *= hyb
            if omega != 0:
                vk += mf_e.get_k(mol_e, dm, hermi=0, omega=omega) * (alpha-hyb)
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
            vj = mf_e.get_j(mol_e, (dmzoo, dmxpy+dmxpy.T), hermi=1)
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
    for t in mf.components.keys():
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

    z1, mo_e1, f1 = cphf.solve_nos1(fvind, mf.mo_energy, mo_occ, wvo,
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

    if mf.xc_e.upper() == 'HF':
        vj, vk = td_grad_e.get_jk(mol_e, (oo0, dmz1doo+dmz1doo.T, dmxpy+dmxpy.T,
                                  dmxmy-dmxmy.T))
        vj = vj.reshape(-1,3,nao_e,nao_e)
        vk = vk.reshape(-1,3,nao_e,nao_e)
        veff1 = -vk
        if singlet:
            veff1 += vj * 2
        else:
            veff1[:2] += vj[:2]*2

    else:

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

    log.timer('CNEO-TDRKS nuclear gradients', *time0)
    return de

def grad_elec_uhf(td_grad, x_y, atmlst=None, max_memory=2000, verbose=logger.INFO):

    log = logger.new_logger(td_grad, verbose)
    time0 = logger.process_clock(), logger.perf_counter()

    mol = td_grad.mol
    mf = td_grad.base._scf
    mf_e = mf.components['e']
    assert isinstance(mf_e, scf.uhf.UHF)
    mol_e = mol.components['e']
    td_grad_e = TDDFT(mf_e).Gradients()

    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    mo_occ = mf.mo_occ

    orbv = {}
    orbo = {}
    nocc = {}
    nvir = {}

    int1e_r_ao = {}
    int1e_r_vo = {}

    for t in mf.components.keys():
        mo_occ[t] = numpy.asarray(mo_occ[t])
        if mo_occ[t].ndim > 1: # unrestricted elec
            assert not t.startswith('n')
            assert mo_occ[t].shape[0] == 2
            occidxa = numpy.where(mo_occ[t][0]>0)[0]
            occidxb = numpy.where(mo_occ[t][1]>0)[0]
            viridxa = numpy.where(mo_occ[t][0]==0)[0]
            viridxb = numpy.where(mo_occ[t][1]==0)[0]
            nocca = len(occidxa)
            noccb = len(occidxb)
            nvira = len(viridxa)
            nvirb = len(viridxb)
            orboa = mo_coeff[t][0][:,occidxa]
            orbob = mo_coeff[t][1][:,occidxb]
            orbva = mo_coeff[t][0][:,viridxa]
            orbvb = mo_coeff[t][1][:,viridxb]
            nao_e = mo_coeff[t][0].shape[0]
            nmoa = nocca + nvira
            nmob = noccb + nvirb
            nocc[t] = (nocca, noccb)
            nvir[t] = (nvira, nvirb)
            orbv[t] = (orbva, orbvb)
            orbo[t] = (orboa, orbob)
        else:
            assert t.startswith('n')
            occidx = numpy.where(mo_occ[t] > 0)[0]
            viridx = numpy.where(mo_occ[t] == 0)[0]
            orbo[t] = mo_coeff[t][:,occidx]
            orbv[t] = mo_coeff[t][:,viridx]
            nocc[t] = len(occidx)
            nvir[t] = len(viridx)
            if t.startswith('n'):
                int1e_r_ao[t] = mf.components[t].int1e_r
                int1e_r_vo[t] = numpy.einsum('ja, ki, xjk->xai', orbv[t],orbo[t], int1e_r_ao[t])

    (xa, xb), (ya, yb) = x_y
    if xa.dtype == 'complex128':
        imag = numpy.max(numpy.abs(xa.imag))
        if imag > 1e-8:
            raise NotImplementedError('complex xy is not supported')
        else:
            xa = xa.real
            xb = xb.real
            ya = ya.real
            yb = yb.real
    xpya = (xa+ya).reshape(nocca,nvira).T
    xpyb = (xb+yb).reshape(noccb,nvirb).T
    xmya = (xa-ya).reshape(nocca,nvira).T
    xmyb = (xb-yb).reshape(noccb,nvirb).T

    dvva = numpy.einsum('ai,bi->ab', xpya, xpya) + numpy.einsum('ai,bi->ab', xmya, xmya)
    dvvb = numpy.einsum('ai,bi->ab', xpyb, xpyb) + numpy.einsum('ai,bi->ab', xmyb, xmyb)
    dooa =-numpy.einsum('ai,aj->ij', xpya, xpya) - numpy.einsum('ai,aj->ij', xmya, xmya)
    doob =-numpy.einsum('ai,aj->ij', xpyb, xpyb) - numpy.einsum('ai,aj->ij', xmyb, xmyb)
    dmxpya = reduce(numpy.dot, (orbva, xpya, orboa.T))
    dmxpyb = reduce(numpy.dot, (orbvb, xpyb, orbob.T))
    dmxmya = reduce(numpy.dot, (orbva, xmya, orboa.T))
    dmxmyb = reduce(numpy.dot, (orbvb, xmyb, orbob.T))
    dmzooa = reduce(numpy.dot, (orboa, dooa, orboa.T))
    dmzoob = reduce(numpy.dot, (orbob, doob, orbob.T))
    dmzooa+= reduce(numpy.dot, (orbva, dvva, orbva.T))
    dmzoob+= reduce(numpy.dot, (orbvb, dvvb, orbvb.T))

    if mf.xc_e.upper() == 'HF':
        vj, vk = mf_e.get_jk(mol_e, (dmzooa, dmxpya+dmxpya.T, dmxmya-dmxmya.T,
                             dmzoob, dmxpyb+dmxpyb.T, dmxmyb-dmxmyb.T), hermi=0)
        vj = vj.reshape(2,3,nao_e,nao_e)
        vk = vk.reshape(2,3,nao_e,nao_e)
        veff0doo = vj[0,0]+vj[1,0] - vk[:,0]
        wvoa = reduce(numpy.dot, (orbva.T, veff0doo[0], orboa)) * 2
        wvob = reduce(numpy.dot, (orbvb.T, veff0doo[1], orbob)) * 2
        veff = vj[0,1]+vj[1,1] - vk[:,1]
        veff0mopa = reduce(numpy.dot, (mo_coeff['e'][0].T, veff[0], mo_coeff['e'][0]))
        veff0mopb = reduce(numpy.dot, (mo_coeff['e'][1].T, veff[1], mo_coeff['e'][1]))
        wvoa -= numpy.einsum('ki,ai->ak', veff0mopa[:nocca,:nocca], xpya) * 2
        wvob -= numpy.einsum('ki,ai->ak', veff0mopb[:noccb,:noccb], xpyb) * 2
        wvoa += numpy.einsum('ac,ai->ci', veff0mopa[nocca:,nocca:], xpya) * 2
        wvob += numpy.einsum('ac,ai->ci', veff0mopb[noccb:,noccb:], xpyb) * 2
        veff = -vk[:,2]
        veff0moma = reduce(numpy.dot, (mo_coeff['e'][0].T, veff[0], mo_coeff['e'][0]))
        veff0momb = reduce(numpy.dot, (mo_coeff['e'][1].T, veff[1], mo_coeff['e'][1]))
        wvoa -= numpy.einsum('ki,ai->ak', veff0moma[:nocca,:nocca], xmya) * 2
        wvob -= numpy.einsum('ki,ai->ak', veff0momb[:noccb,:noccb], xmyb) * 2
        wvoa += numpy.einsum('ac,ai->ci', veff0moma[nocca:,nocca:], xmya) * 2
        wvob += numpy.einsum('ac,ai->ci', veff0momb[noccb:,noccb:], xmyb) * 2

    else:
        ni = mf_e._numint
        ni.libxc.test_deriv_order(mf_e.xc, 3, raise_error=True)
        omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf_e.xc, mol_e.spin)
        rho0, vxc, fxc = ni.cache_xc_kernel(mf_e.mol, mf_e.grids, mf_e.xc,
                                            mo_coeff['e'], mo_occ['e'], spin=1)
        f1vo, f1oo, vxc1, k1ao = \
                tduks._contract_xc_kernel(td_grad_e, mf_e.xc, (dmxpya,dmxpyb),
                                    (dmzooa,dmzoob), True, True, max_memory)

        if ni.libxc.is_hybrid_xc(mf_e.xc):
            dm = (dmzooa, dmxpya+dmxpya.T, dmxmya-dmxmya.T,
                dmzoob, dmxpyb+dmxpyb.T, dmxmyb-dmxmyb.T)
            vj, vk = mf_e.get_jk(mol_e, dm, hermi=0)
            vk *= hyb
            if omega != 0:
                vk += mf.get_k(mol_e, dm, hermi=0, omega=omega) * (alpha-hyb)
            vj = vj.reshape(2,3,nao_e,nao_e)
            vk = vk.reshape(2,3,nao_e,nao_e)

            veff0doo = vj[0,0]+vj[1,0] - vk[:,0] + f1oo[:,0] + k1ao[:,0] * 2
            wvoa = reduce(numpy.dot, (orbva.T, veff0doo[0], orboa)) * 2
            wvob = reduce(numpy.dot, (orbvb.T, veff0doo[1], orbob)) * 2
            veff = vj[0,1]+vj[1,1] - vk[:,1] + f1vo[:,0] * 2
            veff0mopa = reduce(numpy.dot, (mo_coeff['e'][0].T, veff[0], mo_coeff['e'][0]))
            veff0mopb = reduce(numpy.dot, (mo_coeff['e'][1].T, veff[1], mo_coeff['e'][1]))
            wvoa -= numpy.einsum('ki,ai->ak', veff0mopa[:nocca,:nocca], xpya) * 2
            wvob -= numpy.einsum('ki,ai->ak', veff0mopb[:noccb,:noccb], xpyb) * 2
            wvoa += numpy.einsum('ac,ai->ci', veff0mopa[nocca:,nocca:], xpya) * 2
            wvob += numpy.einsum('ac,ai->ci', veff0mopb[noccb:,noccb:], xpyb) * 2
            veff = -vk[:,2]
            veff0moma = reduce(numpy.dot, (mo_coeff['e'][0].T, veff[0], mo_coeff['e'][0]))
            veff0momb = reduce(numpy.dot, (mo_coeff['e'][1].T, veff[1], mo_coeff['e'][1]))
            wvoa -= numpy.einsum('ki,ai->ak', veff0moma[:nocca,:nocca], xmya) * 2
            wvob -= numpy.einsum('ki,ai->ak', veff0momb[:noccb,:noccb], xmyb) * 2
            wvoa += numpy.einsum('ac,ai->ci', veff0moma[nocca:,nocca:], xmya) * 2
            wvob += numpy.einsum('ac,ai->ci', veff0momb[noccb:,noccb:], xmyb) * 2
        else:
            dm = (dmzooa, dmxpya+dmxpya.T,
                dmzoob, dmxpyb+dmxpyb.T)
            vj = mf.get_j(mol_e, dm, hermi=1).reshape(2,2,nao_e,nao_e)

            veff0doo = vj[0,0]+vj[1,0] + f1oo[:,0] + k1ao[:,0] * 2
            wvoa = reduce(numpy.dot, (orbva.T, veff0doo[0], orboa)) * 2
            wvob = reduce(numpy.dot, (orbvb.T, veff0doo[1], orbob)) * 2
            veff = vj[0,1]+vj[1,1] + f1vo[:,0] * 2
            veff0mopa = reduce(numpy.dot, (mo_coeff['e'][0].T, veff[0], mo_coeff['e'][0]))
            veff0mopb = reduce(numpy.dot, (mo_coeff['e'][1].T, veff[1], mo_coeff['e'][1]))
            wvoa -= numpy.einsum('ki,ai->ak', veff0mopa[:nocca,:nocca], xpya) * 2
            wvob -= numpy.einsum('ki,ai->ak', veff0mopb[:noccb,:noccb], xpyb) * 2
            wvoa += numpy.einsum('ac,ai->ci', veff0mopa[nocca:,nocca:], xpya) * 2
            wvob += numpy.einsum('ac,ai->ci', veff0mopb[noccb:,noccb:], xpyb) * 2
            veff0moma = numpy.zeros((nmoa,nmoa))
            veff0momb = numpy.zeros((nmob,nmob))

    wvo = {}
    wvo['e'] = (wvoa, wvob)

    dm = {}
    for t in mf.components.keys():
        dm[t] = None
    dm['e'] = numpy.array([dmzooa, dmzoob])
    for t_pair, interaction in mf.interactions.items():
        p1, p2 = t_pair
        if p1.startswith('e'):
            vj_en = interaction.get_vint(dm)
            wvo_n = reduce(numpy.dot, (orbv[p2].T, vj_en[p2], orbo[p2]))
            wvo[p2] = wvo_n

    vresp = mf.gen_response(hermi=1)
    def fvind(mo1, f1):
        '''
        mo1['e']: (2Za, 2Zb)
        mo1['n']: (Zn)
        '''
        dm = {}
        dm1 = numpy.empty((2,nao_e,nao_e))
        x = mo1['e']
        xa = x[0,:nvira*nocca].reshape(nvira,nocca)
        xb = x[0,nvira*nocca:].reshape(nvirb,noccb)
        dma = reduce(numpy.dot, (orbva, xa, orboa.T))
        dmb = reduce(numpy.dot, (orbvb, xb, orbob.T))
        dm1[0] = dma + dma.T
        dm1[1] = dmb + dmb.T
        dm['e'] = dm1
        for t in mo1.keys():
            if t.startswith('n'):
                mo1[t] = mo1[t].reshape(nvir[t],nocc[t])
                dm_t = reduce(numpy.dot, (orbv[t], mo1[t]*2, orbo[t].T))
                dm[t] = dm_t + dm_t.T

        v1ao = vresp(dm)
        v1 = {}
        v1a = reduce(numpy.dot, (orbva.T, v1ao['e'][0], orboa))
        v1b = reduce(numpy.dot, (orbvb.T, v1ao['e'][1], orbob))
        v1['e'] = numpy.hstack((v1a.ravel(), v1b.ravel()))

        for t in f1.keys():
            f1[t] = f1[t].ravel()
        f_add = get_fock_add_cdft(f1, int1e_r_ao, fac=2.0)

        for t in v1ao.keys():
            if t.startswith('n'):
                v1ao[t] /= 2.0
                v1ao[t] += f_add[t]
                v1[t] = reduce(numpy.dot, (orbv[t].T, v1ao[t], orbo[t])).ravel()

        rfn = position_analysis(mo1, int1e_r_vo)

        return v1, rfn

    z1, mo_e1, f1 = cphf.solve_nos1(fvind, mo_energy, mo_occ, wvo,
                          with_f1=True,
                          max_cycle=td_grad.cphf_max_cycle,
                          tol=td_grad.cphf_conv_tol,
                          verbose=verbose)
    time1 = log.timer('Z-vector using UCPHF solver', *time0)

    dm_z = {}
    z1a, z1b = z1['e']
    z1a = z1a.reshape(nvira, nocca)
    z1b = z1b.reshape(nvirb, noccb)
    z1ao = numpy.empty((2,nao_e,nao_e))
    z1ao[0] = reduce(numpy.dot, (orbva, z1a, orboa.T))
    z1ao[1] = reduce(numpy.dot, (orbvb, z1b, orbob.T))
    dm_z['e'] = (z1ao+z1ao.transpose(0,2,1)) * .5
    for t in z1.keys():
        if t.startswith('n'):
            z1[t] = z1[t].reshape(nvir[t], nocc[t])
            z1ao_t = reduce(numpy.dot, (orbv[t], z1[t], orbo[t].T))
            dm_z[t] = z1ao_t + z1ao_t.T

    veff = vresp(dm_z)
    veff_e = veff['e']

    im0a = numpy.zeros((nmoa,nmoa))
    im0b = numpy.zeros((nmob,nmob))
    im0a[:nocca,:nocca] = reduce(numpy.dot, (orboa.T, veff0doo[0]+veff_e[0], orboa)) * .5
    im0b[:noccb,:noccb] = reduce(numpy.dot, (orbob.T, veff0doo[1]+veff_e[1], orbob)) * .5
    im0a[:nocca,:nocca]+= numpy.einsum('ak,ai->ki', veff0mopa[nocca:,:nocca], xpya) * .5
    im0b[:noccb,:noccb]+= numpy.einsum('ak,ai->ki', veff0mopb[noccb:,:noccb], xpyb) * .5
    im0a[:nocca,:nocca]+= numpy.einsum('ak,ai->ki', veff0moma[nocca:,:nocca], xmya) * .5
    im0b[:noccb,:noccb]+= numpy.einsum('ak,ai->ki', veff0momb[noccb:,:noccb], xmyb) * .5
    im0a[nocca:,nocca:] = numpy.einsum('ci,ai->ac', veff0mopa[nocca:,:nocca], xpya) * .5
    im0b[noccb:,noccb:] = numpy.einsum('ci,ai->ac', veff0mopb[noccb:,:noccb], xpyb) * .5
    im0a[nocca:,nocca:]+= numpy.einsum('ci,ai->ac', veff0moma[nocca:,:nocca], xmya) * .5
    im0b[noccb:,noccb:]+= numpy.einsum('ci,ai->ac', veff0momb[noccb:,:noccb], xmyb) * .5
    im0a[nocca:,:nocca] = numpy.einsum('ki,ai->ak', veff0mopa[:nocca,:nocca], xpya)
    im0b[noccb:,:noccb] = numpy.einsum('ki,ai->ak', veff0mopb[:noccb,:noccb], xpyb)
    im0a[nocca:,:nocca]+= numpy.einsum('ki,ai->ak', veff0moma[:nocca,:nocca], xmya)
    im0b[noccb:,:noccb]+= numpy.einsum('ki,ai->ak', veff0momb[:noccb,:noccb], xmyb)

    mo_energy_e = mo_energy['e']
    zeta_a = (mo_energy_e[0][:,None] + mo_energy_e[0]) * .5
    zeta_b = (mo_energy_e[1][:,None] + mo_energy_e[1]) * .5
    zeta_a[nocca:,:nocca] = mo_energy_e[0][:nocca]
    zeta_b[noccb:,:noccb] = mo_energy_e[1][:noccb]
    zeta_a[:nocca,nocca:] = mo_energy_e[0][nocca:]
    zeta_b[:noccb,noccb:] = mo_energy_e[1][noccb:]
    dm1a = numpy.zeros((nmoa,nmoa))
    dm1b = numpy.zeros((nmob,nmob))
    dm1a[:nocca,:nocca] = dooa * .5
    dm1b[:noccb,:noccb] = doob * .5
    dm1a[nocca:,nocca:] = dvva * .5
    dm1b[noccb:,noccb:] = dvvb * .5
    dm1a[nocca:,:nocca] = z1a * .5
    dm1b[noccb:,:noccb] = z1b * .5
    dm1a[:nocca,:nocca] += numpy.eye(nocca) # for ground state
    dm1b[:noccb,:noccb] += numpy.eye(noccb)
    im0a = reduce(numpy.dot, (mo_coeff['e'][0], im0a+zeta_a*dm1a, mo_coeff['e'][0].T))
    im0b = reduce(numpy.dot, (mo_coeff['e'][1], im0b+zeta_b*dm1b, mo_coeff['e'][1].T))
    im0 = im0a + im0b

    # Initialize hcore_deriv with the underlying SCF object because some
    # extensions (e.g. QM/MM, solvent) modifies the SCF object only.
    mf_grad = td_grad.base._scf.nuc_grad_method()
    mf_grad_e = mf_grad.components['e']
    hcore_deriv = {}
    for t, comp in mf_grad.components.items():
        hcore_deriv[t] = comp.hcore_generator(mol.components[t])
    s1 = mf_grad_e.get_ovlp(mol_e)

    dmz1dooa = z1ao[0] + dmzooa
    dmz1doob = z1ao[1] + dmzoob
    oo0a = reduce(numpy.dot, (orboa, orboa.T))
    oo0b = reduce(numpy.dot, (orbob, orbob.T))
    as_dm1 = oo0a + oo0b + (dmz1dooa + dmz1doob) * .5

    if mf.xc_e.upper() == 'HF':
        vj, vk = td_grad.get_jk(mol, (oo0a, dmz1dooa+dmz1dooa.T, dmxpya+dmxpya.T, dmxmya-dmxmya.T,
                                  oo0b, dmz1doob+dmz1doob.T, dmxpyb+dmxpyb.T, dmxmyb-dmxmyb.T))
        vj = vj.reshape(2,4,3,nao_e,nao_e)
        vk = vk.reshape(2,4,3,nao_e,nao_e)
        veff1a, veff1b = vj[0] + vj[1] - vk

    else:
        if ni.libxc.is_hybrid_xc(mf_e.xc):
            dm = (oo0a, dmz1dooa+dmz1dooa.T, dmxpya+dmxpya.T, dmxmya-dmxmya.T,
                oo0b, dmz1doob+dmz1doob.T, dmxpyb+dmxpyb.T, dmxmyb-dmxmyb.T)
            vj, vk = td_grad_e.get_jk(mol_e, dm)
            vj = vj.reshape(2,4,3,nao_e,nao_e)
            vk = vk.reshape(2,4,3,nao_e,nao_e) * hyb
            if omega != 0:
                vk += td_grad.get_k(mol, dm, omega=omega).reshape(2,4,3,nao_e,nao_e) * (alpha-hyb)
            veff1 = vj[0] + vj[1] - vk
        else:
            dm = (oo0a, dmz1dooa+dmz1dooa.T, dmxpya+dmxpya.T,
                oo0b, dmz1doob+dmz1doob.T, dmxpyb+dmxpyb.T)
            vj = td_grad.get_j(mol, dm).reshape(2,3,3,nao_e,nao_e)
            veff1 = numpy.zeros((2,4,3,nao_e,nao_e))
            veff1[:,:3] = vj[0] + vj[1]

        fxcz1 = tduks._contract_xc_kernel(td_grad_e, mf_e.xc, z1ao, None,
                                    False, False, max_memory)[0]

        veff1[:,0] += vxc1[:,1:]
        veff1[:,1] +=(f1oo[:,1:] + fxcz1[:,1:] + k1ao[:,1:]*2)*2 # *2 for dmz1doo+dmz1oo.T
        veff1[:,2] += f1vo[:,1:] * 2
        veff1a, veff1b = veff1

    time1 = log.timer('2e AO integral derivatives', *time1)

    if atmlst is None:
        atmlst = range(mol.natm)
    offsetdic = mol_e.offset_nr_by_atom()
    de = numpy.zeros((len(atmlst),3))
    dm_gs = mf.make_rdm1()
    for k, ka in enumerate(atmlst):
        shl0, shl1, p0, p1 = offsetdic[ka]

        # Ground state gradients
        h1ao_e = hcore_deriv['e'](ka)

        de[k] += numpy.einsum('xpq,pq->x', veff1a[0,:,p0:p1], oo0a[p0:p1])
        de[k] += numpy.einsum('xpq,pq->x', veff1b[0,:,p0:p1], oo0b[p0:p1])
        de[k] += numpy.einsum('xpq,qp->x', veff1a[0,:,p0:p1], oo0a[:,p0:p1])
        de[k] += numpy.einsum('xpq,qp->x', veff1b[0,:,p0:p1], oo0b[:,p0:p1])

        de[k] += numpy.einsum('xpq,pq->x', veff1a[0,:,p0:p1], dmz1dooa[p0:p1]) * .5
        de[k] += numpy.einsum('xpq,pq->x', veff1b[0,:,p0:p1], dmz1doob[p0:p1]) * .5
        de[k] += numpy.einsum('xpq,qp->x', veff1a[0,:,p0:p1], dmz1dooa[:,p0:p1]) * .5
        de[k] += numpy.einsum('xpq,qp->x', veff1b[0,:,p0:p1], dmz1doob[:,p0:p1]) * .5

        de[k] -= numpy.einsum('xpq,pq->x', s1[:,p0:p1], im0[p0:p1])
        de[k] -= numpy.einsum('xqp,pq->x', s1[:,p0:p1], im0[:,p0:p1])

        de[k] += numpy.einsum('xij,ij->x', veff1a[1,:,p0:p1], oo0a[p0:p1]) * .5
        de[k] += numpy.einsum('xij,ij->x', veff1b[1,:,p0:p1], oo0b[p0:p1]) * .5
        de[k] += numpy.einsum('xij,ij->x', veff1a[2,:,p0:p1], dmxpya[p0:p1,:])
        de[k] += numpy.einsum('xij,ij->x', veff1b[2,:,p0:p1], dmxpyb[p0:p1,:])
        de[k] += numpy.einsum('xij,ij->x', veff1a[3,:,p0:p1], dmxmya[p0:p1,:])
        de[k] += numpy.einsum('xij,ij->x', veff1b[3,:,p0:p1], dmxmyb[p0:p1,:])
        de[k] += numpy.einsum('xji,ij->x', veff1a[2,:,p0:p1], dmxpya[:,p0:p1])
        de[k] += numpy.einsum('xji,ij->x', veff1b[2,:,p0:p1], dmxpyb[:,p0:p1])
        de[k] -= numpy.einsum('xji,ij->x', veff1a[3,:,p0:p1], dmxmya[:,p0:p1])
        de[k] -= numpy.einsum('xji,ij->x', veff1b[3,:,p0:p1], dmxmyb[:,p0:p1])

        z1ao_e = 0.0
        dm_e = dm_gs['e'][0] + dm_gs['e'][1]

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
                                (dm_e, dmz1dooa+dmz1doob), scripts=['ijkl,lk->ij','ijkl,lk->ij'],
                                intor='int2e_ip1', aosym='s2kl', comp=3)
                    v1ne = [_v * charge for _v in v1ne]
                    h1ao_n += v1ne[0] + v1ne[0].transpose(0,2,1)
                    z1ao_n += v1ne[1]

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

        de[k] += numpy.einsum('xpq,pq->x', h1ao_e, as_dm1)
        de[k] += numpy.einsum('xij,ij->x', z1ao_e, dm_e[p0:p1])

        de[k] += td_grad.extra_force(ka, locals())

    log.timer('CNEO-TDUKS nuclear gradients', *time0)
    return de


def as_scanner(td_grad, state=1):
    '''Generating a nuclear gradients scanner/solver (for geometry optimizer).

    The returned solver is a function. This function requires one argument
    "mol" as input and returns energy and first order nuclear derivatives.

    Modified from grad.tdrhf.as_scanner
    '''
    if isinstance(td_grad, lib.GradScanner):
        return td_grad

    if state == 0:
        return td_grad.base._scf.nuc_grad_method().as_scanner()

    logger.info(td_grad, 'Create scanner for %s', td_grad.__class__)
    name = td_grad.__class__.__name__ + CTDSCF_GradScanner.__name_mixin__
    return lib.set_class(CTDSCF_GradScanner(td_grad, state),
                         (CTDSCF_GradScanner, td_grad.__class__), name)

class CTDSCF_GradScanner(lib.GradScanner):
    ''' Modified from grad.tdrhf.TDSCF_GradScanner

    The only change: assert mol_or_geom.__class__ == neo.Mole (was gto.Mole)
    '''
    _keys = {'e_tot'}

    def __init__(self, g, state):
        lib.GradScanner.__init__(self, g)
        if state is not None:
            self.state = state

    def __call__(self, mol_or_geom, state=None, **kwargs):
        if isinstance(mol_or_geom, gto.MoleBase):
            assert mol_or_geom.__class__ == neo.Mole
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
# TODO: Check root flip.  Maybe avoid the initial guess in TDHF otherwise
# large error may be found in the excited states amplitudes
        de = self.kernel(state=state, **kwargs)
        e_tot = self.e_tot[state-1]
        return e_tot, de

    @property
    def converged(self):
        td_scanner = self.base
        return all((td_scanner._scf.converged,
                    td_scanner.converged[self.state]))


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

    def grad_elec(self, xy, singlet, atmlst=None):
        '''Electronic and quantum nuclear part of CNEO-TDDFT nuclear gradients'''

        mf = self.base._scf
        if isinstance(mf.components['e'], scf.uhf.UHF):
            return grad_elec_uhf(self, xy, atmlst, self.max_memory, self.verbose)
        else:
            return grad_elec_rhf(self, xy, singlet, atmlst, self.max_memory, self.verbose)

    as_scanner = as_scanner

Grad = Gradients
neo.ctddft.CTDDirect.Gradients = neo.ctddft.CTDDFT.Gradients = lib.class_as_method(Gradients)
