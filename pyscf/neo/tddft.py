import numpy
from pyscf.tdscf import rhf, uhf
from pyscf.dft.numint import eval_ao, eval_rho
from pyscf.dft.gen_grid import make_mask
from pyscf import lib
from pyscf import scf
from pyscf import __config__
from pyscf.lib import logger
from pyscf.data import nist

REAL_EIG_THRESHOLD = getattr(__config__, 'tdscf_rhf_TDDFT_pick_eig_threshold', 1e-4)

def eval_fxc(epc, rho_e, rho_p):
    epc_type = None
    if isinstance(epc, str):
        epc_type = epc
    elif isinstance(epc, dict):
        if "epc_type" not in epc:
            epc_type = '17-2'
        else:
            epc_type = epc["epc_type"]
    else:
        raise TypeError('Only string or dictionary is allowed for epc')

    if epc_type == '17-1':
        a = 2.35
        b = 2.4
        c = 3.2
    elif epc_type == '17-2':
        a = 2.35
        b = 2.4
        c = 6.6
    elif epc_type == '18-1':
        a = 1.8
        b = 0.1
        c = 0.03
    elif epc_type == '18-2':
        a = 3.9
        b = 0.5
        c = 0.06
    elif epc_type == '17' or epc_type == '18':
        a = epc["a"]
        b = epc["b"]
        c = epc["c"]
    else:
        raise ValueError('Unsupported type of epc %s', epc_type)
    
    if epc_type.startswith('17'):
        rho_product = numpy.multiply(rho_e, rho_p)
        denominator = 4 * numpy.sqrt(rho_product) * numpy.power(a+c*rho_product-b*numpy.sqrt(rho_product), 3) 
        idx = numpy.where(denominator==0)
        denominator[idx] = 1.

        numerator_common = -3*a*b - 3*b*c*rho_product + b**2*numpy.sqrt(rho_product) + 8*a*c*numpy.sqrt(rho_product)

        ee_numerator = numpy.multiply(numpy.square(rho_p) , numerator_common)
        pp_numerator = numpy.multiply(numpy.square(rho_e) , numerator_common)
        ep_numerator = -4*a**2*numpy.sqrt(rho_product) - b*numpy.multiply(rho_product, c*rho_product + b*numpy.sqrt(rho_product)) + a*numpy.multiply(rho_product, 3*b + 4*c*numpy.sqrt(rho_product))
        
        f_ee = numpy.multiply(ee_numerator, 1 / denominator)
        f_pp = numpy.multiply(pp_numerator, 1 / denominator)
        f_ep = numpy.multiply(ep_numerator, 1 / denominator)

        f_ee[idx] = 0.
        f_pp[idx] = 0.
        # f_ep[idx] = -1.0 / a
        f_ep[idx] = 0.

    elif epc_type.startswith('18'):
        raise NotImplementedError('%s', epc_type)
    else:
        raise ValueError('Unsupported type of epc %s', epc_type)
    
    return f_ee, f_pp, f_ep

def init_guess(mf, nstates):
    mf_elec = mf.mf_elec
    unrestricted = mf.unrestricted
    if unrestricted:
        x0_e = uhf.TDA(mf_elec).init_guess(mf_elec, nstates=nstates)
    else:
        x0_e = rhf.TDA(mf_elec).init_guess(mf_elec, nstates=nstates)
    y0_e = numpy.zeros_like(x0_e)
    x0 = numpy.hstack((x0_e, y0_e))
    x1 = numpy.hstack((y0_e, x0_e))
    for i in range(mf.mol.nuc_num):
        mf_nuc = mf.mf_nuc[i]
        nocc_p = mf_nuc.mo_coeff[:,mf_nuc.mo_occ>0].shape[1]
        nvir_p = mf_nuc.mo_coeff.shape[1] - nocc_p
        x0_p = numpy.zeros((x0_e.shape[0],nocc_p*nvir_p))
        y0_p = numpy.zeros_like(x0_p)
        x0 = numpy.hstack((x0, x0_p, y0_p))
        x1 = numpy.hstack((x1, x0_p, y0_p))

    return numpy.asarray(numpy.vstack((x0,x1)))

# def elec_dm_e_response(mf_elec, singlet = True):
#     return mf_elec.gen_response(singlet=singlet, hermi=0)
    
def nuc_dm_n_response_epc(mf_nuc):
    '''
    Self interaction for one nucleus
    '''
    nao = mf_nuc.mol.nao_nr()
    def vind(dms_nuc):
        ndm = dms_nuc.shape[0]
        vmat = numpy.zeros((ndm, nao, nao))

        return vmat
    return vind

def nuc_dm_e_response(mf,i):
    '''
    Nuclear Coulomb matrix from electronic density matrix
    '''
    def vind(dms_elec):
        vj = mf.get_j_n_dm_e(i, dms_elec)
        if dms_elec.shape[0] == 1:
            nao_v = vj.shape[-1]
            vj = vj.reshape((-1,nao_v,nao_v))

        return vj 
    
    return vind

def elec_dm_n_response(mf,i):
    '''
    Electronic Coulomb matrix from nuclear density matrix
    '''
    def vind(dms_nuc):
        vj = mf.get_j_e_dm_n(i,dms_nuc)
        if dms_nuc.shape[0] == 1:
            nao_v = vj.shape[-1]
            vj = vj.reshape((-1,nao_v,nao_v))

        return vj
             
    return vind

def dm_n_dm_n_response(mf,i,j):
    '''
    Nuclear i Coulomb matrix from nuclear density matrix j
    '''
    nao = mf.mol.nuc[i].nao_nr()
    def vind(dmj_nuc):
        vj = mf.get_j_nn(i,j,dmj_nuc)
        if dmj_nuc.shape[0] == 1:
            nao_v = vj.shape[-1]
            vj = vj.reshape((-1,nao_v,nao_v))
        return vj
    
    return vind

def get_epc_iajb_rhf(mf, reshape=False):

    mf_elec = mf.mf_elec
    mo_coeff_e = mf_elec.mo_coeff
    mo_occ_e = mf_elec.mo_occ
    occidx_e = numpy.where(mo_occ_e==2)[0]
    viridx_e = numpy.where(mo_occ_e==0)[0]
    nocc_e = len(occidx_e)
    nvir_e = len(viridx_e)
    orbv_e = mo_coeff_e[:,viridx_e]
    orbo_e = mo_coeff_e[:,occidx_e]

    nao = mf.mol.elec.nao_nr()
    grids = mf.mf_elec.grids
    ni = mf.mf_elec._numint
    
    iajb_e = numpy.zeros((nocc_e, nvir_e, nocc_e, nvir_e))

    nuc_num = mf.mol.nuc_num
    nocc_p = []
    nvir_p = []
    orbv_p = []
    orbo_p = []
    iajb_pe = []
    # iajb_ep = []
    iajb_pp = []
    iajb_p = []

    for i in range(nuc_num):
        mf_nuc = mf.mf_nuc[i]
        mo_coeff_p = mf_nuc.mo_coeff
        mo_energy_p = mf_nuc.mo_energy
        mo_occ_p = mf_nuc.mo_occ
        occidx_p = numpy.where(mo_occ_p==1)[0]
        viridx_p = numpy.where(mo_occ_p==0)[0]
        nocc_p.append(len(occidx_p))
        nvir_p.append(len(viridx_p))
        orbv_p.append(mo_coeff_p[:,viridx_p])
        orbo_p.append(mo_coeff_p[:,occidx_p])
        iajb_pp.append([None]*nuc_num)
        iajb_pe.append(numpy.zeros((nocc_p[i], nvir_p[i], nocc_e, nvir_e)))
        # iajb_ep.append(numpy.zeros((nocc_e, nvir_e, nocc_p[i], nvir_p[i])))
        iajb_p.append(numpy.zeros((nocc_p[i], nvir_p[i], nocc_p[i], nvir_p[i])))

    for i in range(nuc_num):
        for j in range(i+1, nuc_num):
            iajb_pp[i][j] = numpy.zeros((nocc_p[i], nvir_p[i], nocc_p[j], nvir_p[j]))

    for ao, mask, weight, coords in ni.block_loop(mf.mol.elec,grids,nao):
        
        ao_elec = ao
        rho_e = eval_rho(mf.mol.elec, ao_elec, mf.dm_elec)
        w_ov_ps = []
        rho_ov_ps = []

        for i in range(nuc_num):
            ao_nuc = eval_ao(mf.mol.nuc[i], coords)
            rho_p = eval_rho(mf.mol.nuc[i], ao_nuc, mf.dm_nuc[i])
            rho_p[rho_p<0.] = 0.
            f_ee, f_pp, f_ep = eval_fxc(mf.epc,rho_e, rho_p)

            rho_o_p = lib.einsum('rp,pi->ri', ao_nuc, orbo_p[i])
            rho_v_p = lib.einsum('rp,pi->ri', ao_nuc, orbv_p[i])
            rho_ov_p = numpy.einsum('ri,ra->ria', rho_o_p, rho_v_p)
            rho_ov_ps.append(rho_ov_p)
            rho_o_e = lib.einsum('rp,pi->ri', ao_elec, orbo_e)
            rho_v_e = lib.einsum('rp,pi->ri', ao_elec, orbv_e)
            rho_ov_e = numpy.einsum('ri,ra->ria', rho_o_e, rho_v_e)

            w_ov_pe = numpy.einsum('ria,r->ria', rho_ov_e, f_ep*weight)
            iajb_pe[i] += lib.einsum('ria,rjb->iajb', rho_ov_p, w_ov_pe)
            
            w_ov_p = numpy.einsum('ria,r->ria', rho_ov_p, f_pp*weight)
            iajb_p[i] += lib.einsum('ria,rjb->iajb', rho_ov_p, w_ov_p)
            w_ov_ps.append(w_ov_p)

            w_ov_e = numpy.einsum('ria,r->ria', rho_ov_e, f_ee*weight)
            iajb_e += lib.einsum('ria,rjb->iajb', rho_ov_e, w_ov_e) * 2

        for i in range(nuc_num):
            for j in range(i+1, nuc_num):
                iajb_pp[i][j] += lib.einsum('ria,rjb->iajb', rho_ov_ps[i], w_ov_ps[j])
    
    # for i in range(nuc_num):
    #     iajb_ep[i] = iajb_pe[i].transpose(2,3,0,1)

    if reshape:
        for i in range(nuc_num):
            iajb_p[i] = iajb_p[i].reshape((nocc_p[i]*nvir_p[i], nocc_p[i]*nvir_p[i]))
            iajb_pe[i] = iajb_pe[i].reshape((nocc_p[i]*nvir_p[i], nocc_e*nvir_e))
            for j in range(i+1, nuc_num):
                iajb_pp[i][j] = iajb_pp[i][j].reshape((nocc_p[i]*nvir_p[i], nocc_p[j]*nvir_p[j]))
        iajb_e = iajb_e.reshape((nocc_e*nvir_e, nocc_e*nvir_e))


    return iajb_e, iajb_pe, iajb_p

def get_epc_iajb_uhf(mf, reshape=False):
    mf_elec = mf.mf_elec

    mo_coeff_e = mf_elec.mo_coeff
    mo_occ_e = mf_elec.mo_occ
    occidxa = numpy.where(mo_occ_e[0]>0)[0]
    occidxb = numpy.where(mo_occ_e[1]>0)[0]
    viridxa = numpy.where(mo_occ_e[0]==0)[0]
    viridxb = numpy.where(mo_occ_e[1]==0)[0]
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)
    orboa = mo_coeff_e[0][:,occidxa]
    orbob = mo_coeff_e[1][:,occidxb]
    orbva = mo_coeff_e[0][:,viridxa]
    orbvb = mo_coeff_e[1][:,viridxb]

    nao = mf.mol.elec.nao_nr()
    grids = mf.mf_elec.grids
    ni = mf.mf_elec._numint

    iajb_aa = numpy.zeros((nocca, nvira, nocca, nvira))
    iajb_ab = numpy.zeros((nocca, nvira, noccb, nvirb))
    iajb_bb = numpy.zeros((noccb, nvirb, noccb, nvirb))
    
    nuc_num = mf.mol.nuc_num
    nocc_p = []
    nvir_p = []
    orbv_p = []
    orbo_p = []
    iajb_pe_a = []
    iajb_pe_b = []
    iajb_p = []
    
    for i in range(nuc_num):
        mf_nuc = mf.mf_nuc[i]
        mo_coeff_p = mf_nuc.mo_coeff
        mo_occ_p = mf_nuc.mo_occ
        occidx_p = numpy.where(mo_occ_p==1)[0]
        viridx_p = numpy.where(mo_occ_p==0)[0]
        nocc_p.append(len(occidx_p))
        nvir_p.append(len(viridx_p))
        orbv_p.append(mo_coeff_p[:,viridx_p])
        orbo_p.append(mo_coeff_p[:,occidx_p])
        iajb_pe_a.append(numpy.zeros((nocc_p[i], nvir_p[i], nocca, nvira)))
        iajb_pe_b.append(numpy.zeros((nocc_p[i], nvir_p[i], noccb, nvirb)))
        iajb_p.append(numpy.zeros((nocc_p[i], nvir_p[i], nocc_p[i], nvir_p[i])))

    for ao, mask, weight, coords in ni.block_loop(mf.mol.elec,grids,nao):
        
        ao_elec = ao
        rho_e = eval_rho(mf.mol.elec, ao_elec, mf.dm_elec)

        for i in range(nuc_num):
            ao_nuc = eval_ao(mf.mol.nuc[i], coords)
            rho_p = eval_rho(mf.mol.nuc[i], ao_nuc, mf.dm_nuc[i])
            rho_p[rho_p<0.] = 0.
            f_ee, f_pp, f_ep = eval_fxc(mf.epc, rho_e, rho_p)

            rho_o_p = lib.einsum('rp,pi->ri', ao_nuc, orbo_p[i])
            rho_v_p = lib.einsum('rp,pi->ri', ao_nuc, orbv_p[i])
            rho_ov_p = numpy.einsum('ri,ra->ria', rho_o_p, rho_v_p)
            rho_o_a = lib.einsum('rp,pi->ri', ao_elec, orboa)
            rho_v_a = lib.einsum('rp,pi->ri', ao_elec, orbva)
            rho_ov_a = numpy.einsum('ri,ra->ria', rho_o_a, rho_v_a)
            rho_o_b = lib.einsum('rp,pi->ri', ao_elec, orbob)
            rho_v_b = lib.einsum('rp,pi->ri', ao_elec, orbvb)
            rho_ov_b = numpy.einsum('ri,ra->ria', rho_o_b, rho_v_b)

            w_ov_pe_a = numpy.einsum('ria,r->ria', rho_ov_a, f_ep*weight)
            iajb_pe_a[i] += lib.einsum('ria,rjb->iajb', rho_ov_p, w_ov_pe_a)

            w_ov_pe_b = numpy.einsum('ria,r->ria', rho_ov_b, f_ep*weight)
            iajb_pe_b[i] += lib.einsum('ria,rjb->iajb', rho_ov_p, w_ov_pe_b)
            
            w_ov_p = numpy.einsum('ria,r->ria', rho_ov_p, f_pp*weight)
            iajb_p[i] += lib.einsum('ria,rjb->iajb', rho_ov_p, w_ov_p)

            w_ov_a = numpy.einsum('ria,r->ria', rho_ov_a, f_ee*weight)
            iajb_aa += lib.einsum('ria,rjb->iajb', rho_ov_a, w_ov_a)

            w_ov_b = numpy.einsum('ria,r->ria', rho_ov_b, f_ee*weight)
            iajb_bb += lib.einsum('ria,rjb->iajb', rho_ov_b, w_ov_b)
            iajb_ab += lib.einsum('ria,rjb->iajb', rho_ov_a, w_ov_b)

    if reshape:
        for i in range(nuc_num):
            iajb_pe_a[i] = iajb_pe_a[i].reshape((nocc_p[i]*nvir_p[i], nocca*nvira))
            iajb_pe_b[i] = iajb_pe_b[i].reshape((nocc_p[i]*nvir_p[i], noccb*nvirb))
            iajb_p[i] = iajb_p[i].reshape((nocc_p[i]*nvir_p[i], nocc_p[i]*nvir_p[i]))

    return iajb_aa, iajb_bb, iajb_ab, iajb_pe_a, iajb_pe_b, iajb_p

def get_tdrhf_add_epc(xs_e, ys_e, xs_ps, ys_ps, iajb_e, iajb_p, iajb_ep, iajb_pe):
        xys_e = xs_e + ys_e
        abcc_epc = numpy.einsum('iajb,njb->nia',iajb_e,xys_e)
        ccab_epc = []
        nuc_num = len(iajb_ep)
        for i in range(nuc_num):
            xys_p = xs_ps[i] + ys_ps[i]
            ccab_epc.append(numpy.einsum('iajb,njb->nia',iajb_p[i],xys_p))
            ccab_epc[i] += numpy.sqrt(2)*numpy.einsum('iajb,njb->nia',iajb_pe[i],xys_e)
            abcc_epc += numpy.sqrt(2)*numpy.einsum('iajb,njb->nia',iajb_ep[i],xys_p)

        return abcc_epc,ccab_epc

def get_tduhf_add_epc(xa, xb, ya, yb, x_ps, y_ps,
                      iajb_aa, iajb_bb, iajb_ab, iajb_ba, 
                      iajb_ep_a, iajb_pe_a, iajb_ep_b, iajb_pe_b, iajb_p):
    xya = xa + ya
    xyb = xb + yb

    abcc_epc_a = numpy.einsum('iajb,njb->nia',iajb_aa, xya)
    abcc_epc_a += numpy.einsum('iajb,njb->nia',iajb_ab, xyb)

    abcc_epc_b = numpy.einsum('iajb,njb->nia', iajb_ba, xya)
    abcc_epc_b += numpy.einsum('iajb,njb->nia', iajb_bb, xyb)

    ccab_epc = []
    nuc_num = len(iajb_ep_a)
    for i in range(nuc_num):
        xys_p = x_ps[i] + y_ps[i]

        ccab_epc.append(numpy.einsum('iajb,njb->nia', iajb_pe_a[i], xya))
        ccab_epc[i] += numpy.einsum('iajb,njb->nia', iajb_pe_b[i], xyb)
        ccab_epc[i] += numpy.einsum('iajb,njb->nia', iajb_p[i], xys_p)

        abcc_epc_a += numpy.einsum('iajb,njb->nia', iajb_ep_a[i], xys_p)
        abcc_epc_b += numpy.einsum('iajb,njb->nia', iajb_ep_b[i], xys_p)


    return abcc_epc_a, abcc_epc_b, ccab_epc

def get_tdrhf_operation(mf, singlet=True):

    mf_elec = mf.mf_elec
    mo_coeff_e = mf_elec.mo_coeff
    assert (mo_coeff_e.dtype == numpy.double)
    mo_energy_e = mf_elec.mo_energy
    mo_occ_e = mf_elec.mo_occ
    occidx_e = numpy.where(mo_occ_e==2)[0]
    viridx_e = numpy.where(mo_occ_e==0)[0]
    nocc_e = len(occidx_e)
    nvir_e = len(viridx_e)
    orbv_e = mo_coeff_e[:,viridx_e]
    orbo_e = mo_coeff_e[:,occidx_e]
    foo_e = numpy.diag(mo_energy_e[occidx_e])
    fvv_e = numpy.diag(mo_energy_e[viridx_e])
    hdiag_e = fvv_e.diagonal() - foo_e.diagonal()[:,None]
    hdiag = numpy.hstack((hdiag_e.ravel(), -hdiag_e.ravel()))

    nuc_num = mf.mol.nuc_num
    nocc_p = []
    nvir_p = []
    orbv_p = []
    orbo_p = []
    foo_p = []
    fvv_p = []
    vresp_e_dm_n = []
    vresp_n_dm_e = []
    vresp_n_n = []
    vresp_n_dm_n = []

    for i in range(nuc_num):
        mf_nuc = mf.mf_nuc[i]
        mo_coeff_p = mf_nuc.mo_coeff
        assert (mo_coeff_p.dtype == numpy.double)
        mo_energy_p = mf_nuc.mo_energy
        mo_occ_p = mf_nuc.mo_occ
        occidx_p = numpy.where(mo_occ_p==1)[0]
        viridx_p = numpy.where(mo_occ_p==0)[0]
        nocc_p.append(len(occidx_p))
        nvir_p.append(len(viridx_p))
        orbv_p.append(mo_coeff_p[:,viridx_p])
        orbo_p.append(mo_coeff_p[:,occidx_p])
        foo_p.append(numpy.diag(mo_energy_p[occidx_p]))
        fvv_p.append(numpy.diag(mo_energy_p[viridx_p]))

        vresp_e_dm_n.append(elec_dm_n_response(mf,i))
        vresp_n_dm_e.append(nuc_dm_e_response(mf,i))
        vresp_n_n.append([None]*nuc_num)
        vresp_n_dm_n.append(nuc_dm_n_response_epc(mf_nuc))
        for j in range(nuc_num):
            vresp_n_n[i][j] = dm_n_dm_n_response(mf,i,j)
        hdiag_p = fvv_p[-1].diagonal() - foo_p[-1].diagonal()[:,None]
        hdiag = numpy.hstack((hdiag, hdiag_p.ravel(), -hdiag_p.ravel()))
    
    vresp_e = mf_elec.gen_response(singlet=singlet, hermi=0)

    epc = mf.epc
    if epc is not None:
        iajb_e, iajb_pe, iajb_p = get_epc_iajb_rhf(mf,reshape=False)
        iajb_ep = []
        for i in range(nuc_num):
            iajb_ep.append(iajb_pe[i].transpose(2,3,0,1))

    def add_epc(xs_e, ys_e, xs_ps, ys_ps):
        xys_e = xs_e + ys_e
        abcc_epc = numpy.einsum('iajb,njb->nia',iajb_e,xys_e)
        ccab_epc = []
        for i in range(nuc_num):
            xys_p = xs_ps[i] + ys_ps[i]
            ccab_epc.append(numpy.einsum('iajb,njb->nia',iajb_p[i],xys_p))
            ccab_epc[i] += numpy.sqrt(2)*numpy.einsum('iajb,njb->nia',iajb_pe[i],xys_e)
            abcc_epc += numpy.sqrt(2)*numpy.einsum('iajb,njb->nia',iajb_ep[i],xys_p)

        return abcc_epc,ccab_epc

    def vind_elec_elec(xs_e, ys_e, dms_elec):
    
        v1ao = vresp_e(dms_elec)
        v1ov = lib.einsum('xpq,po,qv->xov', v1ao, orbo_e.conj(), orbv_e)
        v1vo = lib.einsum('xpq,qo,pv->xov', v1ao, orbo_e, orbv_e.conj())
        v1ov += lib.einsum('xqs,sp->xqp', xs_e, fvv_e)  # AX
        v1ov -= lib.einsum('xpr,sp->xsr', xs_e, foo_e)  # AX
        v1vo += lib.einsum('xqs,sp->xqp', ys_e, fvv_e)  # AY
        v1vo -= lib.einsum('xpr,sp->xsr', ys_e, foo_e)  # AY

        return v1ov, v1vo
    
    def vind_elec_nuc(i, dms_nuc):

        v1ao_elec_dmn = vresp_e_dm_n[i](dms_nuc)
        v1ov_ep = numpy.sqrt(2)*lib.einsum('xpq,po,qv->xov', v1ao_elec_dmn, orbo_e.conj(), orbv_e)

        return v1ov_ep
    
    def vind_nuc_elec(i, dms_elec):

        v1ao_nuc_dme = vresp_n_dm_e[i](dms_elec)
        v1ov_pe = 0.5*numpy.sqrt(2)*lib.einsum('xpq,po,qv->xov', v1ao_nuc_dme, orbo_p[i].conj(), orbv_p[i])

        return v1ov_pe
    
    def vind_nuc(xs_p, ys_p, i, dm_nuc):

        v1ao_nuc = vresp_n_dm_n[i](dm_nuc)
        v1ov_nuc = lib.einsum('xpq,po,qv->xov', v1ao_nuc, orbo_p[i].conj(), orbv_p[i])
        v1vo_nuc = lib.einsum('xpq,qo,pv->xov', v1ao_nuc, orbo_p[i], orbv_p[i].conj())

        v1ov_nuc += lib.einsum('xqs,sp->xqp', xs_p, fvv_p[i])
        v1ov_nuc -= lib.einsum('xpr,sp->xsr', xs_p, foo_p[i])
        v1vo_nuc += lib.einsum('xqs,sp->xqp', ys_p, fvv_p[i])
        v1vo_nuc -= lib.einsum('xpr,sp->xsr', ys_p, foo_p[i])

        return v1ov_nuc, v1vo_nuc
    
    def vind_nuc1_nuc2(i,j,dms_nuc2):
        v1ao_dm1_dm2 = vresp_n_n[i][j](dms_nuc2)
        v1ov_pp = lib.einsum('xpq,po,qv->xov', v1ao_dm1_dm2, orbo_p[i].conj(), orbv_p[i])
        return v1ov_pp
    
    def vind(xys):
        xys = numpy.asarray(xys)
        nz = xys.shape[0]
        xys_e = xys[:,:2*(nocc_e*nvir_e)]
        xys_ps = xys[:,2*(nocc_e*nvir_e):]
        
        xys_e = xys_e.reshape(-1,2,nocc_e,nvir_e)
        xs_e, ys_e = xys_e.transpose(1,0,2,3)
        dms_elec  = lib.einsum('xov,qv,po->xpq', xs_e*2, orbv_e.conj(), orbo_e)
        dms_elec += lib.einsum('xov,pv,qo->xpq', ys_e*2, orbv_e, orbo_e.conj())
        
        dms_nuc = []
        xs_ps = []
        ys_ps = []
        for i in range(nuc_num):
            nocci = nocc_p[i]
            nviri = nvir_p[i]
            xys_p = xys_ps[:,:2*(nocci*nviri)]
            xys_ps = xys_ps[:,2*(nocci*nviri):]
            xys_p = xys_p.reshape(-1,2,nocci,nviri)
            xs_pi, ys_pi = xys_p.transpose(1,0,2,3)
            dms_nuci = lib.einsum('xov,qv,po->xpq', xs_pi, orbv_p[i].conj(), orbo_p[i])
            dms_nuci += lib.einsum('xov,qv,po->xpq', ys_pi, orbv_p[i].conj(), orbo_p[i])
            dms_nuc.append(dms_nuci)
            xs_ps.append(xs_pi)
            ys_ps.append(ys_pi)

        abcc, bacc = vind_elec_elec(xs_e, ys_e, dms_elec)    # AeXe+BeYe; BeXe+AeYe
        if epc is not None:
            abcc_epc,ccab_epc = get_tdrhf_add_epc(xs_e, ys_e, xs_ps, ys_ps, iajb_e, iajb_p, iajb_ep, iajb_pe)
            abcc += abcc_epc
            bacc += abcc_epc
            
        for i in range(nuc_num):
            v1ov_nuci, v1vo_nuci = vind_nuc(xs_ps[i], ys_ps[i], i, dms_nuc[i])
            v1ov_ep = vind_elec_nuc(i,dms_nuc[i])
            abcc += v1ov_ep
            bacc += v1ov_ep
            
            v1ov_pe = vind_nuc_elec(i,dms_elec)
            ccabi = v1ov_pe + v1ov_nuci
            ccbai = v1ov_pe + v1vo_nuci
            if epc is not None:
                ccabi += ccab_epc[i]
                ccbai += ccab_epc[i]
            for j in range(nuc_num):
                if j!=i:
                    v1ov_pp = vind_nuc1_nuc2(i,j,dms_nuc[j])
                    ccabi += v1ov_pp
                    ccbai += v1ov_pp
            
            ccabi = ccabi.reshape(nz,-1)
            ccbai = ccbai.reshape(nz,-1)        
            if i==0:
                hx_nuc = numpy.hstack((ccabi, -ccbai))
            else:
                hx_nuc = numpy.hstack((hx_nuc, ccabi, -ccbai))
                
        abcc = abcc.reshape(nz,-1)
        bacc = bacc.reshape(nz,-1)
        hx = numpy.hstack((abcc, -bacc, hx_nuc))
        return hx
    
    return vind, hdiag

def get_tduhf_operation(mf):

    mf_elec = mf.mf_elec

    mo_coeff_e = mf_elec.mo_coeff
    mo_energy_e = mf_elec.mo_energy
    mo_occ_e = mf_elec.mo_occ
    occidxa = numpy.where(mo_occ_e[0]>0)[0]
    occidxb = numpy.where(mo_occ_e[1]>0)[0]
    viridxa = numpy.where(mo_occ_e[0]==0)[0]
    viridxb = numpy.where(mo_occ_e[1]==0)[0]
    nocca = len(occidxa)
    noccb = len(occidxb)
    nvira = len(viridxa)
    nvirb = len(viridxb)
    orboa = mo_coeff_e[0][:,occidxa]
    orbob = mo_coeff_e[1][:,occidxb]
    orbva = mo_coeff_e[0][:,viridxa]
    orbvb = mo_coeff_e[1][:,viridxb]
    e_ia_a = mo_energy_e[0][viridxa] - mo_energy_e[0][occidxa,None]
    e_ia_b = mo_energy_e[1][viridxb] - mo_energy_e[1][occidxb,None]
    e_ia = hdiag_e = numpy.hstack((e_ia_a.ravel(), e_ia_b.ravel()))
    hdiag = numpy.hstack((hdiag_e, -hdiag_e))
    vresp_e = mf_elec.gen_response(hermi=0)
    
    nuc_num = mf.mol.nuc_num
    nocc_p = []
    nvir_p = []
    orbv_p = []
    orbo_p = []
    foo_p = []
    fvv_p = []
    vresp_e_dm_n = []
    vresp_n_dm_e = []
    vresp_n_n = []
    for i in range(nuc_num):
        mf_nuc = mf.mf_nuc[i]
        mo_coeff_p = mf_nuc.mo_coeff
        mo_energy_p = mf_nuc.mo_energy
        mo_occ_p = mf_nuc.mo_occ
        occidx_p = numpy.where(mo_occ_p==1)[0]
        viridx_p = numpy.where(mo_occ_p==0)[0]
        nocc_p.append(len(occidx_p))
        nvir_p.append(len(viridx_p))
        orbv_p.append(mo_coeff_p[:,viridx_p])
        orbo_p.append(mo_coeff_p[:,occidx_p])
        foo_p.append(numpy.diag(mo_energy_p[occidx_p]))
        fvv_p.append(numpy.diag(mo_energy_p[viridx_p]))
        vresp_e_dm_n.append(elec_dm_n_response(mf,i))
        vresp_n_dm_e.append(nuc_dm_e_response(mf,i))
        vresp_n_n.append([None]*nuc_num)
        for j in range(nuc_num):
            vresp_n_n[i][j] = dm_n_dm_n_response(mf,i,j)
        hdiag_p = fvv_p[-1].diagonal() - foo_p[-1].diagonal()[:,None]
        hdiag = numpy.hstack((hdiag, hdiag_p.ravel(), -hdiag_p.ravel()))

    epc = mf.epc

    if epc is not None:
        iajb_aa, iajb_bb, iajb_ab, iajb_pe_a, iajb_pe_b, iajb_p = get_epc_iajb_uhf(mf, reshape=False)
        iajb_ba = iajb_ab.transpose(2,3,0,1)
        iajb_ep_a = []
        iajb_ep_b = []
        for i in range(len(iajb_pe_a)):
            iajb_ep_a.append(iajb_pe_a[i].transpose(2,3,0,1))
            iajb_ep_b.append(iajb_pe_b[i].transpose(2,3,0,1))

    def vind_elec_elec(dmsa, dmsb):

        v1ao = vresp_e(numpy.asarray((dmsa,dmsb)))

        v1aov = lib.einsum('xpq,po,qv->xov', v1ao[0], orboa.conj(), orbva)
        v1avo = lib.einsum('xpq,qo,pv->xov', v1ao[0], orboa, orbva.conj())
        v1bov = lib.einsum('xpq,po,qv->xov', v1ao[1], orbob.conj(), orbvb)
        v1bvo = lib.einsum('xpq,qo,pv->xov', v1ao[1], orbob, orbvb.conj())

        return v1aov, v1bov, v1avo, v1bvo
    
    def vind_elec_nuc(i, dms_nuc):
        v1ao_elec_dmn = vresp_e_dm_n[i](dms_nuc)
        v1aov_ep = lib.einsum('xpq,po,qv->xov', v1ao_elec_dmn, orboa.conj(), orbva)
        v1bov_ep = lib.einsum('xpq,po,qv->xov', v1ao_elec_dmn, orbob.conj(), orbvb)

        return v1aov_ep, v1bov_ep
    
    def vind_nuc_elec(i, dms_elec):

        v1ao_nuc_dme = vresp_n_dm_e[i](dms_elec)
        v1ov_pe = lib.einsum('xpq,po,qv->xov', v1ao_nuc_dme, orbo_p[i].conj(), orbv_p[i])

        return v1ov_pe
    
    def vind_nuc(xs_p, ys_p, i):

        v1ov_nuc = lib.einsum('xqs,sp->xqp', xs_p, fvv_p[i])
        v1ov_nuc -= lib.einsum('xpr,sp->xsr', xs_p, foo_p[i])

        v1vo_nuc = lib.einsum('xqs,sp->xqp', ys_p, fvv_p[i])
        v1vo_nuc -= lib.einsum('xpr,sp->xsr', ys_p, foo_p[i])

        return v1ov_nuc, v1vo_nuc
    
    def vind_nuc1_nuc2(i,j,dms_nuc2):
        v1ao_dm1_dm2 = vresp_n_n[i][j](dms_nuc2)
        v1ov_pp = lib.einsum('xpq,po,qv->xov', v1ao_dm1_dm2, orbo_p[i].conj(), orbv_p[i])
        return v1ov_pp
    
    def vind(xys):
        xys = numpy.asarray(xys)
        nz = xys.shape[0]
        xys_e = xys[:,:2*(nocca*nvira+noccb*nvirb)]
        xys_ps = xys[:,2*(nocca*nvira+noccb*nvirb):]

        xs_e, ys_e = xys_e.reshape(nz,2,-1).transpose(1,0,2)
        xa = xs_e[:,:nocca*nvira].reshape(nz,nocca,nvira)
        xb = xs_e[:,nocca*nvira:].reshape(nz,noccb,nvirb)
        ya = ys_e[:,:nocca*nvira].reshape(nz,nocca,nvira)
        yb = ys_e[:,nocca*nvira:].reshape(nz,noccb,nvirb)
        # dms = AX + BY
        dmsa  = lib.einsum('xov,qv,po->xpq', xa, orbva.conj(), orboa)
        dmsb  = lib.einsum('xov,qv,po->xpq', xb, orbvb.conj(), orbob)
        dmsa += lib.einsum('xov,pv,qo->xpq', ya, orbva, orboa.conj())
        dmsb += lib.einsum('xov,pv,qo->xpq', yb, orbvb, orbob.conj())
        dms_elec = dmsa + dmsb
        v1aov, v1bov, v1avo, v1bvo = vind_elec_elec(dmsa, dmsb)

        dms_nuc = []
        v1ov_nuc = []
        v1vo_nuc = []
        xs_ps = []
        ys_ps = []
        for i in range(nuc_num):
            nocci = nocc_p[i]
            nviri = nvir_p[i]
            xys_p = xys_ps[:,:2*(nocci*nviri)]
            xys_ps = xys_ps[:,2*(nocci*nviri):]
            xys_p = xys_p.reshape(-1,2,nocci,nviri)
            xs_pi, ys_pi = xys_p.transpose(1,0,2,3)
            dms_nuci = lib.einsum('xov,qv,po->xpq', xs_pi, orbv_p[i].conj(), orbo_p[i])
            dms_nuci += lib.einsum('xov,qv,po->xpq', ys_pi, orbv_p[i].conj(), orbo_p[i])
            v1ov_nuci, v1vo_nuci = vind_nuc(xs_pi, ys_pi, i)
            
            dms_nuc.append(dms_nuci)
            v1ov_nuc.append(v1ov_nuci)
            v1vo_nuc.append(v1vo_nuci)
            xs_ps.append(xs_pi)
            ys_ps.append(ys_pi)

        if epc is not None:
            abcc_epc_a, abcc_epc_b, ccab_epc = get_tduhf_add_epc(xa, xb, ya, yb, xs_ps, ys_ps,
                                                                 iajb_aa, iajb_bb, iajb_ab, iajb_ba,
                                                                 iajb_ep_a, iajb_pe_a, iajb_ep_b, iajb_pe_b, iajb_p)
            v1aov += abcc_epc_a
            v1avo += abcc_epc_a
            v1bov += abcc_epc_b
            v1bvo += abcc_epc_b

        for i in range(nuc_num):
            v1aov_ep, v1bov_ep = vind_elec_nuc(i, dms_nuc[i])
            v1aov += v1aov_ep
            v1bov += v1bov_ep
            v1avo += v1aov_ep
            v1bvo += v1bov_ep

            v1ov_pe = vind_nuc_elec(i,dms_elec)
            ccabi = v1ov_pe + v1ov_nuc[i]
            ccbai = v1ov_pe + v1vo_nuc[i]
            if epc is not None:
                ccabi += ccab_epc[i]
                ccbai += ccab_epc[i]
            for j in range(nuc_num):
                if j!=i:
                    v1ov_pp = vind_nuc1_nuc2(i,j,dms_nuc[j])
                    ccabi += v1ov_pp
                    ccbai += v1ov_pp
            
            ccabi = ccabi.reshape(nz,-1)
            ccbai = ccbai.reshape(nz,-1)        
            if i==0:
                hx_nuc = numpy.hstack((ccabi, -ccbai))
            else:
                hx_nuc = numpy.hstack((hx_nuc, ccabi, -ccbai))

        abcc = xs_e * e_ia
        bacc = ys_e * e_ia
        abcc[:,:nocca*nvira] += v1aov.reshape(nz,-1)
        abcc[:,nocca*nvira:] += v1bov.reshape(nz,-1)        
        bacc[:,:nocca*nvira] += v1avo.reshape(nz,-1)
        bacc[:,nocca*nvira:] += v1bvo.reshape(nz,-1)

        hx = numpy.hstack((abcc, -bacc, hx_nuc))
        return hx
    return vind, hdiag

def pickeig(w, v, nroots, envs):
    realidx = numpy.where((abs(w.imag) < 1e-4) &
                            (w.real > 1e-3))[0]
    # If the complex eigenvalue has small imaginary part, both the
    # real part and the imaginary part of the eigenvector can
    # approximately be used as the "real" eigen solutions.
    return lib.linalg_helper._eigs_cmplx2real(w, v, realidx,
                                                real_eigenvectors=True)

def remove_linear_dep(mf, threshold = 1e-7):
    mf.mf_elec = scf.addons.remove_linear_dep(mf.mf_elec, threshold=threshold)
    for i in range(mf.mol.nuc_num):
        mf.mf_nuc[i] = scf.addons.remove_linear_dep(mf.mf_nuc[i], threshold=threshold)

class TDDFT(lib.StreamObject):
    conv_tol = getattr(__config__, 'tdscf_rhf_TDA_conv_tol', 1e-9)
    nstates = getattr(__config__, 'tdscf_rhf_TDA_nstates', 3)
    singlet = getattr(__config__, 'tdscf_rhf_TDA_singlet', True)
    lindep = getattr(__config__, 'tdscf_rhf_TDA_lindep', 1e-12)
    level_shift = getattr(__config__, 'tdscf_rhf_TDA_level_shift', 0)
    max_space = getattr(__config__, 'tdscf_rhf_TDA_max_space', 50)
    max_cycle = getattr(__config__, 'tdscf_rhf_TDA_max_cycle', 100)
    # Low excitation filter to avoid numerical instability
    positive_eig_threshold = getattr(__config__, 'tdscf_rhf_TDDFT_positive_eig_threshold', 1e-3)

    def __init__(self, mf):
        self.verbose = mf.verbose
        self.stdout = mf.mf_elec.stdout
        self.mol = mf.mol
        self._scf = mf
        self.max_memory = mf.max_memory
        self.chkfile = mf.chkfile
        self.unrestricted = mf.unrestricted
        if self.unrestricted:
            self.singlet = None

        self.wfnsym = None

        # xy = (X,Y), normalized to 1/2: 2(XX-YY) = 1
        # In TDA, Y = 0
        self.converged = None
        self.e = None
        self.x1 = None

        keys = set(('conv_tol', 'nstates', 'singlet', 'lindep', 'level_shift',
                    'max_space', 'max_cycle'))
        self._keys = set(self.__dict__.keys()).union(keys)

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
    
    def get_precond(self, hdiag):
        def precond(x, e, x0):
            diagd = hdiag - (e-self.level_shift)
            diagd[abs(diagd)<1e-8] = 1e-8
            return x/diagd
        return precond
    
    def gen_vind(self, mf=None):
        if mf is None:
            mf = self._scf
        if self.unrestricted:
            # self.singlet = None
            return get_tduhf_operation(mf)
        else:
            # singlet = mf.mf_elec.singlet
            return get_tdrhf_operation(mf, singlet=self.singlet)
        
    def init_guess(self, mf, nstates=None):
        return init_guess(mf, nstates)
    
    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        if not all(self.converged):
            logger.note(self, 'TD-SCF states %s not converged.',
                        [i for i, x in enumerate(self.converged) if not x])
        logger.note(self, 'Excited State energies (eV)\n%s', self.e * nist.HARTREE2EV)
        return self
    
    def kernel(self, x0=None, nstates=None):
        cpu0 = (logger.process_clock(), logger.perf_counter())
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates

        if x0 is None:
            x0 = self.init_guess(self._scf, self.nstates)
        
        log = logger.Logger(self.stdout, self.verbose)
        vind, hdiag = self.gen_vind(self._scf)
        precond = self.get_precond(hdiag)

        def pickeig(w, v, nroots, envs):
            realidx = numpy.where((abs(w.imag) < REAL_EIG_THRESHOLD) &
                                  (w.real > self.positive_eig_threshold))[0]
            return lib.linalg_helper._eigs_cmplx2real(w, v, realidx,
                                                      real_eigenvectors=True)
        
        self.converged, w, x1 = \
                lib.davidson_nosym1(vind, x0, precond,
                                    tol=self.conv_tol,
                                    nroots=nstates, lindep=self.lindep,
                                    max_cycle=self.max_cycle,
                                    max_space=self.max_space, pick=pickeig,
                                    verbose=log)
        
        self.e = w
        self.x1 = x1

        log.timer('TDDFT', *cpu0)
        self._finalize()
        return self.e, self.x1