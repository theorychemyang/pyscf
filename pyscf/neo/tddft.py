import numpy
from pyscf.tdscf import rhf, uhf
from pyscf.dft.numint import eval_ao, eval_rho
from pyscf import lib
from pyscf import scf
from pyscf import __config__
from pyscf.lib import logger
from pyscf.data import nist
from pyscf import neo, scf

REAL_EIG_THRESHOLD = getattr(__config__, 'tdscf_rhf_TDDFT_pick_eig_threshold', 1e-4)

def eval_fxc(epc, rho_e, rho_p):
    '''
    Evaluate seccond-order derivatives of a epc functional
    '''
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
        f_ep[idx] = 0.

    elif epc_type.startswith('18'):
        raise NotImplementedError('%s', epc_type)
    else:
        raise ValueError('Unsupported type of epc %s', epc_type)
    
    return f_ee, f_pp, f_ep

def init_guess(mf, nstates):
    mol = mf.mol
    mf_elec = mf.components['e']
    unrestricted = isinstance(mf_elec,scf.uhf.UHF)
    if unrestricted:
        x0_e = uhf.TDA(mf_elec).init_guess(mf_elec, nstates=nstates)
    else:
        x0_e = rhf.TDA(mf_elec).init_guess(mf_elec, nstates=nstates)
    y0_e = numpy.zeros_like(x0_e)
    x0 = numpy.hstack((x0_e, y0_e))
    x1 = numpy.hstack((y0_e, x0_e))
    for i in range(mol.natm):
        if mol._quantum_nuc[i]:
            mf_nuc = mf.components[f'n{i}']
            nocc_p = mf_nuc.mo_coeff[:,mf_nuc.mo_occ>0].shape[1]
            nvir_p = mf_nuc.mo_coeff.shape[1] - nocc_p
            x0_p = numpy.zeros((x0_e.shape[0],nocc_p*nvir_p))
            y0_p = numpy.zeros_like(x0_p)
            x0 = numpy.hstack((x0, x0_p, y0_p))
            x1 = numpy.hstack((x1, x0_p, y0_p))

    return numpy.asarray(numpy.vstack((x0,x1)))
    
def get_epc_iajb_rhf(mf, reshape=False):

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    orbv = {}
    orbo = {}
    nocc = {}
    nvir = {}


    for t in mf.components.keys():
        occidx = numpy.where(mo_occ[t] > 0)[0]
        viridx = numpy.where(mo_occ[t] == 0)[0]
        orbo[t] = mo_coeff[t][:,occidx]
        orbv[t] = mo_coeff[t][:,viridx]
        nocc[t] = len(occidx)
        nvir[t] = len(viridx)

    nao = mo_coeff['e'].shape[0]
    grids = mf.components['e'].grids
    ni = mf.components['e']._numint

    iajb = {}
    iajb_int = {}
    for t in mf.components.keys():
        iajb[t] = numpy.zeros((nocc[t], nvir[t], nocc[t], nvir[t]))

    for t_pair in mf.interactions.keys():
        t1, t2 = t_pair
        iajb_int[t_pair] = numpy.zeros((nocc[t1], nvir[t1], nocc[t2], nvir[t2]))

    dm = mf.make_rdm1()
    for _ao, mask, weight, coords in ni.block_loop(mf.mol.components['e'],grids,nao):

        rho = {}
        rho_ov = {}

        for t in mf.components.keys():
            if t.startswith('e'):
                ao = _ao
            else:
                ao = eval_ao(mf.mol.components[t], coords)

            _rho = eval_rho(mf.mol.components[t], ao, dm[t])
            if t.startswith('n'):
                _rho[_rho<0.] = 0.
            rho[t] = _rho
            rho_o = lib.einsum('rp,pi->ri', ao, orbo[t])
            rho_v = lib.einsum('rp,pi->ri', ao, orbv[t])
            rho_ov[t] = numpy.einsum('ri,ra->ria', rho_o, rho_v)

        for (t1, t2) in mf.interactions.keys():
            if t1.startswith('e'):
                f_ee, f_pp, f_ep = eval_fxc(mf.epc, rho[t1], rho[t2])
                w_ov_ep = numpy.einsum('ria,r->ria', rho_ov[t2], f_ep*weight)
                w_ov_p = numpy.einsum('ria,r->ria', rho_ov[t2], f_pp*weight)
                w_ov_e = numpy.einsum('ria,r->ria', rho_ov[t1], f_ee*weight)

                iajb[t1] += lib.einsum('ria,rjb->iajb', rho_ov[t1], w_ov_e) * 2
                iajb[t2] += lib.einsum('ria,rjb->iajb', rho_ov[t2], w_ov_p)
                iajb_int[(t1, t2)] += lib.einsum('ria,rjb->iajb', rho_ov[t1], w_ov_ep)
    

    if reshape:
        for t in iajb.keys():
            iajb[t] = iajb[t].reshape((nocc[t]*nvir[t], nocc[t]*nvir[t]))
        for t_pair in iajb_int.keys():
            t1, t2 = t_pair
            iajb_int[t_pair] = iajb_int[t_pair].reshape((nocc[t1]*nvir[t1], nocc[t2]*nvir[t2]))

    return iajb, iajb_int

def get_epc_iajb_uhf(mf, reshape=False):
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    orbv = {}
    orbo = {}
    nocc = {}
    nvir = {}

    assert isinstance(mf.components['e'], scf.uhf.UHF)


    for t in mf.components.keys():
        mo_occ[t] = numpy.asarray(mo_occ[t])
        if t.startswith('e'):
            occidxa = numpy.where(mo_occ[t][0] > 0)[0]
            occidxb = numpy.where(mo_occ[t][1] > 0)[0]
            viridxa = numpy.where(mo_occ[t][0] == 0)[0]
            viridxb = numpy.where(mo_occ[t][1] == 0)[0]
            nocc[t] = [len(occidxa), len(occidxb)]
            nvir[t] = [len(viridxa), len(viridxb)]
            orbo[t] = [mo_coeff[t][0][:,occidxa], mo_coeff[t][1][:,occidxb]]
            orbv[t] = [mo_coeff[t][0][:,viridxa], mo_coeff[t][1][:,viridxb]]
        else:       
            occidx = numpy.where(mo_occ[t] > 0)[0]
            viridx = numpy.where(mo_occ[t] == 0)[0]
            orbo[t] = mo_coeff[t][:,occidx]
            orbv[t] = mo_coeff[t][:,viridx]
            nocc[t] = len(occidx)
            nvir[t] = len(viridx)

    nao = mf.mol.components['e'].nao_nr()
    grids = mf.components['e'].grids
    ni = mf.components['e']._numint

    iajb = {}
    iajb_int = {}

    for t in mf.components.keys():
        if t.startswith('e'):
            aa = numpy.zeros((nocc[t][0], nvir[t][0], nocc[t][0], nvir[t][0]))
            ab = numpy.zeros((nocc[t][0], nvir[t][0], nocc[t][1], nvir[t][1]))
            bb = numpy.zeros((nocc[t][1], nvir[t][1], nocc[t][1], nvir[t][1]))
            iajb[t] = [aa, ab, bb]
        else:
            iajb[t] = numpy.zeros((nocc[t], nvir[t], nocc[t], nvir[t]))

    for t_pair in mf.interactions.keys():
        t1, t2 = t_pair
        if t1.startswith('e'):
            ep_a = numpy.zeros((nocc[t1][0], nvir[t1][0], nocc[t2], nvir[t2]))
            ep_b = numpy.zeros((nocc[t1][1], nvir[t1][1], nocc[t2], nvir[t2]))
            iajb_int[t_pair] = [ep_a, ep_b]
    
    dm = mf.make_rdm1()
    dm['e'] = dm['e'][0] + dm['e'][1]
    for _ao, mask, weight, coords in ni.block_loop(mf.mol.components['e'],grids,nao):

        rho = {}
        rho_ov = {}

        for t in mf.components.keys():
            if t.startswith('e'):
                ao = _ao
            else:
                ao = eval_ao(mf.mol.components[t], coords)

            _rho = eval_rho(mf.mol.components[t], ao, dm[t])
            if t.startswith('n'):
                _rho[_rho<0.] = 0.
            rho[t] = _rho

            if t.startswith('e'):
                rho_o_a = lib.einsum('rp,pi->ri', ao, orbo[t][0])
                rho_v_a = lib.einsum('rp,pi->ri', ao, orbv[t][0])
                rho_o_b = lib.einsum('rp,pi->ri', ao, orbo[t][1])
                rho_v_b = lib.einsum('rp,pi->ri', ao, orbv[t][1])
                rho_ov[t] = [numpy.einsum('ri,ra->ria', rho_o_a, rho_v_a),\
                            numpy.einsum('ri,ra->ria', rho_o_b, rho_v_b)]
            else:
                rho_o = lib.einsum('rp,pi->ri', ao, orbo[t])
                rho_v = lib.einsum('rp,pi->ri', ao, orbv[t])
                rho_ov[t] = numpy.einsum('ri,ra->ria', rho_o, rho_v)

        for (t1, t2) in mf.interactions.keys():
            if t1.startswith('e'):
                f_ee, f_pp, f_ep = eval_fxc(mf.epc, rho[t1], rho[t2])
                w_ov_ep = numpy.einsum('ria,r->ria', rho_ov[t2], f_ep*weight)
                w_ov_p = numpy.einsum('ria,r->ria', rho_ov[t2], f_pp*weight)
                w_ov_a = numpy.einsum('ria,r->ria', rho_ov[t1][0], f_ee*weight)
                w_ov_b = numpy.einsum('ria,r->ria', rho_ov[t1][1], f_ee*weight)

                iajb[t1][0] += lib.einsum('ria,rjb->iajb', rho_ov[t1][0], w_ov_a)
                iajb[t1][1] += lib.einsum('ria,rjb->iajb', rho_ov[t1][0], w_ov_b)
                iajb[t1][2] += lib.einsum('ria,rjb->iajb', rho_ov[t1][1], w_ov_b)
                iajb[t2] += lib.einsum('ria,rjb->iajb', rho_ov[t2], w_ov_p)
                iajb_int[(t1, t2)][0] += lib.einsum('ria,rjb->iajb', rho_ov[t1][0], w_ov_ep)
                iajb_int[(t1, t2)][1] += lib.einsum('ria,rjb->iajb', rho_ov[t1][1], w_ov_ep)
        
    if reshape:
        for t, comp in iajb.items():
            if t.startswith('n'):
                iajb[t] = iajb[t].reshape((nocc[t]*nvir[t], nocc[t]*nvir[t]))
        for t_pair, comp in iajb_int.items():
            t1, t2 = t_pair
            assert t1.startswith('e')
            for i in range(len(comp)):
                iajb_int[t_pair][i] = comp[i].reshape((nocc[t1][i]*nvir[t1][i], nocc[t2]*nvir[t2]))

    return iajb, iajb_int

def get_tdrhf_add_epc(xs, ys, iajb, iajb_int):
    epc = {}
    xys = {}
    for t in iajb.keys():
        xys[t] = xs[t] + ys[t]
        epc[t] = numpy.einsum('iajb,njb->nia',iajb[t],xys[t])

    for (t1,t2), comp in iajb_int.items():
        if t1.startswith('e'):
            epc[t1] += numpy.einsum('iajb,njb->nia',comp,xys[t2]) * numpy.sqrt(2)
            _comp = comp.transpose(2,3,0,1)
            epc[t2] += numpy.einsum('iajb,njb->nia',_comp,xys[t1]) * numpy.sqrt(2)

    return epc

def get_tduhf_add_epc(xs, ys, iajb, iajb_int):
    epc = {}
    xys = {}
    for t in iajb.keys():
        if t.startswith('e'):
            xys[t] = [xs[t][0] + ys[t][0], xs[t][1] + ys[t][1]]
            epca = numpy.einsum('iajb,njb->nia',iajb[t][0], xys[t][0])
            epca += numpy.einsum('iajb,njb->nia',iajb[t][1], xys[t][1])
            _iajb = iajb[t][1].transpose(2,3,0,1)
            epcb = numpy.einsum('iajb,njb->nia', _iajb, xys[t][0])
            epcb += numpy.einsum('iajb,njb->nia', iajb[t][1], xys[t][1])
            epc[t] = [epca, epcb]
        else:
            xys[t] = xs[t] + ys[t]
            epc[t] = numpy.einsum('iajb,njb->nia', iajb[t], xys[t])

    for (t1, t2), comp in iajb_int.items():
        if t1.startswith('e'):
            epc[t1][0] += numpy.einsum('iajb,njb->nia', comp[0], xys[t2])
            epc[t1][1] += numpy.einsum('iajb,njb->nia', comp[1], xys[t2])
            _comp = comp[0].transpose(2,3,0,1)
            epc[t2] += numpy.einsum('iajb,njb->nia', _comp, xys[t1][0])
            _comp = comp[1].transpose(2,3,0,1)
            epc[t2] += numpy.einsum('iajb,njb->nia', _comp, xys[t1][1])

    return epc

def gen_tdrhf_operation(mf):
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    mo_energy = mf.mo_energy
    orbv = {}
    orbo = {}
    nocc = {}
    nvir = {}
    nov = {}
    foo = {}
    fvv = {}
    hdiag = []

    has_epc = False
    if isinstance(mf, neo.KS):
        if mf.epc is not None:
            has_epc = True

    for t in mf.components.keys():
        occidx = numpy.where(mo_occ[t] > 0)[0]
        viridx = numpy.where(mo_occ[t] == 0)[0]
        orbo[t] = mo_coeff[t][:,occidx]
        orbv[t] = mo_coeff[t][:,viridx]
        nocc[t] = len(occidx)
        nvir[t] = len(viridx)
        nov[t] = nocc[t]*nvir[t]
        foo[t] = numpy.diag(mo_energy[t][occidx])
        fvv[t] = numpy.diag(mo_energy[t][viridx])
        _hdiag = fvv[t].diagonal() - foo[t].diagonal()[:,None]
        hdiag.append(numpy.hstack((_hdiag.ravel(),-_hdiag.ravel())))

    vresp = mf.gen_response(hermi=0, no_epc=True)
    if has_epc:
        iajb, iajb_int = get_epc_iajb_rhf(mf, reshape=False)

    def vind(xys):
        xys = numpy.asarray(xys)
        nz, tot_size = xys.shape
        xs = {}
        ys = {}
        dms = {}
        offset = 0
        for t in mf.components.keys():
            xy = xys[:,offset:offset+2*nov[t]]
            offset += 2*nov[t]
            xy = xy.reshape(-1,2,nocc[t],nvir[t])
            xs[t], ys[t] = xy.transpose(1,0,2,3)
            dms[t] = lib.einsum('xov,pv,qo->xpq', xs[t], orbv[t], orbo[t].conj())
            dms[t] += lib.einsum('xov,qv,po->xpq', ys[t], orbv[t].conj(), orbo[t])

            if t.startswith('e'):
                dms[t] *= numpy.sqrt(2)

        assert (offset == tot_size)

        v1ao = vresp(dms)
        v1ao['e'] = v1ao['e'] * numpy.sqrt(2)
        if has_epc:
            epc = get_tdrhf_add_epc(xs, ys, iajb, iajb_int)

        v1 = []
        for t in mf.components.keys():
            v1_top = lib.einsum('xpq,qo,pv->xov', v1ao[t], orbo[t], orbv[t].conj())
            v1_top += lib.einsum('xqs,sp->xqp', xs[t], fvv[t])
            v1_top -= lib.einsum('xpr,sp->xsr', xs[t], foo[t])

            v1_bot = lib.einsum('xpq,po,qv->xov', v1ao[t], orbo[t].conj(), orbv[t])
            v1_bot += lib.einsum('xqs,sp->xqp', ys[t], fvv[t])
            v1_bot -= lib.einsum('xpr,sp->xsr', ys[t], foo[t])
            if has_epc:
                v1_top += epc[t]
                v1_bot += epc[t]

            v1.append(numpy.hstack((v1_top.reshape(nz,-1), -v1_bot.reshape(nz,-1))))

        hx = numpy.hstack(v1)

        return hx
    
    return vind, numpy.hstack(hdiag)

def gen_tduhf_operation(mf):
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    mo_energy = mf.mo_energy
    orbv = {}
    orbo = {}
    nocc = {}
    nvir = {}
    nov = {}
    foo = {}
    fvv = {}
    hdiag = []

    assert isinstance(mf.components['e'], scf.uhf.UHF)

    has_epc = False
    if isinstance(mf, neo.KS):
        if mf.epc is not None:
            has_epc = True

    for t in mf.components.keys():
        mo_occ[t] = numpy.asarray(mo_occ[t])
        if t.startswith('e'):
            occidxa = numpy.where(mo_occ[t][0] > 0)[0]
            occidxb = numpy.where(mo_occ[t][1] > 0)[0]
            viridxa = numpy.where(mo_occ[t][0] == 0)[0]
            viridxb = numpy.where(mo_occ[t][1] == 0)[0]
            nocc[t] = [len(occidxa), len(occidxb)]
            nvir[t] = [len(viridxa), len(viridxb)]
            nov[t] = [nocc[t][0]*nvir[t][0], nocc[t][1]*nvir[t][1]]
            orbo[t] = [mo_coeff[t][0][:,occidxa], mo_coeff[t][1][:,occidxb]]
            orbv[t] = [mo_coeff[t][0][:,viridxa], mo_coeff[t][1][:,viridxb]]
            foo[t] = [numpy.diag(mo_energy[t][0][occidxa]), numpy.diag(mo_energy[t][1][occidxb])]
            fvv[t] = [numpy.diag(mo_energy[t][0][viridxa]), numpy.diag(mo_energy[t][1][viridxb])]
            for i in range(2):
                _hdiag = fvv[t][i].diagonal() - foo[t][i].diagonal()[:,None]
                hdiag.append(numpy.hstack((_hdiag.ravel(),-_hdiag.ravel())))
        else:       
            occidx = numpy.where(mo_occ[t] > 0)[0]
            viridx = numpy.where(mo_occ[t] == 0)[0]
            orbo[t] = mo_coeff[t][:,occidx]
            orbv[t] = mo_coeff[t][:,viridx]
            nocc[t] = len(occidx)
            nvir[t] = len(viridx)
            nov[t] = nocc[t]*nvir[t]
            foo[t] = numpy.diag(mo_energy[t][occidx])
            fvv[t] = numpy.diag(mo_energy[t][viridx])
            _hdiag = fvv[t].diagonal() - foo[t].diagonal()[:,None]
            hdiag.append(numpy.hstack((_hdiag.ravel(),-_hdiag.ravel())))

    vresp = mf.gen_response(hermi=0, no_epc=True)
    if has_epc:
        iajb, iajb_int = get_epc_iajb_uhf(mf, reshape=False)

    def vind(xys):
        xys = numpy.asarray(xys)
        nz, tot_size = xys.shape
        xs = {}
        ys = {}
        dms = {}
        offset = 0
        for t in mf.components.keys():
            if t.startswith('e'):
                xy = xys[:,offset:offset+2*(nov[t][0]+nov[t][1])]
                x, y = xy.reshape(nz,2,-1).transpose(1,0,2)
                xa = x[:,:nov[t][0]].reshape(nz,nocc[t][0],nvir[t][0])
                xb = x[:,nov[t][0]:].reshape(nz,nocc[t][1],nvir[t][1])
                ya = y[:,:nov[t][0]].reshape(nz,nocc[t][0],nvir[t][0])
                yb = y[:,nov[t][0]:].reshape(nz,nocc[t][1],nvir[t][1])
                dmsa  = lib.einsum('xov,pv,qo->xpq', xa, orbv[t][0].conj(), orbo[t][0])
                dmsb  = lib.einsum('xov,pv,qo->xpq', xb, orbv[t][1].conj(), orbo[t][1])
                dmsa += lib.einsum('xov,qv,po->xpq', ya, orbv[t][0], orbo[t][0].conj())
                dmsb += lib.einsum('xov,qv,po->xpq', yb, orbv[t][1], orbo[t][1].conj())
                dms[t] = numpy.asarray((dmsa, dmsb))
                xs[t] = [xa, xb]
                ys[t] = [ya, yb]
                offset += 2*(nov[t][0]+nov[t][1])

            else:
                xy = xys[:,offset:offset+2*nov[t]]
                offset += 2*nov[t]
                xy = xy.reshape(-1,2,nocc[t],nvir[t])
                xs[t], ys[t] = xy.transpose(1,0,2,3)
                dms[t] = lib.einsum('xov,pv,qo->xpq', xs[t], orbv[t], orbo[t].conj())
                dms[t] += lib.einsum('xov,qv,po->xpq', ys[t], orbv[t].conj(), orbo[t])

        assert (offset == tot_size)
        v1ao = vresp(dms)
        if has_epc:
            epc = get_tduhf_add_epc(xs, ys, iajb, iajb_int)

        v1 = []
        for t in mf.components.keys():
            if t.startswith('e'):
                v1a_top = lib.einsum('xpq,qo,pv->xov', v1ao[t][0], orbo[t][0], orbv[t][0].conj())
                v1a_top += lib.einsum('xqs,sp->xqp', xs[t][0], fvv[t][0])
                v1a_top -= lib.einsum('xpr,sp->xsr', xs[t][0], foo[t][0])

                v1b_top = lib.einsum('xpq,qo,pv->xov', v1ao[t][1], orbo[t][1], orbv[t][1].conj())
                v1b_top += lib.einsum('xqs,sp->xqp', xs[t][1], fvv[t][1])
                v1b_top -= lib.einsum('xpr,sp->xsr', xs[t][1], foo[t][1])

                v1a_bot = lib.einsum('xpq,po,qv->xov', v1ao[t][0], orbo[t][0].conj(), orbv[t][0])
                v1a_bot += lib.einsum('xqs,sp->xqp', ys[t][0], fvv[t][0])
                v1a_bot -= lib.einsum('xpr,sp->xsr', ys[t][0], foo[t][0])

                v1b_bot = lib.einsum('xpq,po,qv->xov', v1ao[t][1], orbo[t][1].conj(), orbv[t][1])
                v1b_bot += lib.einsum('xqs,sp->xqp', ys[t][1], fvv[t][1])
                v1b_bot -= lib.einsum('xpr,sp->xsr', ys[t][1], foo[t][1])

                if has_epc:
                    v1a_top += epc[t][0]
                    v1a_bot += epc[t][0]
                    v1b_top += epc[t][1]
                    v1b_bot += epc[t][1]

                v1.append(numpy.hstack((v1a_top.reshape(nz,-1), v1b_top.reshape(nz,-1),\
                                        -v1a_bot.reshape(nz,-1), -v1b_bot.reshape(nz,-1))))
                
            else:
                v1_top = lib.einsum('xpq,qo,pv->xov', v1ao[t], orbo[t].conj(), orbv[t])
                v1_top += lib.einsum('xqs,sp->xqp', xs[t], fvv[t])
                v1_top -= lib.einsum('xpr,sp->xsr', xs[t], foo[t])

                v1_bot = lib.einsum('xpq,po,qv->xov', v1ao[t], orbo[t], orbv[t].conj())
                v1_bot += lib.einsum('xqs,sp->xqp', ys[t], fvv[t])
                v1_bot -= lib.einsum('xpr,sp->xsr', ys[t], foo[t])

                if has_epc:
                    v1_top += epc[t]
                    v1_bot += epc[t]

                v1.append(numpy.hstack((v1_top.reshape(nz,-1), -v1_bot.reshape(nz,-1))))
            
        hx = numpy.hstack(v1)
        return hx
    
    return vind, numpy.hstack(hdiag)

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
    '''
    Examples:

    >>> from pyscf import neo
    >>> from pyscf.neo import tddft
    >>> mol = neo.M(atom='H 0 0 0; C 0 0 1.067; N 0 0 2.213', basis='631g', 
                    quantum_nuc = ['H'], nuc_basis = 'pb4d')
    >>> mf = neo.HF(mol)
    >>> mf.scf()
    >>> td_mf = tddft.TDDFT(mf)
    >>> td_mf.kernel(nstates=5)
    Excited State energies (eV)
    [0.62060056 0.62060056 0.69023232 1.24762233 1.33973627]
    '''
    
    conv_tol = getattr(__config__, 'tdscf_rhf_TDA_conv_tol', 1e-9)
    nstates = getattr(__config__, 'tdscf_rhf_TDA_nstates', 3)
    lindep = getattr(__config__, 'tdscf_rhf_TDA_lindep', 1e-12)
    level_shift = getattr(__config__, 'tdscf_rhf_TDA_level_shift', 0)
    max_space = getattr(__config__, 'tdscf_rhf_TDA_max_space', 50)
    max_cycle = getattr(__config__, 'tdscf_rhf_TDA_max_cycle', 100)
    # Low excitation filter to avoid numerical instability
    positive_eig_threshold = getattr(__config__, 'tdscf_rhf_TDDFT_positive_eig_threshold', 1e-3)

    def __init__(self, mf):
        self.verbose = mf.verbose
        self.stdout = mf.components['e'].stdout
        self.mol = mf.mol
        self._scf = mf
        self.max_memory = mf.max_memory
        self.chkfile = mf.chkfile
        self.unrestricted = isinstance(mf.components['e'], scf.uhf.UHF)

        self.wfnsym = None

        self.converged = None
        self.e = None
        self.x1 = None

        keys = set(('conv_tol', 'nstates', 'lindep', 'level_shift',
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
            return gen_tduhf_operation(mf)
        else:
            return gen_tdrhf_operation(mf)
        
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