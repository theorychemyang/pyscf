import numpy
from pyscf.tdscf import rhf
from pyscf.dft.numint import eval_ao, eval_rho
from pyscf import lib
from pyscf import scf
from pyscf.lib import logger
from pyscf.data import nist
from pyscf import neo
from pyscf.tdscf._lr_eig import eig as lr_eig, real_eig

def _normalize(x1, mo_occ, log):
    ''' Normalize the NEO-TDDFT eigenvectors

    <Xe|Xe> - <Ye|Ye> + <Xp|Xp> - <Yp|Yp> = 1

    Args:
        x1: list of 1D array
            Each 1D array is organized as: [Xe, Xn, Ye, Yn]
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
        xs, ys = z1.reshape((2, -1))
        offset = 0
        norm = .0
        zs = {}
        xys = {}
        for t in sorted_keys:
            x = xs[offset:offset+nov[t]]
            y = ys[offset:offset+nov[t]]
            offset += nov[t]
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

        ee_numerator = numpy.multiply(numpy.square(rho_p), numerator_common)
        pp_numerator = numpy.multiply(numpy.square(rho_e), numerator_common)
        ep_numerator = (
            -4 * a**2 * numpy.sqrt(rho_product)
            - b * numpy.multiply(rho_product, c * rho_product + b * numpy.sqrt(rho_product))
            + a * numpy.multiply(rho_product, 3 * b + 4 * c * numpy.sqrt(rho_product))
        )

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

def get_init_guess(mf, nstates, deg_eia_thresh, tda=False):
    ''' Generate initial guess for NEO-TDDFT '''

    mo_occ = mf.mo_occ
    mo_energy = mf.mo_energy
    nov = {}
    e_ia = []
    total_size = 0
    for t in mf.components.keys():
        mo_occ[t] = numpy.array(mo_occ[t])
        if mo_occ[t].ndim > 1:
            assert not t.startswith('n')
            assert mo_occ[t].shape[0] == 2
            occidxa = mo_occ[t][0] > 0
            occidxb = mo_occ[t][1] > 0
            viridxa = ~occidxa
            viridxb = ~occidxb
            nocca = numpy.count_nonzero(occidxa)
            noccb = numpy.count_nonzero(occidxb)
            nvira = numpy.count_nonzero(viridxa)
            nvirb = numpy.count_nonzero(viridxb)
            mo_energy[t] = numpy.asarray(mo_energy[t])
            assert mo_energy[t].ndim > 1 and mo_energy[t].shape[0] == 2
            e_ia.append((mo_energy[t][0][viridxa] - mo_energy[t][0][occidxa,None]).ravel())
            e_ia.append((mo_energy[t][1][viridxb] - mo_energy[t][1][occidxb,None]).ravel())
            nov[t] = nocca*nvira + noccb*nvirb
        else:
            occidx = numpy.where(mo_occ[t] > 0)[0]
            viridx = numpy.where(mo_occ[t] == 0)[0]
            e_ia.append((mo_energy[t][viridx] - mo_energy[t][occidx,None]).ravel())
            nov[t] = e_ia[-1].size
        total_size += nov[t]
    e_ia = numpy.concatenate(e_ia)
    nstates = min(nstates, total_size)
    e_threshold = numpy.partition(e_ia, nstates-1)[nstates-1]
    e_threshold += deg_eia_thresh
    idx = numpy.where(e_ia <= e_threshold)[0]
    x0 = numpy.zeros((idx.size, total_size))
    for i, j in enumerate(idx):
        x0[i, j] = 1  # Koopmans' excitations

    if tda:
        return x0
    else:
        y0 = numpy.zeros_like(x0)
        return numpy.hstack([x0, y0])


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

    if mf._epc_n_types is None:
        n_types = []

        for t_pair, interaction in mf.interactions.items():
            if interaction._need_epc():
                if t_pair[0].startswith('n'):
                    n_type  = t_pair[0]
                else:
                    n_type  = t_pair[1]
                n_types.append(n_type)
        mf._epc_n_types = n_types
    else:
        n_types = mf._epc_n_types

    if len(n_types) == 0:
        raise ValueError('No epc detected')

    iajb = {}
    iajb_int = {}
    for t in mf.components.keys():
        iajb[t] = numpy.zeros((nocc[t], nvir[t], nocc[t], nvir[t]))

    t1 = 'e'
    for t2 in n_types:
        iajb_int[(t1, t2)] = numpy.zeros((nocc[t1], nvir[t1], nocc[t2], nvir[t2]))

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

        for t_pair in iajb_int.keys():
            t1, t2 = t_pair
            assert t1.startswith('e')
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

    if mf._epc_n_types is None:
        n_types = []

        for t_pair, interaction in mf.interactions.items():
            if interaction._need_epc():
                if t_pair[0].startswith('n'):
                    n_type  = t_pair[0]
                else:
                    n_type  = t_pair[1]
                n_types.append(n_type)
        mf._epc_n_types = n_types
    else:
        n_types = mf._epc_n_types

    if len(n_types) == 0:
        raise ValueError('No epc detected')

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

    t1 = 'e'
    for t2 in n_types:
        ep_a = numpy.zeros((nocc[t1][0], nvir[t1][0], nocc[t2], nvir[t2]))
        ep_b = numpy.zeros((nocc[t1][1], nvir[t1][1], nocc[t2], nvir[t2]))
        iajb_int[(t1, t2)] = [ep_a, ep_b]

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
                rho_ov[t] = [numpy.einsum('ri,ra->ria', rho_o_a, rho_v_a),
                            numpy.einsum('ri,ra->ria', rho_o_b, rho_v_b)]
            else:
                rho_o = lib.einsum('rp,pi->ri', ao, orbo[t])
                rho_v = lib.einsum('rp,pi->ri', ao, orbv[t])
                rho_ov[t] = numpy.einsum('ri,ra->ria', rho_o, rho_v)

        for (t1, t2) in iajb_int.keys():
            assert t1.startswith('e')
            # TODO: cache reusable electronic quantities
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
            epc[t2] += numpy.einsum('jbia,njb->nia',comp,xys[t1]) * numpy.sqrt(2)

    return epc

def get_tduhf_add_epc(xs, ys, iajb, iajb_int):
    epc = {}
    xys = {}
    for t in iajb.keys():
        if t.startswith('e'):
            xys[t] = [xs[t][0] + ys[t][0], xs[t][1] + ys[t][1]]
            epca = numpy.einsum('iajb,njb->nia',iajb[t][0], xys[t][0])
            epca += numpy.einsum('iajb,njb->nia',iajb[t][1], xys[t][1])
            epcb = numpy.einsum('jbia,njb->nia', iajb[t][1], xys[t][0])
            epcb += numpy.einsum('iajb,njb->nia', iajb[t][2], xys[t][1])
            epc[t] = [epca, epcb]
        else:
            xys[t] = xs[t] + ys[t]
            epc[t] = numpy.einsum('iajb,njb->nia', iajb[t], xys[t])

    for (t1, t2), comp in iajb_int.items():
        if t1.startswith('e'):
            epc[t1][0] += numpy.einsum('iajb,njb->nia', comp[0], xys[t2])
            epc[t1][1] += numpy.einsum('iajb,njb->nia', comp[1], xys[t2])
            epc[t2] += numpy.einsum('jbia,njb->nia', comp[0], xys[t1][0])
            epc[t2] += numpy.einsum('jbia,njb->nia', comp[1], xys[t1][1])

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
    hdiag_top = []
    hdiag_bot = []

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
        hdiag_top.append(_hdiag.ravel())
        hdiag_bot.append(-_hdiag.ravel())

    vresp = mf.gen_response(hermi=0, no_epc=True)
    # TODO: Integrate EPC response calculation into _gen_neo_response
    # Currently, the EPC response in Davidson algorithm is computed
    # via straightforward matrix multiplication.
    if has_epc:
        iajb, iajb_int = get_epc_iajb_rhf(mf, reshape=False)

    def vind(xys):
        xys = numpy.asarray(xys)
        nz, tot_size = xys.shape
        xs_arr, ys_arr = numpy.array_split(xys, 2, axis=1)
        xs = {}
        ys = {}
        dms = {}
        offset = 0
        for t in mf.components.keys():
            x_arr = xs_arr[:,offset:offset+nov[t]]
            y_arr = ys_arr[:,offset:offset+nov[t]]
            offset += nov[t]
            xs[t] = x_arr.reshape((-1, nocc[t], nvir[t]))
            ys[t] = y_arr.reshape((-1, nocc[t], nvir[t]))
            dms[t] = lib.einsum('xov,pv,qo->xpq', xs[t], orbv[t], orbo[t].conj())
            dms[t] += lib.einsum('xov,qv,po->xpq', ys[t], orbv[t].conj(), orbo[t])

            if t.startswith('e'):
                dms[t] *= numpy.sqrt(2)

        assert (offset == tot_size // 2)

        v1ao = vresp(dms)
        v1ao['e'] = v1ao['e'] * numpy.sqrt(2)
        if has_epc:
            epc = get_tdrhf_add_epc(xs, ys, iajb, iajb_int)

        v1_tops = []
        v1_bots = []
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

            v1_tops.append(v1_top.reshape(nz, -1))
            v1_bots.append(-v1_bot.reshape(nz, -1))

        hx = numpy.hstack(v1_tops + v1_bots)

        return hx

    return vind, numpy.hstack(hdiag_top + hdiag_bot)

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
    hdiag_top = []
    hdiag_bot = []

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
                hdiag_top.append(_hdiag.ravel())
                hdiag_bot.append(-_hdiag.ravel())
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
            hdiag_top.append(_hdiag.ravel())
            hdiag_bot.append(-_hdiag.ravel())

    vresp = mf.gen_response(hermi=0, no_epc=True)
    # TODO: Integrate EPC response calculation into _gen_neo_response
    # Currently, the EPC response in Davidson algorithm is computed
    # via straightforward matrix multiplication.
    if has_epc:
        iajb, iajb_int = get_epc_iajb_uhf(mf, reshape=False)

    def vind(xys):
        xys = numpy.asarray(xys)
        nz, tot_size = xys.shape
        xs_arr, ys_arr = numpy.array_split(xys, 2, axis=1)
        xs = {}
        ys = {}
        dms = {}
        offset = 0
        for t in mf.components.keys():
            if t.startswith('e'):
                x_arr = xs_arr[:,offset:offset+(nov[t][0]+nov[t][1])]
                y_arr = ys_arr[:,offset:offset+(nov[t][0]+nov[t][1])]
                xa = x_arr[:,:nov[t][0]].reshape(nz,nocc[t][0],nvir[t][0])
                xb = x_arr[:,nov[t][0]:].reshape(nz,nocc[t][1],nvir[t][1])
                ya = y_arr[:,:nov[t][0]].reshape(nz,nocc[t][0],nvir[t][0])
                yb = y_arr[:,nov[t][0]:].reshape(nz,nocc[t][1],nvir[t][1])
                dmsa  = lib.einsum('xov,pv,qo->xpq', xa, orbv[t][0].conj(), orbo[t][0])
                dmsb  = lib.einsum('xov,pv,qo->xpq', xb, orbv[t][1].conj(), orbo[t][1])
                dmsa += lib.einsum('xov,qv,po->xpq', ya, orbv[t][0], orbo[t][0].conj())
                dmsb += lib.einsum('xov,qv,po->xpq', yb, orbv[t][1], orbo[t][1].conj())
                dms[t] = numpy.asarray((dmsa, dmsb))
                xs[t] = [xa, xb]
                ys[t] = [ya, yb]
                offset += (nov[t][0]+nov[t][1])

            else:
                x_arr = xs_arr[:,offset:offset+nov[t]]
                y_arr = ys_arr[:,offset:offset+nov[t]]
                offset += nov[t]
                xs[t] = x_arr.reshape((-1, nocc[t], nvir[t]))
                ys[t] = y_arr.reshape((-1, nocc[t], nvir[t]))
                dms[t] = lib.einsum('xov,pv,qo->xpq', xs[t], orbv[t], orbo[t].conj())
                dms[t] += lib.einsum('xov,qv,po->xpq', ys[t], orbv[t].conj(), orbo[t])

        assert (offset == tot_size // 2)
        v1ao = vresp(dms)
        if has_epc:
            epc = get_tduhf_add_epc(xs, ys, iajb, iajb_int)

        v1_tops = []
        v1_bots = []
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

                v1_tops.append(numpy.hstack((v1a_top.reshape(nz,-1), v1b_top.reshape(nz,-1))))
                v1_bots.append(numpy.hstack((-v1a_bot.reshape(nz,-1), -v1b_bot.reshape(nz,-1))))

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

                v1_tops.append(v1_top.reshape(nz, -1))
                v1_bots.append(-v1_bot.reshape(nz, -1))

        hx = numpy.hstack(v1_tops + v1_bots)
        return hx

    return vind, numpy.hstack(hdiag_top + hdiag_bot)

class TDDFT(rhf.TDBase):
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

    def gen_vind(self, mf=None):
        if mf is None:
            mf = self._scf
        if isinstance(mf.components['e'], scf.uhf.UHF):
            return gen_tduhf_operation(mf)
        else:
            return gen_tdrhf_operation(mf)

    def get_init_guess(self, mf, nstates=None):
        if nstates is None:
            nstates = self.nstates
        else:
            self.nstates = nstates
        return get_init_guess(mf, nstates, self.deg_eia_thresh)

    def _finalize(self):
        '''Hook for dumping results and clearing up the object.'''
        if not all(self.converged):
            logger.note(self, 'NEO-TDDFT states %s not converged.',
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
            x0 = self.get_init_guess(self._scf, self.nstates)

        log = logger.Logger(self.stdout, self.verbose)
        vind, hdiag = self.gen_vind(self._scf)
        precond = self.get_precond(hdiag)

        real_system = self._scf.mo_coeff['e'][0].dtype == numpy.double

        if real_system:
            eig = real_eig
            pickeig = None
        else:
            eig = lr_eig
            # We only need positive eigenvalues
            def pickeig(w, v, nroots, envs):
                realidx = numpy.where((abs(w.imag) < rhf.REAL_EIG_THRESHOLD) &
                                      (w.real > self.positive_eig_threshold))[0]
                # If the complex eigenvalue has small imaginary part, both the
                # real part and the imaginary part of the eigenvector can
                # approximately be used as the "real" eigen solutions.
                return lib.linalg_helper._eigs_cmplx2real(w, v, realidx, real_system)

        self.converged, self.e, x1 = eig(
            vind, x0, precond, tol_residual=self.conv_tol, lindep=self.lindep,
            nroots=nstates, pick=pickeig, max_cycle=self.max_cycle,
            max_memory=self.max_memory, verbose=log)

        self.xy = _normalize(x1, self._scf.mo_occ, log)

        log.timer('NEO-TDDFT Davidson', *cpu0)
        self._finalize()
        return self.e, self.xy
