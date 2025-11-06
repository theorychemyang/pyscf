# N-2 resolution method for multicomponent FCI
import numpy
import scipy
from pyscf import lib
from pyscf import ao2mo
from pyscf.fci import cistring
from pyscf import neo
from pyscf.neo import cdavidson
from pyscf.neo.fci_n_resolution import make_hdiag, make_rdiag
from pyscf.neo.fci_n_resolution import FCI as _FCI
from pyscf.fci.fci_uhf_slow_n_minus_2_resolution import gen_des_des_str_index
from pyscf.lib import logger
from pyscf.lib import misc

def gen_des_des_str_index_2(orb_list, nelec):
    '''A temporary hack for double occupation beyond 64 orbitals'''
    if nelec < 2:
        return None
    if nelec != 2:
        raise NotImplementedError("This path is for nelec == 2")

    orb_list = numpy.asarray(list(orb_list), dtype=numpy.int32)
    norb = len(orb_list)

    # No valid pairs -> empty table
    if norb < 2:
        return numpy.zeros((1, 0, 4), dtype=numpy.int32)

    m = norb * (norb - 1) # meaningful rows
    body = numpy.zeros((m, 4), dtype=numpy.int32)

    off = 0
    base_s = 0
    for q in range(1, norb):
        p = numpy.arange(q, dtype=numpy.int32) # 0..q-1
        cnt = 2 * q

        # Interleave: (q,p,-1), (p,q,+1) with s = base_s + p
        body[off:off+cnt:2, 0] = q
        body[off:off+cnt:2, 1] = p
        body[off:off+cnt:2, 2] = base_s + p
        body[off:off+cnt:2, 3] = -1

        body[off+1:off+cnt:2, 0] = p
        body[off+1:off+cnt:2, 1] = q
        body[off+1:off+cnt:2, 2] = base_s + p
        body[off+1:off+cnt:2, 3] = 1

        off += cnt
        base_s += q

    # Pad to norb*norb
    total = norb * norb
    if m < total:
        pad = numpy.zeros((total - m, 4), dtype=numpy.int32)
        out = numpy.vstack((body, pad))
    else:
        out = body

    return out.reshape(1, total, 4)

def gen_des_str_index_2(orb_list, nelec):
    '''A temporary hack low occupation beyond 64 orbitals'''
    orb_list = numpy.asarray(list(orb_list), dtype=numpy.int32)
    norb = len(orb_list)

    if nelec == 1:
        if norb < 1:
            return numpy.zeros((0, 1, 4), dtype=numpy.int32)
        out = numpy.zeros((norb, 1, 4), dtype=numpy.int32)
        # column 1 stores orbital index; last column = 1
        out[:, 0, 1] = numpy.arange(norb, dtype=numpy.int32)
        out[:, 0, 3] = 1
        return out

    if nelec == 2:
        if norb < 2:
            return numpy.zeros((0, 2, 4), dtype=numpy.int32)

        n_pairs = norb * (norb - 1) // 2
        out = numpy.empty((n_pairs, 2, 4), dtype=numpy.int32)

        k = 0
        for q in range(1, norb):
            i = numpy.arange(q, dtype=numpy.int32) # i = 0..q-1
            rows = i.size

            # [0, i, q, -1]
            out[k:k+rows, 0, 0] = 0
            out[k:k+rows, 0, 1] = i
            out[k:k+rows, 0, 2] = q
            out[k:k+rows, 0, 3] = -1

            # [0, q, i, +1]
            out[k:k+rows, 1, 0] = 0
            out[k:k+rows, 1, 1] = q
            out[k:k+rows, 1, 2] = i
            out[k:k+rows, 1, 3] = 1

            k += rows

        return out

    raise NotImplementedError("gen_des_str_index_2 supports only nelec == 1 or 2")

def contract(h1, h2, fcivec, norb, nparticle, dd_index=None, d_index=None,
             r1=None):
    ndim = len(norb)
    if dd_index is None:
        dd_index = []
        for i in range(ndim):
            if nparticle[i] > 1:
                try:
                    dd_index_ = gen_des_des_str_index(range(norb[i]), nparticle[i])
                except NotImplementedError:
                    if nparticle[i] == 2:
                        dd_index_ = gen_des_des_str_index_2(range(norb[i]), nparticle[i])
                    else:
                        raise NotImplementedError('64 orbitals or more and more than 2 occupation')
                dd_index.append(dd_index_)
            else:
                dd_index.append(None)
    if d_index is None:
        d_index = []
        for i in range(ndim):
            if nparticle[i] > 0:
                try:
                    d_index_ = cistring.gen_des_str_index(range(norb[i]), nparticle[i])
                except NotImplementedError:
                    if nparticle[i] <= 2:
                        d_index_ = gen_des_str_index_2(range(norb[i]), nparticle[i])
                    else:
                        raise NotImplementedError('64 orbitals or more and more than 2 occupation')
                d_index.append(d_index_)
            else:
                d_index.append(None)
    ndim = len(norb)
    dim = []
    for i in range(ndim):
        dim.append(cistring.num_strings(norb[i], nparticle[i]))
    ci0 = fcivec.reshape(dim)

    fcinew = numpy.zeros_like(ci0, dtype=fcivec.dtype)
    if r1 is not None:
        total_dim = 0
        for i in range(ndim):
            if r1[i] is not None:
                if r1[i].ndim == 2:
                    r1[i] = r1[i].reshape((1,)+r1[i].shape)
                total_dim += r1[i].shape[0]
        r_fcinew = numpy.zeros((total_dim,)+ci0.shape, dtype=fcivec.dtype)

    t1_cache = []
    n = 0
    for k in range(ndim):
        if dd_index[k] is not None:
            assert h2[k][k] is not None
            h2_ = ao2mo.restore(1, h2[k][k], norb[k])
            m = len(dd_index[k])
            dim2 = dim.copy()
            dim2[k] = m
            t1 = numpy.zeros((norb[k],norb[k]) + tuple(dim2), dtype=fcivec.dtype)
            str0_indices = [slice(None)] * ndim
            str1_indices = [slice(None)] * ndim
            for str1, tab in enumerate(dd_index[k]):
                str1_indices[k] = str1
                str1_indices_tuple = tuple(str1_indices)
                for i, j, str0, sign in tab:
                    str0_indices[k] = str0
                    t1[(i,j)+str1_indices_tuple] += sign * ci0[tuple(str0_indices)]

            g1 = lib.einsum('pqrs,qsA->prA', h2_.reshape([norb[k]]*4),
                            t1.reshape([norb[k]]*2+[-1]))
            g1 = g1.reshape([norb[k]]*2+dim2)
            t1 = None

            for str1, tab in enumerate(dd_index[k]):
                str1_indices[k] = str1
                str1_indices_tuple = tuple(str1_indices)
                for i, j, str0, sign in tab:
                    str0_indices[k] = str0
                    fcinew[tuple(str0_indices)] += sign * g1[(i,j)+str1_indices_tuple]
            g1 = None

        if d_index[k] is not None:
            assert h1[k] is not None
            m = cistring.num_strings(norb[k], nparticle[k]-1)
            dim2 = dim.copy()
            dim2[k] = m
            t1 = numpy.zeros((norb[k],) + tuple(dim2), dtype=fcivec.dtype)
            str0_indices = [slice(None)] * ndim
            str1_indices = [slice(None)] * ndim
            for str0, tab in enumerate(d_index[k]):
                str0_indices[k] = str0
                str0_indices_tuple = tuple(str0_indices)
                for _, i, str1, sign in tab:
                    str1_indices[k] = str1
                    t1[(i,)+tuple(str1_indices)] += sign * ci0[str0_indices_tuple]
            t1_cache.append(t1)

            g1 = lib.einsum('pq,qA->pA', h1[k], t1.reshape((norb[k], -1)))
            g1 = g1.reshape([norb[k]]+dim2)

            for str0, tab in enumerate(d_index[k]):
                str0_indices[k] = str0
                str0_indices_tuple = tuple(str0_indices)
                for _, i, str1, sign in tab:
                    str1_indices[k] = str1
                    fcinew[str0_indices_tuple] += sign * g1[(i,)+tuple(str1_indices)]
            g1 = None
            if r1 is not None and r1[k] is not None:
                sub_dim = r1[k].shape[0]
                g1 = lib.einsum('xpq,qA->xpA', r1[k], t1.reshape((norb[k], -1)))
                g1 = g1.reshape([-1,norb[k]]+dim2)

                for str0, tab in enumerate(d_index[k]):
                    str0_indices[k] = str0
                    r_fcinew_idx = (slice(n, n+sub_dim),) + tuple(str0_indices)
                    for _, i, str1, sign in tab:
                        str1_indices[k] = str1
                        g1_idx = (slice(None), i) + tuple(str1_indices)
                        r_fcinew[r_fcinew_idx] += sign * g1[g1_idx]
                n += sub_dim
                g1 = None
        else:
            t1_cache.append(None)

    done = [[False] * ndim for _ in range(ndim)]
    for k in range(ndim):
        for l in range(k+1, ndim):
            if (h2[k][l] is not None or h2[l][k] is not None) and d_index[k] is not None \
                    and d_index[l] is not None and not done[k][l]:
                m1 = cistring.num_strings(norb[k], nparticle[k]-1)
                m2 = cistring.num_strings(norb[l], nparticle[l]-1)
                dim2 = dim.copy()
                dim2[k] = m1
                dim2[l] = m2
                t1 = numpy.zeros((norb[k],norb[l]) + tuple(dim2), dtype=fcivec.dtype)

                t1_k = t1_cache[k]
                str0_indices = [slice(None)] * (ndim + 1)
                str1_indices = [slice(None)] * (ndim + 2)
                for str0, tab in enumerate(d_index[l]):
                    str0_indices[l + 1] = str0
                    str0_indices_tuple = tuple(str0_indices)
                    for _, i, str1, sign in tab:
                        str1_indices[1] = i
                        str1_indices[l + 2] = str1
                        t1[tuple(str1_indices)] += sign * t1_k[str0_indices_tuple]

                if h2[k][l] is not None:
                    g1 = lib.einsum('pqrs,qsA->prA', h2[k][l].reshape([norb[k]]*2+[norb[l]]*2),
                                    t1.reshape((norb[k], norb[l], -1)))
                else:
                    g1 = lib.einsum('rspq,qsA->prA', h2[l][k].reshape([norb[l]]*2+[norb[k]]*2),
                                    t1.reshape((norb[k], norb[l], -1)))
                g1 = g1.reshape([norb[k], norb[l]]+dim2)
                dim3 = dim.copy()
                dim3[l] = m2
                t1 = numpy.zeros((norb[l],) + tuple(dim3), dtype=fcivec.dtype)

                str0_indices = [slice(None)] * (ndim + 1)
                str1_indices = [slice(None)] * (ndim + 2)
                for str0, tab in enumerate(d_index[k]):
                    str0_indices[k + 1] = str0
                    str0_indices_tuple = tuple(str0_indices)
                    for _, i, str1, sign in tab:
                        str1_indices[0] = i
                        str1_indices[k + 2] = str1
                        t1[str0_indices_tuple] += sign * g1[tuple(str1_indices)]
                g1 = None
                str0_indices = [slice(None)] * ndim
                str1_indices = [slice(None)] * (ndim + 1)
                for str0, tab in enumerate(d_index[l]):
                    str0_indices[l] = str0
                    str0_indices_tuple = tuple(str0_indices)
                    for _, i, str1, sign in tab:
                        str1_indices[0] = i
                        str1_indices[l + 1] = str1
                        fcinew[str0_indices_tuple] += sign * t1[tuple(str1_indices)]
                done[k][l] = done[l][k] = True
    if r1 is None:
        return fcinew.reshape(fcivec.shape)
    else:
        return fcinew.reshape(fcivec.shape), r_fcinew.reshape((-1,)+fcivec.shape)

def contract_1e(h1, fcivec, norb, nparticle, d_index=None):
    '''Contract only for the one-body part of quantum nuclei'''
    ndim = len(norb)
    if d_index is None:
        d_index = []
        for i in range(ndim):
            if nparticle[i] > 0:
                try:
                    d_index_ = cistring.gen_des_str_index(range(norb[i]), nparticle[i])
                except NotImplementedError:
                    if nparticle[i] == 1:
                        d_index_ = numpy.zeros((norb[i], 1, 4), dtype=numpy.int32)
                        d_index_[:,:,-1] = 1
                        d_index_[:,:,1] = numpy.arange(norb[i], dtype=numpy.int32).reshape(-1,1)
                    else:
                        raise NotImplementedError('64 orbitals or more and not 1 occupation')
                d_index.append(d_index_)
            else:
                d_index.append(None)
    ndim = len(norb)
    dim = []
    for i in range(ndim):
        dim.append(cistring.num_strings(norb[i], nparticle[i]))
    ci0 = fcivec.reshape(dim)

    total_dim = 0
    for i in range(2, ndim):
        if h1[i].ndim == 2:
            h1[i] = h1[i].reshape((1,)+h1[i].shape)
        total_dim += h1[i].shape[0]
    fcinew = numpy.zeros((total_dim,)+ci0.shape, dtype=fcivec.dtype)

    n = 0
    for k in range(2, ndim):
        if d_index[k] is not None:
            assert h1[k] is not None
            sub_dim = h1[k].shape[0]
            m = cistring.num_strings(norb[k], nparticle[k]-1)
            dim2 = dim.copy()
            dim2[k] = m
            t1 = numpy.zeros((norb[k],) + tuple(dim2), dtype=fcivec.dtype)
            str0_indices = [slice(None)] * ndim
            str1_indices = [slice(None)] * ndim
            for str0, tab in enumerate(d_index[k]):
                str0_indices[k] = str0
                str0_indices_tuple = tuple(str0_indices)
                for _, i, str1, sign in tab:
                    str1_indices[k] = str1
                    t1[(i,)+tuple(str1_indices)] += sign * ci0[str0_indices_tuple]

            g1 = lib.einsum('xpq,qA->xpA', h1[k], t1.reshape((norb[k], -1)))
            g1 = g1.reshape([-1,norb[k]]+dim2)

            for str0, tab in enumerate(d_index[k]):
                str0_indices[k] = str0
                fcinew_idx = (slice(n, n+sub_dim),) + tuple(str0_indices)
                for _, i, str1, sign in tab:
                    str1_indices[k] = str1
                    g1_idx = (slice(None), i) + tuple(str1_indices)
                    fcinew[fcinew_idx] += sign * g1[g1_idx]
            n += sub_dim
            g1 = None
    return fcinew.reshape((total_dim,)+fcivec.shape)

def kernel(h1, g2, norb, nparticle, ecore=0, ci0=None, hdiag=None, nroots=1,
           r1=None, rdiag=None, f0=None, conv_tol=1e-12, lindep=1e-14,
           max_cycle=250, max_space=24, max_memory=260000, verbose=logger.DEBUG1,
           constraint_start_space=4, auto_bounds=True, gtol=1e-12, rtol=1e-12):
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(misc.StreamObject.stdout, verbose)

    h2 = [[None] * len(norb) for _ in range(len(norb))]
    for i in range(len(norb)):
        for j in range(len(norb)):
            if g2[i][j] is not None:
                if i == j:
                    h2[i][j] = g2[i][j] * 0.5
                else:
                    h2[i][j] = g2[i][j]

    if hdiag is None:
        hdiag = make_hdiag(h1, g2, norb, nparticle)
    if ci0 is None:
        log.debug('N-2 resolution method')
        dim = []
        for i in range(len(norb)):
            dim.append(cistring.num_strings(norb[i], nparticle[i]))
        log.debug(f'FCI vector shape: {dim}')
        log.debug(f'FCI dimension: {hdiag.size}')
        addrs = numpy.argpartition(hdiag, nroots-1)[:nroots]
        ci0 = []
        for addr in addrs:
            log.debug(f'{hdiag[addr]=}')
            ci0_ = numpy.zeros(hdiag.size)
            ci0_[addr] = 1
            ci0.append(ci0_)

    if r1 is None:
        def hop(c):
            hc = contract(h1, h2, c, norb, nparticle)
            return hc.reshape(-1)
        precond = lambda x, e, *args: x/(hdiag-e+1e-4)
        t0 = logger.perf_counter()
        converged, e, c = lib.davidson1(lambda xs: [hop(x) for x in xs],
                                        ci0, precond, tol=conv_tol, lindep=lindep,
                                        max_cycle=max_cycle, max_space=max_space,
                                        max_memory=max_memory, nroots=nroots,
                                        verbose=verbose)
        log.debug(f'davidson: {logger.perf_counter() - t0} seconds')
        if converged[0]:
            log.note(f'FCI Davidson converged! Energy = {e[0]+ecore:.15g}')
        else:
            log.note('FCI Davidson did not converge according to current setting.')
            log.note(f'Energy = {e[0]+ecore:.15g}')
        return e+ecore, c
    else:
        def hop(c):
            hc, rc = contract(h1, h2, c, norb, nparticle, r1=r1)
            return hc.reshape(-1), rc.reshape(rc.shape[0],-1)
        if rdiag is None:
            rdiag = make_rdiag(r1, norb, nparticle)
        t0 = logger.perf_counter()
        converged, e, c, f = cdavidson.davidson1(lambda xs: [list(t) for t in zip(*[hop(x) for x in xs])],
                                                 ci0, f0.reshape(-1), hdiag, rdiag,
                                                 tol=conv_tol, lindep=lindep,
                                                 max_cycle=max_cycle, max_space=max_space,
                                                 max_memory=max_memory, nroots=nroots,
                                                 verbose=verbose,
                                                 constraint_start_space=constraint_start_space,
                                                 auto_bounds=auto_bounds, gtol=gtol, rtol=rtol)
        log.debug(f'davidson: {logger.perf_counter() - t0} seconds')
        if converged[0]:
            log.note(f'C-FCI Davidson converged! Energy = {e[0]+ecore:.15g}')
        else:
            log.note('C-FCI Davidson did not converge according to current setting.')
            log.note(f'Energy = {e[0]+ecore:.15g}')
        return e+ecore, c, f

def energy(h1, g2, fcivec, norb, nparticle, ecore=0):
    h2 = [[None] * len(norb) for _ in range(len(norb))]
    for i in range(len(norb)):
        for j in range(len(norb)):
            if g2[i][j] is not None:
                if i == j:
                    h2[i][j] = g2[i][j] * 0.5
                else:
                    h2[i][j] = g2[i][j]
    ci1 = contract(h1, h2, fcivec, norb, nparticle)
    return numpy.dot(fcivec, ci1) + ecore

def integrals(mf):
    from functools import reduce
    mol = mf.mol
    nelec = mol.elec.nelec
    h1e_a = None
    if nelec[0] > 0:
        h1e_a = reduce(numpy.dot, (mf.mf_elec.mo_coeff[0].T, mf.mf_elec.get_hcore(), mf.mf_elec.mo_coeff[0]))
    h1e_b = None
    if nelec[1] > 0:
        h1e_b = reduce(numpy.dot, (mf.mf_elec.mo_coeff[1].T, mf.mf_elec.get_hcore(), mf.mf_elec.mo_coeff[1]))
    h1 = [h1e_a, h1e_b]
    for i in range(mol.nuc_num):
        h1n = reduce(numpy.dot, (mf.mf_nuc[i].mo_coeff.T, mf.mf_nuc[i].get_hcore(), mf.mf_nuc[i].mo_coeff))
        h1.append(h1n)

    g2 = [[None] * (2+mol.nuc_num) for _ in range(2+mol.nuc_num)]
    if nelec[0] > 1:
        eri_ee_aa = ao2mo.kernel(mf.mf_elec._eri,
                                 (mf.mf_elec.mo_coeff[0], mf.mf_elec.mo_coeff[0],
                                  mf.mf_elec.mo_coeff[0], mf.mf_elec.mo_coeff[0]),
                                 compact=False)
        g2[0][0] = eri_ee_aa
    if nelec[1] > 1:
        eri_ee_bb = ao2mo.kernel(mf.mf_elec._eri,
                                 (mf.mf_elec.mo_coeff[1], mf.mf_elec.mo_coeff[1],
                                  mf.mf_elec.mo_coeff[1], mf.mf_elec.mo_coeff[1]),
                                 compact=False)
        g2[1][1] = eri_ee_bb
    if nelec[0] > 0 and nelec[1] > 0:
        eri_ee_ab = ao2mo.kernel(mf.mf_elec._eri,
                                 (mf.mf_elec.mo_coeff[0], mf.mf_elec.mo_coeff[0],
                                  mf.mf_elec.mo_coeff[1], mf.mf_elec.mo_coeff[1]),
                                 compact=False)
        g2[0][1] = eri_ee_ab

    for i in range(mol.nuc_num):
        ia = mol.nuc[i].atom_index
        charge = mol.atom_charge(ia)
        if nelec[0] > 0:
            eri_ne = -charge * ao2mo.kernel(mf._eri_ne[i],
                                            (mf.mf_nuc[i].mo_coeff, mf.mf_nuc[i].mo_coeff,
                                             mf.mf_elec.mo_coeff[0], mf.mf_elec.mo_coeff[0]),
                                            compact=False)
            g2[i+2][0] = eri_ne
        if nelec[1] > 0:
            eri_ne = -charge * ao2mo.kernel(mf._eri_ne[i],
                                            (mf.mf_nuc[i].mo_coeff, mf.mf_nuc[i].mo_coeff,
                                             mf.mf_elec.mo_coeff[1], mf.mf_elec.mo_coeff[1]),
                                            compact=False)
            g2[i+2][1] = eri_ne

    for i in range(mol.nuc_num):
        ia = mol.nuc[i].atom_index
        charge_i = mol.atom_charge(ia)
        for j in range(i):
            ja = mol.nuc[j].atom_index
            charge = charge_i * mol.atom_charge(ja)
            eri_nn = charge * ao2mo.kernel(mf._eri_nn[j][i],
                                           (mf.mf_nuc[j].mo_coeff, mf.mf_nuc[j].mo_coeff,
                                            mf.mf_nuc[i].mo_coeff, mf.mf_nuc[i].mo_coeff),
                                           compact=False)
            g2[j+2][i+2] = eri_nn
    return h1, g2

def FCI(mf, kernel=kernel, integrals=integrals, energy=energy):
    return _FCI(mf, kernel=kernel, integrals=integrals, energy=energy)


if __name__ == '__main__':
    mol = neo.M(atom='H 0 0 0; H 0 1.0 0; H 1.0 0 0; He 1.0 1.0 0', basis='6-31G',
                nuc_basis='1s1p1d', charge=0, spin=1)
    mol.verbose = 0
    mol.output = None

    mf = neo.HF(mol, unrestricted=True)
    mf.conv_tol_grad = 1e-7
    mf.kernel()
    print(f'HF energy: {mf.e_tot}', flush=True)
    t0 = logger.perf_counter()
    e0 = _FCI(mf).kernel()[0]
    print(f'N resolution FCI energy: {e0}, time: {logger.perf_counter() - t0} s')
    t0 = logger.perf_counter()
    e1 = FCI(mf).kernel()[0]
    print(f'N-2 resolution FCI energy: {e1}, difference: {e1 - e0}, time: {logger.perf_counter() - t0} s')
