# N resolution method for multicomponent FCI
from functools import reduce
import numpy
import scipy
from pyscf import lib
from pyscf import ao2mo
from pyscf.fci import cistring
from pyscf import neo, scf
from pyscf.neo import cdavidson
from pyscf.lib import logger
from pyscf.lib import misc

def contract(h1, h2, fcivec, norb, nparticle, link_index=None, r1=None):
    ndim = len(norb)
    if link_index is None:
        link_index = []
        for i in range(ndim):
            link_index.append(cistring.gen_linkstr_index(range(norb[i]), nparticle[i]))
    dim = []
    for i in range(ndim):
        dim.append(cistring.num_strings(norb[i], nparticle[i]))
    ci0 = fcivec.reshape(dim)

    t1 = []
    for k in range(ndim):
        t1_this = numpy.zeros((norb[k],norb[k])+tuple(dim), dtype=fcivec.dtype)
        str0_indices = [slice(None)] * ndim
        str1_indices = [slice(None)] * ndim
        for str0, tab in enumerate(link_index[k]):
            str0_indices[k] = str0
            str0_indices_tuple = tuple(str0_indices)
            for a, i, str1, sign in tab:
                str1_indices[k] = str1
                t1_this[tuple([a,i]+str1_indices)] += sign * ci0[str0_indices_tuple]
        t1.append(t1_this)

    norb_e = norb[0]
    h2e_aa = ao2mo.restore(1, h2[0][0], norb_e)
    h2e_ab = ao2mo.restore(1, h2[0][1], norb_e)
    h2e_bb = ao2mo.restore(1, h2[1][1], norb_e)

    g1 = lib.einsum('bjai,aiA->bjA', h2e_aa.reshape([norb_e]*4),
                    t1[0].reshape([norb_e]*2+[-1])) \
       + lib.einsum('bjai,aiA->bjA', h2e_ab.reshape([norb_e]*4),
                    t1[1].reshape([norb_e]*2+[-1]))
    g1 = g1.reshape([norb_e]*2+dim)

    fcinew = numpy.zeros_like(ci0, dtype=fcivec.dtype)
    for str0, tab in enumerate(link_index[0]):
        for a, i, str1, sign in tab:
            fcinew[str1] += sign * g1[a,i,str0]

    g1 = lib.einsum('bjai,aiA->bjA', h2e_bb.reshape([norb_e]*4),
                    t1[1].reshape([norb_e]*2+[-1])) \
       + lib.einsum('aibj,aiA->bjA', h2e_ab.reshape([norb_e]*4),
                    t1[0].reshape([norb_e]*2+[-1]))
    g1 = g1.reshape([norb_e]*2+dim)

    for str0, tab in enumerate(link_index[1]):
        for a, i, str1, sign in tab:
            fcinew[:,str1] += sign * g1[a,i,:,str0]

    for i in range(2, ndim):
        fcinew += numpy.dot(h1[i].reshape(-1), t1[i].reshape(-1,fcivec.size)).reshape(fcinew.shape)

    done = [[False] * ndim for _ in range(ndim)]
    for k in range(ndim):
        for l in range(ndim):
            if k != l and (k >= 2 or l >= 2) and h2[k][l] is not None and not done[k][l]:
                if k < l:
                    g1 = lib.einsum('aibj,aiA->bjA', h2[k][l].reshape([norb[k]]*2+[norb[l]]*2),
                                    t1[k].reshape([norb[k]]*2+[-1]))
                    g1 = g1.reshape([norb[l]]*2+dim)
                    str0_indices = [slice(None)] * ndim
                    str1_indices = [slice(None)] * ndim
                    for str0, tab in enumerate(link_index[l]):
                        str0_indices[l] = str0
                        for a, i, str1, sign in tab:
                            str1_indices[l] = str1
                            fcinew[tuple(str1_indices)] += sign * g1[tuple([a,i]+str0_indices)]
                else:
                    g1 = lib.einsum('bjai,aiA->bjA', h2[k][l].reshape([norb[k]]*2+[norb[l]]*2),
                                    t1[l].reshape([norb[l]]*2+[-1]))
                    g1 = g1.reshape([norb[k]]*2+dim)
                    str0_indices = [slice(None)] * ndim
                    str1_indices = [slice(None)] * ndim
                    for str0, tab in enumerate(link_index[k]):
                        str0_indices[k] = str0
                        for a, i, str1, sign in tab:
                            str1_indices[k] = str1
                            fcinew[tuple(str1_indices)] += sign * g1[tuple([a,i]+str0_indices)]
                done[k][l] = done[l][k] = True
    if r1 is not None:
        total_dim = 0
        for i in range(ndim):
            if r1[i] is not None:
                if r1[i].ndim == 2:
                    r1[i] = r1[i].reshape((1,)+r1[i].shape)
                total_dim += r1[i].shape[0]
        r_fcinew = numpy.empty((total_dim,)+ci0.shape, dtype=fcivec.dtype)
        n = 0
        for i in range(ndim):
            if r1[i] is not None:
                for x in range(r1[i].shape[0]):
                    r_fcinew[n] = numpy.dot(r1[i][x].reshape(-1),
                                            t1[i].reshape(-1,fcivec.size)).reshape(r_fcinew.shape[1:])
                    n += 1
        return fcinew.reshape(fcivec.shape), r_fcinew.reshape((total_dim,)+fcivec.shape)
    else:
        return fcinew.reshape(fcivec.shape)

def contract_1e(h1, fcivec, norb, nparticle, link_index=None):
    '''Contract only for the one-body part of quantum nuclei'''
    ndim = len(norb)
    if link_index is None:
        link_index = [None, None]
        for i in range(2, ndim):
            link_index.append(cistring.gen_linkstr_index(range(norb[i]), nparticle[i]))
    dim = []
    for i in range(ndim):
        dim.append(cistring.num_strings(norb[i], nparticle[i]))
    ci0 = fcivec.reshape(dim)

    t1 = [None, None]
    for k in range(2, ndim):
        t1_this = numpy.zeros((norb[k],norb[k])+tuple(dim), dtype=fcivec.dtype)
        str0_indices = [slice(None)] * ndim
        str1_indices = [slice(None)] * ndim
        for str0, tab in enumerate(link_index[k]):
            str0_indices[k] = str0
            str0_indices_tuple = tuple(str0_indices)
            for a, i, str1, sign in tab:
                str1_indices[k] = str1
                t1_this[tuple([a,i]+str1_indices)] += sign * ci0[str0_indices_tuple]
        t1.append(t1_this)

    total_dim = 0
    for i in range(2, ndim):
        if h1[i].ndim == 2:
            h1[i] = h1[i].reshape((1,)+h1[i].shape)
        total_dim += h1[i].shape[0]
    fcinew = numpy.empty((total_dim,)+ci0.shape, dtype=fcivec.dtype)
    n = 0
    for i in range(2, ndim):
        for x in range(h1[i].shape[0]):
            fcinew[n] = numpy.dot(h1[i][x].reshape(-1),
                                  t1[i].reshape(-1,fcivec.size)).reshape(fcinew.shape[1:])
            n += 1
    return fcinew.reshape((total_dim,)+fcivec.shape)

def absorb_h1e(h1e, eri, norb, nelec, fac=1):
    '''Modify 2e Hamiltonian to include 1e Hamiltonian contribution.
    '''
    if not isinstance(nelec, (int, numpy.integer)):
        nelec = sum(nelec)
    h1e_a, h1e_b = h1e
    h2e_aa = ao2mo.restore(1, eri[0].copy(), norb).astype(h1e_a.dtype, copy=False)
    h2e_ab = ao2mo.restore(1, eri[1].copy(), norb).astype(h1e_a.dtype, copy=False)
    h2e_bb = ao2mo.restore(1, eri[2].copy(), norb).astype(h1e_a.dtype, copy=False)
    f1e_a = h1e_a - numpy.einsum('jiik->jk', h2e_aa) * .5
    f1e_b = h1e_b - numpy.einsum('jiik->jk', h2e_bb) * .5
    f1e_a *= 1./(nelec+1e-100)
    f1e_b *= 1./(nelec+1e-100)
    for k in range(norb):
        h2e_aa[:,:,k,k] += f1e_a
        h2e_aa[k,k,:,:] += f1e_a
        h2e_ab[:,:,k,k] += f1e_a
        h2e_ab[k,k,:,:] += f1e_b
        h2e_bb[:,:,k,k] += f1e_b
        h2e_bb[k,k,:,:] += f1e_b
    return (h2e_aa * fac, h2e_ab * fac, h2e_bb * fac)

def make_hdiag(h1, g2, norb, nparticle, verbose=logger.DEBUG1):
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(misc.StreamObject.stdout, verbose)

    t0 = logger.perf_counter()
    ndim = len(norb)
    dim = []
    for i in range(ndim):
        dim.append(cistring.num_strings(norb[i], nparticle[i]))
    hdiag = numpy.zeros(tuple(dim))
    occslists = []
    for i in range(ndim):
        occslists.append(cistring.gen_occslst(range(norb[i]), nparticle[i]))
    jdiag = [[None] * ndim for _ in range(ndim)]
    kdiag = [None] * ndim
    done = [[False] * ndim for _ in range(ndim)]
    if g2 is not None:
        for k in range(ndim):
            for l in range(ndim):
                if g2[k][l] is not None and not done[k][l]:
                    if k == l:
                        g2_ = ao2mo.restore(1, g2[k][l], norb[k])
                        jdiag[k][l] = numpy.einsum('iijj->ij', g2_)
                        kdiag[k] = numpy.einsum('ijji->ij', g2_)
                    else:
                        jdiag[k][l] = numpy.einsum('iijj->ij', g2[k][l].reshape([norb[k]]*2+[norb[l]]*2))
                    done[k][l] = done[l][k] = True

    for i in range(ndim):
        if h1[i] is not None or jdiag[i][i] is not None or kdiag[i] is not None:
            str0_indices = [slice(None)] * ndim
            for str0, occ in enumerate(occslists[i]):
                str0_indices[i] = str0
                e1 = 0
                if h1[i] is not None:
                    e1 = h1[i][occ,occ].sum()
                e2 = 0
                if jdiag[i][i] is not None:
                    e2 += jdiag[i][i][occ][:,occ].sum()
                if kdiag[i] is not None:
                    e2 -= kdiag[i][occ][:,occ].sum()
                hdiag[tuple(str0_indices)] += e1 + e2*.5

    done = [[False] * ndim for _ in range(ndim)]
    for k in range(ndim):
        for l in range(ndim):
            if k != l and jdiag[k][l] is not None and not done[k][l]:
                str0_indices = [slice(None)] * ndim
                jdiag_ = jdiag[k][l]
                for str0, aocc in enumerate(occslists[k]):
                    str0_indices[k] = str0
                    for str1, bocc in enumerate(occslists[l]):
                        e2 = jdiag_[aocc][:,bocc].sum()
                        str0_indices[l] = str1
                        hdiag[tuple(str0_indices)] += e2
                done[k][l] = done[l][k] = True
    log.debug(f'make_hdiag: {logger.perf_counter() - t0} seconds')
    return hdiag.reshape(-1)

def make_rdiag(r1, norb, nparticle):
    ndim = len(norb)
    dim = []
    for i in range(ndim):
        dim.append(cistring.num_strings(norb[i], nparticle[i]))
    total_dim = 0
    for i in range(ndim):
        if r1[i] is not None:
            if r1[i].ndim == 2:
                r1[i] = r1[i].reshape((1,)+r1[i].shape)
            total_dim += r1[i].shape[0]
    rdiag = numpy.zeros((total_dim,)+tuple(dim))

    n = 0
    for i in range(ndim):
        if r1[i] is not None:
            occslist = cistring.gen_occslst(range(norb[i]), nparticle[i])
            str0_indices = [slice(None)] * ndim
            for str0, occ in enumerate(occslist):
                str0_indices[i] = str0
                for x in range(r1[i].shape[0]):
                    rdiag[n+x][tuple(str0_indices)] += r1[i][x][occ,occ].sum()
            n += r1[i].shape[0]

    return rdiag.reshape(total_dim,-1)

def kernel(h1, g2, norb, nparticle, ecore=0, ci0=None, hdiag=None, nroots=1,
           r1=None, rdiag=None, f0=None, conv_tol=1e-12, lindep=1e-14,
           max_cycle=250, max_space=24, max_memory=260000, verbose=logger.DEBUG1,
           constraint_start_space=4, auto_bounds=True, gtol=1e-12, rtol=1e-12):
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(misc.StreamObject.stdout, verbose)

    h2 = [[None] * len(norb) for _ in range(len(norb))]
    h2[0][0], h2[0][1], h2[1][1] = absorb_h1e(h1[:2], (g2[0][0], g2[0][1], g2[1][1]),
                                              norb[0], (nparticle[0], nparticle[1]), .5)
    for i in range(len(norb)):
        for j in range(len(norb)):
            if i >= 2 or j >= 2:
                h2[i][j] = g2[i][j]

    if hdiag is None:
        hdiag = make_hdiag(h1, g2, norb, nparticle)
    if ci0 is None:
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
    h2[0][0], h2[0][1], h2[1][1] = absorb_h1e(h1[:2], (g2[0][0], g2[0][1], g2[1][1]),
                                              norb[0], (nparticle[0], nparticle[1]), .5)
    for i in range(len(norb)):
        for j in range(len(norb)):
            if i >= 2 or j >= 2:
                h2[i][j] = g2[i][j]
    ci1 = contract(h1, h2, fcivec, norb, nparticle)
    return numpy.dot(fcivec, ci1) + ecore

# dm_pq = <|p^+ q|>
def make_rdm1(fcivec, index, norb, nparticle):
    ndim = len(norb)
    link_index = cistring.gen_linkstr_index(range(norb[index]), nparticle[index])
    dim = []
    for i in range(ndim):
        dim.append(cistring.num_strings(norb[i], nparticle[i]))
    fcivec = fcivec.reshape(dim)
    rdm1 = numpy.zeros((norb[index],norb[index]))
    str0_indices = [slice(None)] * ndim
    str1_indices = [slice(None)] * ndim
    for str0, tab in enumerate(link_index):
        str0_indices[index] = str0
        str0_indices_tuple = tuple(str0_indices)
        for a, i, str1, sign in tab:
            str1_indices[index] = str1
            rdm1[a,i] += sign * numpy.dot(fcivec[tuple(str1_indices)].reshape(-1),
                                          fcivec[str0_indices_tuple].reshape(-1))
    return rdm1

def make_rdm2(fcivec, index1, index2, norb, nparticle):
    assert index1 != index2
    ndim = len(norb)
    link_index1 = cistring.gen_linkstr_index(range(norb[index1]), nparticle[index1])
    link_index2 = cistring.gen_linkstr_index(range(norb[index2]), nparticle[index2])
    dim = []
    for i in range(ndim):
        dim.append(cistring.num_strings(norb[i], nparticle[i]))
    fcivec = fcivec.reshape(dim)
    rdm2 = numpy.zeros((norb[index1],norb[index1],norb[index2],norb[index2]))
    str0_indices = [slice(None)] * ndim
    str1_indices = [slice(None)] * ndim
    for str01, tab1 in enumerate(link_index1):
        str0_indices[index1] = str01
        for p, q, str11, sign1 in tab1:
            str1_indices[index1] = str11
            for str02, tab2 in enumerate(link_index2):
                str0_indices[index2] = str02
                str0_indices_tuple = tuple(str0_indices)
                for r, s, str12, sign2 in tab2:
                    str1_indices[index2] = str12
                    rdm2[p,q,r,s] += sign1 * sign2 \
                                     * numpy.dot(fcivec[tuple(str1_indices)].reshape(-1),
                                                 fcivec[str0_indices_tuple].reshape(-1))
    return rdm2

def energy_decomp(h1, g2, fcivec, norb, nparticle, verbose=logger.NOTE):
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(misc.StreamObject.stdout, verbose)

    ndim = len(norb)
    for i in range(ndim):
        if h1[i] is not None:
            rdm1 = make_rdm1(fcivec, i, norb, nparticle)
            e1 = numpy.dot(h1[i].reshape(-1), rdm1.reshape(-1))
            log.note(f'1-body energy for {i}-th particle: {e1}')
    done = [[False] * ndim for _ in range(ndim)]
    for i in range(ndim):
        for j in range(ndim):
            if i != j and g2[i][j] is not None and not done[i][j]:
                rdm2 = make_rdm2(fcivec, i, j, norb, nparticle)
                e2 = numpy.dot(g2[i][j].reshape(-1), rdm2.reshape(-1))
                log.note(f'2-body energy between {i}-th particle and {j}-th particle: {e2}')
                done[i][j] = done[j][i] = True

def entropy(indices, fcivec, norb, nparticle):
    r"""Subspace von Neumann entropy.
    indices means the indices you want for the subspace entropy.
    For example, if we have a system of [0, 1, 2, 3],
    indices = [2]
    means we will first get rho[2] = \sum_{0,1,3} rho[0,1,2,3]
    then calculate the entropy via -\sum \lambda ln(\lambda);
    indices = [0,1,2]
    means we will first get rho[0,1,2] = \sum_{3} rho[0,1,2,3]
    then calculate the entropy.
    """
    if isinstance(indices, (int, numpy.integer)):
        indices = [indices]
    ndim = len(norb)
    dim = []
    size = 1
    for i in range(ndim):
        n = cistring.num_strings(norb[i], nparticle[i])
        dim.append(n)
        if i in indices:
            size *= n
    fcivec = fcivec.reshape(dim)

    sum_dims = [i for i in range(ndim) if i not in indices]

    input_subscripts1 = ''.join(chr(97 + i) for i in range(ndim))
    input_subscripts2 = ''.join(chr(97 + i) if i in sum_dims
                                else chr(97 + ndim + i)
                                for i in range(ndim))
    output_subscripts = ''.join(chr(97 + i) for i in indices) \
                      + ''.join(chr(97 + ndim + i) for i in indices)

    einsum_str = f'{input_subscripts1},{input_subscripts2}->{output_subscripts}'
    rdm = numpy.einsum(einsum_str, fcivec, fcivec)
    w = scipy.linalg.eigh(rdm.reshape(size,size), eigvals_only=True)
    w = w[w>1e-16]
    return -(w * numpy.log(w)).sum()

def integrals(mf):
    mol = mf.mol
    h1e_a = reduce(numpy.dot, (mf.mf_elec.mo_coeff[0].T, mf.mf_elec.get_hcore(), mf.mf_elec.mo_coeff[0]))
    h1e_b = reduce(numpy.dot, (mf.mf_elec.mo_coeff[1].T, mf.mf_elec.get_hcore(), mf.mf_elec.mo_coeff[1]))
    h1 = [h1e_a, h1e_b]
    for i in range(mol.nuc_num):
        h1n = reduce(numpy.dot, (mf.mf_nuc[i].mo_coeff.T, mf.mf_nuc[i].get_hcore(), mf.mf_nuc[i].mo_coeff))
        h1.append(h1n)
    eri_ee_aa = ao2mo.kernel(mf.mf_elec._eri,
                             (mf.mf_elec.mo_coeff[0], mf.mf_elec.mo_coeff[0],
                              mf.mf_elec.mo_coeff[0], mf.mf_elec.mo_coeff[0]),
                             compact=False)
    eri_ee_ab = ao2mo.kernel(mf.mf_elec._eri,
                             (mf.mf_elec.mo_coeff[0], mf.mf_elec.mo_coeff[0],
                              mf.mf_elec.mo_coeff[1], mf.mf_elec.mo_coeff[1]),
                             compact=False)
    eri_ee_bb = ao2mo.kernel(mf.mf_elec._eri,
                             (mf.mf_elec.mo_coeff[1], mf.mf_elec.mo_coeff[1],
                              mf.mf_elec.mo_coeff[1], mf.mf_elec.mo_coeff[1]),
                             compact=False)
    g2 = [[None] * (2+mol.nuc_num) for _ in range(2+mol.nuc_num)]
    g2[0][0] = eri_ee_aa
    g2[0][1] = eri_ee_ab
    g2[1][1] = eri_ee_bb
    for i in range(mol.nuc_num):
        ia = mol.nuc[i].atom_index
        charge = mol.atom_charge(ia)
        eri_ne = -charge * ao2mo.kernel(mf._eri_ne[i],
                                        (mf.mf_nuc[i].mo_coeff, mf.mf_nuc[i].mo_coeff,
                                         mf.mf_elec.mo_coeff[0], mf.mf_elec.mo_coeff[0]),
                                        compact=False)
        g2[i+2][0] = eri_ne
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

def symmetry_finder(mol, verbose=logger.DEBUG1):
    '''A very simple symmetry finder that does not use point-group symmetry
    or symmetry axes'''
    from pyscf.hessian.thermo import rotation_const, _get_rotor_type

    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(misc.StreamObject.stdout, verbose)

    # find if this is a linear molecule
    mass = mol.mass
    atom_coords = mol.atom_coords()
    natm_mm = 0
    if mol.mm_mol is not None:
        natm_mm = mol.mm_mol.natm
        atom_coords = numpy.vstack([atom_coords, mol.mm_mol.atom_coords()])
        mass = numpy.concatenate((mass, numpy.ones(natm_mm)))
    mass_center = numpy.einsum('z,zx->x', mass, atom_coords) / mass.sum()
    atom_coords = atom_coords - mass_center
    rot_const = rotation_const(mass, atom_coords, 'GHz')
    rotor_type = _get_rotor_type(rot_const)
    symm = None
    if rotor_type == 'LINEAR':
        symm = 'LINEAR'
        # if a linear molecule, detect if the molecule is along x/y/z axis
        if numpy.abs(atom_coords[:,0]).max() < 1e-6:
            if numpy.abs(atom_coords[:,1]).max() < 1e-6:
                axis = 2
            elif numpy.abs(atom_coords[:,2]).max() < 1e-6:
                axis = 1
        elif numpy.abs(atom_coords[:,1]).max() < 1e-6:
            if numpy.abs(atom_coords[:,2]).max() < 1e-6:
                axis = 0
        else:
            # if not along an axis, warn
            log.note('This molecule is linear, but was not put along x/y/z axis. Symmetry will be OFF.')
            return None, None
        log.debug(f'Linear molecule along {chr(axis+88)}')
    else:
        # see if a planar molecule in a special plane
        symm = 'PLANAR'
        if numpy.abs(atom_coords[:,0]).max() < 1e-6:
            axis = 0
        elif numpy.abs(atom_coords[:,1]).max() < 1e-6:
            axis = 1
        elif numpy.abs(atom_coords[:,2]).max() < 1e-6:
            axis = 2
        else:
            # if not in a special plane, warn
            if mol.natm + natm_mm == 3:
                log.note('This molecule is planar, but was not put in xy/yz/xz planes. Symmetry will be OFF.')
            else:
                log.note('Not a molecule that symmetry is easy to exploit. Symmetry will be OFF.')
            return None, None
        log.debug(f'Planar molecule perpendicular to {chr(axis+88)}')
    return symm, axis

def FCI(mf, kernel=kernel, integrals=integrals, energy=energy, fci_verbose=logger.DEBUG1):
    assert isinstance(mf.mf_elec, scf.uhf.UHF)

    if isinstance(fci_verbose, logger.Logger):
        log = fci_verbose
    else:
        log = logger.Logger(misc.StreamObject.stdout, fci_verbose)

    norb_e = mf.mf_elec.mo_coeff[0].shape[1]
    mol = mf.mol
    nelec = mol.elec.nelec
    norb = [norb_e, norb_e]
    nparticle = [nelec[0], nelec[1]]
    for i in range(mol.nuc_num):
        norb_n = mf.mf_nuc[i].mo_coeff.shape[1]
        norb.append(norb_n)
        nparticle.append(1)
    log.debug(f'{norb=}')
    log.debug(f'{nparticle=}')

    is_cneo = False
    if isinstance(mf, neo.CDFT):
        is_cneo = True
        log.debug('CNEO-FCI')
    else:
        log.debug('Unconstrained NEO-FCI')

    h1, g2 = integrals(mf)

    ecore = mf.energy_nuc()

    class CISolver():
        def __init__(self):
            # pyscf.fci uses 1e-10, and the default of lib.davidson1 is 1e-12.
            # Choose a tight convergence
            self.conv_tol = 1e-12
            self.lindep = 1e-14 # Default
            # CNEO-FCI usually requires <100 for easy cases, and for challenging cases
            # it can use more than 100 cycles. Set 250 as the default in case one might
            # want to experiment with other parameters that can sometimes cause >200 cycles.
            self.max_cycle = 250
            self.max_space = 24 # double the default value. More memory, but better convergence
            self.max_memory = 260000 # 260000 is good for H2 cc-pV6Z & PB6H
            self.verbose = logger.DEBUG1
            self.nroots = 1
            # NOTE: `symmetry` here is a very basic symmetry implementation to reduce the
            # dimensionality of the position constraint, and it does not utilize the new position
            # deviation matrices along symmetry axes. It requires the molecule to be aligned to
            # major axes in the Cartesian coordinate.
            # TODO: adapt to the new symmetry code of CNEO-DFT.
            # TODO: implement the FCI code that utilizes wave function symmetry?
            self.symmetry = True
        def kernel(self, h1=h1, g2=g2, norb=norb, nparticle=nparticle,
                   ecore=ecore):
            self.e, self.c = kernel(h1, g2, norb, nparticle, ecore,
                                    nroots=self.nroots,
                                    conv_tol=self.conv_tol,
                                    lindep=self.lindep,
                                    max_cycle=self.max_cycle,
                                    max_space=self.max_space,
                                    max_memory=self.max_memory,
                                    verbose=self.verbose)
            return self.e[0], self.c[0]
        def entropy(self, indices, fcivec=None, norb=norb, nparticle=nparticle):
            if fcivec is None:
                fcivec = self.c[0]
            return entropy(indices, fcivec, norb, nparticle)
        def energy_decomp(self, h1=h1, g2=g2, fcivec=None, norb=norb, nparticle=nparticle):
            if fcivec is None:
                fcivec = self.c[0]
            return energy_decomp(h1, g2, fcivec, norb, nparticle)
        def make_rdm1(self, index, fcivec=None, norb=norb, nparticle=nparticle, ao_repr=False):
            assert index >= 0 and index < len(norb)
            if fcivec is None:
                fcivec = self.c[0]
            rdm1 = make_rdm1(fcivec, index, norb, nparticle)
            if ao_repr:
                if index == 0:
                    coeff = mf.mf_elec.mo_coeff[0]
                elif index == 1:
                    coeff = mf.mf_elec.mo_coeff[1]
                else:
                    coeff = mf.mf_nuc[index-2].mo_coeff
                rdm1 = reduce(numpy.dot, (coeff, rdm1, coeff.T))
            return rdm1
        def make_natorbs(self, index, fcivec=None, norb=norb, nparticle=nparticle):
            assert index >= 0 and index < len(norb)
            if fcivec is None:
                fcivec = self.c[0]
            rdm1 = make_rdm1(fcivec, index, norb, nparticle)
            if index == 0:
                coeff = mf.mf_elec.mo_coeff[0]
            elif index == 1:
                coeff = mf.mf_elec.mo_coeff[1]
            else:
                coeff = mf.mf_nuc[index-2].mo_coeff
            # pyscf.mp.dfmp2_native
            eigval, eigvec = numpy.linalg.eigh(rdm1)
            natocc = numpy.flip(eigval)
            w = natocc[natocc>1e-16]
            logger.note(self, f'Natocc entropy: {-(w * numpy.log(w)).sum()}')
            natorb = lib.dot(coeff, numpy.fliplr(eigvec))
            return natocc, natorb

    natm_mm = 0
    if mol.mm_mol is not None:
        natm_mm = mol.mm_mol.natm
    if is_cneo and mol.natm + natm_mm > 1:
        r1 = [None, None]
        for i in range(mol.nuc_num):
            r1n = []
            for x in range(mf.mf_nuc[i].int1e_r.shape[0]):
                r1n.append(reduce(numpy.dot, (mf.mf_nuc[i].mo_coeff.T, mf.mf_nuc[i].int1e_r[x],
                                              mf.mf_nuc[i].mo_coeff)))
            r1n = numpy.array(r1n)
            r1.append(r1n)
        # get initial f from CNEO-HF
        # NOTE: for high angular momentum basis, CNEO-HF guess can be bad. Zero guess?
        f = numpy.zeros((mol.nuc_num, 3))
        for i in range(mol.nuc_num):
            ia = mol.nuc[i].atom_index
            f[i] = mf.f[ia]
        symm, axis = symmetry_finder(mol)
        if symm:
            if symm == 'LINEAR':
                # retain only axis direction
                f = f[:,axis]
                for i in range(mol.nuc_num):
                    r1[i+2] = r1[i+2][axis]
            else:
                f = numpy.delete(f, axis, axis=1)
                # remove axis direction
                for i in range(mol.nuc_num):
                    r1[i+2] = numpy.delete(r1[i+2], axis, axis=0)
        class CCISolver(CISolver):
            def __init__(self):
                super().__init__()
                # Minimal start_space is 2, but 4 is usually more advisable
                # because in the cases when f is large, space 2 or 3 often reach
                # the bound and the optimization is basically wasted.
                self.constraint_start_space = 4
                # Bound the optimization to avoid insane Lagrange multiplier values
                self.auto_bounds = True
                # gtol=1e-12 here is usually tight enough to result in <1e-15 r error
                # for hydrogen. gtol=1e-15 in neo.cdft is for heavy nuclei, but we
                # probably won't care about heavy nuclei in FCI.
                self.gtol = 1e-12
                # rtol can be tighter but in most cases the convergence is already
                # much tighter than 1e-12. rtol is mostly useful when linear dependency
                # occurs, and in that case 1e-9 r error should not be considered converged.
                self.rtol = 1e-12
            def kernel(self, h1=h1, r1=r1, g2=g2, norb=norb, nparticle=nparticle,
                       ecore=ecore):
                self.e, self.c, self.f = kernel(h1, g2, norb, nparticle, ecore,
                                                nroots=self.nroots, r1=r1, f0=f,
                                                conv_tol=self.conv_tol,
                                                lindep=self.lindep,
                                                max_cycle=self.max_cycle,
                                                max_space=self.max_space,
                                                max_memory=self.max_memory,
                                                verbose=self.verbose,
                                                constraint_start_space=self.constraint_start_space,
                                                auto_bounds=self.auto_bounds,
                                                gtol=self.gtol, rtol=self.rtol)
                return self.e[0], self.c[0], self.f
        cisolver = CCISolver()
    else:
        cisolver = CISolver()
    return cisolver


if __name__ == '__main__':
    mol = neo.M(atom='H 0 0 0', basis='aug-ccpvdz',
                nuc_basis='pb4d', charge=0, spin=1)
    mol.verbose = 0
    mol.output = None

    mf = neo.HF(mol, unrestricted=True)
    mf.conv_tol_grad = 1e-7
    mf.kernel()
    print(f'HF energy: {mf.e_tot}', flush=True)
    e1 = FCI(mf).kernel()[0]
    print(f'FCI energy: {e1}, difference with benchmark: {e1 - -0.4777448729395}')
