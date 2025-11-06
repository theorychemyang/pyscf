import numpy
from pyscf import lib
from pyscf import ao2mo
from pyscf.fci import cistring, selected_ci

# original N-resolution
def contract_2e(eri, fcivec, norb, nelec, link_index=None):
    '''Compute E_{pq}E_{rs}|CI>'''
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    if link_index is None:
        link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
        link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)
    else:
        link_indexa, link_indexb = link_index
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    ci0 = fcivec.reshape(na,nb)
    t1a = numpy.zeros((norb,norb,na,nb), dtype=fcivec.dtype)
    t1b = numpy.zeros((norb,norb,na,nb), dtype=fcivec.dtype)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            t1a[a,i,str1] += sign * ci0[str0]
    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            t1b[a,i,:,str1] += sign * ci0[:,str0]

    g2e_aa = ao2mo.restore(1, eri[0], norb)
    g2e_ab = ao2mo.restore(1, eri[1], norb)
    g2e_bb = ao2mo.restore(1, eri[2], norb)

    g1a = lib.einsum('bjai,aiAB->bjAB', g2e_aa.reshape([norb]*4), t1a) \
          + lib.einsum('bjai,aiAB->bjAB', g2e_ab.reshape([norb]*4), t1b)

    fcinew = numpy.zeros_like(ci0, dtype=fcivec.dtype)
    for str0, tab in enumerate(link_indexa):
        for a, i, str1, sign in tab:
            fcinew[str1] += sign * g1a[a,i,str0]
    g1a = None

    g1b = lib.einsum('bjai,aiAB->bjAB', g2e_bb.reshape([norb]*4), t1b) \
          + lib.einsum('aibj,aiAB->bjAB', g2e_ab.reshape([norb]*4), t1a)
    t1a = None
    t1b = None

    for str0, tab in enumerate(link_indexb):
        for a, i, str1, sign in tab:
            fcinew[:,str1] += sign * g1b[a,i,:,str0]
    return fcinew.reshape(fcivec.shape)

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

# (N-2)-resolution
def contract_2(h1e, eri, fcivec, norb, nelec, link_index=None):
    '''Compute p^+ r^+ s q|CI>'''
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    dd_indexa = gen_des_des_str_index(range(norb), neleca)
    dd_indexb = gen_des_des_str_index(range(norb), nelecb)
    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    ci0 = fcivec.reshape(na,nb)

    h1e_a, h1e_b = h1e
    g2e_aa = ao2mo.restore(1, eri[0], norb)
    g2e_ab = ao2mo.restore(1, eri[1], norb)
    g2e_bb = ao2mo.restore(1, eri[2], norb)

    fcinew = numpy.zeros_like(ci0, dtype=fcivec.dtype)

    if dd_indexa is not None:
        ma = len(dd_indexa)
        t1a = numpy.zeros((norb,norb,ma,nb), dtype=fcivec.dtype)
        for str1, tab in enumerate(dd_indexa):
            for i, j, str0, sign in tab:
                t1a[i,j,str1] += sign * ci0[str0]

        g1a = lib.einsum('pqrs,qsAB->prAB', g2e_aa.reshape([norb]*4), t1a)
        t1a = None

        for str1, tab in enumerate(dd_indexa):
            for i, j, str0, sign in tab:
                fcinew[str0] += sign * g1a[i,j,str1]
        g1a = None

    if dd_indexb is not None:
        mb = len(dd_indexb)
        t1b = numpy.zeros((norb,norb,na,mb), dtype=fcivec.dtype)
        for str1, tab in enumerate(dd_indexb):
            for i, j, str0, sign in tab:
                t1b[i,j,:,str1] += sign * ci0[:,str0]

        g1b = lib.einsum('pqrs,qsAB->prAB', g2e_bb.reshape([norb]*4), t1b)
        t1b = None

        for str1, tab in enumerate(dd_indexb):
            for i, j, str0, sign in tab:
                fcinew[:,str0] += sign * g1b[i,j,:,str1]
        g1b = None

    if neleca > 0:
        d_indexa = cistring.gen_des_str_index(range(norb), neleca)
    else:
        d_indexa = None
    if nelecb > 0:
        d_indexb = cistring.gen_des_str_index(range(norb), nelecb)
    else:
        d_indexb = None

    if d_indexa is not None:
        ma = cistring.num_strings(norb, neleca-1)
        t1a = numpy.zeros((norb,ma,nb), dtype=fcivec.dtype)
        for str0, tab in enumerate(d_indexa):
            for _, i, str1, sign in tab:
                t1a[i,str1] += sign * ci0[str0]

        g1a = lib.einsum('pq,qAB->pAB', h1e_a, t1a)
        t1a = None

        for str0, tab in enumerate(d_indexa):
            for _, i, str1, sign in tab:
                fcinew[str0] += sign * g1a[i,str1]
        g1a = None

    if d_indexb is not None:
        mb = cistring.num_strings(norb, nelecb-1)
        t1b = numpy.zeros((norb,na,mb), dtype=fcivec.dtype)
        for str0, tab in enumerate(d_indexb):
            for _, i, str1, sign in tab:
                t1b[i,:,str1] += sign * ci0[:,str0]

        g1b = lib.einsum('pq,qAB->pAB', h1e_b, t1b)

        for str0, tab in enumerate(d_indexb):
            for _, i, str1, sign in tab:
                fcinew[:,str0] += sign * g1b[i,:,str1]
        g1b = None

    if d_indexa is not None and d_indexb is not None:
        t1ab = numpy.zeros((norb,norb,ma,mb), dtype=fcivec.dtype)
        for str0, tab in enumerate(d_indexa):
            for _, i, str1, sign in tab:
                t1ab[i,:,str1] += sign * t1b[:,str0]
        t1b = None

        g1ab = lib.einsum('pqrs,qsAB->prAB', g2e_ab.reshape([norb]*4), t1ab)

        t1ab = numpy.zeros((norb,na,mb), dtype=fcivec.dtype)
        for str0, tab in enumerate(d_indexa):
            for _, i, str1, sign in tab:
                t1ab[:,str0] += sign * g1ab[i,:,str1]
        g1ab = None
        for str0, tab in enumerate(d_indexb):
            for _, i, str1, sign in tab:
                fcinew[:,str0] += sign * t1ab[i,:,str1]
    return fcinew.reshape(fcivec.shape)

def gen_des_des_str_index(orb_list, nelec):
    if nelec < 2:
        return None
    strs = cistring.make_strings(orb_list, nelec)
    if isinstance(strs, cistring.OIndexList):
        raise NotImplementedError('System with 64 orbitals or more')

    norb = len(orb_list)
    return selected_ci.des_des_linkstr(strs, norb, nelec)

def make_hdiag(h1e, eri, norb, nelec, opt=None):
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    h1e_a, h1e_b = h1e
    g2e_aa = ao2mo.restore(1, eri[0], norb)
    g2e_ab = ao2mo.restore(1, eri[1], norb)
    g2e_bb = ao2mo.restore(1, eri[2], norb)

    occslista = occslistb = cistring.gen_occslst(range(norb), neleca)
    if neleca != nelecb:
        occslistb = cistring.gen_occslst(range(norb), nelecb)
    jdiag_aa = numpy.einsum('iijj->ij',g2e_aa)
    jdiag_ab = numpy.einsum('iijj->ij',g2e_ab)
    jdiag_bb = numpy.einsum('iijj->ij',g2e_bb)
    kdiag_aa = numpy.einsum('ijji->ij',g2e_aa)
    kdiag_bb = numpy.einsum('ijji->ij',g2e_bb)
    hdiag = []
    for aocc in occslista:
        for bocc in occslistb:
            e1 = h1e_a[aocc,aocc].sum() + h1e_b[bocc,bocc].sum()
            e2 = jdiag_aa[aocc][:,aocc].sum() + jdiag_ab[aocc][:,bocc].sum() * 2 \
               + jdiag_bb[bocc][:,bocc].sum() \
               - kdiag_aa[aocc][:,aocc].sum() - kdiag_bb[bocc][:,bocc].sum()
            hdiag.append(e1 + e2*.5)
    return numpy.array(hdiag)

def kernel(h1e, eri, norb, nelec, ecore=0):
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec
    h2e = absorb_h1e(h1e, eri, norb, nelec, .5)

    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    ci0 = numpy.zeros((na,nb))
    ci0[0,0] = 1

    def hop(c):
        hc = contract_2e(h2e, c, norb, nelec)
        return hc.reshape(-1)
    hdiag = make_hdiag(h1e, eri, norb, nelec)
    precond = lambda x, e, *args: x/(hdiag-e+1e-4)
    e, c = lib.davidson(hop, ci0.reshape(-1), precond)
    return e+ecore

def kernel_2(h1e, eri, norb, nelec, ecore=0):
    if isinstance(nelec, (int, numpy.integer)):
        nelecb = nelec//2
        neleca = nelec - nelecb
    else:
        neleca, nelecb = nelec

    na = cistring.num_strings(norb, neleca)
    nb = cistring.num_strings(norb, nelecb)
    ci0 = numpy.zeros((na,nb))
    ci0[0,0] = 1

    # eri_ab is not scaled by 0.5 because it cancels with the factor 2
    # from (pq in alpha, rs in beta) + (pq in beta, rs in alpha)
    # and eri_ba is not explicitly passed
    g2e = (eri[0] * 0.5, eri[1], eri[2] * 0.5)

    def hop(c):
        hc = contract_2(h1e, g2e, c, norb, nelec)
        return hc.reshape(-1)
    hdiag = make_hdiag(h1e, eri, norb, nelec)
    precond = lambda x, e, *args: x/(hdiag-e+1e-4)
    e, c = lib.davidson(hop, ci0.reshape(-1), precond)
    return e+ecore

if __name__ == '__main__':
    from functools import reduce
    from pyscf import gto
    from pyscf import scf

    mol = gto.M(atom='Li 0 0 0; He 0 0 1.0', basis='6-31G', charge=0, spin=1)
    mol.verbose = 0
    mol.output = None

    mf = scf.UHF(mol)
    mf.conv_tol_grad = 1e-6
    mf.kernel()
    print(f'{mf.e_tot=}')
    norb = mf.mo_coeff[0].shape[1]
    nelec = mol.nelec
    h1e_a = reduce(numpy.dot, (mf.mo_coeff[0].T, mf.get_hcore(), mf.mo_coeff[0]))
    h1e_b = reduce(numpy.dot, (mf.mo_coeff[1].T, mf.get_hcore(), mf.mo_coeff[1]))
    eri_aa = ao2mo.kernel(mf._eri, (mf.mo_coeff[0], mf.mo_coeff[0], mf.mo_coeff[0], mf.mo_coeff[0]), compact=False)
    eri_ab = ao2mo.kernel(mf._eri, (mf.mo_coeff[0], mf.mo_coeff[0], mf.mo_coeff[1], mf.mo_coeff[1]), compact=False)
    eri_bb = ao2mo.kernel(mf._eri, (mf.mo_coeff[1], mf.mo_coeff[1], mf.mo_coeff[1], mf.mo_coeff[1]), compact=False)

    #e1 = kernel((h1e_a, h1e_b), (eri_aa, eri_ab, eri_bb), norb, nelec, ecore=mf.energy_nuc())
    e1 = kernel_2((h1e_a, h1e_b), (eri_aa, eri_ab, eri_bb), norb, nelec, ecore=mf.energy_nuc())
    from pyscf import fci
    print(e1, e1 - fci.FCI(mf).kernel()[0])
