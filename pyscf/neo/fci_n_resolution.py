# N resolution method for multicomponent FCI
import numpy
import scipy
from pyscf import lib
from pyscf import ao2mo
from pyscf.fci import cistring
from pyscf import neo

def contract(h1, h2, fcivec, norb, nparticle, link_index=None):
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

def make_hdiag(h1, g2, norb, nparticle, opt=None):
    h1e_a = h1[0]
    h1e_b = h1[1]
    g2e_aa = ao2mo.restore(1, g2[0][0], norb[0])
    g2e_ab = ao2mo.restore(1, g2[0][1], norb[0])
    g2e_bb = ao2mo.restore(1, g2[1][1], norb[0])

    ndim = len(norb)
    occslists = []
    for i in range(ndim):
        occslists.append(cistring.gen_occslst(range(norb[i]), nparticle[i]))
    occslista = occslists[0]
    occslistb = occslists[1]
    jdiag_aa = numpy.einsum('iijj->ij',g2e_aa)
    jdiag_ab = numpy.einsum('iijj->ij',g2e_ab)
    jdiag_bb = numpy.einsum('iijj->ij',g2e_bb)
    kdiag_aa = numpy.einsum('ijji->ij',g2e_aa)
    kdiag_bb = numpy.einsum('ijji->ij',g2e_bb)
    jdiag = [[None] * ndim for _ in range(ndim)]
    done = [[False] * ndim for _ in range(ndim)]
    for k in range(ndim):
        for l in range(ndim):
            if k != l and (k >= 2 or l >= 2) and g2[k][l] is not None and not done[k][l]:
                jdiag[k][l] = numpy.einsum('iijj->ij',g2[k][l].reshape([norb[k]]*2+[norb[l]]*2))
                done[k][l] = done[l][k] = True

    def nested_loop(lists, prev_occ, current_index=2, result=0.0):
        if current_index == len(lists):
            hdiag.append(result)
            return

        for occ in lists[current_index]:
            e1n = h1[current_index][occ,occ].sum()
            e2n = 0.0
            for previous_index, occ_ in enumerate(prev_occ):
                if jdiag[previous_index][current_index] is not None:
                    e2n += jdiag[previous_index][current_index][occ_][:,occ].sum()
                else:
                    e2n += jdiag[current_index][previous_index][occ][:,occ_].sum()
            nested_loop(lists, prev_occ + [occ], current_index + 1, result + e1n + e2n)

    hdiag = []
    for aocc in occslista:
        e1a = h1e_a[aocc,aocc].sum()
        e2a = jdiag_aa[aocc][:,aocc].sum() - kdiag_aa[aocc][:,aocc].sum()
        for bocc in occslistb:
            e1 = e1a + h1e_b[bocc,bocc].sum()
            e2 = e2a + jdiag_ab[aocc][:,bocc].sum() * 2 \
                 + jdiag_bb[bocc][:,bocc].sum() - kdiag_bb[bocc][:,bocc].sum()
            nested_loop(occslists, [aocc, bocc], current_index=2, result=e1+e2*.5)
    return numpy.array(hdiag)

def kernel(h1, g2, norb, nparticle, ecore=0, ci0=None):
    h2 = [[None] * len(norb) for _ in range(len(norb))]
    h2[0][0], h2[0][1], h2[1][1] = absorb_h1e(h1[:2], (g2[0][0], g2[0][1], g2[1][1]),
                                              norb[0], (nparticle[0], nparticle[1]), .5)
    for i in range(len(norb)):
        for j in range(len(norb)):
            if i >= 2 or j >= 2:
                h2[i][j] = g2[i][j]

    if ci0 is None:
        dim = []
        for i in range(len(norb)):
            dim.append(cistring.num_strings(norb[i], nparticle[i]))
        print(f'FCI vector shape: {dim}', flush=True)
        ci0 = numpy.zeros(tuple(dim))
        print(f'FCI dimension: {ci0.size}', flush=True)
        mem = 0
        for i in range(len(norb)):
            mem += norb[i]**2
        mem += max(max(norb[2:])**2, norb[0]**2*2)
        print(f'Peak memory estimation: {mem * ci0.size * 8/1e9} GB')
        ci0[tuple(0 for _ in range(ci0.ndim))] = 1

    def hop(c):
        hc = contract(h1, h2, c, norb, nparticle)
        return hc.reshape(-1)
    hdiag = make_hdiag(h1, g2, norb, nparticle)
    print(f'{hdiag[:4]=}', flush=True)
    precond = lambda x, e, *args: x/(hdiag-e+1e-4)
    converged, e, c = lib.davidson1(lambda xs: [hop(x) for x in xs],
                                    ci0.reshape(-1), precond, max_cycle=100,
                                    verbose=10)
    if converged[0]:
        print('FCI Davidson converged!')
    else:
        print('FCI Davidson did not converge according to current setting.')
    return e[0]+ecore, c[0]

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
            rdm1[a,i] += sign * numpy.dot(fcivec[tuple(str1_indices)].ravel(),
                                          fcivec[str0_indices_tuple].ravel())
    return rdm1

def entropy(indices, fcivec, norb, nparticle):
    """Subspace von Neumann entropy.
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

def FCI(mf):
    from functools import reduce
    assert mf.unrestricted
    norb_e = mf.mf_elec.mo_coeff[0].shape[1]
    mol = mf.mol
    nelec = mol.elec.nelec
    norb = [norb_e, norb_e]
    nparticle = [nelec[0], nelec[1]]
    for i in range(mol.nuc_num):
        norb_n = mf.mf_nuc[i].mo_coeff.shape[1]
        norb.append(norb_n)
        nparticle.append(1)
    print(f'{norb=}', flush=True)
    print(f'{nparticle=}', flush=True)

    is_cneo = False
    if isinstance(mf, neo.CDFT):
        is_cneo = True
        print('CNEO-FCI')
    else:
        print('Unconstrained NEO-FCI')

    h1e_a = reduce(numpy.dot, (mf.mf_elec.mo_coeff[0].T, mf.mf_elec.hcore_static, mf.mf_elec.mo_coeff[0]))
    h1e_b = reduce(numpy.dot, (mf.mf_elec.mo_coeff[1].T, mf.mf_elec.hcore_static, mf.mf_elec.mo_coeff[1]))
    h1 = [h1e_a, h1e_b]
    for i in range(mol.nuc_num):
        h1n = reduce(numpy.dot, (mf.mf_nuc[i].mo_coeff.T, mf.mf_nuc[i].hcore_static, mf.mf_nuc[i].mo_coeff))
        h1.append(h1n)
    r1 = []
    if is_cneo:
        for i in range(mol.nuc_num):
            r1n = []
            for x in range(mf.mf_nuc[i].int1e_r.shape[0]):
                r1n.append(reduce(numpy.dot, (mf.mf_nuc[i].mo_coeff.T, mf.mf_nuc[i].int1e_r[x], mf.mf_nuc[i].mo_coeff)))
            r1n = numpy.array(r1n)
            r1.append(r1n)
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
    g2 = [[None] * len(norb) for _ in range(len(norb))]
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

    ecore = mf.mf_elec.energy_nuc()
    if is_cneo:
        class CCISolver():
            def position_analysis(self, f, h1, r1, g2, norb, nparticle, ecore):
                f = f.reshape(len(norb)-2,-1)
                f1 = [h1[0], h1[1]]
                for i in range(len(norb)-2):
                    f1.append(h1[i+2] + numpy.einsum('xij,x->ij', r1[i], f[i]))
                self.e, self.c = kernel(f1, g2, norb, nparticle, ecore, self.c)
                dr = []
                for i in range(len(norb)-2):
                    rdm1 = make_rdm1(self.c, i+2, norb, nparticle)
                    dr.append(numpy.einsum('xij,ij->x', r1[i], rdm1))
                dr = numpy.array(dr).ravel()
                print()
                print(f'CNEO| {f=}')
                print(f'CNEO| lowest eigenvalue of H+f(r-R)={self.e}')
                print(f'CNEO| max|dr|={numpy.abs(dr).max()}')
                print()
                return dr

            def kernel(self, h1=h1, r1=r1, g2=g2, norb=norb, nparticle=nparticle,
                       ecore=ecore):
                # get initial f from CNEO-HF
                f = numpy.zeros((mol.nuc_num, 3))
                for i in range(mol.nuc_num):
                    ia = mol.nuc[i].atom_index
                    f[i] = mf.f[ia]
                print(f'Initial f: {f}')
                self.c = None
                opt = scipy.optimize.root(self.position_analysis, f,
                                          args=(h1, r1, g2, norb, nparticle, ecore),
                                          method='hybr',
                                          options={'xtol': 1e-15})
                print(f'CNEO| Final: Optimized f: {f.reshape(mol.nuc_num,-1)}')
                print(f'CNEO| Final: Position deviation: {opt.fun.reshape(mol.nuc_num,-1)}')
                print(f'CNEO| Final: Position deviation max: {numpy.abs(opt.fun).max()}')
                eigenvalue = self.e
                print(f'CNEO| Energy directly from the eigenvalue of H+f(r-R) matrix: {self.e}')
                h2 = [[None] * len(norb) for _ in range(len(norb))]
                h2[0][0], h2[0][1], h2[1][1] = absorb_h1e(h1[:2], (g2[0][0], g2[0][1], g2[1][1]),
                                                          norb[0], (nparticle[0], nparticle[1]), .5)
                for i in range(len(norb)):
                    for j in range(len(norb)):
                        if i >= 2 or j >= 2:
                            h2[i][j] = g2[i][j]
                ci1 = contract(h1, h2, self.c, norb, nparticle)
                self.e = numpy.dot(self.c, ci1)
                print(f'CNEO| Energy recalculated using c^T*H*c: {self.e}, difference={self.e-eigenvalue}')
                return self.e, self.c, f
            def entropy(self, indices, fcivec=None, norb=norb, nparticle=nparticle):
                if fcivec is None:
                    fcivec = self.c
                return entropy(indices, fcivec, norb, nparticle)
        cisolver = CCISolver()
    else:
        class CISolver():
            def kernel(self, h1=h1, g2=g2, norb=norb, nparticle=nparticle,
                       ecore=ecore):
                self.e, self.c = kernel(h1, g2, norb, nparticle, ecore)
                return self.e, self.c
            def entropy(self, indices, fcivec=None, norb=norb, nparticle=nparticle):
                if fcivec is None:
                    fcivec = self.c
                return entropy(indices, fcivec, norb, nparticle)
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
