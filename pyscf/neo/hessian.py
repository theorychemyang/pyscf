#!/usr/bin/env python

'''
Analytical Hessian for constrained nuclear-electronic orbitals
'''
import numpy
from functools import reduce
from pyscf import gto, lib
from pyscf.data import nist
from pyscf.lib import logger
from pyscf.hessian.thermo import harmonic_analysis, \
                                 rotation_const, \
                                 rotational_symmetry_number, \
                                 _get_rotor_type
from pyscf.neo import cphf, ucphf
from pyscf.scf.jk import get_jk

# import _response_functions to load gen_response methods in CDFT class
from pyscf.neo import _response_functions # noqa


def hess_cneo(hessobj, mo1, h1ao,
              atmlst=None, max_memory=4000, verbose=None):
    '''Additional Hessian terms because of quantum nuclei'''
    mol = hessobj.mol
    mf = hessobj.base

    de2 = hessobj.partial_hess_cneo(atmlst=atmlst, max_memory=max_memory,
                                    verbose=verbose)

    if isinstance(h1ao, str):
        h1ao = lib.chkfile.load(h1ao, 'scf_f1ao_n')
        h1ao = dict([(int(k), h1ao[k]) for k in h1ao])
        for k in h1ao:
            h1ao[k] = dict([(int(l), h1ao[k][l]) for l in h1ao[k]])
    if isinstance(mo1, str):
        mo1 = lib.chkfile.load(mo1, 'scf_mo1_n')
        mo1 = dict([(int(k), mo1[k]) for k in mo1])
        for k in mo1:
            mo1[k] = dict([(int(l), mo1[k][l]) for l in mo1[k]])

    for k in range(mol.nuc_num):
        mf_n = mf.mf_nuc[k]
        mo_coeff = mf_n.mo_coeff
        mo_occ = mf_n.mo_occ
        mocc = mo_coeff[:,mo_occ>0]
        for i0, ia in enumerate(atmlst):
            for j0 in range(i0+1):
                ja = atmlst[j0]
                dm1 = numpy.einsum('ypi,qi->ypq', mo1[ja][k], mocc)
                # 2.0 for +c.c. nuclear orbitals are only singly occupied
                de2[i0,j0] += numpy.einsum('xpq,ypq->xy', h1ao[ia][k], dm1) * 2.0

    for i0, ia in enumerate(atmlst):
        for j0 in range(i0):
            de2[j0,i0] = de2[i0,j0].T

    return de2

def partial_hess_cneo(hessobj, atmlst=None, max_memory=4000, verbose=None):
    '''Partial derivative
    '''
    e1, ej = _partial_hess_ej(hessobj, atmlst=atmlst, max_memory=max_memory,
                              verbose=verbose)
    return e1 + ej

def _partial_hess_ej(hessobj, atmlst=None, max_memory=4000, verbose=None):
    log = logger.new_logger(hessobj, verbose)
    time0 = t1 = (logger.process_clock(), logger.perf_counter())

    mol = hessobj.mol
    mf = hessobj.base
    if hessobj.base.unrestricted:
        nao_e = mf.mf_elec.mo_coeff[0].shape[0]
    else:
        nao_e = mf.mf_elec.mo_coeff.shape[0]

    # hcore contributions from quantum nuclei
    hcore_deriv = []
    for x in mol.nuc:
        hcore_deriv.append(hessobj.hcore_generator(x))

    aoslices = mol.aoslice_by_atom()
    e1 = numpy.zeros((mol.natm,mol.natm,3,3))
    ej = numpy.zeros((mol.natm,mol.natm,3,3))
    dm0_e = mf.dm_elec
    for k in range(mol.nuc_num):
        nuc = mol.nuc[k]
        ka = nuc.atom_index
        charge = mol.atom_charge(ka)
        nao_n = mf.mf_nuc[k].mo_coeff.shape[0]
        dm0_n = mf.dm_nuc[k]
        # <\nabla^2 elec | nuc>, e: all atoms
        vj1e_diag = get_jk((mol.elec, mol.elec, nuc, nuc), dm0_n,
                           scripts='ijkl,lk->ij', intor='int2e_ipip1',
                           aosym='s2kl', comp=9)
        vj1e_diag = -charge * vj1e_diag.reshape(3,3,nao_e,nao_e)
        # <\nabla^2 nuc | elec>, n: ka
        vj1n_diag = get_jk((nuc, nuc, mol.elec, mol.elec), dm0_e,
                           scripts='ijkl,lk->ij', intor='int2e_ipip1',
                           aosym='s2kl', comp=9)
        vj1n_diag = -charge * vj1n_diag.reshape(3,3,nao_n,nao_n)
        for l in range(mol.nuc_num):
            if l != k:
                nuc2 = mol.nuc[l]
                la = nuc2.atom_index
                charge2 = mol.atom_charge(la)
                dm0_n2 = mf.dm_nuc[l]
                # <\nabla^2 nuc | nuc2>, n: ka
                vj1nn = get_jk((nuc, nuc, nuc2, nuc2), dm0_n2,
                               scripts='ijkl,lk->ij', intor='int2e_ipip1',
                               aosym='s2kl', comp=9)
                vj1n_diag += charge * charge2 * vj1nn.reshape(3,3,nao_n,nao_n)
        t1 = log.timer_debug1('contracting int2e_ipip1 for quantum nuc %d' % k, *t1)
        # <\nabla nuc \nabla nuc | elec>, n: ka
        vj2n = get_jk((nuc, nuc, mol.elec, mol.elec), dm0_e,
                      scripts='ijkl,lk->ij', intor='int2e_ipvip1',
                      aosym='s2kl', comp=9)
        vj1n_diag += -charge * vj2n.reshape(3,3,nao_n,nao_n)
        for l in range(mol.nuc_num):
            if l != k:
                nuc2 = mol.nuc[l]
                la = nuc2.atom_index
                charge2 = mol.atom_charge(la)
                dm0_n2 = mf.dm_nuc[l]
                # <\nabla nuc \nabla nuc | nuc2>, n: ka
                vj2nn = get_jk((nuc, nuc, nuc2, nuc2), dm0_n2,
                               scripts='ijkl,lk->ij', intor='int2e_ipvip1',
                               aosym='s2kl', comp=9)
                vj1n_diag += charge * charge2 * vj2nn.reshape(3,3,nao_n,nao_n)
        t1 = log.timer_debug1('contracting int2e_ipvip1 for quantum nuc %d' % k, *t1)

        for i0, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia][2:]
            ej[i0,i0] += numpy.einsum('xypq,pq->xy', vj1e_diag[:,:,p0:p1], dm0_e[p0:p1]) * 2.0
            if ia == ka:
                ej[i0,i0] += numpy.einsum('xypq,pq->xy', vj1n_diag, dm0_n) * 2.0

        # <\nabla elec | \nabla nuc>, e: all atoms, n: ka
        vj1e = get_jk((mol.elec, mol.elec, nuc, nuc), dm0_n, scripts='ijkl,lk->ij',
                      intor='int2e_ip1ip2', aosym='s1', comp=9)
        vj1e = -charge * vj1e.reshape(3,3,nao_e,nao_e)
        t1 = log.timer_debug1('contracting int2e_ip1ip2 e-n for quantum nuc %d' % k, *t1)

        for i0, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia][2:]
            for j0, ja in enumerate(atmlst[:i0+1]):
                q0, q1 = aoslices[ja][2:]
                if ia == ka:
                    # n: ia, e: ja
                    # note the transpose
                    ej[i0,j0] += numpy.einsum('xypq,pq->xy', vj1e[:,:,q0:q1], dm0_e[q0:q1]).T * 4.0
                if ja == ka:
                    # e: ia, n: ja
                    ej[i0,j0] += numpy.einsum('xypq,pq->xy', vj1e[:,:,p0:p1], dm0_e[p0:p1]) * 4.0

        for l in range(k+1, mol.nuc_num): # l > k part is sufficient
            nuc2 = mol.nuc[l]
            la = nuc2.atom_index
            for i0, ia in enumerate(atmlst):
                for j0, ja in enumerate(atmlst[:i0]): # i != j, so i0 instead of i0+1
                    if (ia == ka and ja == la) or (ia == la and ja == ka):
                        charge2 = mol.atom_charge(la)
                        dm0_n2 = mf.dm_nuc[l]
                        # <\nabla nuc | \nabla nuc2>
                        vj1nn = get_jk((nuc, nuc, nuc2, nuc2), dm0_n2,
                                       scripts='ijkl,lk->ij', intor='int2e_ip1ip2',
                                       aosym='s1', comp=9)
                        vj1nn = charge * charge2 * vj1nn.reshape(3,3,nao_n,nao_n)
                        ej[i0,j0] += numpy.einsum('xypq,pq->xy', vj1nn, dm0_n) * 4.0
        t1 = log.timer_debug1('contracting int2e_ip1ip2 for n-n quantum nuc %d' % k, *t1)

        for i0, ia in enumerate(atmlst):
            shl0, shl1, p0, p1 = aoslices[ia]
            shls_slice = (shl0, shl1) + (0, mol.elec.nbas) + (0, nuc.nbas)*2
            # <\nabla elec \nabla elec | nuc>, e1: ia, e2: ja for all atoms
            vj2e = get_jk((mol.elec, mol.elec, nuc, nuc), dm0_n,
                          scripts='ijkl,lk->ij', intor='int2e_ipvip1',
                          aosym='s2kl', comp=9, shls_slice=shls_slice)
            vj2e = -charge * vj2e.reshape(3,3,p1-p0,nao_e)
            t1 = log.timer_debug1('contracting int2e_ipvip1 e-n for quantum nuc %d atom %d'
                                  % (k, ia), *t1)

            for j0, ja in enumerate(atmlst[:i0+1]):
                q0, q1 = aoslices[ja][2:]
                ej[i0,j0] += numpy.einsum('xypq,pq->xy', vj2e[:,:,:,q0:q1], dm0_e[p0:p1,q0:q1]) * 2.0

                h1ao = hcore_deriv[k](ia, ja)
                if isinstance(h1ao, numpy.ndarray):
                    e1[i0,j0] += numpy.einsum('xypq,pq->xy', h1ao, dm0_n)

    for i0, ia in enumerate(atmlst):
        for j0 in range(i0):
            e1[j0,i0] = e1[i0,j0].T
            ej[j0,i0] = ej[i0,j0].T

    log.timer('CNEO partial hessian', *time0)
    return e1, ej

def make_h1(hessobj, mf_e=None, mf_n=None, chkfile=None, atmlst=None,
            verbose=None):
    if mf_e is None: mf_e = hessobj.base.mf_elec
    if mf_n is None: mf_n = hessobj.base.mf_nuc
    mol = hessobj.mol
    if atmlst is None:
        atmlst = range(mol.natm)

    unrestricted = hessobj.base.unrestricted

    # Note: h1ao_e includes v1_ee
    hobj = mf_e.Hessian()
    if hessobj.grid_response is not None:
        hobj.grid_response = hessobj.grid_response
    h1ao_e_or_chkfile = \
            hobj.make_h1(mf_e.mo_coeff, mf_e.mo_occ, chkfile=chkfile,
                         atmlst=atmlst, verbose=verbose)

    if unrestricted:
        h1ao_e_a = [None] * mol.natm
        h1ao_e_b = [None] * mol.natm
    else:
        h1ao_e = [None] * mol.natm
    h1ao_n = [None] * mol.natm

    hcore_deriv = []
    for x in mol.nuc:
        hcore_deriv.append(hessobj.base.nuc_grad_method().hcore_generator(x))

    aoslices = mol.aoslice_by_atom()
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        if unrestricted:
            if isinstance(h1ao_e_or_chkfile, str):
                h1_e_a = lib.chkfile.load(h1ao_e_or_chkfile, 'scf_f1ao/0/%d' % ia)
                h1_e_b = lib.chkfile.load(h1ao_e_or_chkfile, 'scf_f1ao/1/%d' % ia)
            else:
                h1_e_a = h1ao_e_or_chkfile[0][ia]
                h1_e_b = h1ao_e_or_chkfile[1][ia]
        else:
            if isinstance(h1ao_e_or_chkfile, str):
                key = 'scf_f1ao/%d' % ia
                h1_e = lib.chkfile.load(h1ao_e_or_chkfile, key)
            else:
                h1_e = h1ao_e_or_chkfile[ia]
        h1ao_n[ia] = [None] * mol.nuc_num
        for j in range(mol.nuc_num):
            ja = mol.nuc[j].atom_index
            charge = mol.atom_charge(ja)
            # derivative w.r.t. electronic basis center
            shls_slice = (shl0, shl1) + (0, mol.elec.nbas) + (0, mol.nuc[j].nbas)*2
            v1e, v1n = get_jk((mol.elec, mol.elec, mol.nuc[j], mol.nuc[j]),
                              (hessobj.base.dm_nuc[j],
                               hessobj.base.dm_elec[:,p0:p1]),
                              scripts=('ijkl,lk->ij', 'ijkl,ji->kl'),
                              intor='int2e_ip1', aosym='s2kl', comp=3,
                              shls_slice=shls_slice)
            # elec e-n part
            v1e *= charge
            if unrestricted:
                h1_e_a[:,p0:p1] += v1e
                h1_e_a[:,:,p0:p1] += v1e.transpose(0,2,1)
                h1_e_b[:,p0:p1] += v1e
                h1_e_b[:,:,p0:p1] += v1e.transpose(0,2,1)
            else:
                h1_e[:,p0:p1] += v1e
                h1_e[:,:,p0:p1] += v1e.transpose(0,2,1)
            # nuc e-n part
            h1_n = v1n * 2.0 * charge # 2.0 for symmetry in nuclear orbitals
            v1e = None
            v1n = None
            # nuc hcore part
            h1_n += hcore_deriv[j](ia)
            if ja == ia:
                # derivative w.r.t. nuclear basis center
                # e-n part
                v1e, v1n = get_jk((mol.nuc[j], mol.nuc[j], mol.elec, mol.elec),
                                  (hessobj.base.dm_nuc[j], hessobj.base.dm_elec),
                                  scripts=('ijkl,ji->kl', 'ijkl,lk->ij'),
                                  intor='int2e_ip1', aosym='s2kl', comp=3)
                # elec e-n part
                if unrestricted:
                    h1_e_a += v1e * 2.0 * charge
                    h1_e_b += v1e * 2.0 * charge
                else:
                    h1_e += v1e * 2.0 * charge # 2.0 for symmetry in nuclear orbitals
                # nuc e-n part
                v1n *= charge
                h1_n += v1n
                h1_n += v1n.transpose(0,2,1)
                # nuc h1 contribution from other quantum nuclei
                for k in range(mol.nuc_num):
                    if k != j:
                        ka = mol.nuc[k].atom_index
                        v1n = get_jk((mol.nuc[j], mol.nuc[j], mol.nuc[k], mol.nuc[k]),
                                     hessobj.base.dm_nuc[k], scripts='ijkl,lk->ij',
                                     intor='int2e_ip1', aosym='s2kl', comp=3)
                        v1n *= -charge * mol.atom_charge(ka)
                        h1_n += v1n
                        h1_n += v1n.transpose(0,2,1)
            elif mol.quantum_nuc[ia]:
                # ia displacement can be from other quantum nuclei, search for it
                for k in range(mol.nuc_num):
                    ka = mol.nuc[k].atom_index
                    if ka == ia:
                        v1n = get_jk((mol.nuc[k], mol.nuc[k], mol.nuc[j], mol.nuc[j]),
                                     hessobj.base.dm_nuc[k], scripts='ijkl,ji->kl',
                                     intor='int2e_ip1', aosym='s2kl', comp=3)
                        h1_n -= v1n * 2.0 * charge * mol.atom_charge(ka)
            if chkfile is None:
                h1ao_n[ia][j] = h1_n
            else:
                key = 'scf_f1ao_n/%d/%d' % (ia, j)
                lib.chkfile.save(chkfile, key, h1_n)
            v1e = v1n = None
        if unrestricted:
            if chkfile is None:
                h1ao_e_a[ia] = h1_e_a
                h1ao_e_b[ia] = h1_e_b
            else:
                lib.chkfile.save(chkfile, 'scf_f1ao/0/%d' % ia, h1_e_a)
                lib.chkfile.save(chkfile, 'scf_f1ao/1/%d' % ia, h1_e_b)
        else:
            if chkfile is None:
                h1ao_e[ia] = h1_e
            else:
                key = 'scf_f1ao/%d' % ia
                lib.chkfile.save(chkfile, key, h1_e)
    if chkfile is None:
        if unrestricted:
            return (h1ao_e_a,h1ao_e_b), h1ao_n
        else:
            return h1ao_e, h1ao_n
    else:
        return chkfile, chkfile

def get_hcore(mol_n):
    '''Part of the second derivatives of core Hamiltonian'''
    ia = mol_n.atom_index
    mol = mol_n.super_mol
    mass = mol.mass[ia] * nist.ATOMIC_MASS / nist.E_MASS
    charge = mol.atom_charge(ia)
    h1aa = mol_n.intor('int1e_ipipkin', comp=9) / mass
    h1aa += mol_n.intor('int1e_ipkinip', comp=9) / mass
    if mol._pseudo or mol_n._pseudo:
        raise NotImplementedError('Nuclear hessian for GTH PP')
    else:
        h1aa -= mol_n.intor('int1e_ipipnuc', comp=9) * charge
        h1aa -= mol_n.intor('int1e_ipnucip', comp=9) * charge
    if mol.has_ecp():
        assert mol_n.has_ecp()
        h1aa -= mol_n.intor('ECPscalar_ipipnuc', comp=9) * charge
        h1aa -= mol_n.intor('ECPscalar_ipnucip', comp=9) * charge
    nao = h1aa.shape[-1]
    return h1aa.reshape(3,3,nao,nao)

def solve_mo1_rks(mf, h1ao_e_or_chkfile, h1ao_n_or_chkfile,
                  fx=None, atmlst=None, max_memory=4000, verbose=None):
    '''Solve the first order equation

    Kwargs:
        fx : function(dm_mo) => v1_mo
            A function to generate the induced potential.
            See also the function gen_vind_rks.
    '''
    # if using chkfile, then they both should be, and should be the same
    if isinstance(h1ao_e_or_chkfile, str) or \
        isinstance(h1ao_n_or_chkfile, str):
        assert h1ao_e_or_chkfile == h1ao_n_or_chkfile
    mol = mf.mol
    if atmlst is None: atmlst = range(mol.natm)
    mf_e = mf.mf_elec
    mf_n = mf.mf_nuc

    mo_coeff_e = mf_e.mo_coeff
    mo_occ_e = mf_e.mo_occ
    nao_e, nmo_e = mo_coeff_e.shape
    mocc_e = mo_coeff_e[:,mo_occ_e>0]
    nocc_e = mocc_e.shape[1]

    mo_coeff_n = []
    mo_occ_n = []
    nao_n = []
    nmo_n = []
    mocc_n = []
    nocc_n = []
    for i in range(mol.nuc_num):
        mo_coeff_n.append(mf_n[i].mo_coeff)
        mo_occ_n.append(mf_n[i].mo_occ)
        tmp1, tmp2 = mo_coeff_n[-1].shape
        nao_n.append(tmp1)
        nmo_n.append(tmp2)
        mocc_n.append(mo_coeff_n[-1][:,mo_occ_n[-1]>0])
        nocc_n.append(mocc_n[-1].shape[1])

    if fx is None:
        fx = gen_vind_rks(mf)
    s1a = -mf_e.mol.intor('int1e_ipovlp', comp=3)

    def _ao2mo(mat, mo_coeff, mocc):
        return numpy.asarray([reduce(numpy.dot, (mo_coeff.T, x, mocc)) for x in mat])

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, max_memory*.9-mem_now)
    # TODO: blksize does not yet take into account nuclear part memory usage
    blksize = max(2, int(max_memory*1e6/8 / (nmo_e*nocc_e*3*6)))
    mo1s_e = [None] * mol.natm
    e1s_e = [None] * mol.natm
    mo1s_n = [None] * mol.natm
    f1s_n = [None] * mol.natm
    aoslices = mol.aoslice_by_atom()
    for ia0, ia1 in lib.prange(0, len(atmlst), blksize):
        s1vo = []
        h1vo_e = []
        for i0 in range(ia0, ia1):
            ia = atmlst[i0]
            shl0, shl1, p0, p1 = aoslices[ia]
            s1ao = numpy.zeros((3,nao_e,nao_e))
            s1ao[:,p0:p1] += s1a[:,p0:p1]
            s1ao[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)
            s1vo.append(_ao2mo(s1ao, mo_coeff_e, mocc_e))
            if isinstance(h1ao_e_or_chkfile, str):
                key = 'scf_f1ao/%d' % ia
                h1ao_e = lib.chkfile.load(h1ao_e_or_chkfile, key)
            else:
                h1ao_e = h1ao_e_or_chkfile[ia]
            h1vo_e.append(_ao2mo(h1ao_e, mo_coeff_e, mocc_e))
        h1vo_e = numpy.vstack(h1vo_e)
        s1vo = numpy.vstack(s1vo)

        h1vo_n = []
        for j in range(mol.nuc_num):
            h1vo_n_j = []
            for i0 in range(ia0, ia1):
                ia = atmlst[i0]
                if isinstance(h1ao_n_or_chkfile, str):
                    key = 'scf_f1ao_n/%d/%d' % (ia, j)
                    h1ao_n = lib.chkfile.load(h1ao_n_or_chkfile, key)
                else:
                    h1ao_n = h1ao_n_or_chkfile[ia][j]
                h1vo_n_j.append(_ao2mo(h1ao_n, mo_coeff_n[j], mocc_n[j]))
            h1vo_n_j = numpy.vstack(h1vo_n_j)
            h1vo_n.append(h1vo_n_j)

        mo1e, e1e, mo1n, f1n = cphf.solve(fx, mf_e, mf_n, h1vo_e, h1vo_n, s1vo,
                                          with_f1n=True, max_cycle=100, verbose=verbose)
        mo1e = numpy.einsum('pq,xqi->xpi', mo_coeff_e, mo1e).reshape(-1,3,nao_e,nocc_e)
        e1e = e1e.reshape(-1,3,nocc_e,nocc_e)

        for k in range(ia1-ia0):
            ia = atmlst[k+ia0]
            if isinstance(h1ao_e_or_chkfile, str):
                key = 'scf_mo1/%d' % ia
                lib.chkfile.save(h1ao_e_or_chkfile, key, mo1e[k])
            else:
                mo1s_e[ia] = mo1e[k]
            e1s_e[ia] = e1e[k].reshape(3,nocc_e,nocc_e)
            mo1s_n[ia] = [None] * mol.nuc_num
            f1s_n[ia] = [None] * mol.nuc_num
        mo1e = e1e = None

        for j in range(mol.nuc_num):
            mo1n[j] = numpy.einsum('pq,xqi->xpi', mo_coeff_n[j],
                                   mo1n[j]).reshape(-1,3,nao_n[j],nocc_n[j])
            f1n[j] = f1n[j].reshape(-1,3,3)
            for k in range(ia1-ia0):
                ia = atmlst[k+ia0]
                if isinstance(h1ao_n_or_chkfile, str):
                    key = 'scf_mo1_n/%d/%d' % (ia, j)
                    lib.chkfile.save(h1ao_n_or_chkfile, key, mo1n[j][k])
                else:
                    mo1s_n[ia][j] = mo1n[j][k]
                f1s_n[ia][j] = f1n[j][k]
        mo1n = f1n = None

    if isinstance(h1ao_e_or_chkfile, str):
        if isinstance(h1ao_n_or_chkfile, str):
            return h1ao_e_or_chkfile, e1s_e, h1ao_n_or_chkfile, f1s_n
        else:
            return h1ao_e_or_chkfile, e1s_e, mo1s_n, f1s_n
    else:
        if isinstance(h1ao_n_or_chkfile, str):
            return mo1s_e, e1s_e, h1ao_n_or_chkfile, f1s_n
        else:
            return mo1s_e, e1s_e, mo1s_n, f1s_n

def gen_vind_rks(mf):
    mol = mf.mol
    mf_e = mf.mf_elec
    mf_n = mf.mf_nuc

    mo_coeff_e = mf_e.mo_coeff
    mo_occ_e = mf_e.mo_occ

    nao_e, nmo_e = mo_coeff_e.shape
    mocc_e = mo_coeff_e[:,mo_occ_e>0]
    nocc_e = mocc_e.shape[1]

    mo_coeff_n = []
    mo_occ_n = []
    nao_n = []
    nmo_n = []
    mocc_n = []
    nocc_n = []
    for i in range(mol.nuc_num):
        mo_coeff_n.append(mf_n[i].mo_coeff)
        mo_occ_n.append(mf_n[i].mo_occ)
        tmp1, tmp2 = mo_coeff_n[-1].shape
        nao_n.append(tmp1)
        nmo_n.append(tmp2)
        mocc_n.append(mo_coeff_n[-1][:,mo_occ_n[-1]>0])
        nocc_n.append(mocc_n[-1].shape[1])

    vresp = mf.gen_response(hermi=1)
    def fx(mo1e, mo1n, f1n=None):
        mo1e = mo1e.reshape(-1,nmo_e,nocc_e)
        nset = len(mo1e)
        dm1e_symm = numpy.empty((nset,nao_e,nao_e))
        dm1e_partial = numpy.empty((nset,nao_e,nao_e))
        for i, x in enumerate(mo1e):
            # *2 for double occupancy
            dm = reduce(numpy.dot, (mo_coeff_e, x*2, mocc_e.T))
            dm1e_symm[i] = dm + dm.T
            dm1e_partial[i] = dm # without c.c.
        dm1n = [None] * mol.nuc_num
        for i in range(mol.nuc_num):
            mo1n[i] = mo1n[i].reshape(-1,nmo_n[i],nocc_n[i])
            assert nset == len(mo1n[i])
            dm1n[i] = numpy.empty((nset, nao_n[i], nao_n[i]))
            for j, x in enumerate(mo1n[i]):
                # without c.c.
                dm1n[i][j] = reduce(numpy.dot, (mo_coeff_n[i], x, mocc_n[i].T))
        v1e, v1n = vresp(dm1e_symm, dm1e_partial, dm1n)
        v1vo_e = numpy.empty_like(mo1e)
        for i, x in enumerate(v1e):
            v1vo_e[i] = reduce(numpy.dot, (mo_coeff_e.T, x, mocc_e))
        v1vo_n = []
        rfn = None
        if f1n is not None:
            rfn = []
        for i in range(mol.nuc_num):
            v1vo_ni = numpy.empty_like(mo1n[i])
            for j, x in enumerate(v1n[i]):
                v1vo_ni[j] = reduce(numpy.dot, (mo_coeff_n[i].T, x, mocc_n[i]))
            if f1n is not None:
                rvo = reduce(numpy.dot, (mo_coeff_n[i].T, mf_n[i].int1e_r, mocc_n[i]))
                # calculate r * f and add to nuclear fock
                v1vo_ni += numpy.einsum('axi,cx->cai', rvo, f1n[i])
                # store r * mo1, which will lead to equation r * mo1 = 0
                rfn.append(numpy.einsum('axi,cai->cx', rvo, mo1n[i]))
            v1vo_n.append(v1vo_ni)
        return v1vo_e, v1vo_n, rfn
    return fx

def solve_mo1_uks(mf, h1ao_e_or_chkfile, h1ao_n_or_chkfile,
                  fx=None, atmlst=None, max_memory=4000, verbose=None):
    '''Solve the first order equation

    Kwargs:
        fx : function(dm_mo) => v1_mo
            A function to generate the induced potential.
            See also the function gen_vind_uks.
    '''
    # if using chkfile, then they both should be, and should be the same
    if isinstance(h1ao_e_or_chkfile, str) or \
        isinstance(h1ao_n_or_chkfile, str):
        assert h1ao_e_or_chkfile == h1ao_n_or_chkfile
    mol = mf.mol
    if atmlst is None: atmlst = range(mol.natm)
    mf_e = mf.mf_elec
    mf_n = mf.mf_nuc

    mo_coeff_e = mf_e.mo_coeff
    mo_occ_e = mf_e.mo_occ
    nao_e, nmo_e = mo_coeff_e[0].shape
    mocc_e_a = mo_coeff_e[0][:,mo_occ_e[0]>0]
    mocc_e_b = mo_coeff_e[1][:,mo_occ_e[1]>0]
    nocc_e_a = mocc_e_a.shape[1]
    nocc_e_b = mocc_e_b.shape[1]

    mo_coeff_n = []
    mo_occ_n = []
    nao_n = []
    nmo_n = []
    mocc_n = []
    nocc_n = []
    for i in range(mol.nuc_num):
        mo_coeff_n.append(mf_n[i].mo_coeff)
        mo_occ_n.append(mf_n[i].mo_occ)
        tmp1, tmp2 = mo_coeff_n[-1].shape
        nao_n.append(tmp1)
        nmo_n.append(tmp2)
        mocc_n.append(mo_coeff_n[-1][:,mo_occ_n[-1]>0])
        nocc_n.append(mocc_n[-1].shape[1])

    if fx is None:
        fx = gen_vind_uks(mf)
    s1a = -mf_e.mol.intor('int1e_ipovlp', comp=3)

    def _ao2mo(mat, mo_coeff, mocc):
        return numpy.asarray([reduce(numpy.dot, (mo_coeff.T, x, mocc)) for x in mat])

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, max_memory*.9-mem_now)
    # TODO: blksize does not yet take into account nuclear part memory usage
    blksize = max(2, int(max_memory*1e6/8 / (nao_e*(nocc_e_a+nocc_e_b)*3*6)))
    mo1s_e_a = [None] * mol.natm
    mo1s_e_b = [None] * mol.natm
    e1s_e_a = [None] * mol.natm
    e1s_e_b = [None] * mol.natm
    mo1s_n = [None] * mol.natm
    f1s_n = [None] * mol.natm
    aoslices = mol.aoslice_by_atom()
    for ia0, ia1 in lib.prange(0, len(atmlst), blksize):
        s1vo_a = []
        s1vo_b = []
        h1vo_e_a = []
        h1vo_e_b = []
        for i0 in range(ia0, ia1):
            ia = atmlst[i0]
            shl0, shl1, p0, p1 = aoslices[ia]
            s1ao = numpy.zeros((3,nao_e,nao_e))
            s1ao[:,p0:p1] += s1a[:,p0:p1]
            s1ao[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)
            s1vo_a.append(_ao2mo(s1ao, mo_coeff_e[0], mocc_e_a))
            s1vo_b.append(_ao2mo(s1ao, mo_coeff_e[1], mocc_e_b))
            if isinstance(h1ao_e_or_chkfile, str):
                h1ao_e_a = lib.chkfile.load(h1ao_e_or_chkfile, 'scf_f1ao/0/%d'%ia)
                h1ao_e_b = lib.chkfile.load(h1ao_e_or_chkfile, 'scf_f1ao/1/%d'%ia)
            else:
                h1ao_e_a = h1ao_e_or_chkfile[0][ia]
                h1ao_e_b = h1ao_e_or_chkfile[1][ia]
            h1vo_e_a.append(_ao2mo(h1ao_e_a, mo_coeff_e[0], mocc_e_a))
            h1vo_e_b.append(_ao2mo(h1ao_e_b, mo_coeff_e[1], mocc_e_b))
        h1vo_e = (numpy.vstack(h1vo_e_a), numpy.vstack(h1vo_e_b))
        s1vo = (numpy.vstack(s1vo_a), numpy.vstack(s1vo_b))

        h1vo_n = []
        for j in range(mol.nuc_num):
            h1vo_n_j = []
            for i0 in range(ia0, ia1):
                ia = atmlst[i0]
                if isinstance(h1ao_n_or_chkfile, str):
                    key = 'scf_f1ao_n/%d/%d' % (ia, j)
                    h1ao_n = lib.chkfile.load(h1ao_n_or_chkfile, key)
                else:
                    h1ao_n = h1ao_n_or_chkfile[ia][j]
                h1vo_n_j.append(_ao2mo(h1ao_n, mo_coeff_n[j], mocc_n[j]))
            h1vo_n_j = numpy.vstack(h1vo_n_j)
            h1vo_n.append(h1vo_n_j)

        mo1e, e1e, mo1n, f1n = ucphf.solve(fx, mf_e, mf_n, h1vo_e, h1vo_n, s1vo,
                                           with_f1n=True, max_cycle=100, verbose=verbose)
        mo1e_a = numpy.einsum('pq,xqi->xpi', mo_coeff_e[0], mo1e[0]).reshape(-1,3,nao_e,nocc_e_a)
        mo1e_b = numpy.einsum('pq,xqi->xpi', mo_coeff_e[1], mo1e[1]).reshape(-1,3,nao_e,nocc_e_b)
        e1e_a = e1e[0].reshape(-1,3,nocc_e_a,nocc_e_a)
        e1e_b = e1e[1].reshape(-1,3,nocc_e_b,nocc_e_b)

        for k in range(ia1-ia0):
            ia = atmlst[k+ia0]
            if isinstance(h1ao_e_or_chkfile, str):
                lib.chkfile.save(h1ao_e_or_chkfile, 'scf_mo1/0/%d'%ia, mo1e_a[k])
                lib.chkfile.save(h1ao_e_or_chkfile, 'scf_mo1/1/%d'%ia, mo1e_b[k])
            else:
                mo1s_e_a[ia] = mo1e_a[k]
                mo1s_e_b[ia] = mo1e_b[k]
            e1s_e_a[ia] = e1e_a[k].reshape(3,nocc_e_a,nocc_e_a)
            e1s_e_b[ia] = e1e_b[k].reshape(3,nocc_e_b,nocc_e_b)
            mo1s_n[ia] = [None] * mol.nuc_num
            f1s_n[ia] = [None] * mol.nuc_num
        mo1e = e1e = None

        for j in range(mol.nuc_num):
            mo1n[j] = numpy.einsum('pq,xqi->xpi', mo_coeff_n[j],
                                   mo1n[j]).reshape(-1,3,nao_n[j],nocc_n[j])
            f1n[j] = f1n[j].reshape(-1,3,3)
            for k in range(ia1-ia0):
                ia = atmlst[k+ia0]
                if isinstance(h1ao_n_or_chkfile, str):
                    key = 'scf_mo1_n/%d/%d' % (ia, j)
                    lib.chkfile.save(h1ao_n_or_chkfile, key, mo1n[j][k])
                else:
                    mo1s_n[ia][j] = mo1n[j][k]
                f1s_n[ia][j] = f1n[j][k]
        mo1n = f1n = None

    if isinstance(h1ao_e_or_chkfile, str):
        if isinstance(h1ao_n_or_chkfile, str):
            return h1ao_e_or_chkfile, (e1s_e_a,e1s_e_b), h1ao_n_or_chkfile, f1s_n
        else:
            return h1ao_e_or_chkfile, (e1s_e_a,e1s_e_b), mo1s_n, f1s_n
    else:
        if isinstance(h1ao_n_or_chkfile, str):
            return (mo1s_e_a,mo1s_e_b), (e1s_e_a,e1s_e_b), h1ao_n_or_chkfile, f1s_n
        else:
            return (mo1s_e_a,mo1s_e_b), (e1s_e_a,e1s_e_b), mo1s_n, f1s_n

def gen_vind_uks(mf):
    mol = mf.mol
    mf_e = mf.mf_elec
    mf_n = mf.mf_nuc

    mo_coeff_e = mf_e.mo_coeff
    mo_occ_e = mf_e.mo_occ

    nao_e, nmo_e_a = mo_coeff_e[0].shape
    nmo_e_b = mo_coeff_e[1].shape[1]
    mocc_e_a = mo_coeff_e[0][:,mo_occ_e[0]>0]
    mocc_e_b = mo_coeff_e[1][:,mo_occ_e[1]>0]
    nocc_e_a = mocc_e_a.shape[1]
    nocc_e_b = mocc_e_b.shape[1]

    mo_coeff_n = []
    mo_occ_n = []
    nao_n = []
    nmo_n = []
    mocc_n = []
    nocc_n = []
    for i in range(mol.nuc_num):
        mo_coeff_n.append(mf_n[i].mo_coeff)
        mo_occ_n.append(mf_n[i].mo_occ)
        tmp1, tmp2 = mo_coeff_n[-1].shape
        nao_n.append(tmp1)
        nmo_n.append(tmp2)
        mocc_n.append(mo_coeff_n[-1][:,mo_occ_n[-1]>0])
        nocc_n.append(mocc_n[-1].shape[1])

    vresp = mf.gen_response(hermi=1)
    def fx(mo1e, mo1n, f1n=None):
        mo1e = mo1e.reshape(-1,nmo_e_a*nocc_e_a+nmo_e_b*nocc_e_b)
        nset = len(mo1e)
        dm1e_symm = numpy.empty((2,nset,nao_e,nao_e))
        dm1e_partial = numpy.empty((nset,nao_e,nao_e))
        for i, x in enumerate(mo1e):
            xa = x[:nmo_e_a*nocc_e_a].reshape(nmo_e_a,nocc_e_a)
            xb = x[nmo_e_a*nocc_e_a:].reshape(nmo_e_b,nocc_e_b)
            dma = reduce(numpy.dot, (mo_coeff_e[0], xa, mocc_e_a.T))
            dmb = reduce(numpy.dot, (mo_coeff_e[1], xb, mocc_e_b.T))
            dm1e_symm[0,i] = dma + dma.T
            dm1e_symm[1,i] = dmb + dmb.T
            dm1e_partial[i] = dma + dmb # without c.c.
        dm1n = [None] * mol.nuc_num
        for i in range(mol.nuc_num):
            mo1n[i] = mo1n[i].reshape(-1,nmo_n[i],nocc_n[i])
            assert nset == len(mo1n[i])
            dm1n[i] = numpy.empty((nset, nao_n[i], nao_n[i]))
            for j, x in enumerate(mo1n[i]):
                # without c.c.
                dm1n[i][j] = reduce(numpy.dot, (mo_coeff_n[i], x, mocc_n[i].T))
        v1e, v1n = vresp(dm1e_symm, dm1e_partial, dm1n)
        v1vo_e = numpy.empty_like(mo1e)
        for i in range(nset):
            v1vo_e[i,:nmo_e_a*nocc_e_a] = \
                reduce(numpy.dot, (mo_coeff_e[0].T, v1e[0,i], mocc_e_a)).ravel()
            v1vo_e[i,nmo_e_a*nocc_e_a:] = \
                reduce(numpy.dot, (mo_coeff_e[1].T, v1e[1,i], mocc_e_b)).ravel()
        v1vo_n = []
        rfn = None
        if f1n is not None:
            rfn = []
        for i in range(mol.nuc_num):
            v1vo_ni = numpy.empty_like(mo1n[i])
            for j, x in enumerate(v1n[i]):
                v1vo_ni[j] = reduce(numpy.dot, (mo_coeff_n[i].T, x, mocc_n[i]))
            if f1n is not None:
                rvo = reduce(numpy.dot, (mo_coeff_n[i].T, mf_n[i].int1e_r, mocc_n[i]))
                # calculate r * f and add to nuclear fock
                v1vo_ni += numpy.einsum('axi,cx->cai', rvo, f1n[i])
                # store r * mo1, which will lead to equation r * mo1 = 0
                rfn.append(numpy.einsum('axi,cai->cx', rvo, mo1n[i]))
            v1vo_n.append(v1vo_ni)
        return v1vo_e, v1vo_n, rfn
    return fx


class Hessian(lib.StreamObject):
    '''
    Examples::

    >>> from pyscf import neo
    >>> mol = neo.Mole()
    >>> mol.build(atom='H 0 0 0.00; C 0 0 1.064; N 0 0 2.220', basis='ccpvdz')
    >>> mf = neo.CDFT(mol)
    >>> mf.mf_elec.xc = 'b3lyp'
    >>> mf.scf()
    >>> hess = neo.Hessian(mf)
    >>> h = hess.kernel()
    >>> freq_info = hess.harmonic_analysis(mol, h)
    >>> print(freq_info)
    '''

    def __init__(self, scf_method):
        self.base = scf_method
        if self.base.epc is not None:
            raise NotImplementedError('Hessian with epc is not implemented')
        self.verbose = scf_method.verbose
        self.mol = scf_method.mol
        self.chkfile = scf_method.chkfile
        self.max_memory = self.mol.max_memory
        self.grid_response = None

        self.atmlst = range(self.mol.natm)
        self.de = numpy.zeros((0,0,3,3))  # (A,B,dR_A,dR_B)
        self._keys = set(self.__dict__.keys())

    partial_hess_cneo = partial_hess_cneo
    hess_cneo = hess_cneo
    make_h1 = make_h1

    def hcore_generator(self, mol_n):
        mol = mol_n.super_mol
        with_x2c = getattr(self.base, 'with_x2c', None)
        if with_x2c:
            raise NotImplementedError('X2C not supported')
        with_ecp = mol.has_ecp()
        if with_ecp:
            assert mol_n.has_ecp()
            ecp_atoms = set(mol_n._ecpbas[:,gto.ATOM_OF])
        else:
            ecp_atoms = ()
        nao = mol_n.nao_nr()
        ia = mol_n.atom_index
        charge = mol.atom_charge(ia)
        def hcore_deriv(iatm, jatm):
            zi = mol.atom_charge(iatm)
            zj = mol.atom_charge(jatm)
            hcore = 0.0
            if iatm == jatm:
                if iatm == ia:
                    hcore = get_hcore(mol_n)
                elif not mol.quantum_nuc[iatm]:
                    with mol_n.with_rinv_at_nucleus(iatm):
                        hcore = mol_n.intor('int1e_ipiprinv', comp=9)
                        hcore += mol_n.intor('int1e_iprinvip', comp=9)
                        hcore *= zi
                        if with_ecp and iatm in ecp_atoms:
                            # ECP rinv has the same sign as ECP nuc,
                            # unlike regular rinv = -nuc.
                            # reverse the sign to mimic regular rinv
                            hcore -= mol_n.intor('ECPscalar_ipiprinv', comp=9)
                            hcore -= mol_n.intor('ECPscalar_iprinvip', comp=9)
                        hcore = charge * hcore.reshape(3,3,nao,nao)
            else:
                if iatm == ia and not mol.quantum_nuc[jatm]:
                    with mol_n.with_rinv_at_nucleus(jatm):
                        hcore = mol_n.intor('int1e_ipiprinv', comp=9)
                        hcore += mol_n.intor('int1e_iprinvip', comp=9)
                        hcore *= zj
                        if with_ecp and jatm in ecp_atoms:
                            hcore -= mol_n.intor('ECPscalar_ipiprinv', comp=9)
                            hcore -= mol_n.intor('ECPscalar_iprinvip', comp=9)
                        hcore = -charge * hcore.reshape(3,3,nao,nao)
                elif jatm == ia and not mol.quantum_nuc[iatm]:
                    with mol_n.with_rinv_at_nucleus(iatm):
                        hcore = mol_n.intor('int1e_ipiprinv', comp=9)
                        hcore += mol_n.intor('int1e_iprinvip', comp=9)
                        hcore *= zi
                        if with_ecp and iatm in ecp_atoms:
                            hcore -= mol_n.intor('ECPscalar_ipiprinv', comp=9)
                            hcore -= mol_n.intor('ECPscalar_iprinvip', comp=9)
                        hcore = -charge * hcore.reshape(3,3,nao,nao)
            if isinstance(hcore, numpy.ndarray):
                return hcore + hcore.conj().transpose(0,1,3,2)
            else:
                return 0.0
        return hcore_deriv

    def solve_mo1(self, h1ao_e_or_chkfile, h1ao_n_or_chkfile,
                  fx=None, atmlst=None, max_memory=4000, verbose=None):
        if self.base.unrestricted:
            return solve_mo1_uks(self.base, h1ao_e_or_chkfile, h1ao_n_or_chkfile,
                                 fx, atmlst, max_memory, verbose)
        else:
            return solve_mo1_rks(self.base, h1ao_e_or_chkfile, h1ao_n_or_chkfile,
                                 fx, atmlst, max_memory, verbose)

    def hess_elec(self, mo1e, e1e, h1ao_e, max_memory=4000, verbose=None):
        '''Hessian of electrons and classic nuclei in CNEO'''
        hobj = self.base.mf_elec.Hessian()
        if self.grid_response is not None:
            hobj.grid_response = self.grid_response
        de2 = hobj.hess_elec(mo1=mo1e, mo_e1=e1e, h1ao=h1ao_e,
                             max_memory=max_memory, verbose=verbose) \
              + hobj.hess_nuc()
        if self.base.disp is not None:
            self.base.mf_elec.disp = self.base.disp
            de2 += hobj.get_dispersion()
        return de2

    def kernel(self, atmlst=None):
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        log = logger.new_logger(self, self.verbose)
        time0 = t1 = (logger.process_clock(), logger.perf_counter())
        h1ao_e, h1ao_n = self.make_h1(chkfile=self.chkfile, atmlst=atmlst,
                                      verbose=log)
        t1 = log.timer_debug1('making H1', *time0)

        mo1e, e1e, mo1n, _ = self.solve_mo1(h1ao_e, h1ao_n, atmlst=atmlst,
                                            max_memory=self.max_memory,
                                            verbose=log)
        t1 = log.timer_debug1('solving MO1', *t1)
        de = self.hess_elec(mo1e, e1e, h1ao_e, max_memory=self.max_memory,
                            verbose=log)
        t1 = log.timer_debug1('electronic part Hessian', *t1)
        de += self.hess_cneo(mo1n, h1ao_n, atmlst=atmlst,
                             max_memory=self.max_memory, verbose=log)
        t1 = log.timer_debug1('nuclear part Hessian', *t1)
        self.de = de
        log.timer('CNEO hessian', *time0)
        return self.de
    hess = kernel

    def harmonic_analysis(self, mol, hess, exclude_trans=True, exclude_rot=True,
                          imaginary_freq=True, mass=None):
        if mass is None:
            mass = mol.mass
        return harmonic_analysis(mol, hess, exclude_trans=exclude_trans,
                                 exclude_rot=exclude_rot, imaginary_freq=imaginary_freq,
                                 mass=mass)

    def thermo(self, model, freq, temperature=298.15, pressure=101325):
        '''Copy from pyscf.hessian.thermo.thermo only to change the definition of mass.
        It should support mass input just like harmonic_analysis'''
        mol = model.mol
        atom_coords = mol.atom_coords()
        mass = mol.mass # NOTE: only this line is different from pyscf.hessian.thermo.thermo
        mass_center = numpy.einsum('z,zx->x', mass, atom_coords) / mass.sum()
        atom_coords = atom_coords - mass_center

        kB = nist.BOLTZMANN
        h = nist.PLANCK
        # c = nist.LIGHT_SPEED_SI
        # beta = 1. / (kB * temperature)
        R_Eh = kB*nist.AVOGADRO / (nist.HARTREE2J * nist.AVOGADRO)

        results = {}
        results['temperature'] = (temperature, 'K')
        results['pressure'] = (pressure, 'Pa')

        E0 = model.e_tot
        results['E0'] = (E0, 'Eh')

        # Electronic part
        results['S_elec' ] = (R_Eh * numpy.log(mol.multiplicity), 'Eh/K')
        results['Cv_elec'] = results['Cp_elec'] = (0, 'Eh/K')
        results['E_elec' ] = results['H_elec' ] = (E0, 'Eh')

        # Translational part. See also https://cccbdb.nist.gov/thermo.asp for the
        # partition function q_trans
        mass_tot = mass.sum() * nist.ATOMIC_MASS
        q_trans = ((2.0 * numpy.pi * mass_tot * kB * temperature / h**2)**1.5
                   * kB * temperature / pressure)
        results['S_trans' ] = (R_Eh * (2.5 + numpy.log(q_trans)), 'Eh/K')
        results['Cv_trans'] = (1.5 * R_Eh, 'Eh/K')
        results['Cp_trans'] = (2.5 * R_Eh, 'Eh/K')
        results['E_trans' ] = (1.5 * R_Eh * temperature, 'Eh')
        results['H_trans' ] = (2.5 * R_Eh * temperature, 'Eh')

        # Rotational part
        rot_const = rotation_const(mass, atom_coords, 'GHz')
        results['rot_const'] = (rot_const, 'GHz')
        rotor_type = _get_rotor_type(rot_const)

        sym_number = rotational_symmetry_number(mol)
        results['sym_number'] = (sym_number, '')

        # partition function q_rot (https://cccbdb.nist.gov/thermo.asp)
        if rotor_type == 'ATOM':
            results['S_rot' ] = (0, 'Eh/K')
            results['Cv_rot'] = results['Cp_rot'] = (0, 'Eh/K')
            results['E_rot' ] = results['H_rot' ] = (0, 'Eh')
        elif rotor_type == 'LINEAR':
            B = rot_const[1] * 1e9
            q_rot = kB * temperature / (sym_number * h * B)
            results['S_rot' ] = (R_Eh * (1 + numpy.log(q_rot)), 'Eh/K')
            results['Cv_rot'] = results['Cp_rot'] = (R_Eh, 'Eh/K')
            results['E_rot' ] = results['H_rot' ] = (R_Eh * temperature, 'Eh')
        else:
            ABC = rot_const * 1e9
            q_rot = ((kB*temperature/h)**1.5 * numpy.pi**.5
                     / (sym_number * numpy.prod(ABC)**.5))
            results['S_rot' ] = (R_Eh * (1.5 + numpy.log(q_rot)), 'Eh/K')
            results['Cv_rot'] = results['Cp_rot'] = (1.5 * R_Eh, 'Eh/K')
            results['E_rot' ] = results['H_rot' ] = (1.5 * R_Eh * temperature, 'Eh')

        # Vibrational part.
        au2hz = (nist.HARTREE2J / (nist.ATOMIC_MASS * nist.BOHR_SI**2))**.5 / (2 * numpy.pi)
        idx = freq.real > 0
        vib_temperature = freq.real[idx] * au2hz * h / kB
        # reduced_temperature
        rt = vib_temperature / max(1e-14, temperature)
        e = numpy.exp(-rt)

        ZPE = R_Eh * .5 * vib_temperature.sum()
        results['ZPE'] = (ZPE, 'Eh')

        results['S_vib' ] = (R_Eh * (rt*e/(1-e) - numpy.log(1-e)).sum(), 'Eh/K')
        results['Cv_vib'] = results['Cp_vib'] = (R_Eh * (e * rt**2/(1-e)**2).sum(), 'Eh/K')
        results['E_vib' ] = results['H_vib' ] = \
                (ZPE + R_Eh * temperature * (rt * e / (1-e)).sum(), 'Eh')

        results['G_elec' ] = (results['H_elec' ][0] - temperature * results['S_elec' ][0], 'Eh')
        results['G_trans'] = (results['H_trans'][0] - temperature * results['S_trans'][0], 'Eh')
        results['G_rot'  ] = (results['H_rot'  ][0] - temperature * results['S_rot'  ][0], 'Eh')
        results['G_vib'  ] = (results['H_vib'  ][0] - temperature * results['S_vib'  ][0], 'Eh')

        def _sum(f):
            keys = ('elec', 'trans', 'rot', 'vib')
            return sum(results.get(f+'_'+key, (0,))[0] for key in keys)
        results['S_tot' ] = (_sum('S' ), 'Eh/K')
        results['Cv_tot'] = (_sum('Cv'), 'Eh/K')
        results['Cp_tot'] = (_sum('Cp'), 'Eh/K')
        results['E_0K' ]  = (E0 + ZPE, 'Eh')
        results['E_tot' ] = (_sum('E'), 'Eh')
        results['H_tot' ] = (_sum('H'), 'Eh')
        results['G_tot' ] = (_sum('G'), 'Eh')

        return results


from pyscf.neo import cdft
cdft.CDFT.Hessian = lib.class_as_method(Hessian)
