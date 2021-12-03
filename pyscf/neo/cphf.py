#!/usr/bin/env python

'''
Coupled perturbed Hartree-Fock for constrained nuclear-electronic orbital method
'''

import numpy
from pyscf import lib, gto, scf
from pyscf.hessian.rks import Hessian
from pyscf.data import nist
from pyscf.lib import logger
from functools import reduce

class CPHF(lib.StreamObject):
    def __init__(self, scf_method):
        self.mol = scf_method.mol
        self.base = scf_method
        self.atmlst = range(self.mol.natm)

        global occidx_e, viridx_e
        occidx_e = self.base.mf_elec.mo_occ > 0
        viridx_e = self.base.mf_elec.mo_occ == 0
        #nset = 3*len(self.atmlst)
        self.verbose = 4

    def ao2mo(self, mf, mat):
        return numpy.asarray([reduce(numpy.dot, (mf.mo_coeff.T, x, mf.mo_coeff[:,mf.mo_occ>0])) for x in mat])

    def mo12ne(self, mo1):
        'transfer mo1 from a 1-d array to mo1_e, mo1_n and f1 with proper shape'

        mo_coeff = self.base.mf_elec.mo_coeff
        mo_occ = self.base.mf_elec.mo_occ
        nao, nmo = mo_coeff.shape
        mocc = mo_coeff[:,mo_occ>0]
        nocc = mocc.shape[1]
        nset = 3*len(self.atmlst)

        n_e = nset * nmo * nocc
        index = n_e
        mo1 = mo1.ravel()
        mo1_e = mo1[:index].reshape(nset, nmo, nocc)

        mo1_n = []
        for i in range(len(self.mol.nuc)):
            mo_coeff = self.base.mf_nuc[i].mo_coeff
            mo_occ = self.base.mf_nuc[i].mo_occ
            mocc = mo_coeff[:,mo_occ>0]
            nocc = mocc.shape[1]
            nao, nmo = mo_coeff.shape
            add = nset * nmo * nocc # nocc = 1 for quantum nuclei actually
            mo1_n.append(mo1[index: index + add].reshape(nset, nmo, nocc))
            index += add

        f1 = []
        for i in range(len(self.mol.nuc)):
            add = nset*3
            f1.append(mo1[index:index + add].reshape(nset, 3))
            index += add

        assert(index == len(mo1))

        return mo1_e, mo1_n, f1

    def get_A_e(self, mo1_e, mo1_n):
        'get the response matrix for electrons'

        mf_e = self.base.mf_elec
        mo_coeff_e = mf_e.mo_coeff
        mo_occ_e = mf_e.mo_occ
        nao_e, nmo_e = mo_coeff_e.shape
        nset = 3*len(self.atmlst)

        # calculate A^e * U^e
        vresp = mf_e.gen_response(mo_coeff_e, mo_occ_e, hermi=1)
        dm1 = numpy.empty((nset, nao_e, nao_e))
        for i, x in enumerate(mo1_e):
            dm = reduce(numpy.dot, (mo_coeff_e, x*2, mo_coeff_e[:, mo_occ_e>0].T)) # *2 for double occupancy
            dm1[i] = dm + dm.T
        v1 = -vresp(dm1)
        v1vo = numpy.empty_like(mo1_e)
        for i, x in enumerate(v1):
            v1vo[i] = reduce(numpy.dot, (mo_coeff_e.T, x, mo_coeff_e[:, mo_occ_e>0]))

        # calculate C^e * U^n
        for i in range(len(self.mol.nuc)):
            mo_coeff_n = self.base.mf_nuc[i].mo_coeff
            mo_occ_n = self.base.mf_nuc[i].mo_occ

            ia = self.mol.nuc[i].atom_index
            C = numpy.empty_like(mo1_e)
            for j, x in enumerate(mo1_n[i]):
                dm_n = numpy.einsum('ij, mi, nj->mn', x, mo_coeff_n, mo_coeff_n[:,mo_occ_n>0])
                v1en_ao = scf.jk.get_jk(
                    (self.mol.elec, self.mol.elec, self.mol.nuc[i], self.mol.nuc[i]), dm_n, scripts='ijkl,lk->ij', intor='int2e', aosym='s4')
                v1en_mo = numpy.einsum('mn, mi, na->ia', v1en_ao, mo_coeff_e, mo_coeff_e[:,mo_occ_e>0])
                C[j] = 2*self.mol.atom_charge(ia)*v1en_mo

            v1vo += C

        return v1vo

    def get_A_n(self, i, mo1_e, mo1_n, f1):
        'get the response of quantum nuclei'
        mf_n = self.base.mf_nuc[i]
        mo_coeff_n = mf_n.mo_coeff
        mo_occ_n = mf_n.mo_occ

        # calculate C^e^T * U^e
        mf_e = self.base.mf_elec
        mo_coeff_e = mf_e.mo_coeff
        mo_occ_e = mf_e.mo_occ

        ia = self.mol.nuc[i].atom_index
        C = numpy.empty_like(mo1_n[i])
        for j, x in enumerate(mo1_e):
            dm_e = numpy.einsum('ij, mi, nj->mn', x, mo_coeff_e, mo_coeff_e[:,mo_occ_e>0])
            v_ne_ao = scf.jk.get_jk((self.mol.nuc[i], self.mol.nuc[i], self.mol.elec,
                                    self.mol.elec), dm_e, scripts='ijkl,lk->ij', intor='int2e', aosym='s4')
            v_ne_mo = numpy.einsum('mn, mi, na->ia', v_ne_ao, mo_coeff_n, mo_coeff_n[:,mo_occ_n>0])
            C[j] = 4*self.mol.atom_charge(ia)*v_ne_mo

        for j in range(self.mol.nuc_num):
            if j != i:
                mf_j = self.base.mf_nuc[j]
                mo_coeff_j = mf_j.mo_coeff
                mo_occ_j = mf_j.mo_occ
                ja = self.mol.nuc[j].atom_index

                for k, x in enumerate(mo1_n[j]):
                    dm_nj = numpy.einsum('ij, mi, nj->mn', x, mo_coeff_j, mo_coeff_j[:,mo_occ_j>0])
                    v_nn_ao = scf.jk.get_jk((self.mol.nuc[i], self.mol.nuc[i], self.mol.nuc[j],
                                            self.mol.nuc[j]), dm_nj, scripts='ijkl,lk->ij', intor='int2e', aosym='s4')
                    v_nn_mo = numpy.einsum('mn,mi,na->ia', v_nn_ao, mo_coeff_n, mo_coeff_n[:,mo_occ_n>0])
                    C[k] -= 2*self.mol.atom_charge(ja)*self.mol.atom_charge(ia)*v_nn_mo

        # calculate R * F
        R_ao = self.mol.nuc[i].intor('int1e_r', comp=3)
        R_mo = numpy.einsum('xmn, mi, na->xia', R_ao, mo_coeff_n, mo_coeff_n[:, mo_occ_n>0])
        C -= numpy.einsum('xia, cx->cia', R_mo, f1[i])

        return C

    def get_R(self, i, mo1_n_i, f1_i):
        'get the R matrix for the i-th quantum nuclei'
        mf_n = self.base.mf_nuc[i]
        mo_coeff = mf_n.mo_coeff
        mo_occ = mf_n.mo_occ

        R_ao = self.mol.nuc[i].intor('int1e_r', comp=3)
        R_mo = numpy.einsum('xmn, mi, na->xia', R_ao, mo_coeff, mo_coeff[:, mo_occ>0])
        R = numpy.einsum('xia, cia ->cx', R_mo, mo1_n_i)*2

        # subtract identity matrix times f because (1+a)x=b is solved actually
        R -= f1_i
        return R

    def full_response(self, mo1):
        'set up the full matrix and multiply by mo1'
        mo1_e, mo1_n, f1 = self.mo12ne(mo1)

        e_a = self.base.mf_elec.mo_energy[viridx_e]
        e_i = self.base.mf_elec.mo_energy[occidx_e]
        e_ai = -1 / lib.direct_sum('a-i->ai', e_a, e_i)

        A_e = self.get_A_e(mo1_e, mo1_n)
        A_e[:,viridx_e,:] *= e_ai
        A_e[:,occidx_e,:] = 0
        response = A_e.ravel()

        for i in range(len(self.mol.nuc)):
            mf_n = self.base.mf_nuc[i]
            mo_occ_n = mf_n.mo_occ
            e_a = mf_n.mo_energy[mo_occ_n == 0]
            e_i = mf_n.mo_energy[mo_occ_n > 0]
            e_ai = -1 / lib.direct_sum('a-i->ai', e_a, e_i)

            A_n = self.get_A_n(i, mo1_e, mo1_n, f1)
            A_n[:,mo_occ_n==0,:] *= e_ai
            A_n[:,mo_occ_n>0,:] = 0
            response = numpy.concatenate((response, A_n.ravel()))
        for i in range(len(self.mol.nuc)):
            response = numpy.concatenate((response, self.get_R(i, mo1_n[i], f1[i]).ravel()))
        return response

    def get_Bmat_elec(self, mf_e):
        'get the B matrix for electrons w.r.t. the displacement of nuclei'
        mol = mf_e.mol

        e_a = mf_e.mo_energy[viridx_e]
        e_i = mf_e.mo_energy[occidx_e]
        e_ai = -1 / lib.direct_sum('a-i->ai', e_a, e_i)
        nvir, nocc = e_ai.shape
        nao, nmo = mf_e.mo_coeff.shape

        # Note: h1ao includes v1_ee
        hobj = mf_e.Hessian()
        h1ao = hobj.make_h1(mf_e.mo_coeff, mf_e.mo_occ) # TODO: use checkfile to save time

        s1a = -mol.intor('int1e_ipovlp', comp=3)
        aoslices = mol.aoslice_by_atom()
        Bs = []
        s1 = []
        for a in self.atmlst:
            shl0, shl1, p0, p1 = aoslices[a, :]

            s1ao = numpy.zeros((3,nao,nao))
            v1ao = numpy.zeros((3,nao,nao))

            s1ao[:,p0:p1] += s1a[:,p0:p1]
            s1ao[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)

            for i in range(len(self.mol.nuc)):
                vi = numpy.zeros((3, nao, nao))
                ia = self.mol.nuc[i].atom_index
                v1_e = -scf.jk.get_jk((self.mol.elec, self.mol.elec, self.mol.nuc[i], self.mol.nuc[i]),
                                      self.base.dm_nuc[i], scripts='ijkl,lk->ij', intor='int2e_ip1', comp=3, aosym='s2kl')
                vi[:,p0:p1] += v1_e[:,p0:p1]
                vi[:,:,p0:p1] += v1_e[:,p0:p1].transpose(0,2,1)
                if ia == a:
                    v1_n = -scf.jk.get_jk((self.mol.nuc[i], self.mol.nuc[i], self.mol.elec, self.mol.elec),
                                          self.base.dm_nuc[i], scripts='ijkl,ji->kl', intor='int2e_ip1', comp=3, aosym='s2kl')
                    vi += v1_n*2

                v1ao -= vi*self.mol.atom_charge(ia)

            s1vo = self.ao2mo(mf_e, s1ao)
            s1.append(s1vo)
            B = self.ao2mo(mf_e, h1ao[a] + v1ao) - s1vo*e_i
            Bs.append(B)
        return numpy.array(Bs).reshape(-1, nmo, nocc), numpy.array(s1).reshape(-1, nmo, nocc)

    def get_Bmat_nuc(self, i):
        'get B matrix for the i-th quantum nuclei w.r.t the the displacement of nuclei'
        mf_n = self.base.mf_nuc[i]
        ia = self.mol.nuc[i].atom_index

        occidx = mf_n.mo_occ > 0
        viridx = mf_n.mo_occ == 0
        e_i = mf_n.mo_energy[occidx]
        e_a = mf_n.mo_energy[viridx]
        e_ai = -1 / lib.direct_sum('a-i->ai', e_a, e_i)
        nvir, nocc = e_ai.shape
        nao, nmo = mf_n.mo_coeff.shape

        Bs = []
        s1 = []
        for a in self.atmlst:
            h1ao = numpy.zeros((3,nao,nao))
            v1ao = numpy.zeros((3,nao,nao))
            s1ao = numpy.zeros((3,nao,nao))
            f1ao = numpy.zeros((3,nao,nao))

            shl0, shl1, p0, p1 = self.mol.aoslice_by_atom()[a, :]
            shls_slice = (shl0, shl1) + (0, self.mol.elec.nbas) + (0, self.mol.nuc[i].nbas)*2
            v1_en = -scf.jk.get_jk((self.mol.elec, self.mol.elec, self.mol.nuc[i], self.mol.nuc[i]), self.base.dm_elec[:,
                                   p0:p1], scripts='ijkl,ji->kl', intor='int2e_ip1', shls_slice = shls_slice, comp=3, aosym='s2kl')
            v1ao -= v1_en*self.mol.atom_charge(ia)*2 # *2 for c.c.

            if ia == a:
                mass = self.mol.mass[ia] * nist.ATOMIC_MASS / nist.E_MASS
                h1n = -self.mol.nuc[i].intor('int1e_ipkin', comp=3)/mass
                h1n += self.mol.nuc[i].intor('int1e_ipnuc', comp=3)*self.mol.atom_charge(ia)
                h1n += h1n.transpose(0, 2, 1)
                h1ao += h1n

                v1_ne = -scf.jk.get_jk((self.mol.nuc[i], self.mol.nuc[i], self.mol.elec, self.mol.elec),
                                       self.base.dm_elec, scripts='ijkl,lk->ij', intor='int2e_ip1', comp=3, aosym='s2kl')
                v1_ne += v1_ne.transpose(0, 2, 1)
                v1ao -= v1_ne*self.mol.atom_charge(ia)

                f1ao = numpy.einsum(
                    'ijkl,j->ikl', -self.mol.nuc[i].intor('int1e_irp').reshape(3, 3, nao, nao), self.base.f[ia])
                f1ao += f1ao.transpose(0, 2, 1)

                s1ao = -self.mol.nuc[i].intor('int1e_ipovlp', comp=3)
                s1ao += s1ao.transpose(0, 2, 1)

                for j in range(len(self.mol.nuc)):
                    if j != i:
                        ja = self.mol.nuc[j].atom_index
                        v1nn = -scf.jk.get_jk((self.mol.nuc[i], self.mol.nuc[i], self.mol.nuc[j], self.mol.nuc[j]),
                                              self.base.dm_nuc[j], scripts='ijkl,lk->ij', intor='int2e_ip1', comp=3, aosym='s2kl')
                        v1nn += v1nn.transpose(0, 2, 1)
                        v1ao += v1nn*self.mol.atom_charge(ia)*self.mol.atom_charge(ja)

            else:
                if self.mol.quantum_nuc[a] == True:
                    for j in range(len(self.mol.nuc)):
                        ja = self.mol.nuc[j].atom_index
                        if ja == a:
                            v1_nn2 = -scf.jk.get_jk((self.mol.nuc[j], self.mol.nuc[j], self.mol.nuc[i], self.mol.nuc[i]),
                                                    self.base.dm_nuc[j], scripts='ijkl,ji->kl', intor='int2e_ip1', comp=3, aosym='s2kl')
                            v1ao += v1_nn2*self.mol.atom_charge(ia)*self.mol.atom_charge(ja)*2
                else:
                    with self.mol.nuc[i].with_rinv_as_nucleus(a):
                        vrinv = self.mol.nuc[i].intor('int1e_iprinv', comp=3) # <\nabla|1/r|>
                        vrinv *= (self.mol.atom_charge(a)*self.mol.atom_charge(ia))
                        vrinv += vrinv.transpose(0, 2, 1)
                    h1ao += vrinv

            s1vo = self.ao2mo(mf_n, s1ao)
            s1.append(s1vo)
            B = self.ao2mo(mf_n, h1ao + v1ao + f1ao) - s1vo*e_i
            Bs.append(B)
        return numpy.array(Bs).reshape(-1, nmo, nocc), numpy.array(s1).reshape(-1, nmo, nocc)

    def kernel(self, max_cycle=30, tol=1e-9, hermi=False):
        'CPHF solver for cNEO'
        e_a = self.base.mf_elec.mo_energy[viridx_e]
        e_i = self.base.mf_elec.mo_energy[occidx_e]
        e_ai = -1 / lib.direct_sum('a-i->ai', e_a, e_i)

        B_e, s1 = self.get_Bmat_elec(self.base.mf_elec)
        B_e[:,viridx_e] *= e_ai
        B_e[:,occidx_e] = -s1[:, occidx_e]*.5
        mo1base = B_e.ravel()

        for i in range(len(self.mol.nuc)):
            mf_n = self.base.mf_nuc[i]
            occidx_n = mf_n.mo_occ > 0
            viridx_n = mf_n.mo_occ == 0
            e_a_n = mf_n.mo_energy[viridx_n]
            e_i_n = mf_n.mo_energy[occidx_n]
            e_ai_n = -1 / lib.direct_sum('a-i->ai', e_a_n, e_i_n)

            B_n, s1 = self.get_Bmat_nuc(i)
            B_n[:,viridx_n] *= e_ai_n
            B_n[:,occidx_n] = -s1[:, occidx_n]*.5

            mo1base = numpy.concatenate((mo1base, B_n.ravel()))

        for i in range(len(self.mol.nuc)):
            ia = self.mol.nuc[i].atom_index
            for j in self.atmlst:
                if ia == j:
                    irp = -self.mol.nuc[i].intor('int1e_irp', comp=9)
                    r = numpy.identity(3) - 2*numpy.einsum('xij,ji->x', irp, self.base.dm_nuc[i]).reshape(3,3)
                    mo1base = numpy.concatenate((mo1base, r.ravel()))
                else:
                    mo1base = numpy.concatenate((mo1base, numpy.zeros(9)))
        logger.info(self, 'The size of CPHF equations: %s', len(mo1base))


        mo1 = lib.krylov(self.full_response, mo1base, tol=tol, max_cycle=max_cycle, hermi=hermi)
        mo1_e, mo1_n, f1 = self.mo12ne(mo1)
        logger.debug(self, 'f1:\n%s', f1)

        v1mo_e = self.get_A_e(mo1_e, mo1_n)
        B_e, s1 = self.get_Bmat_elec(self.base.mf_elec)
        e1_e = B_e[:,occidx_e] + mo1_e[:,occidx_e] * lib.direct_sum('i-j->ij', e_i, e_i) - v1mo_e[:,occidx_e]
        logger.debug(self, 'e1e:\n%s', e1_e)

        e1_n = []
        for i in range(len(self.mol.nuc)):
            mf_n = self.base.mf_nuc[i]
            occidx = mf_n.mo_occ > 0
            viridx = mf_n.mo_occ == 0

            e_i = mf_n.mo_energy[occidx]
            e_a = mf_n.mo_energy[viridx]
            e_ai = -1 / lib.direct_sum('a-i->ai', e_a, e_i)

            v1mo_n = self.get_A_n(i, mo1_e, mo1_n, f1)
            B_n, s1 = self.get_Bmat_nuc(i)
            e1 = B_n[:,occidx] + mo1_n[i][:,occidx] * lib.direct_sum('i-j->ij', e_i, e_i) - v1mo_n[:,occidx]
            logger.debug(self, 'e1n:\n%s', e1)
            e1_n.append(e1)

        return mo1_e, e1_e, mo1_n, e1_n, f1


