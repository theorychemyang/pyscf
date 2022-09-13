#!/usr/bin/env python

'''
Analytical Hessian for constrained nuclear-electronic orbitals
'''
import numpy
from pyscf import lib
from pyscf import scf
from pyscf.lib import logger
from pyscf.data import nist
from pyscf.hessian.thermo import _get_TR, rotation_const, _get_rotor_type
from pyscf.neo.cphf import CPHF
from functools import reduce

class Hessian(lib.StreamObject):
    '''
    Example:

    >>> mol = neo.Mole()
    >>> mol.build(atom='H 0 0 0.00; C 0 0 1.064; N 0 0 2.220', basis='ccpvdz')
    >>> mf = neo.CDFT(mol)
    >>> mf.mf_elec.xc = 'b3lyp'
    >>> mf.scf()

    >>> h = neo.Hessian(mf)
    >>> h.kernel()
    >>> results = hess.harmonic_analysis(mol, h)
    >>> print(results)
    '''

    def __init__(self, scf_method):
        self.mol = scf_method.mol
        self.base = scf_method
        self.atmlst = range(self.mol.natm)
        self.verbose = 4
        if self.base.epc is not None:
            raise NotImplementedError('Hessian with epc is not implemented')


    def hess_elec(self, mo1_e, e1_e):
        'Hessian of electrons and classic nuclei in cNEO'
        hobj = self.base.mf_elec.Hessian()
        de2 = hobj.hess_elec(mo1=mo1_e, mo_e1=e1_e) + hobj.hess_nuc()
        return de2

    def hess_elec_nuc1(self, i, mo1_n, mo1_e):
        'part of hessian for Coulomb interactions between electrons and the i-th quantum nuclei'
        de2 = numpy.zeros((self.mol.natm, self.mol.natm, 3, 3))
        aoslices = self.mol.aoslice_by_atom()

        dm0_e = self.base.dm_elec
        mo_coeff_e = self.base.mf_elec.mo_coeff
        mo_occ_e = self.base.mf_elec.mo_occ
        mocc_e = mo_coeff_e[:, mo_occ_e>0]
        nao_e = mo_coeff_e.shape[0]

        mf_nuc = self.base.mf_nuc[i]
        nuc = mf_nuc.mol
        index = nuc.atom_index
        charge = self.mol.atom_charge(index)
        mo_coeff_n = mf_nuc.mo_coeff
        mo_occ_n = mf_nuc.mo_occ
        mocc_n = mo_coeff_n[:, mo_occ_n>0]
        dm0_n = self.base.dm_nuc[i]

        for i0, ia in enumerate(self.atmlst):
            shl0, shl1, p0, p1 = aoslices[ia]
            shls_slice = (shl0, shl1) + (0, self.mol.elec.nbas) + (0, nuc.nbas)*2

            v1ao = numpy.zeros((3, nao_e, nao_e))
            v1en_ao = -scf.jk.get_jk((self.mol.elec, self.mol.elec, nuc, nuc), dm0_n,
                                     scripts='ijkl,lk->ij', intor='int2e_ip1', comp=3, aosym='s2kl')
            v1ao[:, p0:p1] += v1en_ao[:, p0:p1]
            v1ao[:, :, p0:p1] += v1en_ao[:, p0:p1].transpose(0, 2, 1)

            v1en_ao2 = -scf.jk.get_jk((self.mol.elec, self.mol.elec, nuc, nuc),
                                      dm0_e[:,p0:p1], scripts='ijkl,ji->kl', intor='int2e_ip1',
                                      shls_slice=shls_slice, comp=3, aosym='s2kl')*2

            if ia == index:
                v1en_ao3 = -scf.jk.get_jk((nuc, nuc, self.mol.elec, self.mol.elec), dm0_n,
                                          scripts='ijkl,ji->kl', intor='int2e_ip1', comp=3, aosym='s2kl')*2
                v1en_ao4 = -scf.jk.get_jk((nuc, nuc, self.mol.elec, self.mol.elec), dm0_e,
                                          scripts='ijkl,lk->ij', intor='int2e_ip1', comp=3, aosym='s2kl')
                v1en_ao4 += v1en_ao4.transpose(0, 2, 1)

            for j0 in range(i0+1):
                ja = self.atmlst[j0]
                dm1_e = numpy.einsum('ypi, qi->ypq', mo1_e[ja], mocc_e)
                # *2 for c.c. and *2 for double occupancy of electrons
                de2[i0, j0] -= numpy.einsum('xpq, ypq->xy', v1ao, dm1_e)*4
                dm1_n = numpy.einsum('ypi, qi->ypq', mo1_n[ja], mocc_n)
                de2[i0, j0] -= numpy.einsum('xpq, ypq->xy', v1en_ao2, dm1_n)*2
                if ia == index:
                    de2[i0, j0] -= numpy.einsum('xpq, ypq->xy', v1en_ao3, dm1_e)*4
                    de2[i0, j0] -= numpy.einsum('xpq, ypq->xy', v1en_ao4, dm1_n)*2

            for j0 in range(i0):
                de2[j0, i0] = de2[i0, j0].T
        return de2*charge

    def hess_elec_nuc2(self, i):
        'part of hessian for Coulomb interactions between electrons and the i-th quantum nuclei'
        dm0_e = self.base.dm_elec
        nao_e = dm0_e.shape[0]

        de2 = numpy.zeros((self.mol.natm, self.mol.natm, 3, 3))
        aoslices = self.mol.aoslice_by_atom()

        mf_nuc = self.base.mf_nuc[i]
        nuc = mf_nuc.mol
        index = nuc.atom_index
        charge = self.mol.atom_charge(index)
        dm0_n = self.base.dm_nuc[i]
        i = self.atmlst.index(index)

        hess_naa = scf.jk.get_jk((nuc, nuc, self.mol.elec, self.mol.elec), dm0_e,
                                 scripts='ijkl,lk->ij', intor='int2e_ipip1', comp=9, aosym='s2kl')
        hess_naa += hess_naa.transpose(0, 2, 1)

        hess_eaa = scf.jk.get_jk((self.mol.elec, self.mol.elec, nuc, nuc), dm0_n,
                                 scripts='ijkl,lk->ij', intor='int2e_ipip1', comp=9, aosym='s2kl')

        for i0, ia in enumerate(self.atmlst):
            shl0, shl1, p0, p1 = aoslices[ia]
            shls_slice = (shl0, shl1) + (0, self.mol.elec.nbas) + (0, nuc.nbas)*2

            de2[i0, i0] -= numpy.einsum('xpq,pq->x', hess_eaa[:,p0:p1], dm0_e[p0:p1]).reshape(3,3)*2
            if ia == index:
                de2[i0, i0] -= numpy.einsum('xpq, pq->x', hess_naa, dm0_n).reshape(3, 3)

            hess_eaeb = scf.jk.get_jk((self.mol.elec, self.mol.elec, nuc, nuc), dm0_n,
                                      scripts='ijkl,lk->ij', intor='int2e_ipvip1', comp=9)

            for j0, ja in enumerate(self.atmlst[:i0+1]):
                q0, q1 = aoslices[ja][2:]
                de2[i0,j0] -= numpy.einsum('xpq,pq->x', hess_eaeb[:,p0:p1,q0:q1], dm0_e[p0:p1,q0:q1]).reshape(3, 3)*2
                if index == ia:
                    test = numpy.zeros((9, nao_e, nao_e))
                    hess_naeb = scf.jk.get_jk((nuc, nuc, self.mol.elec, self.mol.elec), dm0_n,
                                              scripts='ijkl,ji->kl', intor='int2e_ip1ip2', comp=9)
                    test[:, q0:q1] += hess_naeb[:, q0:q1]
                    test[:, :, q0:q1] += hess_naeb[:, q0:q1].transpose(0, 2, 1)
                    de2[i0, j0] -= numpy.einsum('xpq,pq->x', test, dm0_e).reshape(3,3)*2

                if index == ja:
                    hess_eanb = scf.jk.get_jk((self.mol.elec, self.mol.elec, nuc, nuc),
                                              dm0_e[:, p0:p1], scripts='ijkl,ji->kl', intor='int2e_ip1ip2', shls_slice=shls_slice, comp=9)
                    de2[i0, j0] -= numpy.einsum('xpq,pq->x', hess_eanb, dm0_n).reshape(3, 3)*4
                if index == ia and index == ja:
                    hess_nanb = scf.jk.get_jk((nuc, nuc, self.mol.elec, self.mol.elec), dm0_n,
                                              scripts='ijkl,ji->kl', intor='int2e_ipvip1', comp=9)
                    de2[i0, j0] -= numpy.einsum('xpq,pq->x', hess_nanb, dm0_e).reshape(3, 3)*2

            for j0 in range(i0):
                de2[j0,i0] = de2[i0,j0].T
        return de2*charge

    def hess_nuc1(self, i, mo1_n, e1_n, f1):
        'part of hessian for the i-th quantum nuclei'
        de2 = numpy.zeros((self.mol.natm, self.mol.natm, 3, 3))
        mf_nuc = self.base.mf_nuc[i]
        index = mf_nuc.mol.atom_index
        mo_coeff = mf_nuc.mo_coeff
        mo_occ = mf_nuc.mo_occ
        mocc = mo_coeff[:, mo_occ>0]
        nao = mo_coeff.shape[0]

        for i0, ia in enumerate(self.atmlst):
            h1a = numpy.zeros((3, nao, nao))

            if ia == index:
                h1a = -mf_nuc.mol.intor('int1e_ipkin', comp=3)/(self.mol.mass[index]*nist.ATOMIC_MASS/nist.E_MASS)
                h1a += mf_nuc.mol.intor('int1e_ipnuc', comp=3)*self.mol.atom_charge(index)
                h1a += h1a.transpose(0, 2, 1)

            elif self.mol.quantum_nuc[ia] == False:
                with mf_nuc.mol.with_rinv_as_nucleus(ia):
                    vrinv = mf_nuc.mol.intor('int1e_iprinv', comp=3)
                    vrinv *= (self.mol.atom_charge(ia)*self.mol.atom_charge(index))
                h1a = vrinv + vrinv.transpose(0,2,1)

            for j0 in range(i0+1):
                ja = self.atmlst[j0]
                dm1_n = numpy.einsum('ypi,qi->ypq', mo1_n[ja], mocc)
                de2[i0, j0] += numpy.einsum('xpq,ypq->xy', h1a, dm1_n)*2 # *2 for c.c. of dm1

            for j0 in range(i0):
                de2[j0, i0] = de2[i0, j0].T

        return de2


    def hess_nuc2(self, i):
        'part of hessian for the i-th quantum nuclei'
        de2 = numpy.zeros((self.mol.natm, self.mol.natm, 3, 3))
        mf_nuc = self.base.mf_nuc[i]
        dm0_n = self.base.dm_nuc[i]
        index = mf_nuc.mol.atom_index
        i = self.atmlst.index(index)
        mass = self.mol.mass[index] * nist.ATOMIC_MASS/nist.E_MASS
        charge = self.mol.atom_charge(index)

        mo_energy = mf_nuc.mo_energy
        mo_coeff = mf_nuc.mo_coeff
        mo_occ = mf_nuc.mo_occ
        mocc = mo_coeff[:, mo_occ>0]
        nao = mo_coeff.shape[0]
        dme0 = numpy.einsum('pi,qi,i->pq', mocc, mocc, mo_energy[mo_occ>0])

        for i0, ia in enumerate(self.atmlst):
            if ia == index:
                h1aa = mf_nuc.mol.intor('int1e_ipipkin', comp=9)/mass
                h1aa += mf_nuc.mol.intor('int1e_ipkinip', comp=9)/mass
                h1aa -= mf_nuc.mol.intor('int1e_ipipnuc', comp=9)*charge
                h1aa -= mf_nuc.mol.intor('int1e_ipnucip', comp=9)*charge
                h1aa = h1aa.reshape(3, 3, nao, nao)
                h1aa += h1aa.transpose(0, 1, 3, 2)

                de2[i0, i0] += numpy.einsum('xypq, pq->xy', h1aa, dm0_n)

                # for j0, ja in enumerate(self.atmlst):
                for j0 in range(i0+1):
                    ja = self.atmlst[j0]
                    if self.mol.quantum_nuc[ja] is False:
                        with mf_nuc.mol.with_rinv_at_nucleus(ja):
                            rinv_ab = -mf_nuc.mol.intor('int1e_ipiprinv', comp=9).reshape(3, 3, nao, nao)
                            rinv_ab -= mf_nuc.mol.intor('int1e_iprinvip', comp=9).reshape(3, 3, nao, nao)
                            rinv_ab *= charge * self.mol.atom_charge(ja)
                        de2[i0, j0] += numpy.einsum('xypq,pq->xy', rinv_ab, dm0_n)*2

            # test: hessian for quantum-classical nucler replusions
            elif self.mol.quantum_nuc[ia] is False:
                with mf_nuc.mol.with_rinv_at_nucleus(ia):
                    rinv_aa = mf_nuc.mol.intor('int1e_ipiprinv', comp=9).reshape(3, 3, nao, nao) \
                              * charge * self.mol.atom_charge(ia)
                    rinv_aa += mf_nuc.mol.intor('int1e_iprinvip', comp=9).reshape(3, 3, nao, nao) \
                               * charge * self.mol.atom_charge(ia)
                de2[i0, i0] += numpy.einsum('xypq,pq->xy', rinv_aa, dm0_n)*2

                for j0 in range(i0+1):
                    ja = self.atmlst[j0]
                    if ja == index:
                        with mf_nuc.mol.with_rinv_as_nucleus(ia):
                            rinv_ab = -mf_nuc.mol.intor('int1e_ipiprinv', comp=9).reshape(3, 3, nao, nao) \
                                      * charge * self.mol.atom_charge(ia)
                            rinv_ab -= mf_nuc.mol.intor('int1e_iprinvip', comp=9).reshape(3, 3, nao, nao) \
                                       * charge * self.mol.atom_charge(ia)
                        de2[i0, j0] += numpy.einsum('xypq,pq->xy', rinv_ab, dm0_n)*2

            for j0 in range(i0):
                de2[j0, i0] = de2[i0, j0].T
        return de2

    def hess_nuc_nuc1(self, mo1_n):
        'part of hessian for Coulomb interactions between quantum nuclei'
        hess = numpy.zeros((self.mol.natm, self.mol.natm, 3, 3))
        for i in range(len(self.mol.nuc)):
            index1 = self.mol.nuc[i].atom_index
            mo_coeff = self.base.mf_nuc[i].mo_coeff
            mo_occ = self.base.mf_nuc[i].mo_occ
            mocc_i = mo_coeff[:, mo_occ>0]
            nao_i = mo_coeff.shape[0]
            for j in range(len(self.mol.nuc)):
                if j != i:
                    index2 = self.mol.nuc[j].atom_index
                    mo_coeff = self.base.mf_nuc[j].mo_coeff
                    mo_occ = self.base.mf_nuc[j].mo_occ
                    mocc_j = mo_coeff[:, mo_occ>0]
                    nao_j = mo_coeff.shape[0]
                    de2 = numpy.zeros((self.mol.natm, self.mol.natm, 3, 3))
                    for i0, ia in enumerate(self.atmlst):
                        for j0 in range(i0+1):
                            ja = self.atmlst[j0]
                            dm1_i = numpy.einsum('ypi,qi->ypq', mo1_n[i][ja], mocc_i)
                            dm1_j = numpy.einsum('ypi,qi->ypq', mo1_n[j][ja], mocc_j)
                            v1 = numpy.zeros((3, nao_j, nao_j))
                            v2 = numpy.zeros((3, nao_i, nao_i))
                            if ia == index1:
                                v1 = -scf.jk.get_jk((self.mol.nuc[i], self.mol.nuc[i], self.mol.nuc[j], self.mol.nuc[j]),
                                                    self.base.dm_nuc[i], scripts='ijkl,ji->kl', intor='int2e_ip1', comp=3, aosym='s2kl')*2
                                v2 = -scf.jk.get_jk((self.mol.nuc[i], self.mol.nuc[i], self.mol.nuc[j], self.mol.nuc[j]),
                                                    self.base.dm_nuc[j], scripts='ijkl,lk->ij', intor='int2e_ip1', comp=3, aosym='s2kl')
                                v2 += v2.transpose(0, 2, 1)
                            elif ia == index2:
                                v1 = -scf.jk.get_jk((self.mol.nuc[j], self.mol.nuc[j], self.mol.nuc[i], self.mol.nuc[i]),
                                                    self.base.dm_nuc[i], scripts='ijkl,lk->ij', intor='int2e_ip1', comp=3, aosym='s2kl')
                                v1 += v1.transpose(0, 2, 1)
                                v2 = -scf.jk.get_jk((self.mol.nuc[j], self.mol.nuc[j], self.mol.nuc[i], self.mol.nuc[i]),
                                                    self.base.dm_nuc[j], scripts='ijkl,ji->kl', intor='int2e_ip1', comp=3, aosym='s2kl')*2

                            de2[i0, j0] += numpy.einsum('xpq,ypq->xy', v1, dm1_j)* \
                                                        self.mol.atom_charge(index1)*self.mol.atom_charge(index2)*2
                            de2[i0, j0] += numpy.einsum('xpq,ypq->xy', v2, dm1_i)* \
                                                        self.mol.atom_charge(index1)*self.mol.atom_charge(index2)*2

                        for j0 in range(i0):
                            de2[j0, i0] = de2[i0, j0].T
                    hess += de2/2
        return hess

    def hess_nuc_nuc2(self):
        'part of hessian for Coulomb interactions between quantum nuclei'
        hess = numpy.zeros((self.mol.natm, self.mol.natm, 3, 3))
        for i in range(len(self.mol.nuc)):
            index_i = self.mol.nuc[i].atom_index
            for j in range(len(self.mol.nuc)):
                index_j = self.mol.nuc[j].atom_index
                if j != i:
                    de2 = numpy.zeros((self.mol.natm, self.mol.natm, 3, 3))
                    for i0, ia in enumerate(self.atmlst):
                        for j0 in range(i0+1):
                            ja = self.atmlst[j0]
                            charge = self.mol.atom_charge(index_i)*self.mol.atom_charge(index_j)
                            if ia == index_i:
                                if ja == index_i:
                                    v2_aa = scf.jk.get_jk((self.mol.nuc[i], self.mol.nuc[i], self.mol.nuc[j], self.mol.nuc[j]),
                                                          self.base.dm_nuc[j], scripts='ijkl,lk->ij', intor='int2e_ipip1', comp=9, aosym='s2kl')
                                    v2_aa += scf.jk.get_jk((self.mol.nuc[i], self.mol.nuc[i], self.mol.nuc[j], self.mol.nuc[j]),
                                                           self.base.dm_nuc[j], scripts='ijkl,lk->ij', intor='int2e_ipvip1', comp=9, aosym='s2kl')
                                    v2_aa += v2_aa.transpose(0, 2, 1)
                                    de2[i0, j0] += numpy.einsum('xpq,pq->x', v2_aa,
                                                                self.base.dm_nuc[i]).reshape(3, 3)*charge
                                elif ja == index_j:
                                    v2_ab = scf.jk.get_jk((self.mol.nuc[i], self.mol.nuc[i], self.mol.nuc[j], self.mol.nuc[j]),
                                                          self.base.dm_nuc[j], scripts='ijkl,lk->ij', intor='int2e_ip1ip2', comp=9)*2
                                    v2_ab += v2_ab.transpose(0, 2, 1)
                                    de2[i0, j0] += numpy.einsum('xpq,pq->x', v2_ab,
                                                                self.base.dm_nuc[i]).reshape(3, 3)*charge
                            elif ia == index_j:
                                if ja == index_j:
                                    v2_aa = scf.jk.get_jk((self.mol.nuc[j], self.mol.nuc[j], self.mol.nuc[i], self.mol.nuc[i]),
                                                          self.base.dm_nuc[i], scripts='ijkl,lk->ij', intor='int2e_ipip1', comp=9, aosym='s2kl')
                                    v2_aa += scf.jk.get_jk((self.mol.nuc[j], self.mol.nuc[j], self.mol.nuc[i], self.mol.nuc[i]),
                                                           self.base.dm_nuc[i], scripts='ijkl,lk->ij', intor='int2e_ipvip1', comp=9, aosym='s2kl')
                                    v2_aa += v2_aa.transpose(0, 2, 1)
                                    de2[i0, j0] += numpy.einsum('xpq,pq->x', v2_aa,
                                                                self.base.dm_nuc[j]).reshape(3, 3)*charge
                                elif ja == index_i:
                                    v2_ab = scf.jk.get_jk((self.mol.nuc[j], self.mol.nuc[j], self.mol.nuc[i], self.mol.nuc[i]),
                                                          self.base.dm_nuc[i], scripts='ijkl,lk->ij', intor='int2e_ip1ip2', comp=9)*2
                                    v2_ab += v2_ab.transpose(0, 2, 1)
                                    de2[i0, j0] += numpy.einsum('xpq,pq->x', v2_ab,
                                                                self.base.dm_nuc[j]).reshape(3, 3)*charge
                        for j0 in range(i0):
                            de2[j0,i0] = de2[i0,j0].T

                    hess += de2/2

        return hess

    def kernel(self):
        cphf = CPHF(self.base)
        mo1_e, e1_e, mo1_n, e1_n, f1 = cphf.kernel()

        mo_coeff_e = self.base.mf_elec.mo_coeff
        mo_occ_e = self.base.mf_elec.mo_occ
        nmo_e = mo_coeff_e.shape[1]
        mocc_e = mo_coeff_e[:,mo_occ_e>0]
        nocc_e = mocc_e.shape[1]

        mo1_e = mo1_e.reshape(-1, 3, nmo_e, nocc_e)
        mo1_e = numpy.einsum('pj,cxji->cxpi', mo_coeff_e, mo1_e)
        e1_e = e1_e.reshape(-1, 3, nocc_e, nocc_e)

        hess = self.hess_elec(mo1_e = mo1_e, e1_e = e1_e)
        logger.debug(self, 'hess_elec:\n%s', hess)

        for i in range(len(self.mol.nuc)):
            mo_coeff_n = self.base.mf_nuc[i].mo_coeff
            mo_occ_n = self.base.mf_nuc[i].mo_occ
            nmo_n = mo_coeff_n.shape[1]
            mocc_n = mo_coeff_n[:,mo_occ_n>0]
            nocc_n = mocc_n.shape[1]

            mo1_n[i] = mo1_n[i].reshape(-1, 3, nmo_n, nocc_n)
            mo1_n[i] = numpy.einsum('pj,cxji->cxpi', mo_coeff_n, mo1_n[i])
            e1_n[i] = e1_n[i].reshape(-1, 3, nocc_n, nocc_n)
            f1[i] = f1[i].reshape(-1, 3, 3)

            hess += self.hess_elec_nuc1(i, mo1_n[i], mo1_e)
            logger.debug(self, 'elec_nuc1:\n%s', self.hess_elec_nuc1(i, mo1_n[i], mo1_e))
            hess += self.hess_elec_nuc2(i)
            logger.debug(self, 'elec_nuc2:\n%s', self.hess_elec_nuc2(i))
            hess += self.hess_nuc1(i, mo1_n[i], e1_n[i], f1[i])
            logger.debug(self, 'hess_nuc1:\n%s', self.hess_nuc1(i, mo1_n[i], e1_n[i], f1[i]))
            hess += self.hess_nuc2(i)
            logger.debug(self, 'hess_nuc2:\n%s', self.hess_nuc2(i))

        hess += self.hess_nuc_nuc1(mo1_n)
        logger.debug(self, 'hess_nuc_nuc1:\n%s', self.hess_nuc_nuc1(mo1_n))
        hess += self.hess_nuc_nuc2()
        logger.debug(self, 'hess_nuc_nuc2:\n%s', self.hess_nuc_nuc2())

        logger.note(self, 'hess:\n%s', hess)
        return hess

    def harmonic_analysis(self, mol, hess, exclude_trans=True, exclude_rot=True,
                          imaginary_freq=True):
        '''Copy from pyscf/pyscf/hessian/thermo.py
        Each column is one mode

        imaginary_freq (boolean): save imaginary_freq as complex number (if True)
        or negative real number (if False)
        copy from pyscf.hessian.thermo
        '''
        results = {}
        atom_coords = mol.atom_coords()
        mass = mol.mass # This line is different from pyscf.hessian.thermo.harmonic_analysis
        mass_center = numpy.einsum('z,zx->x', mass, atom_coords) / mass.sum()
        atom_coords = atom_coords - mass_center
        natm = atom_coords.shape[0]

        mass_hess = numpy.einsum('pqxy,p,q->pqxy', hess, mass**-.5, mass**-.5)
        h = mass_hess.transpose(0,2,1,3).reshape(natm*3,natm*3)

        TR = _get_TR(mass, atom_coords)
        TRspace = []
        if exclude_trans:
            TRspace.append(TR[:3])

        if exclude_rot:
            rot_const = rotation_const(mass, atom_coords)
            rotor_type = _get_rotor_type(rot_const)
            if rotor_type == 'ATOM':
                pass
            elif rotor_type == 'LINEAR':  # linear molecule
                TRspace.append(TR[3:5])
            else:
                TRspace.append(TR[3:])

        if TRspace:
            TRspace = numpy.vstack(TRspace)
            q, r = numpy.linalg.qr(TRspace.T)
            P = numpy.eye(natm * 3) - q.dot(q.T)
            w, v = numpy.linalg.eigh(P)
            bvec = v[:,w > 1e-7]
            h = reduce(numpy.dot, (bvec.T, h, bvec))
            force_const_au, mode = numpy.linalg.eigh(h)
            mode = bvec.dot(mode)
        else:
            force_const_au, mode = numpy.linalg.eigh(h)

        freq_au = numpy.lib.scimath.sqrt(force_const_au)
        results['freq_error'] = numpy.count_nonzero(freq_au.imag > 0)
        if not imaginary_freq and numpy.iscomplexobj(freq_au):
            # save imaginary frequency as negative frequency
            freq_au = freq_au.real - abs(freq_au.imag)

        results['freq_au'] = freq_au
        au2hz = (nist.HARTREE2J / (nist.ATOMIC_MASS * nist.BOHR_SI**2))**.5 / (2 * numpy.pi)
        results['freq_wavenumber'] = freq_wn = freq_au * au2hz / nist.LIGHT_SPEED_SI * 1e-2

        norm_mode = numpy.einsum('z,zri->izr', mass**-.5, mode.reshape(natm,3,-1))
        results['norm_mode'] = norm_mode
        reduced_mass = 1./numpy.einsum('izr,izr->i', norm_mode, norm_mode)
        results['reduced_mass'] = reduced_mass

        # https://en.wikipedia.org/wiki/Vibrational_temperature
        results['vib_temperature'] = freq_au * au2hz * nist.PLANCK / nist.BOLTZMANN

        # force constants
        dyne = 1e-2 * nist.HARTREE2J / nist.BOHR_SI**2
        results['force_const_au'] = force_const_au
        results['force_const_dyne'] = reduced_mass * force_const_au * dyne  #cm^-1/a0^2

        # TODO: IR intensity
        return results


from pyscf.neo import CDFT
CDFT.Hessian = lib.class_as_method(Hessian)
