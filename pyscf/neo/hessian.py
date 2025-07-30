#!/usr/bin/env python

'''
Analytic Hessian for constrained nuclear-electronic orbitals
'''

import numpy
from functools import reduce
from pyscf import gto, lib, neo, scf
from pyscf.data import nist
from pyscf.lib import logger
from pyscf.hessian import rhf as rhf_hessian
from pyscf.hessian.thermo import harmonic_analysis, \
                                 rotation_const, \
                                 rotational_symmetry_number, \
                                 _get_rotor_type
from pyscf.neo import grad, cphf
from pyscf.scf.jk import get_jk

# import _response_functions to load gen_response methods in CDFT class
from pyscf.neo import _response_functions # noqa
# import pyscf.grad.rhf to activate nuc_grad_method method
from pyscf.grad import rhf  # noqa


def general_hessian(hess_method):
    '''Modify gradient method to support general charge and mass.
    Similar to general_scf decorator in neo/hf.py
    '''
    if isinstance(hess_method, ComponentHess):
        return hess_method

    assert (isinstance(hess_method.base, scf.hf.SCF) and
            isinstance(hess_method.base, neo.hf.Component))

    return hess_method.view(lib.make_class((ComponentHess, hess_method.__class__)))

class ComponentHess:
    __name_mixin__ = 'Component'

    def __init__(self, hess_method):
        self.__dict__.update(hess_method.__dict__)

    def get_hcore(self, mol=None):
        '''Part of the second derivatives of core Hamiltonian'''
        if mol is None: mol = self.mol
        h1aa = mol.intor('int1e_ipipkin', comp=9) / self.base.mass
        h1ab = mol.intor('int1e_ipkinip', comp=9) / self.base.mass
        if mol._pseudo:
            raise NotImplementedError('Nuclear hessian for GTH PP')
        else:
            h1aa += mol.intor('int1e_ipipnuc', comp=9) * self.base.charge
            h1ab += mol.intor('int1e_ipnucip', comp=9) * self.base.charge
        if mol.has_ecp():
            h1aa += mol.intor('ECPscalar_ipipnuc', comp=9) * self.base.charge
            h1ab += mol.intor('ECPscalar_ipnucip', comp=9) * self.base.charge
        nao = h1aa.shape[-1]
        return h1aa.reshape(3,3,nao,nao), h1ab.reshape(3,3,nao,nao)

    def hcore_generator(self, mol=None):
        if mol is None: mol = self.mol
        with_x2c = getattr(self.base, 'with_x2c', None)
        if with_x2c:
            raise NotImplementedError('X2C not supported')

        with_ecp = mol.has_ecp()
        if with_ecp:
            ecp_atoms = set(mol._ecpbas[:,gto.ATOM_OF])
        else:
            ecp_atoms = ()
        aoslices = mol.aoslice_by_atom()
        nbas = mol.nbas
        nao = mol.nao_nr()
        h1aa, h1ab = self.get_hcore(mol)
        def get_hcore(iatm, jatm):
            ish0, ish1, i0, i1 = aoslices[iatm]
            jsh0, jsh1, j0, j1 = aoslices[jatm]
            zi = mol.atom_charge(iatm)
            zj = mol.atom_charge(jatm)
            hcore = numpy.zeros((3,3,nao,nao))
            if iatm == jatm:
                if i1 > i0:
                    hcore[:,:,i0:i1] += h1aa[:,:,i0:i1]
                    hcore[:,:,i0:i1,i0:i1] += h1ab[:,:,i0:i1,i0:i1]
                if not mol.super_mol._quantum_nuc[iatm]:
                    with mol.with_rinv_at_nucleus(iatm):
                        rinv2aa = mol.intor('int1e_ipiprinv', comp=9)
                        rinv2ab = mol.intor('int1e_iprinvip', comp=9)
                        rinv2aa *= zi
                        rinv2ab *= zi
                        if with_ecp and iatm in ecp_atoms:
                            # ECP rinv has the same sign as ECP nuc,
                            # unlike regular rinv = -nuc.
                            # Reverse the sign to mimic regular rinv
                            rinv2aa -= mol.intor('ECPscalar_ipiprinv', comp=9)
                            rinv2ab -= mol.intor('ECPscalar_iprinvip', comp=9)
                    rinv2aa = rinv2aa.reshape(3,3,nao,nao) * self.base.charge
                    rinv2ab = rinv2ab.reshape(3,3,nao,nao) * self.base.charge
                    hcore += -rinv2aa - rinv2ab
                    if i1 > i0:
                        hcore[:,:,i0:i1] += rinv2aa[:,:,i0:i1]
                        hcore[:,:,i0:i1] += rinv2ab[:,:,i0:i1]
                        hcore[:,:,:,i0:i1] += rinv2aa[:,:,i0:i1].transpose(0,1,3,2)
                        hcore[:,:,:,i0:i1] += rinv2ab[:,:,:,i0:i1]
            else:
                if i1 > i0 and j1 > j0:
                    hcore[:,:,i0:i1,j0:j1] += h1ab[:,:,i0:i1,j0:j1]
                if not mol.super_mol._quantum_nuc[iatm] and jsh1 > jsh0:
                    with mol.with_rinv_at_nucleus(iatm):
                        shls_slice = (jsh0, jsh1, 0, nbas)
                        rinv2aa = mol.intor('int1e_ipiprinv', comp=9, shls_slice=shls_slice)
                        rinv2ab = mol.intor('int1e_iprinvip', comp=9, shls_slice=shls_slice)
                        rinv2aa *= zi
                        rinv2ab *= zi
                        if with_ecp and iatm in ecp_atoms:
                            rinv2aa -= mol.intor('ECPscalar_ipiprinv', comp=9, shls_slice=shls_slice)
                            rinv2ab -= mol.intor('ECPscalar_iprinvip', comp=9, shls_slice=shls_slice)
                        hcore[:,:,j0:j1] += self.base.charge * rinv2aa.reshape(3,3,j1-j0,nao)
                        hcore[:,:,j0:j1] += self.base.charge \
                                            * rinv2ab.reshape(3,3,j1-j0,nao).transpose(1,0,2,3)

                if not mol.super_mol._quantum_nuc[jatm] and ish1 > ish0:
                    with mol.with_rinv_at_nucleus(jatm):
                        shls_slice = (ish0, ish1, 0, nbas)
                        rinv2aa = mol.intor('int1e_ipiprinv', comp=9, shls_slice=shls_slice)
                        rinv2ab = mol.intor('int1e_iprinvip', comp=9, shls_slice=shls_slice)
                        rinv2aa *= zj
                        rinv2ab *= zj
                        if with_ecp and jatm in ecp_atoms:
                            rinv2aa -= mol.intor('ECPscalar_ipiprinv', comp=9, shls_slice=shls_slice)
                            rinv2ab -= mol.intor('ECPscalar_iprinvip', comp=9, shls_slice=shls_slice)
                        hcore[:,:,i0:i1] += self.base.charge * rinv2aa.reshape(3,3,i1-i0,nao)
                        hcore[:,:,i0:i1] += self.base.charge * rinv2ab.reshape(3,3,i1-i0,nao)
            return hcore + hcore.conj().transpose(0,1,3,2)
        return get_hcore

    def partial_hess_elec(self, mo_energy=None, mo_coeff=None, mo_occ=None,
                          atmlst=None, max_memory=4000, verbose=None):
        if self.base.is_nucleus: # Nucleus does not have self-type interaction
            log = logger.new_logger(self, verbose)
            time0 = (logger.process_clock(), logger.perf_counter())
            # Only hcore part here
            mol = self.mol
            mf = self.base
            if mo_energy is None: mo_energy = mf.mo_energy
            if mo_occ is None:    mo_occ = mf.mo_occ
            if mo_coeff is None:  mo_coeff = mf.mo_coeff
            if atmlst is None: atmlst = range(mol.natm)

            mocc = mo_coeff[:,mo_occ>0]
            dm0 = numpy.dot(mocc, mocc.T)
            # Energy weighted density matrix
            dme0 = numpy.einsum('pi,qi,i->pq', mocc, mocc, mo_energy[mo_occ>0])

            hcore_deriv = self.hcore_generator(mol)
            s1aa, s1ab, _ = rhf_hessian.get_ovlp(mol)

            aoslices = mol.aoslice_by_atom()
            natm = len(atmlst)
            e1 = numpy.zeros((natm, natm, 3, 3))
            for i0, ia in enumerate(atmlst):
                p0, p1 = aoslices[ia][2:]
                e1[i0, i0] -= numpy.einsum('xypq,pq->xy', s1aa[:,:,p0:p1], dme0[p0:p1])*2
                for j0, ja in enumerate(atmlst[:i0+1]):
                    q0, q1 = aoslices[ja][2:]
                    # *2 for +c.c.
                    e1[i0, j0] -= numpy.einsum('xypq,pq->xy', s1ab[:,:,p0:p1,q0:q1], dme0[p0:p1,q0:q1])*2

                    h1ao = hcore_deriv(ia, ja)
                    e1[i0, j0] += numpy.einsum('xypq,pq->xy', h1ao, dm0)

                for j0 in range(i0):
                    e1[j0, i0] = e1[i0, j0].T
            log.timer('CNEO nuclear partial hessian', *time0)
            return e1
        else:
            assert abs(self.base.charge) == 1
            return super().partial_hess_elec(mo_energy, mo_coeff, mo_occ, atmlst,
                                             max_memory, verbose)

    def hess_elec(self, mo_energy=None, mo_coeff=None, mo_occ=None,
                  mo1=None, mo_e1=None, h1ao=None,
                  atmlst=None, max_memory=4000, verbose=None):
        # Make sure hess_elec does not trigger make_h1 and solve_mo1
        assert mo1 is not None and mo_e1 is not None and h1ao is not None

        if not self.base.is_nucleus:
            return super().hess_elec(mo_energy, mo_coeff, mo_occ,
                                     mo1, mo_e1, h1ao,
                                     atmlst, max_memory,verbose)

        # NOTE: The only goal to copy hess_elec here is
        # to change to single occupation for quantum nuclei
        log = logger.new_logger(self, verbose)
        time0 = (logger.process_clock(), logger.perf_counter())

        mol = self.mol
        mf = self.base
        if mo_energy is None: mo_energy = mf.mo_energy
        if mo_occ is None:    mo_occ = mf.mo_occ
        if mo_coeff is None:  mo_coeff = mf.mo_coeff
        if atmlst is None: atmlst = range(mol.natm)

        de2 = self.partial_hess_elec(mo_energy, mo_coeff, mo_occ, atmlst,
                                     max_memory, log)

        nao = mo_coeff.shape[0]
        mocc = mo_coeff[:,mo_occ>0]
        s1a = -mol.intor('int1e_ipovlp', comp=3)

        aoslices = mol.aoslice_by_atom()
        for i0, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia][2:]
            s1ao = numpy.zeros((3,nao,nao))
            if p1 > p0:
                s1ao[:,p0:p1] += s1a[:,p0:p1]
                s1ao[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)
            s1oo = numpy.einsum('xpq,pi,qj->xij', s1ao, mocc, mocc)

            for j0 in range(i0+1):
                ja = atmlst[j0]
                # *2 for +c.c. Nuclear orbitals are only singly occupied
                dm1 = numpy.einsum('ypi,qi->ypq', mo1[ja], mocc)
                de2[i0,j0] += numpy.einsum('xpq,ypq->xy', h1ao[ia], dm1) * 2
                dm1 = numpy.einsum('ypi,qi,i->ypq', mo1[ja], mocc, mo_energy[mo_occ>0])
                de2[i0,j0] -= numpy.einsum('xpq,ypq->xy', s1ao, dm1) * 2
                de2[i0,j0] -= numpy.einsum('xpq,ypq->xy', s1oo, mo_e1[ja])

            for j0 in range(i0):
                de2[j0,i0] = de2[i0,j0].T

        log.timer('CNEO nuclear hessian', *time0)
        return de2

    def solve_mo1(self, mo_energy, mo_coeff, mo_occ, h1ao,
                  fx=None, atmlst=None, max_memory=4000, verbose=None):
        raise AttributeError

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        raise AttributeError

def hess_elec(hessobj, mo_energy=None, mo_coeff=None, mo_occ=None,
              mo1=None, mo_e1=None, h1ao=None,
              atmlst=None, max_memory=4000, verbose=None):
    log = logger.new_logger(hessobj, verbose)

    mol = hessobj.mol
    mf = hessobj.base
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    if atmlst is None: atmlst = range(mol.natm)

    de2 = hessobj.partial_hess_int(mo_coeff, mo_occ, atmlst, log)
    time0 = t1 = (logger.process_clock(), logger.perf_counter())

    if h1ao is None:
        h1ao = hessobj.make_h1(mo_coeff, mo_occ, None, atmlst, log)
        t1 = log.timer_debug1('making H1', *time0)
    if mo1 is None or mo_e1 is None:
        mo1, mo_e1 = hessobj.solve_mo1(mo_energy, mo_coeff, mo_occ, h1ao,
                                       None, atmlst, max_memory, log)
        hessobj.mo1 = mo1 # Store mo1 for IR intensity
        t1 = log.timer_debug1('solving MO1', *t1)

    for t, comp in hessobj.components.items():
        de2 += comp.hess_elec(mo_energy[t], mo_coeff[t], mo_occ[t],
                              mo1[t], mo_e1[t], h1ao[t],
                              atmlst, max_memory, log)

    log.timer('CNEO hessian', *time0)
    return de2

def partial_hess_int(hessobj, mo_coeff, mo_occ, atmlst=None, verbose=None):
    '''Partial derivative due to inter-type interactions'''
    log = logger.new_logger(hessobj, verbose)
    time0 = t1 = (logger.process_clock(), logger.perf_counter())

    mol = hessobj.mol
    if atmlst is None:
        atmlst = range(mol.natm)

    natm = len(atmlst)
    ej = numpy.zeros((natm, natm, 3, 3))

    mf = hessobj.base
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    for (t1, t2), interaction in mf.interactions.items():
        comp1 = mf.components[t1]
        comp2 = mf.components[t2]
        mol1 = comp1.mol
        mol2 = comp2.mol
        nao1 = mol1.nao
        nao2 = mol2.nao
        aoslices1 = mol1.aoslice_by_atom()
        aoslices2 = mol2.aoslice_by_atom()
        dm1 = dm0[t1]
        if interaction.mf1_unrestricted:
            assert dm1.ndim > 2 and dm1.shape[0] == 2
            dm1 = dm1[0] + dm1[1]
        dm2 = dm0[t2]
        if interaction.mf2_unrestricted:
            assert dm2.ndim > 2 and dm2.shape[0] == 2
            dm2 = dm2[0] + dm2[1]
        for i0, ia in enumerate(atmlst):
            shl0_1, shl1_1, p0_1, p1_1 = aoslices1[ia]
            shl0_2, shl1_2, p0_2, p1_2 = aoslices2[ia]
            shls_slice1 = (shl0_1, shl1_1) + (0, mol1.nbas) + (0, mol2.nbas)*2
            shls_slice2 = (shl0_2, shl1_2) + (0, mol2.nbas) + (0, mol1.nbas)*2
            # <\nabla^2 mol1 mol1 | mol2 mol2>
            if shl1_1 > shl0_1:
                vj1_diag = get_jk((mol1, mol1, mol2, mol2),
                                  comp1.charge * comp2.charge * dm2,
                                  scripts='ijkl,lk->ij', intor='int2e_ipip1',
                                  aosym='s2kl', comp=9, shls_slice=shls_slice1)
                vj1_diag = vj1_diag.reshape(3,3,p1_1-p0_1,nao1)
                ej[i0, i0] += numpy.einsum('xypq,pq->xy', vj1_diag, dm1[p0_1:p1_1])*2
            # <\nabla^2 mol2 mol2| mol1 mol1>
            if shl1_2 > shl0_2:
                vj2_diag = get_jk((mol2, mol2, mol1, mol1),
                                  comp1.charge * comp2.charge * dm1,
                                  scripts='ijkl,lk->ij', intor='int2e_ipip1',
                                  aosym='s2kl', comp=9, shls_slice=shls_slice2)
                vj2_diag = vj2_diag.reshape(3,3,p1_2-p0_2,nao2)
                ej[i0, i0] += numpy.einsum('xypq,pq->xy', vj2_diag, dm2[p0_2:p1_2])*2
            # <\nabla mol1 \nabla mol1 | mol2 mol2>
            if shl1_1 > shl0_1:
                vj1 = get_jk((mol1, mol1, mol2, mol2),
                             comp1.charge * comp2.charge * dm2,
                             scripts='ijkl,lk->ij', intor='int2e_ipvip1',
                             aosym='s2kl', comp=9,
                             shls_slice=shls_slice1)
                vj1 = vj1.reshape(3,3,p1_1-p0_1,nao1)
                for j0, ja in enumerate(atmlst[:i0+1]):
                    q0, q1 = aoslices1[ja][2:]
                    if q1 == q0:
                        continue
                    # *2 for +c.c.
                    ej[i0, j0] += numpy.einsum('xypq,pq->xy', vj1[:,:,:,q0:q1],
                                               dm1[p0_1:p1_1,q0:q1])*2
            # <\nabla mol2 \nabla mol2 | mol1 mol1>
            if shl1_2 > shl0_2:
                vj2 = get_jk((mol2, mol2, mol1, mol1),
                             comp1.charge * comp2.charge * dm1,
                             scripts='ijkl,lk->ij', intor='int2e_ipvip1',
                             aosym='s2kl', comp=9,
                             shls_slice=shls_slice2)
                vj2 = vj2.reshape(3,3,p1_2-p0_2,nao2)
                for j0, ja in enumerate(atmlst[:i0+1]):
                    q0, q1 = aoslices2[ja][2:]
                    if q1 == q0:
                        continue
                    # *2 for +c.c.
                    ej[i0, j0] += numpy.einsum('xypq,pq->xy', vj2[:,:,:,q0:q1],
                                               dm2[p0_2:p1_2,q0:q1])*2
            for j0, ja in enumerate(atmlst[:i0+1]):
                calculated = False
                # <\nabla mol1 mol1| \nabla mol2 mol2>
                if shl1_1 > shl0_1:
                    shl0_j, shl1_j, q0, q1 = aoslices2[ja]
                    if shl1_j > shl0_j:
                        shls_slice = (shl0_1, shl1_1) + (0, mol1.nbas) \
                                      + (shl0_j, shl1_j) + (0, mol2.nbas)
                        vj1 = get_jk((mol1, mol1, mol2, mol2),
                                     comp1.charge * comp2.charge * dm2[:,q0:q1],
                                     scripts='ijkl,lk->ij', intor='int2e_ip1ip2',
                                     aosym='s1', comp=9,
                                     shls_slice=shls_slice)
                        vj1 = vj1.reshape(3,3,p1_1-p0_1,nao1)
                        ip1ip2 = numpy.einsum('xypq,pq->xy', vj1, dm1[p0_1:p1_1])*4
                        if i0 == j0:
                            # Diagonal: skip (22|11) calculation and double (11|22)
                            ej[i0, j0] += ip1ip2 + ip1ip2.T
                            calculated = True
                        else:
                            ej[i0, j0] += ip1ip2
                # <\nabla mol2 mol2| \nabla mol1 mol1>
                if not calculated and shl1_2 > shl0_2:
                    shl0_j, shl1_j, q0, q1 = aoslices1[ja]
                    if shl1_j > shl0_j:
                        shls_slice = (shl0_2, shl1_2) + (0, mol2.nbas) \
                                      + (shl0_j, shl1_j) + (0, mol1.nbas)
                        vj2 = get_jk((mol2, mol2, mol1, mol1),
                                     comp1.charge * comp2.charge * dm1[:,q0:q1],
                                     scripts='ijkl,lk->ij', intor='int2e_ip1ip2',
                                     aosym='s1', comp=9,
                                     shls_slice=shls_slice)
                        vj2 = vj2.reshape(3,3,p1_2-p0_2,nao2)
                        ej[i0, j0] += numpy.einsum('xypq,pq->xy', vj2, dm2[p0_2:p1_2])*4

            for j0 in range(i0):
                ej[j0, i0] = ej[i0, j0].T

    log.timer('CNEO interaction partial hessian', *time0)
    return ej

def make_h1(hessobj, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
    mol = hessobj.mol
    if atmlst is None:
        atmlst = range(mol.natm)

    h1ao = {}
    for t, comp in hessobj.components.items():
        if not comp.base.is_nucleus:
            # Get base h1ao for electrons
            h1ao[t] = numpy.asarray(comp.make_h1(mo_coeff[t], mo_occ[t], chkfile, atmlst, verbose))
        else:
            # Get base h1ao for quantum nuclei
            # There is no self J/K so comp.make_h1 is not used
            hcore_deriv = grad.general_grad(comp.base.nuc_grad_method())\
                          .hcore_generator(mol.components[t])
            h1ao[t] = [None] * mol.natm
            for i0, ia in enumerate(atmlst):
                h1ao[t][ia] = hcore_deriv(ia)

    mf = hessobj.base
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    # h1ao due to inter-type interactions
    for (t1, t2), interaction in mf.interactions.items():
        comp1 = mf.components[t1]
        comp2 = mf.components[t2]
        mol1 = comp1.mol
        mol2 = comp2.mol
        aoslices1 = mol1.aoslice_by_atom()
        aoslices2 = mol2.aoslice_by_atom()
        dm1 = dm0[t1]
        if interaction.mf1_unrestricted:
            assert dm1.ndim > 2 and dm1.shape[0] == 2
            dm1 = dm1[0] + dm1[1]
        dm2 = dm0[t2]
        if interaction.mf2_unrestricted:
            assert dm2.ndim > 2 and dm2.shape[0] == 2
            dm2 = dm2[0] + dm2[1]
        for i0, ia in enumerate(atmlst):
            shl0, shl1, p0, p1 = aoslices1[ia]
            # Derivative w.r.t. mol1
            if shl1 > shl0:
                shls_slice = (shl0, shl1) + (0, mol1.nbas) + (0, mol2.nbas)*2
                v1, v2 = get_jk((mol1, mol1, mol2, mol2),
                                (-comp1.charge * comp2.charge * dm2,
                                 -comp1.charge * comp2.charge * dm1[:,p0:p1]),
                                scripts=('ijkl,lk->ij', 'ijkl,ji->kl'),
                                intor='int2e_ip1', aosym='s2kl', comp=3,
                                shls_slice=shls_slice)
                if interaction.mf1_unrestricted:
                    h1ao[t1][:,ia,:,p0:p1] += v1
                    h1ao[t1][:,ia,:,:,p0:p1] += v1.transpose(0,2,1)
                else:
                    h1ao[t1][ia][:,p0:p1] += v1
                    h1ao[t1][ia][:,:,p0:p1] += v1.transpose(0,2,1)
                if interaction.mf2_unrestricted:
                    h1ao[t2][:,ia] += v2 + v2.transpose(0,2,1)
                else:
                    h1ao[t2][ia] += v2 + v2.transpose(0,2,1)
            shl0, shl1, p0, p1 = aoslices2[ia]
            # Derivative w.r.t. mol2
            if shl1 > shl0:
                shls_slice = (shl0, shl1) + (0, mol2.nbas) + (0, mol1.nbas)*2
                v2, v1 = get_jk((mol2, mol2, mol1, mol1),
                                (-comp1.charge * comp2.charge * dm1,
                                 -comp1.charge * comp2.charge * dm2[:,p0:p1]),
                                scripts=('ijkl,lk->ij', 'ijkl,ji->kl'),
                                intor='int2e_ip1', aosym='s2kl', comp=3,
                                shls_slice=shls_slice)
                if interaction.mf2_unrestricted:
                    h1ao[t2][:,ia,:,p0:p1] += v2
                    h1ao[t2][:,ia,:,:,p0:p1] += v2.transpose(0,2,1)
                else:
                    h1ao[t2][ia][:,p0:p1] += v2
                    h1ao[t2][ia][:,:,p0:p1] += v2.transpose(0,2,1)
                if interaction.mf1_unrestricted:
                    h1ao[t1][:,ia] += v1 + v1.transpose(0,2,1)
                else:
                    h1ao[t1][ia] += v1 + v1.transpose(0,2,1)
    return h1ao

def solve_mo1(mf, mo_energy, mo_coeff, mo_occ, h1ao,
              fx=None, atmlst=None, max_memory=4000, verbose=None,
              max_cycle=100, level_shift=0):
    '''Solve the first order equation

    Kwargs:
        fx : function(dm_mo) => v1_mo
            A function to generate the induced potential.
            See also the function gen_vind.
    '''
    mol = mf.mol
    if atmlst is None: atmlst = range(mol.natm)

    nao = {}
    nmo = {}
    mocc = {}
    nocc = {}
    is_component_unrestricted = {}
    for t in mo_coeff.keys():
        mo_coeff[t] = numpy.asarray(mo_coeff[t])
        if mo_coeff[t].ndim > 2: # unrestricted
            assert not t.startswith('n')
            assert mo_coeff[t].shape[0] == 2
            is_component_unrestricted[t] = True
            nao[t], nmoa = mo_coeff[t][0].shape
            nmob = mo_coeff[t][1].shape[1]
            nmo[t] = (nmoa, nmob)
            mo_occ[t] = numpy.asarray(mo_occ[t])
            assert mo_occ[t].ndim > 1 and mo_occ[t].shape[0] == 2
            mocca = mo_coeff[t][0][:,mo_occ[t][0]>0]
            moccb = mo_coeff[t][1][:,mo_occ[t][1]>0]
            mocc[t] = (mocca, moccb)
            nocca = mocca.shape[1]
            noccb = moccb.shape[1]
            nocc[t] = (nocca, noccb)
        else: # restricted
            is_component_unrestricted[t] = False
            nao[t], nmo[t] = mo_coeff[t].shape
            mocc[t] = mo_coeff[t][:,mo_occ[t]>0]
            nocc[t] = mocc[t].shape[1]

    debug = False
    if isinstance(verbose, int):
        if verbose >= logger.DEBUG1:
            debug = True
    elif hasattr(verbose, 'verbose'):
        if verbose.verbose >= logger.DEBUG1:
            debug = True
    if fx is None:
        fx = gen_vind(mf, mo_coeff, mo_occ, debug=debug)

    s1a = {}
    for t, comp in mol.components.items():
        s1a[t] = -comp.intor('int1e_ipovlp', comp=3)

    def _ao2mo(mat, mo_coeff, mocc):
        return numpy.asarray([reduce(numpy.dot, (mo_coeff.T, x, mocc)) for x in mat])

    mo1s = {}
    e1s = {}
    for t in mo_coeff.keys():
        if is_component_unrestricted[t]:
            mo1s[t] = [[None] * mol.natm, [None] * mol.natm]
            e1s[t] = [[None] * mol.natm, [None] * mol.natm]
        else:
            mo1s[t] = [None] * mol.natm
            e1s[t] = [None] * mol.natm
    mem_now = lib.current_memory()[0]
    max_memory = max(2000, max_memory*.9-mem_now)
    nmo_nocc = 0
    for t in mo_coeff.keys():
        if is_component_unrestricted[t]:
            nmo_nocc += nao[t]*(nocc[t][0]+nocc[t][1])
        else:
            nmo_nocc += nmo[t]*nocc[t]
        if t.startswith('n'):
            nmo_nocc += 3 # additional equations for CNEO
    # Change 6 to 8 because need to copy mo1base/(v in vind_vo) in cphf.solve
    blksize = max(2, int(max_memory*1e6/8 / (nmo_nocc*3*8)))
    for ia0, ia1 in lib.prange(0, len(atmlst), blksize):
        s1vo = {}
        h1vo = {}
        for t, comp in mol.components.items():
            s1voa = []
            s1vob = []
            h1voa = []
            h1vob = []
            aoslices = comp.aoslice_by_atom()
            for i0 in range(ia0, ia1):
                ia = atmlst[i0]
                p0, p1 = aoslices[ia][2:]
                s1ao = numpy.zeros((3,nao[t],nao[t]))
                s1ao[:,p0:p1] += s1a[t][:,p0:p1]
                s1ao[:,:,p0:p1] += s1a[t][:,p0:p1].transpose(0,2,1)
                if is_component_unrestricted[t]:
                    s1voa.append(_ao2mo(s1ao, mo_coeff[t][0], mocc[t][0]))
                    s1vob.append(_ao2mo(s1ao, mo_coeff[t][1], mocc[t][1]))
                    h1voa.append(_ao2mo(h1ao[t][0,ia], mo_coeff[t][0], mocc[t][0]))
                    h1vob.append(_ao2mo(h1ao[t][1,ia], mo_coeff[t][1], mocc[t][1]))
                else:
                    s1voa.append(_ao2mo(s1ao, mo_coeff[t], mocc[t]))
                    h1voa.append(_ao2mo(h1ao[t][ia], mo_coeff[t], mocc[t]))

            if is_component_unrestricted[t]:
                h1vo[t] = (numpy.vstack(h1voa), numpy.vstack(h1vob))
                s1vo[t] = (numpy.vstack(s1voa), numpy.vstack(s1vob))
            else:
                h1vo[t] = numpy.vstack(h1voa)
                s1vo[t] = numpy.vstack(s1voa)

        tol = mf.conv_tol_cpscf * (ia1 - ia0)
        mo1, e1, _ = cphf.solve(fx, mo_energy, mo_occ, h1vo, s1vo,
                                with_f1=True, verbose=verbose,
                                max_cycle=max_cycle, level_shift=level_shift, tol=tol)
        for t, comp in mo1.items():
            if is_component_unrestricted[t]:
                mo1a = numpy.einsum('pq,xqi->xpi', mo_coeff[t][0], comp[0]).reshape(-1,3,nao[t],nocc[t][0])
                mo1b = numpy.einsum('pq,xqi->xpi', mo_coeff[t][1], comp[1]).reshape(-1,3,nao[t],nocc[t][1])
                e1a = e1[t][0].reshape(-1,3,nocc[t][0],nocc[t][0])
                e1b = e1[t][1].reshape(-1,3,nocc[t][1],nocc[t][1])
                for k in range(ia1-ia0):
                    ia = atmlst[k+ia0]
                    mo1s[t][0][ia] = mo1a[k]
                    mo1s[t][1][ia] = mo1b[k]
                    e1s[t][0][ia] = e1a[k].reshape(3,nocc[t][0],nocc[t][0])
                    e1s[t][1][ia] = e1b[k].reshape(3,nocc[t][1],nocc[t][1])
                mo1a = mo1b = e1a = e1b = None
            else:
                mo1a = numpy.einsum('pq,xqi->xpi', mo_coeff[t], comp).reshape(-1,3,nao[t],nocc[t])
                e1a = e1[t].reshape(-1,3,nocc[t],nocc[t])
                for k in range(ia1-ia0):
                    ia = atmlst[k+ia0]
                    mo1s[t][ia] = mo1a[k]
                    e1s[t][ia] = e1a[k].reshape(3,nocc[t],nocc[t])
                mo1a = e1a = None
            mo1[t] = e1[t] = None
        mo1 = e1 = None
    return mo1s, e1s

def gen_vind(mf, mo_coeff, mo_occ, debug=False):
    nao = {}
    nmo = {}
    mocc = {}
    nocc = {}
    is_component_unrestricted = {}
    for t in mo_coeff.keys():
        mo_coeff[t] = numpy.asarray(mo_coeff[t])
        if mo_coeff[t].ndim > 2: # unrestricted
            assert not t.startswith('n')
            assert mo_coeff[t].shape[0] == 2
            is_component_unrestricted[t] = True
            nao[t], nmoa = mo_coeff[t][0].shape
            nmob = mo_coeff[t][1].shape[1]
            nmo[t] = (nmoa, nmob)
            mo_occ[t] = numpy.asarray(mo_occ[t])
            assert mo_occ[t].ndim > 1 and mo_occ[t].shape[0] == 2
            mocca = mo_coeff[t][0][:,mo_occ[t][0]>0]
            moccb = mo_coeff[t][1][:,mo_occ[t][1]>0]
            mocc[t] = (mocca, moccb)
            nocca = mocca.shape[1]
            noccb = moccb.shape[1]
            nocc[t] = (nocca, noccb)
        else: # restricted
            is_component_unrestricted[t] = False
            nao[t], nmo[t] = mo_coeff[t].shape
            mocc[t] = mo_coeff[t][:,mo_occ[t]>0]
            nocc[t] = mocc[t].shape[1]
    vresp = mf.gen_response(mo_coeff, mo_occ, hermi=1)
    def fx(mo1, f1=None):
        dm1 = {}
        for t, comp in mo1.items():
            if is_component_unrestricted[t]:
                nmoa, nmob = nmo[t]
                mocca, moccb = mocc[t]
                nocca, noccb = nocc[t]
                comp = comp.reshape(-1,nmoa*nocca+nmob*noccb)
                nset = len(comp)
                dm1[t] = numpy.empty((2,nset,nao[t],nao[t]))
                for i, x in enumerate(comp):
                    xa = x[:nmoa*nocca].reshape(nmoa,nocca)
                    xb = x[nmoa*nocca:].reshape(nmob,noccb)
                    dma = reduce(numpy.dot, (mo_coeff[t][0], xa, mocca.T))
                    dmb = reduce(numpy.dot, (mo_coeff[t][1], xb, moccb.T))
                    dm1[t][0,i] = dma + dma.T
                    dm1[t][1,i] = dmb + dmb.T
            else:
                comp = comp.reshape(-1,nmo[t],nocc[t])
                nset = len(comp)
                dm1[t] = numpy.empty((nset,nao[t],nao[t]))
                for i, x in enumerate(comp):
                    if t.startswith('n'):
                        # quantum nuclei are always singly occupied
                        dm = reduce(numpy.dot, (mo_coeff[t], x, mocc[t].T))
                    else:
                        # *2 for double occupancy (RHF electrons)
                        dm = reduce(numpy.dot, (mo_coeff[t], x*2, mocc[t].T))
                    dm1[t][i] = dm + dm.T
        v1 = vresp(dm1)
        v1vo = {}
        if f1 is None:
            r1vo = None
        else:
            r1vo = {}
        for t, comp in mo1.items():
            if is_component_unrestricted[t]:
                nmoa, nmob = nmo[t]
                mocca, moccb = mocc[t]
                nocca, noccb = nocc[t]
                comp = comp.reshape(-1,nmoa*nocca+nmob*noccb)
                nset = len(comp)
                v1vo[t] = numpy.empty_like(comp)
                for i in range(nset):
                    v1vo[t][i,:nmoa*nocca] = reduce(numpy.dot, (mo_coeff[t][0].T, v1[t][0,i], mocca)).ravel()
                    v1vo[t][i,nmoa*nocca:] = reduce(numpy.dot, (mo_coeff[t][1].T, v1[t][1,i], moccb)).ravel()
            else:
                comp = comp.reshape(-1,nmo[t],nocc[t])
                v1vo[t] = numpy.empty_like(comp)
                for i, x in enumerate(v1[t]):
                    v1vo[t][i] = reduce(numpy.dot, (mo_coeff[t].T, x, mocc[t]))
            if f1 is not None and t in f1 and t.startswith('n'):
                # DEBUG: Verify nuclear dm1 * int1e_r
                if debug:
                    position = numpy.einsum('aij,xij->ax', dm1[t], mf.components[t].int1e_r)
                    print(f'[DEBUG] norm(dm1 * int1e_r) for {t}: {numpy.linalg.norm(position)}')
                rvo = numpy.empty((3,nmo[t],nocc[t]))
                for i, x in enumerate(mf.components[t].int1e_r):
                    rvo[i] = reduce(numpy.dot, (mo_coeff[t].T, x, mocc[t]))
                # Calculate f1 * r and add to nuclear Fock derivative
                v1vo[t] += numpy.einsum('ax,xpi->api', f1[t], rvo)
                # Store r * mo1, which will lead to equation r * mo1 = 0
                r1vo[t] = numpy.einsum('api,xpi->ax', comp, rvo)
        return v1vo, r1vo
    return fx

def dipole_grad(hessobj, mo1=None):
    'Gradients for molecular dipole moment with CNEO'
    mol = hessobj.mol
    mf = hessobj.base
    mf_e = mf.components['e']
    if isinstance(mf_e, scf.uhf.UHF):
        return None # dipole_grad is not yet implemented for UKS
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    natm = mol.natm
    atmlst = range(natm)
    de = numpy.zeros((natm, 3, 3))
    for i in range(natm): # contribution from nuclei
        de[i] = numpy.eye(3) * mol.atom_charge(i)

    if mo1 is None:
        mo1 = hessobj.mo1 # mo1 might be available if hessobj.kernel() has been run

    if mo1 is None:
        h1ao = hessobj.make_h1(mo_coeff, mo_occ, None, atmlst, hessobj.verbose)
        mo1, mo_e1 = hessobj.solve_mo1(mo_energy, mo_coeff, mo_occ, h1ao,
                                       None, atmlst, hessobj.max_memory,
                                       hessobj.verbose)
        hessobj.mo1 = mo1 # Store mo1

    # contribution from electrons
    mol_e = mol.components['e']
    nao_e = mol_e.nao

    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    charge_center = numpy.einsum('i,ix->x', charges, coords) / charges.sum()

    with mol_e.with_common_orig(charge_center):
        int1e_irp = - mol_e.intor("int1e_irp", comp=9)
        int1e_r = mol_e.intor_symmetric("int1e_r", comp=3)

    dm = mf.make_rdm1()
    dm_e = dm['e']
    if isinstance(mf_e, scf.uhf.UHF):
        assert dm_e.ndim > 2 and dm_e.shape[0] == 2
        dm_e = dm_e[0] + dm_e[1]
    aoslices = mol_e.aoslice_by_atom()
    for a in range(natm):
        p0, p1 = aoslices[a][2:]
        h2ao = numpy.zeros((9, nao_e, nao_e))
        h2ao[:,:,p0:p1] += int1e_irp[:,:,p0:p1] # nable is on ket in int1e_irp
        h2ao[:,p0:p1] += int1e_irp[:,:,p0:p1].transpose(0, 2, 1)
        de[a] -= numpy.einsum('xuv,uv->x',h2ao, dm_e).reshape(3, 3).T

    dm1e = numpy.einsum('Axui,vi->Axuv', numpy.array(mo1['e']), mf_e.mo_coeff[:, mf_e.mo_occ > 0])
    de -= 4 * numpy.einsum('Axuv,tuv->Axt', dm1e, int1e_r)
    #mo1_grad = numpy.einsum("up,uv,Axvi->Axpi", mf_e.mo_coeff, mf_e.get_ovlp(), mo1['e'])
    #h1_dip = numpy.einsum("tuv,up,vi->tpi", int1e_r, mf_e.mo_coeff, mf_e.mo_coeff[:, mf_e.mo_occ > 0])
    #de -= 4 * numpy.einsum("tpi,Axpi->Axt", h1_dip, mo1_grad)


    '''
    # contributions from quantum nuclei
    for i in range(mol.nuc_num):
        ia = mol.nuc[i].atom_index
        nao_n = mol.nuc[i].nao
        int1n_irp = mol.nuc[i].intor("int1e_irp").reshape(3, 3, nao_n, nao_n)
        dm_n = dm[f'n{ia}']
        de[ia] -= numpy.einsum("xtuv,uv->xt", int1n_irp, dm_n) * 2

        mf_n = mf.mf_nuc[i]
        dm1n = numpy.einsum('Axui,vi->Axuv', numpy.array(mo1[f'n{ia}']), mf_n.mo_coeff[:, mf_n.mo_occ > 0])
        int1n_r = mol.nuc[i].intor_symmetric("int1e_r")
        de += 4 * numpy.einsum('Axuv,tuv->Axt', dm1n, int1n_r)
    '''

    return de

class Hessian(rhf_hessian.HessianBase):
    '''
    Examples::

    >>> from pyscf import neo
    >>> mol = neo.M(atom='H 0 0 0.00; C 0 0 1.064; N 0 0 2.220', basis='ccpvdz')
    >>> mf = neo.CDFT(mol, xc='b3lyp')
    >>> mf.scf()
    >>> hess = mf.Hessian()
    >>> h = hess.kernel()
    >>> freq_info = hess.harmonic_analysis(mol, h)
    >>> print(freq_info)
    '''

    def __init__(self, scf_method):
        super().__init__(scf_method)
        if self.base.epc is not None:
            raise NotImplementedError('Hessian with epc is not implemented')
        self.components = {}
        for t, comp in self.base.components.items():
            self.components[t] = general_hessian(comp.Hessian())
        self.max_cycle = 100 # bump up from 50 as (C)NEO-CPHF is harder to converge
        self.mo1 = None # Store mo1 for IR intensity
        self._keys = self._keys.union(['components', 'mo1'])

    partial_hess_int = partial_hess_int
    hess_elec = hess_elec
    make_h1 = make_h1

    def solve_mo1(self, mo_energy, mo_coeff, mo_occ, h1ao,
                  fx=None, atmlst=None, max_memory=4000, verbose=None):
        return solve_mo1(self.base, mo_energy, mo_coeff, mo_occ, h1ao,
                         fx, atmlst, max_memory, verbose,
                         max_cycle=self.max_cycle, level_shift=self.level_shift)

    def hess_nuc(self, mol=None, atmlst=None):
        if mol is None:
            mol = self.mol
        mol_e = mol.components['e']
        return self.components['e'].hess_nuc(mol_e, atmlst)

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        de = self.hess_elec(mo_energy, mo_coeff, mo_occ, atmlst=atmlst)
        self.de = de + self.hess_nuc(self.mol, atmlst=atmlst)
        if self.base.do_disp():
            self.de += self.components['e'].get_dispersion()
        return self.de
    hess = kernel

    def harmonic_analysis(self, mol, hess, exclude_trans=True, exclude_rot=True,
                          imaginary_freq=True, mass=None, intensity=True):
        if mass is None:
            mass = mol.mass

        results = harmonic_analysis(mol, hess, exclude_trans=exclude_trans,
                                    exclude_rot=exclude_rot, imaginary_freq=imaginary_freq,
                                    mass=mass)
        if intensity is True:
            'unit: km/mol'

            modes = results["norm_mode"].reshape(-1, mol.natm * 3)
            #indices = numpy.asarray(range(mol.natm))

            #im = numpy.repeat(mass[indices]**-0.5, 3)
            #modes = numpy.einsum('in,n->in', modes, im) # Un-mass-weight eigenvectors


            # TODO: UKS case
            dipole_de = dipole_grad(self, self.mo1)
            if dipole_de is not None:
                dipole_de = dipole_de.reshape(-1, 3)
                de_q = numpy.einsum('nt, in->it', dipole_de, modes) # dipole gradients w.r.t normal coordinates

                # Conversion factor from atomic units to (D/Angstrom)^2/amu.
                # 1 (D/Angstrom)^2/amu = 42.255 km/mol
                # import qcelemental as qcel
                # conv = qcel.constants.conversion_factor("(e^2 * bohr^2)/(bohr^2 * atomic_unit_of_mass)",
                #                                         "debye^2 / (angstrom^2 * amu)")
                # or
                # from ase import units
                # conv = (1.0 / units.Debye)**2 * units._amu / units._me
                # conv = 42055.45033345739
                # ir_inten = numpy.einsum("qt, qt -> q", de_q, de_q) * conv * 1e-3 * numpy.pi / 3

                # from atomic units to km/mol
                # Ref: J Comput Chem 23: 895–910, 2002, Eq. 13-14
                from scipy.constants import physical_constants
                alpha = physical_constants["fine-structure constant"][0]
                amu = physical_constants["atomic mass constant"][0]
                m_e = physical_constants["electron mass"][0]
                N_A = physical_constants["Avogadro constant"][0]
                a_0 = physical_constants["Bohr radius"][0]

                unit_kmmol = alpha**2 * (1e-3 / amu) * m_e * N_A * numpy.pi * a_0 / 3
                ir_inten = numpy.einsum("qt, qt -> q", de_q, de_q) * unit_kmmol

                results['intensity'] = ir_inten

        return results

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
