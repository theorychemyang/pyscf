#!/usr/bin/env python

'''
Analytic gradient for density-fitting interaction Coulomb in CDFT
'''

import numpy
import scipy
from pyscf import lib
from pyscf.scf.jk import get_jk
from pyscf.grad import rhf as rhf_grad
from pyscf.neo import grad
from pyscf.lib import logger
from pyscf.df.grad.rhf import _int3c_wrapper, balance_partition


def get_cross_j(mol_e, mol_n, auxmol_e, dm_e, dm_n, atmlst, max_memory,
                auxbasis_response=True):
    """Calculate the cross J terms for molecular gradient calculations
    in CNEO density-fitting.

    This function computes the Coulomb contribution to the gradient arising from
    the interaction between two different molecular systems (e and n) while
    density-fitting is applied to electrons.

    Returns
    -------
    numpy.ndarray
        The gradient contribution with shape (len(atmlst), 3).
    """

    def get_packed_dm(nao, dm):
        assert dm.ndim == 2 # Does not support multiple dm's yet
        idx = numpy.arange(nao)
        idx = idx * (idx+1) // 2 + idx
        dm_tril = dm + dm.T
        dm_tril = lib.pack_tril(dm_tril)
        dm_tril[idx] *= .5
        return dm_tril

    def process_rhoj_block(get_int3c, mol, dm_tril, ao_ranges, aux_loc):
        naux = aux_loc[-1]
        rhoj = numpy.zeros(naux)
        for shl0, shl1, _ in ao_ranges:
            int3c = get_int3c((0, mol.nbas, 0, mol.nbas, shl0, shl1))
            p0, p1 = aux_loc[shl0], aux_loc[shl1]
            rhoj[p0:p1] = lib.einsum('wp,w->p', int3c, dm_tril)
            int3c = None
        return rhoj

    def process_vj_block(get_int3c, mol, rhoj, ao_ranges, aux_loc):
        nao = mol.nao
        vj = numpy.zeros((3, nao, nao))
        for shl0, shl1, _ in ao_ranges:
            int3c = get_int3c((0, mol.nbas, 0, mol.nbas, shl0, shl1))
            p0, p1 = aux_loc[shl0], aux_loc[shl1]
            vj += lib.einsum('xijp,p->xij', int3c, rhoj[p0:p1])
            int3c = None
        return vj

    get_int3c_s2_e = _int3c_wrapper(mol_e, auxmol_e, 'int3c2e', 's2ij')
    get_int3c_s2_n = _int3c_wrapper(mol_n, auxmol_e, 'int3c2e', 's2ij')
    get_int3c_ip1_e = _int3c_wrapper(mol_e, auxmol_e, 'int3c2e_ip1', 's1')
    get_int3c_ip1_n = _int3c_wrapper(mol_n, auxmol_e, 'int3c2e_ip1', 's1')
    get_int3c_ip2_e = _int3c_wrapper(mol_e, auxmol_e, 'int3c2e_ip2', 's2ij')
    get_int3c_ip2_n = _int3c_wrapper(mol_n, auxmol_e, 'int3c2e_ip2', 's2ij')

    nao_e = mol_e.nao
    nao_n = mol_n.nao
    naux = auxmol_e.nao
    dm_e = numpy.asarray(dm_e)
    dm_n = numpy.asarray(dm_n)

    dm_tril_e = get_packed_dm(nao_e, dm_e)
    dm_tril_n = get_packed_dm(nao_n, dm_n)

    aux_loc = auxmol_e.ao_loc
    max_memory_ = max_memory - lib.current_memory()[0]
    blksize = int(min(max(max_memory_ * .5e6/8 / (nao_e**2*3), 20), naux, 240))
    ao_ranges_e = balance_partition(aux_loc, blksize)

    # (i,j|P) for e
    rhoj_e = process_rhoj_block(get_int3c_s2_e, mol_e, dm_tril_e, ao_ranges_e, aux_loc)

    max_memory_ = max_memory - lib.current_memory()[0]
    blksize = int(min(max(max_memory_ * .5e6/8 / (nao_n**2*3), 20), naux, 240))
    ao_ranges_n = balance_partition(aux_loc, blksize)

    # (I,J|P) for n
    rhoj_n = process_rhoj_block(get_int3c_s2_n, mol_n, dm_tril_n, ao_ranges_n, aux_loc)

    # (P|Q)
    int2c = auxmol_e.intor('int2c2e', aosym='s1')
    rhoj_e = scipy.linalg.solve(int2c, rhoj_e.T, assume_a='pos').T
    rhoj_n = scipy.linalg.solve(int2c, rhoj_n.T, assume_a='pos').T
    int2c = None

    de = numpy.zeros((len(atmlst), 3))
    # (d/dX i,j|P)
    vj = process_vj_block(get_int3c_ip1_e, mol_e, rhoj_n, ao_ranges_e, aux_loc)
    aoslices = mol_e.aoslice_by_atom()

    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia, 2:]
        de[k] -= 2 * lib.einsum('xij,ij->x', vj[:, p0:p1], dm_e[p0:p1])

    # (d/dX I,J|P)
    vj = process_vj_block(get_int3c_ip1_n, mol_n, rhoj_e, ao_ranges_n, aux_loc)
    aoslices = mol_n.aoslice_by_atom()

    for k, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia, 2:]
        de[k] -= 2 * lib.einsum('xij,ij->x', vj[:, p0:p1], dm_n[p0:p1])

    vj = None

    if auxbasis_response:
        # (i,j|d/dX P) and (I,J|d/dX P)
        vjaux = numpy.empty((3, naux))
        for shl0, shl1, _ in ao_ranges_e:
            int3c = get_int3c_ip2_e((0, mol_e.nbas, 0, mol_e.nbas, shl0, shl1))  # (i,j|P)
            p0, p1 = aux_loc[shl0], aux_loc[shl1]
            vjaux[:, p0:p1] = lib.einsum('xwp,w,p->xp',
                                         int3c, dm_tril_e, rhoj_n[p0:p1])
            int3c = None

        for shl0, shl1, _ in ao_ranges_n:
            int3c = get_int3c_ip2_n((0, mol_n.nbas, 0, mol_n.nbas, shl0, shl1))  # (I,J|P)
            p0, p1 = aux_loc[shl0], aux_loc[shl1]
            vjaux[:, p0:p1] += lib.einsum('xwp,w,p->xp',
                                          int3c, dm_tril_n, rhoj_e[p0:p1])
            int3c = None

        # (d/dX P|Q)
        int2c_e1 = auxmol_e.intor('int2c2e_ip1', aosym='s1')
        vjaux -= lib.einsum('xpq,p,q->xp', int2c_e1, rhoj_e, rhoj_n) +\
                 lib.einsum('xpq,p,q->xp', int2c_e1, rhoj_n, rhoj_e)

        auxslices = auxmol_e.aoslice_by_atom()
        de -= numpy.array([vjaux[:, p0:p1].sum(axis=1)
                           for p0, p1 in auxslices[:, 2:]])
    return de

def grad_int(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    '''Calcuate gradient for inter-component density-fitting Coulomb interactions'''
    mf = mf_grad.base
    mol = mf_grad.mol

    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_occ is None:
        mo_occ = mf.mo_occ
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff

    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)

    dm0 = mf.make_rdm1(mo_coeff, mo_occ)

    if atmlst is None:
        atmlst = range(mol.natm)

    de = numpy.zeros((len(atmlst), 3))

    for (t1, t2), interaction in mf.interactions.items():
        comp1 = mf.components[t1]
        comp2 = mf.components[t2]
        dm1 = dm0[t1]
        if interaction.mf1_unrestricted:
            assert dm1.ndim > 2 and dm1.shape[0] == 2
            dm1 = dm1[0] + dm1[1]
        dm2 = dm0[t2]
        if interaction.mf2_unrestricted:
            assert dm2.ndim > 2 and dm2.shape[0] == 2
            dm2 = dm2[0] + dm2[1]
        mol1 = comp1.mol
        mol2 = comp2.mol

        check_ne = 0  # 0: not ne; 1: t1 is e t2 is n; 2: t1 is n t2 is e
        if mf_grad.base.df_ne:
            if t1 == 'e' and t2.startswith('n'):
                check_ne = 1
            elif t2 == 'e' and t1.startswith('n'):
                check_ne = 2

        if check_ne != 0:
            if check_ne == 1:
                mol_e, mol_n = mol1, mol2
                auxmol_e = comp1.with_df.auxmol
                dm_e, dm_n = dm1, dm2
            else:
                mol_e, mol_n = mol2, mol1
                auxmol_e = comp2.with_df.auxmol
                dm_e, dm_n = dm2, dm1

            de += get_cross_j(mol_e, mol_n, auxmol_e, dm_e, dm_n, atmlst,
                              mf_grad.max_memory, mf_grad.auxbasis_response) *\
                              comp1.charge * comp2.charge
        else:
            de += grad.grad_pair_int(mol1, mol2, dm1, dm2,
                                     comp1.charge, comp2.charge, atmlst)

    if log.verbose >= logger.DEBUG:
        log.debug('gradients of Coulomb interaction')
        rhf_grad._write(log, mol, de, atmlst)

    return de


class Gradients(grad.Gradients):
    '''Analtic graident for density-fitting CDFT'''

    def __init__(self, mf):
        super().__init__(mf)

    auxbasis_response = True
    grad_int = grad_int

Grad = Gradients
