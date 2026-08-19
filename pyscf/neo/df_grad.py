#!/usr/bin/env python

'''
Analytic gradient for density-fitting interaction Coulomb in CDFT
'''

import numpy
import scipy
from pyscf import lib
from pyscf.grad import rhf as rhf_grad
from pyscf.neo import grad
from pyscf.lib import logger
from pyscf.df.grad.rhf import _int3c_wrapper, balance_partition

class _ElectronicGradWithoutJ:
    '''Electronic component gradient with J supplied by the global DF path.'''

    def get_j(self, mol=None, dm=None, hermi=0, omega=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.base.make_rdm1()
        nao = mol.nao
        dms = numpy.asarray(dm)
        out_shape = dms.shape[:-2] + (3,) + dms.shape[-2:]
        dms = dms.reshape(-1,nao,nao)
        nset = dms.shape[0]
        vj = numpy.zeros((nset,3,nao,nao))
        if self.auxbasis_response:
            vjaux = numpy.zeros((nset,nset,mol.natm,3))
            vj = lib.tag_array(vj.reshape(out_shape), aux=numpy.array(vjaux))
        else:
            vj = vj.reshape(out_shape)
        return vj

    def get_jk(self, mol=None, dm=None, hermi=0, with_j=True, with_k=True,
               omega=None):
        vj = vk = None
        if with_j:
            vj = self.get_j(mol, dm, hermi)
        if with_k:
            vk = super().get_jk(mol, dm, hermi, False, True, omega)[1]
        return vj, vk


def get_j(mols, auxmol, dms, charges, atmlst, max_memory, df_nn=False,
          auxbasis_response=True):
    """Calculate the global density-fitting J gradient in CNEO.

    Each component projection and derivative three-center integral is evaluated
    once.  Nuclear self-J is excluded; when ``df_nn`` is false, all nuclear-
    nuclear terms are excluded.

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

    assert mols.keys() == dms.keys() == charges.keys()
    naux = auxmol.nao
    aux_loc = auxmol.ao_loc
    dm_tril = {}
    ao_ranges = {}
    get_int3c_ip1 = {}
    get_int3c_ip2 = {}
    rhoj_raw = []
    keys = list(mols)
    for t in keys:
        mol_t = mols[t]
        dm_t = numpy.asarray(dms[t])
        dm_tril[t] = get_packed_dm(mol_t.nao, dm_t)
        max_memory_ = max_memory - lib.current_memory()[0]
        blksize = int(min(max(max_memory_ * .5e6/8 / (mol_t.nao**2*3), 20),
                          naux, 240))
        ao_ranges[t] = balance_partition(aux_loc, blksize)
        get_int3c_s2 = _int3c_wrapper(mol_t, auxmol, 'int3c2e', 's2ij')
        get_int3c_ip1[t] = _int3c_wrapper(mol_t, auxmol, 'int3c2e_ip1', 's1')
        get_int3c_ip2[t] = _int3c_wrapper(mol_t, auxmol, 'int3c2e_ip2', 's2ij')
        rhoj_raw.append(process_rhoj_block(get_int3c_s2, mol_t, dm_tril[t],
                                           ao_ranges[t], aux_loc))

    # (P|Q)
    int2c = auxmol.intor('int2c2e', aosym='s1')
    rhoj = scipy.linalg.solve(int2c, numpy.asarray(rhoj_raw).T,
                              assume_a='pos').T
    int2c = None
    rhoj = {t: rhoj[i] for i, t in enumerate(keys)}
    rhoj_total = sum(charges[t] * rhoj[t] for t in keys)
    rhoj_e = rhoj['e'] * charges['e']
    # The potential for each nucleus excludes its own fitted density.  Without
    # DF-NN, nuclei see only the electronic fitted density.
    rhoj_out = {}
    for t in keys:
        if t == 'e':
            rhoj_out[t] = rhoj_total
        elif df_nn:
            rhoj_out[t] = rhoj_total - charges[t] * rhoj[t]
        else:
            rhoj_out[t] = rhoj_e

    de = numpy.zeros((len(atmlst), 3))
    # (d/dX i,j|P)
    for t in keys:
        mol_t = mols[t]
        vj = process_vj_block(get_int3c_ip1[t], mol_t, rhoj_out[t],
                              ao_ranges[t], aux_loc) * charges[t]
        aoslices = mol_t.aoslice_by_atom()
        dm_t = numpy.asarray(dms[t])
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia, 2:]
            de[k] -= 2 * lib.einsum('xij,ij->x', vj[:, p0:p1], dm_t[p0:p1])

    vj = None

    if auxbasis_response:
        # (i,j|d/dX P)
        vjaux = numpy.zeros((3, naux))
        for t in keys:
            mol_t = mols[t]
            for shl0, shl1, _ in ao_ranges[t]:
                int3c = get_int3c_ip2[t]((0, mol_t.nbas, 0, mol_t.nbas, shl0, shl1))
                p0, p1 = aux_loc[shl0], aux_loc[shl1]
                vjaux[:, p0:p1] += lib.einsum('xwp,w,p->xp',
                                              int3c, dm_tril[t],
                                              rhoj_out[t][p0:p1]) * charges[t]
                int3c = None

        # (d/dX P|Q)
        int2c_e1 = auxmol.intor('int2c2e_ip1', aosym='s1')
        vjaux -= lib.einsum('xpq,tp,tq->xp', int2c_e1,
                            numpy.asarray([charges[t] * rhoj[t] for t in keys]),
                            numpy.asarray([rhoj_out[t] for t in keys]))

        auxslices = auxmol.aoslice_by_atom()
        de -= numpy.array([vjaux[:, p0:p1].sum(axis=1)
                           for p0, p1 in auxslices[:, 2:]])[list(atmlst)]
    return de

def grad_int(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    '''Calculate gradient for inter-component density-fitting Coulomb interactions'''
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
    mols = {}
    dms = {}
    charges = {}
    if mf.df_ne:
        for t, comp in mf.components.items():
            dm_t = dm0[t]
            if mf.with_df._unrestricted[t]:
                assert dm_t.ndim > 2 and dm_t.shape[0] == 2
                dm_t = dm_t[0] + dm_t[1]
            mols[t] = comp.mol
            dms[t] = dm_t
            charges[t] = comp.charge

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

        if not (mf.df_ne and ('e' in (t1, t2) or mf.df_nn)):
            de += grad.grad_pair_int(mol1, mol2, dm1, dm2,
                                     comp1.charge, comp2.charge, atmlst)

    if mf.df_ne:
        t0 = (logger.process_clock(), logger.perf_counter())
        with_df = mf.with_df
        if with_df._auxmol_atom_major is None:
            with_df._auxmol_atom_major = with_df.make_auxmol_atom_major()
        auxmol = with_df._auxmol_atom_major

        de += get_j(mols, auxmol, dms, charges, atmlst,
                    mf_grad.max_memory, mf.df_nn,
                    mf_grad.auxbasis_response)
        logger.timer(mf_grad, 'df grad vj', *t0)

    if log.verbose >= logger.DEBUG:
        log.debug('gradients of Coulomb interaction')
        rhf_grad._write(log, mol, de, atmlst)

    return de


class Gradients(grad.Gradients):
    '''Analytic gradient for density-fitting CDFT'''

    auxbasis_response = True
    grad_int = grad_int

    def __init__(self, mf):
        super().__init__(mf)
        comp_e = self.components.get('e', None)
        # Global J is evaluated with all components in grad_int.  The
        # electronic component keeps using its ordinary e-only DF object for K.
        if (comp_e is not None and mf.df_ne and
            not isinstance(comp_e, _ElectronicGradWithoutJ)):
            self.components['e'] = comp_e.view(lib.make_class(
                (_ElectronicGradWithoutJ, comp_e.__class__)))

    def reset(self, mol=None):
        super().reset(mol)
        comp_e = self.components.get('e', None)
        if (comp_e is not None and self.base.df_ne and
            not isinstance(comp_e, _ElectronicGradWithoutJ)):
            self.components['e'] = comp_e.view(lib.make_class(
                (_ElectronicGradWithoutJ, comp_e.__class__)))
        return self

Grad = Gradients
