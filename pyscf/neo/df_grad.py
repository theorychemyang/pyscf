#!/usr/bin/env python

'''
Analytic gradient for density-fitting interaction Coulomb in CDFT
'''

import ctypes
from itertools import product
import numpy
import scipy
from pyscf import lib
from pyscf.grad import rhf as rhf_grad
from pyscf.neo import grad
from pyscf.lib import logger
from pyscf.df.grad.rhf import _int3c_wrapper, balance_partition
from pyscf.df.grad import rhf as df_rhf_grad
from pyscf.df import addons
from pyscf.ao2mo import _ao2mo

class _ElectronicGradWithoutJ:
    '''Electronic gradient with global J evaluated by the NEO interaction builder.

    CPU gradient assembly contracts AO derivative matrices with the density.
    The joint integral evaluator supplies full-range vk, including its .aux
    response tag. Additional range-separated K uses the electronic DF object.
    '''

    def __init__(self, comp, *, vk=None):
        self.__dict__.update(comp.__dict__)
        # This local worker supplies the AO K derivative from joint integral work.
        self.vk = vk

    def get_j(self, mol=None, dm=None, hermi=0, omega=None):
        # Global J supplies Coulomb derivatives; preserve the electronic
        # gradient API's zero-J shape and auxiliary-response tag.
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
            if self.vk is not None and (omega is None or omega == 0):
                # Conventional DFT scales K in place; preserve the supplied derivative.
                vk = self.vk.copy()
                if self.auxbasis_response:
                    vk = lib.tag_array(vk, aux=self.vk.aux.copy())
            else:
                vk = super().get_jk(mol, dm, hermi, False, True, omega)[1]
        return vj, vk



def _cho_solve_rhok(mf_grad, mol, auxmol, orbol, orbor):
    '''Return raw J projections and metric-solved K factors.

    Copied from df.grad.rhf._cho_solve_rhojk. The NEO delta preserves rhoj
    in raw auxiliary coordinates for the subsequent global metric solve.
    '''
    # Begin df.grad.rhf._cho_solve_rhojk forward contraction and HDF storage.
    nset = len (orbol)
    nao, naux = mol.nao, auxmol.nao
    nbas, nauxbas = mol.nbas, auxmol.nbas
    ao_loc = mol.ao_loc
    nocc = [o.shape[-1] for o in orbor]

    int2c = auxmol.intor('int2c2e', aosym='s1')
    solve_j2c = df_rhf_grad._gen_metric_solver(int2c)
    int2c = None
    get_int3c_s1 = _int3c_wrapper(mol, auxmol, 'int3c2e', 's1')
    rhoj = numpy.zeros((nset,naux))
    f_rhok = lib.H5TmpFile()
    t1 = (logger.process_clock (), logger.perf_counter ())
    max_memory = mf_grad.max_memory - lib.current_memory()[0]
    blksize = max_memory * .5e6/8 / (naux*nao)
    mol_ao_ranges = balance_partition(ao_loc, blksize)
    nsteps = len(mol_ao_ranges)
    t2 = t1
    for istep, (shl0, shl1, nd) in enumerate(mol_ao_ranges):
        int3c = get_int3c_s1((0, nbas, shl0, shl1, 0, nauxbas))
        t2 = logger.timer_debug1 (mf_grad, 'NEO df grad intor (P|mn)', *t2)
        p0, p1 = ao_loc[shl0], ao_loc[shl1]
        for i in range(nset):
            # MRH 05/21/2020: De-vectorize this because array contiguity -> multithread efficiency
            v = lib.dot(int3c.reshape (nao, -1, order='F').T, orbor[i]).reshape (naux, (p1-p0)*nocc[i])
            t2 = logger.timer_debug1 (mf_grad, 'NEO df grad einsum (P|mn) u_ni N_i = v_Pmi', *t2)
            rhoj[i] += numpy.dot (v, orbol[i][p0:p1].ravel ())
            t2 = logger.timer_debug1 (mf_grad, 'NEO df grad einsum v_Pmi u_mi = rho_P', *t2)
            v = solve_j2c(v)
            t2 = logger.timer_debug1 (mf_grad, 'NEO df grad cho_solve (P|Q) D_Qmi = v_Pmi', *t2)
            f_rhok['%s/%s'%(i,istep)] = v.reshape(naux,p1-p0,-1)
            t2 = logger.timer_debug1 (mf_grad, 'NEO df grad cache D_Pmi (m <-> i transpose upon retrieval)', *t2)
        int3c = v = None
    # NEO delta: retain raw rhoj for the global solve; rhok uses the e-only metric.
    int2c = None
    t1 = logger.timer_debug1 (mf_grad, 'NEO df grad electronic J source and K factors', *t1)
    class get_rhok_class :
        def __init__(self, my_f):
            self.f_rhok = my_f
        def __call__(self, set_id, p0, p1):
            buf = numpy.empty((p1-p0,nocc[set_id],nao))
            col1 = 0
            for istep in range(nsteps):
                dat = self.f_rhok['%s/%s'%(set_id,istep)][p0:p1]
                col0, col1 = col1, col1 + dat.shape[1]
                buf[:p1-p0,:,col0:col1] = dat.transpose(0,2,1)
            return buf
    get_rhok = get_rhok_class (f_rhok)
    # End df.grad.rhf._cho_solve_rhojk forward contraction and HDF storage.
    # Coulomb uses the spin sum; K factors retain their spin sets.
    return rhoj.sum(axis=0), get_rhok


def get_jk(mols, auxmol, dms, charges, atmlst, max_memory, df_nn=False,
           auxbasis_response=True, electronic_grad=None):
    """Calculate global J gradients and the requested electronic K derivative.

    Each component projection and derivative three-center integral is evaluated
    once.  Nuclear self-J is excluded; when ``df_nn`` is false, all nuclear-
    nuclear terms are excluded.

    ``dms['e']`` is the original electronic density, including its spin
    channels and MO tags. J uses its spin sum; K uses the original density.

    ``electronic_grad`` is the electronic component's DF gradient object,
    or None. Supplying it shares electronic integral tiles with K and returns
    both the global-J atom gradient and electronic AO K derivative.

    Returns
    -------
    numpy.ndarray or tuple
        Global Coulomb atom gradient, shape (len(atmlst), 3). With
        ``electronic_grad``, also return the AO K derivative with the
        auxiliary-response tag expected by conventional DFT gradient get_jk.
    """

    # K needs the original spin channels and MO tags; J needs the spin sum.
    dm_e = dms['e']
    if dm_e.ndim == 3:
        assert dm_e.shape[0] == 2
        dms = dms.copy()
        dms['e'] = dm_e[0] + dm_e[1]

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
            if electronic_grad is not None and mol is mols['e'] and p0 < auxmol_e.nao:
                # Begin df.grad.rhf.get_jk ip1 K contraction. Embedding its
                # internal loop shares global J's electronic integral tile.
                int3c = int3c.transpose (0,3,2,1)
                for i in range(nset):
                    tmp = numpy.empty ((3,p1-p0,nocc[i],nao), dtype=orbol[0].dtype)
                    fdrv(ftrans1, fmmm1,
                         tmp.ctypes.data_as(ctypes.c_void_p),
                         int3c.ctypes.data_as(ctypes.c_void_p),
                         orbol[i].ctypes.data_as(ctypes.c_void_p),
                         ctypes.c_int (3*(p1-p0)), ctypes.c_int (nao),
                         (ctypes.c_int*4)(0, nocc[i], 0, nao),
                         null, ctypes.c_int(0))
                    rhok = get_rhok (i, p0, p1)
                    vk[i] += lib.einsum('xpoi,pok->xik', tmp, rhok)
                    rhok = tmp = None
                # End pyscf.df.grad.rhf.get_jk ip1 K contraction.
            int3c = None
        return vj

    assert mols.keys() == dms.keys() == charges.keys()
    naux = auxmol.nao
    aux_loc = auxmol.ao_loc
    if electronic_grad is not None:
        mol_e = mols['e']
        e_df = electronic_grad.base.with_df
        auxmol_e = e_df.auxmol
        if auxmol_e is None:
            auxmol_e = addons.make_auxmol(mol_e, e_df.auxbasis)
        # Follow df.grad.rhf.get_jk/_cho_solve_rhojk for electronic K factors.
        orbol, orbor = df_rhf_grad._decompose_rdm1(electronic_grad, mol_e, dm_e)
        nocc = [o.shape[-1] for o in orbor]
        nset = len(nocc)
        rhoj_raw_e, get_rhok = _cho_solve_rhok(
            electronic_grad, mol_e, auxmol_e, orbol, orbor)
        vk = numpy.zeros((nset,3,mol_e.nao,mol_e.nao))
        fdrv = _ao2mo.libao2mo.AO2MOnr_e2_drv
        fmmm1 = _ao2mo.libao2mo.AO2MOmmm_bra_nr_s1
        ftrans1 = _ao2mo.libao2mo.AO2MOtranse2_nr_s1
        null = lib.c_null_ptr()
        if auxbasis_response:
            # df.grad.rhf.get_jk occupied-pair factors use only e auxiliaries.
            rhok_oo = []
            for i, j in product(range(nset), repeat=2):
                tmp = numpy.empty((auxmol_e.nao,nocc[i],nocc[j]))
                max_memory_ = max_memory - lib.current_memory()[0]
                blksize = int(min(max(max_memory_ * .5e6/8 /
                                     (mol_e.nao*max(nocc)), 20), auxmol_e.nao))
                for p0, p1 in lib.prange(0, auxmol_e.nao, blksize):
                    rhok = get_rhok(i, p0, p1).reshape((p1-p0)*nocc[i], mol_e.nao)
                    tmp[p0:p1] = lib.dot(rhok, orbol[j]).reshape(p1-p0,nocc[i],nocc[j])
                rhok_oo.append(tmp)
            vkaux = numpy.zeros((nset,nset,3,auxmol_e.nao))
            fmmm2 = _ao2mo.libao2mo.AO2MOmmm_bra_nr_s2
            ftrans2 = _ao2mo.libao2mo.AO2MOtranse2_nr_s2
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
        if t == 'e' and electronic_grad is not None:
            # Keep K tiles inside the contiguous electronic auxiliary prefix.
            nbas_e = auxmol_e.nbas
            ao_ranges[t] = balance_partition(aux_loc[:nbas_e+1], blksize)
            ao_ranges[t] += [(shl0+nbas_e, shl1+nbas_e, nL)
                             for shl0, shl1, nL in balance_partition(
                                 aux_loc[nbas_e:]-aux_loc[nbas_e], blksize)]
        get_int3c_s2 = _int3c_wrapper(mol_t, auxmol, 'int3c2e', 's2ij')
        get_int3c_ip1[t] = _int3c_wrapper(mol_t, auxmol, 'int3c2e_ip1', 's1')
        get_int3c_ip2[t] = _int3c_wrapper(mol_t, auxmol, 'int3c2e_ip2', 's2ij')
        if t == 'e' and electronic_grad is not None:
            # Electronic auxiliary projections come from the K forward pass.
            # Evaluate only the added nuclear auxiliary shells for this source.
            rhoj_t = numpy.zeros(naux)
            rhoj_t[:auxmol_e.nao] = rhoj_raw_e
            for shl0, shl1, _ in ao_ranges[t]:
                if shl0 < nbas_e:
                    continue
                int3c = get_int3c_s2((0, mol_t.nbas, 0, mol_t.nbas, shl0, shl1))
                p0, p1 = aux_loc[shl0], aux_loc[shl1]
                rhoj_t[p0:p1] = lib.einsum('wp,w->p', int3c, dm_tril[t])
            rhoj_raw.append(rhoj_t)
        else:
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
                if t == 'e' and electronic_grad is not None and p0 < auxmol_e.nao:
                    # Begin df.grad.rhf.get_jk ip2 K contraction, sharing
                    # global J's electronic-AO/electronic-auxiliary tile.
                    for i in range(nset):
                        buf = numpy.empty((3,p1-p0,nocc[i],mol_e.nao), dtype=orbol[i].dtype)
                        fdrv(ftrans2, fmmm2,
                             buf.ctypes.data_as(ctypes.c_void_p),
                             int3c.ctypes.data_as(ctypes.c_void_p),
                             orbol[i].ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_int(3*(p1-p0)), ctypes.c_int(mol_e.nao),
                             (ctypes.c_int*4)(0, nocc[i], 0, mol_e.nao),
                             null, ctypes.c_int(0))
                        for j in range(nset):
                            ints_ij = lib.dot(buf.reshape(-1, mol_e.nao), orbor[j])
                            ints_ij = ints_ij.reshape(3,p1-p0,nocc[i],nocc[j])
                            vkaux[i,j,:,p0:p1] += lib.einsum(
                                'xpij,pij->xp', ints_ij, rhok_oo[i*nset+j][p0:p1])
                    buf = ints_ij = None
                    # End pyscf.df.grad.rhf.get_jk ip2 K contraction.
                int3c = None

        # (d/dX P|Q)
        int2c_e1 = auxmol.intor('int2c2e_ip1', aosym='s1')
        if electronic_grad is not None:
            # df.grad.rhf.get_jk metric derivative: its electronic principal
            # block is already present in the global metric derivative.
            int2c_e1_e = int2c_e1[:,:auxmol_e.nao,:auxmol_e.nao]
            for i, j in product(range(nset), repeat=2):
                tmp = lib.einsum('pij,qji->pq', rhok_oo[i*nset+j], rhok_oo[j*nset+i])
                vkaux[i,j] -= lib.einsum('xpq,pq->xp', int2c_e1_e, tmp)
        vjaux -= lib.einsum('xpq,tp,tq->xp', int2c_e1,
                            numpy.asarray([charges[t] * rhoj[t] for t in keys]),
                            numpy.asarray([rhoj_out[t] for t in keys]))

        auxslices = auxmol.aoslice_by_atom()
        de_aux = numpy.array([vjaux[:, p0:p1].sum(axis=1)
                              for p0, p1 in auxslices[:, 2:]])
        if electronic_grad is not None:
            # Component-major auxiliaries repeat the physical atom list for
            # each component. Sum their forces onto the original atom indices.
            de_aux = de_aux.reshape(-1, mols['e'].natm, 3).sum(axis=0)
        de -= de_aux[list(atmlst)]
    if electronic_grad is not None:
        out_shape = numpy.asarray(dm_e).shape[:-2] + (3,mol_e.nao,mol_e.nao)
        vk = -vk.reshape(out_shape)
        if auxbasis_response:
            auxslices_e = auxmol_e.aoslice_by_atom()
            vkaux = numpy.array([-vkaux[:,:,:,p0:p1].sum(axis=3)
                                 for p0, p1 in auxslices_e[:,2:]])
            vkaux = numpy.ascontiguousarray(vkaux.transpose(1,2,0,3))
            vk = lib.tag_array(vk, aux=vkaux)
        return de, vk
    return de

def grad_int(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None,
             electronic_grad=None):
    '''Return the global Coulomb atom gradient.

    Supplying electronic_grad also requests the electronic AO K derivative,
    including its auxiliary-response tag. The caller selects compatible
    fitting settings before requesting joint electronic integral evaluation.
    '''
    mf = mf_grad.base
    mol = mf_grad.mol

    with_k = electronic_grad is not None
    vk = None
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
            mols[t] = comp.mol
            dms[t] = dm_t
            charges[t] = comp.charge

    if not mf.df_ne:
        de += grad.grad_eri(mf_grad, dm0, mf.interactions.items(), atmlst)
    elif not mf.df_nn:
        eri_interactions = [(pair, interaction) for pair, interaction in mf.interactions.items()
                            if 'e' not in pair]
        if eri_interactions:
            de += grad.grad_eri(mf_grad, dm0, eri_interactions, atmlst)

    if mf.df_ne:
        t0 = (logger.process_clock(), logger.perf_counter())
        with_df = mf.with_df
        if electronic_grad is not None:
            auxmol = with_df.auxmol
            if auxmol is None:
                auxmol = with_df.make_auxmol()
        else:
            if with_df._auxmol_atom_major is None:
                with_df._auxmol_atom_major = with_df.make_auxmol_atom_major()
            auxmol = with_df._auxmol_atom_major

        result = get_jk(mols, auxmol, dms, charges, atmlst,
                        mf_grad.max_memory, mf.df_nn,
                        mf_grad.auxbasis_response, electronic_grad=electronic_grad)
        if electronic_grad is None:
            de += result
        else:
            de += result[0]
            vk = result[1]
        logger.timer(mf_grad, 'df grad vj', *t0)

    if log.verbose >= logger.DEBUG:
        log.debug('gradients of Coulomb interaction')
        rhf_grad._write(log, mol, de, atmlst)

    if with_k:
        return de, vk
    return de


class Gradients(grad.Gradients):
    '''Analytic gradient for density-fitting CDFT'''

    auxbasis_response = True
    grad_int = grad_int

    def grad_elec(self, mo_energy, mo_coeff, mo_occ, atmlst=None):
        comp = self.components['e']
        mf = self.base
        mf_e = comp.base
        share_ints = (mf.df_ne and mf.with_df.df_ne_scheme == 'global' and
                      not mf_e.only_dfj and
                      mf_e.with_df.auxbasis == mf.with_df.auxbasis and
                      comp.auxbasis_response == self.auxbasis_response)
        if share_ints and hasattr(mf_e, '_numint'):
            omega, alpha, hyb = mf_e._numint.rsh_and_hybrid_coeff(mf_e.xc, spin=mf_e.mol.spin)
            share_ints = mf_e._numint.libxc.is_hybrid_xc(mf_e.xc)
            share_ints = share_ints and (omega == 0 or (alpha != 0 and hyb != 0))
        if share_ints:
            de, vk = self.grad_int(mo_energy, mo_coeff, mo_occ, atmlst,
                                   electronic_grad=comp)
        else:
            de = self.grad_int(mo_energy, mo_coeff, mo_occ, atmlst)
            vk = None
        electronic_grad = comp
        if mf.df_ne:
            electronic_grad = lib.set_class(
                _ElectronicGradWithoutJ(comp, vk=vk),
                (_ElectronicGradWithoutJ, comp.__class__))
        return super().grad_elec(mo_energy, mo_coeff, mo_occ, atmlst,
                                 electronic_grad=electronic_grad, interaction_grad=de)


Grad = Gradients
