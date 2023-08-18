#!/usr/bin/env python

'''
Nuclear Electronic Orbital Hartree-Fock (NEO-HF)
'''

import copy
import ctypes
import numpy
import scipy
import warnings
import h5py
from pyscf import df, gto, lib, neo, scf
from pyscf.data import nist
from pyscf.lib import logger
from pyscf.scf import _vhf, chkfile
from pyscf.scf.hf import TIGHT_GRAD_CONV_TOL
from pyscf.qmmm.itrf import qmmm_for_scf


def dot_eri_dm(eri, dms, nao_v=None, eri_dot_dm=True):
    assert(eri.dtype == numpy.double)
    eri = numpy.asarray(eri, order='C')
    dms = numpy.asarray(dms, order='C')
    dms_shape = dms.shape
    nao_dm = dms_shape[-1]
    if nao_v is None:
        nao_v = nao_dm

    dms = dms.reshape(-1,nao_dm,nao_dm)
    n_dm = dms.shape[0]

    vj = numpy.zeros((n_dm,nao_v,nao_v))

    dmsptr = []
    vjkptr = []
    fjkptr = []

    npair_v = nao_v*(nao_v+1)//2
    npair_dm = nao_dm*(nao_dm+1)//2
    if eri.ndim == 2 and npair_v*npair_dm == eri.size: # 4-fold symmetry eri
        if eri_dot_dm: # 'ijkl,kl->ij'
            fdrv = getattr(_vhf.libcvhf, 'CVHFnrs4_incore_drv_diff_size_v_dm')
            fvj = _vhf._fpointer('CVHFics4_kl_s2ij_diff_size')
        else: # 'ijkl,ij->kl'
            fdrv = getattr(_vhf.libcvhf, 'CVHFnrs4_incore_drv_diff_size_dm_v')
            fvj = _vhf._fpointer('CVHFics4_ij_s2kl_diff_size')
        for i, dm in enumerate(dms):
            dmsptr.append(dm.ctypes.data_as(ctypes.c_void_p))
            vjkptr.append(vj[i].ctypes.data_as(ctypes.c_void_p))
            fjkptr.append(fvj)
    else:
        raise RuntimeError('Array shape not consistent: nao_v %s, DM %s, eri %s'
                           % (nao_v, dms_shape, eri.shape))

    n_ops = len(dmsptr)
    fdrv(eri.ctypes.data_as(ctypes.c_void_p),
         (ctypes.c_void_p*n_ops)(*dmsptr), (ctypes.c_void_p*n_ops)(*vjkptr),
         ctypes.c_int(n_ops), ctypes.c_int(nao_v), ctypes.c_int(nao_dm),
         (ctypes.c_void_p*n_ops)(*fjkptr))

    for i in range(n_dm):
        lib.hermi_triu(vj[i], 1, inplace=True)
    if n_dm == 1:
        vj = vj.reshape((nao_v,nao_v))
    else:
        vj = vj.reshape((n_dm,nao_v,nao_v))
    return vj

def _is_mem_enough(nao1, nao2, max_memory):
    return nao1**2*nao2**2*2/1e6+lib.current_memory()[0] < max_memory*.95

def _build_eri_ne(mol_elec, mol_nuc):
    eri_ne = None
    if mol_elec.super_mol.incore_anyway or \
        _is_mem_enough(mol_elec.nao_nr(), mol_nuc.nao_nr(), mol_elec.super_mol.max_memory):
        atm, bas, env = gto.conc_env(mol_nuc._atm, mol_nuc._bas, mol_nuc._env,
                                     mol_elec._atm, mol_elec._bas, mol_elec._env)
        intor_name = 'int2e_sph'
        if mol_elec.super_mol.cart is True:
            intor_name = 'int2e_cart'
        eri_ne = \
            gto.moleintor.getints(intor_name, atm, bas, env,
                                  shls_slice=(0, mol_nuc._bas.shape[0], 0, mol_nuc._bas.shape[0],
                                              mol_nuc._bas.shape[0],
                                              mol_nuc._bas.shape[0] + mol_elec._bas.shape[0],
                                              mol_nuc._bas.shape[0],
                                              mol_nuc._bas.shape[0] + mol_elec._bas.shape[0]),
                                  aosym='s4')
    return eri_ne

def _build_eri_nn(mol_nuc1, mol_nuc2):
    '''idx mole_nuc2 > mole_nuc1'''
    eri_nn = None
    if mol_nuc1.super_mol.incore_anyway or \
        _is_mem_enough(mol_nuc1.nao_nr(), mol_nuc2.nao_nr(), mol_nuc1.super_mol.max_memory):
        atm, bas, env = gto.conc_env(mol_nuc1._atm, mol_nuc1._bas, mol_nuc1._env,
                                     mol_nuc2._atm, mol_nuc2._bas, mol_nuc2._env)
        intor_name = 'int2e_sph'
        if mol_nuc1.super_mol.cart is True:
            intor_name = 'int2e_cart'
        eri_nn = \
            gto.moleintor.getints(intor_name, atm, bas, env,
                                  shls_slice=(0, mol_nuc1._bas.shape[0], 0, mol_nuc1._bas.shape[0],
                                              mol_nuc1._bas.shape[0],
                                              mol_nuc1._bas.shape[0] + mol_nuc2._bas.shape[0],
                                              mol_nuc1._bas.shape[0],
                                              mol_nuc1._bas.shape[0] + mol_nuc2._bas.shape[0]),
                                  aosym='s4')
    return eri_nn

def get_j_e_dm_n(idx_nuc, dm_n, mol_elec=None, mol_nuc=None, eri_ne=None):
    '''Get electronic Coulomb matrix from e-n interaction'''
    ia = mol_nuc.atom_index
    charge = mol_elec.super_mol.atom_charge(ia)
    if eri_ne is not None and isinstance(eri_ne, (tuple, list)):
        if eri_ne[idx_nuc] is None:
            eri_ne[idx_nuc] = _build_eri_ne(mol_elec, mol_nuc)
        if eri_ne[idx_nuc] is not None:
            return -charge * dot_eri_dm(eri_ne[idx_nuc], dm_n, nao_v=mol_elec.nao_nr(),
                                        eri_dot_dm=False)
    warnings.warn("Direct Vee is used, might be slow. Check PYSCF_MAX_MEMORY?")
    return -charge * scf.jk.get_jk((mol_elec, mol_elec, mol_nuc, mol_nuc),
                                   dm_n, scripts='ijkl,lk->ij',
                                   intor='int2e', aosym='s4')

def get_j_n_dm_e(idx_nuc, dm_e, mol_elec=None, mol_nuc=None, eri_ne=None):
    '''Get nuclear Coulomb matrix from e-n interaction'''
    ia = mol_nuc.atom_index
    charge = mol_elec.super_mol.atom_charge(ia)
    if eri_ne is not None and isinstance(eri_ne, (tuple, list)):
        if eri_ne[idx_nuc] is None:
            eri_ne[idx_nuc] = _build_eri_ne(mol_elec, mol_nuc)
        if eri_ne[idx_nuc] is not None:
            return -charge * dot_eri_dm(eri_ne[idx_nuc], dm_e, nao_v=mol_nuc.nao_nr(),
                                        eri_dot_dm=True)
    warnings.warn("Direct Vee is used, might be slow. Check PYSCF_MAX_MEMORY?")
    return -charge * scf.jk.get_jk((mol_nuc, mol_nuc, mol_elec, mol_elec),
                                   dm_e, scripts='ijkl,lk->ij',
                                   intor='int2e', aosym='s4')

def get_j_nn(idx1, idx2, dm_n2, mol_nuc1=None, mol_nuc2=None, eri_nn=None):
    '''Get nuclear Coulomb matrix from n-n interaction'''
    ia = mol_nuc1.atom_index
    ja = mol_nuc2.atom_index
    charge = mol_nuc1.super_mol.atom_charge(ia) * mol_nuc2.super_mol.atom_charge(ja)
    if eri_nn is not None and isinstance(eri_nn, (tuple, list)):
        if idx1 < idx2:
            if eri_nn[idx1][idx2] is None:
                eri_nn[idx1][idx2] = _build_eri_nn(mol_nuc1, mol_nuc2)
            if eri_nn[idx1][idx2] is not None:
                return charge * dot_eri_dm(eri_nn[idx1][idx2], dm_n2,
                                           nao_v=mol_nuc1.nao_nr(),
                                           eri_dot_dm=True)
        elif idx1 > idx2:
            if eri_nn[idx2][idx1] is None:
                eri_nn[idx2][idx1] = _build_eri_nn(mol_nuc2, mol_nuc1)
            if eri_nn[idx2][idx1] is not None:
                return charge * dot_eri_dm(eri_nn[idx2][idx1], dm_n2,
                                           nao_v=mol_nuc1.nao_nr(),
                                           eri_dot_dm=False)
        elif idx1 == idx2:
            return 0.0
    warnings.warn("Direct Vee is used, might be slow. Check PYSCF_MAX_MEMORY?")
    return charge * scf.jk.get_jk((mol_nuc1, mol_nuc1, mol_nuc2, mol_nuc2),
                                  dm_n2, scripts='ijkl,lk->ij',
                                  intor='int2e', aosym='s4')

def hcore_nuc_qmmm(mm_mol, mol_n, charge):
    coords = mm_mol.atom_coords()
    charges = mm_mol.atom_charges()
    nao = mol_n.nao
    max_memory = mol_n.super_mol.max_memory - lib.current_memory()[0]
    blksize = int(min(max_memory*1e6/8/nao**2, 200))
    blksize = max(blksize, 1)
    v = 0
    if mm_mol.charge_model == 'gaussian':
        expnts = mm_mol.get_zetas()

        if mol_n.cart:
            intor = 'int3c2e_cart'
        else:
            intor = 'int3c2e_sph'
        cintopt = gto.moleintor.make_cintopt(mol_n._atm, mol_n._bas,
                                             mol_n._env, intor)
        for i0, i1 in lib.prange(0, charges.size, blksize):
            fakemol = gto.fakemol_for_charges(coords[i0:i1], expnts[i0:i1])
            j3c = df.incore.aux_e2(mol_n, fakemol, intor=intor,
                                   aosym='s2ij', cintopt=cintopt)
            v += numpy.einsum('xk,k->x', j3c, -charges[i0:i1])
        v = lib.unpack_tril(v)
    else: # point-charge model
        for i0, i1 in lib.prange(0, charges.size, blksize):
            j3c = mol_n.intor('int1e_grids', hermi=1, grids=coords[i0:i1])
            v += numpy.einsum('kpq,k->pq', j3c, -charges[i0:i1])
    return -charge * v

def get_hcore_nuc(mol, mf_nuc, dm_elec, dm_nuc, mol_elec=None,
                  mol_nuc=None, eri_ne=None, eri_nn=None):
    '''Get the core Hamiltonian for quantum nucleus.'''
    super_mol = mol.super_mol
    if mol_elec is None: mol_elec = super_mol.elec
    if mol_nuc is None: mol_nuc = super_mol.nuc
    ia = mol.atom_index
    # create the static part of hcore (true hcore) for the first time
    # and cache it
    if mf_nuc.hcore_static is None:
        # the mass of the quantum nucleus in a.u.
        mass = super_mol.mass[ia] * nist.ATOMIC_MASS / nist.E_MASS
        charge = super_mol.atom_charge(ia)
        # nuclear kinetic energy and Coulomb interactions with classical nuclei
        mf_nuc.hcore_static = mol.intor_symmetric('int1e_kin') / mass
        mf_nuc.hcore_static -= mol.intor_symmetric('int1e_nuc') * charge
        # QMMM part
        if super_mol.mm_mol is not None:
            mf_nuc.hcore_static += hcore_nuc_qmmm(super_mol.mm_mol, mol, charge)
    h = 0
    # find the index of mol
    # TODO: store this information
    i = 0
    found = False
    for i in range(len(mol_nuc)):
        if ia == mol_nuc[i].atom_index:
            found = True
            break
    if not found:
        raise RuntimeError('Failed to find the index of the quantum nucleus')
    # Coulomb interaction between the quantum nucleus and electrons
    ne_U = 0.0
    if dm_elec.ndim > 2:
        jcross = get_j_n_dm_e(i, dm_elec[0]+dm_elec[1], mol_elec=mol_elec,
                              mol_nuc=mol, eri_ne=eri_ne)
    else:
        jcross = get_j_n_dm_e(i, dm_elec, mol_elec=mol_elec, mol_nuc=mol,
                              eri_ne=eri_ne)
    h += jcross
    if isinstance(dm_nuc[i], numpy.ndarray):
        ne_U = numpy.einsum('ij,ji', jcross, dm_nuc[i]) # n-e Coulomb energy
    # Coulomb interactions between quantum nuclei
    nn_U = 0.0
    if super_mol.nuc_num > 1: # n-n exists only when there are 2 or more
        jcross = 0.0
        assert len(dm_nuc) == len(mol_nuc) == super_mol.nuc_num
        for j in range(super_mol.nuc_num):
            ja = mol_nuc[j].atom_index
            if ja != ia and isinstance(dm_nuc[j], numpy.ndarray):
                jcross += get_j_nn(i, j, dm_nuc[j], mol_nuc1=mol,
                                   mol_nuc2=mol_nuc[j], eri_nn=eri_nn)
        h += jcross
        if isinstance(dm_nuc[i], numpy.ndarray):
            nn_U = numpy.einsum('ij,ji', jcross, dm_nuc[i]) # n-n Coulomb energy
    # attach n-e and n-n Coulomb to h to use later when calculating the total energy
    h = lib.tag_array(mf_nuc.hcore_static + h, ne_U=ne_U, nn_U=nn_U)
    return h

def get_occ_nuc(mf):
    '''Label the occupation for each orbital of the quantum nucleus'''
    def get_occ(mo_energy=mf.mo_energy, mo_coeff=mf.mo_coeff):
        e_idx = numpy.argsort(mo_energy)
        mo_occ = numpy.zeros(mo_energy.size)
        mo_occ[e_idx[mf.occ_state]] = mf.mol.nnuc # 1 or fractional
        return mo_occ
    return get_occ

def get_occ_elec(mf):
    '''To support fractional occupations'''
    def get_occ(mo_energy=mf.mo_energy, mo_coeff=mf.mo_coeff):
        if mo_energy is None: mo_energy = mf.mo_energy
        e_idx_a = numpy.argsort(mo_energy[0])
        e_idx_b = numpy.argsort(mo_energy[1])
        e_sort_a = mo_energy[0][e_idx_a]
        e_sort_b = mo_energy[1][e_idx_b]
        nmo = mo_energy[0].size
        n_a, n_b = mf.nelec
        mo_occ = numpy.zeros_like(mo_energy)
        # change the homo occupation to fractional
        if n_a > n_b or n_b <= 0:
            mo_occ[0,e_idx_a[:n_a - 1]] = 1
            mo_occ[0,e_idx_a[n_a - 1]] = mf.mol.nhomo
            mo_occ[1,e_idx_b[:n_b]] = 1
        else:
            mo_occ[1,e_idx_b[:n_b - 1]] = 1
            mo_occ[1,e_idx_b[n_b - 1]] = mf.mol.nhomo
            mo_occ[0,e_idx_a[:n_a]] = 1
        if mf.verbose >= logger.INFO and n_a < nmo and n_b > 0 and n_b < nmo:
            if e_sort_a[n_a-1]+1e-3 > e_sort_a[n_a]:
                logger.warn(mf, 'alpha nocc = %d  HOMO %.15g >= LUMO %.15g',
                            n_a, e_sort_a[n_a-1], e_sort_a[n_a])
            else:
                logger.info(mf, '  alpha nocc = %d  HOMO = %.15g  LUMO = %.15g',
                            n_a, e_sort_a[n_a-1], e_sort_a[n_a])

            if e_sort_b[n_b-1]+1e-3 > e_sort_b[n_b]:
                logger.warn(mf, 'beta  nocc = %d  HOMO %.15g >= LUMO %.15g',
                            n_b, e_sort_b[n_b-1], e_sort_b[n_b])
            else:
                logger.info(mf, '  beta  nocc = %d  HOMO = %.15g  LUMO = %.15g',
                            n_b, e_sort_b[n_b-1], e_sort_b[n_b])

            if e_sort_a[n_a-1]+1e-3 > e_sort_b[n_b]:
                logger.warn(mf, 'system HOMO %.15g >= system LUMO %.15g',
                            e_sort_b[n_a-1], e_sort_b[n_b])

            numpy.set_printoptions(threshold=nmo)
            logger.debug(mf, '  alpha mo_energy =\n%s', mo_energy[0])
            logger.debug(mf, '  beta  mo_energy =\n%s', mo_energy[1])
            numpy.set_printoptions(threshold=1000)

        if mo_coeff is not None and mf.verbose >= logger.DEBUG:
            ss, s = mf.spin_square((mo_coeff[0][:,mo_occ[0]>0],
                                    mo_coeff[1][:,mo_occ[1]>0]), mf.get_ovlp())
            logger.debug(mf, 'multiplicity <S^2> = %.8g  2S+1 = %.8g', ss, s)
        return mo_occ
    return get_occ

def get_init_guess_nuc(mf, mol):
    '''Generate initial guess density matrix for the quantum nucleus

        Returns:
        Density matrix, 2D ndarray
    '''
    # TODO: SCF atom guess for quantum nuclei?
    h1n = mf.get_hcore(mol)
    s1n = mol.intor_symmetric('int1e_ovlp')
    mo_energy, mo_coeff = mf.eig(h1n, s1n)
    mo_occ = mf.get_occ(mo_energy, mo_coeff)
    return mf.make_rdm1(mo_coeff, mo_occ)

def get_hcore_elec(mol, mf_elec, dm_nuc, mol_nuc=None, eri_ne=None):
    '''Get the core Hamiltonian for electrons in NEO'''
    super_mol = mol.super_mol
    if mol_nuc is None: mol_nuc = super_mol.nuc
    # create the static part of hcore (true hcore) for the first time
    # and cache it
    if mf_elec.hcore_static is None:
        mf_elec.hcore_static = mf_elec.__class__.get_hcore(mf_elec, mol)
    j = 0
    # Coulomb interactions between electrons and all quantum nuclei
    for i in range(super_mol.nuc_num):
        if isinstance(dm_nuc[i], numpy.ndarray):
            j += get_j_e_dm_n(i, dm_nuc[i], mol_elec=mol, mol_nuc=mol_nuc[i],
                              eri_ne=eri_ne)
    return mf_elec.hcore_static + j

def get_veff_nuc_bare(mol):
    return numpy.zeros((mol.nao_nr(), mol.nao_nr()))

def get_fock(mf, h1e=None, s1e=None, vhf=None, dm=None, cycle=-1, diis=None,
             diis_start_cycle=None, level_shift_factor=None, damp_factor=None,
             diis_pos='both', diis_type=3):
    mol = mf.mol
    if mf.dm_elec is None:
        mf.dm_elec = mf.mf_elec.make_rdm1()
    for i in range(mol.nuc_num):
        if mf.dm_nuc[i] is None:
            mf.dm_nuc[i] = mf.mf_nuc[i].make_rdm1()
    if dm is None:
        if mf.dm_elec.ndim > 2: # UHF/UKS
            dm = [mf.dm_elec[0], mf.dm_elec[1]]
        else:
            dm = [mf.dm_elec]
        dm += mf.dm_nuc
    if s1e is None:
        s1e_e = mf.mf_elec.get_ovlp()
        if mf.dm_elec.ndim > 2: # UHF/UKS
            s1e = [s1e_e, s1e_e]
        else:
            s1e = [s1e_e]
        for i in range(mol.nuc_num):
            s1e.append(mf.mf_nuc[i].get_ovlp())
    if h1e is None:
        h1e = [mf.mf_elec.get_hcore()] \
              + [mf.mf_nuc[i].get_hcore() for i in range(mol.nuc_num)]
    if vhf is None:
        vhf = [mf.mf_elec.get_veff(mol.elec, mf.dm_elec)] \
              + [mf.mf_nuc[i].get_veff(mol.nuc[i], mf.dm_nuc[i])
                 for i in range(mol.nuc_num)]

    if mf.dm_elec.ndim > 2: # UHF/UKS
        f = [h1e[0] + vhf[0][0], h1e[0] + vhf[0][1]]
        start = 2
    else:
        f = [h1e[0] + vhf[0]]
        start = 1
    for i in range(mol.nuc_num):
        f.append(h1e[i + 1] + vhf[i + 1])

    # CNEO constraint term
    f0 = None
    fock_add = None
    if isinstance(mf, neo.CDFT):
        f0 = copy.deepcopy(f) # Fock without constraint term
        # NOTE: an important change is that even if not using DIIS,
        # we still optimize f. This helps with final extra cycle convergence
        if diis_pos == 'pre' or diis_pos == 'both' or \
                (cycle < 0 and diis is None):
            # optimize f in cNEO
            for i in range(mol.nuc_num):
                ia = mf.mf_nuc[i].mol.atom_index
                opt = scipy.optimize.root(mf.position_analysis, mf.f[ia],
                                          args=(mf.mf_nuc[i], f0[start + i],
                                                s1e[start + i]), method='hybr')
                logger.debug(mf, 'Lagrange multiplier of %s(%i) atom: %s' %
                             (mf.mol.atom_symbol(ia), ia, mf.f[ia]))
                logger.debug(mf, 'Position deviation: %s', opt.fun)
        fock_add = mf.get_fock_add_cdft()
        for i in range(mol.nuc_num):
            f[start + i] = f0[start + i] + fock_add[i]

    if cycle < 0 and diis is None:  # Not inside the SCF iteration
        return f

    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if damp_factor is None:
        damp_factor = mf.damp

    shifta = shiftb = shiftp = 0.0
    if isinstance(level_shift_factor, (tuple, list, numpy.ndarray)):
        if mf.dm_elec.ndim > 2:
            shifta, shiftb, shiftp = level_shift_factor
        else:
            shifta, shiftp = level_shift_factor
    else:
        shifta = shiftb = shiftp = level_shift_factor
    dampa = dampb = dampp = 0.0
    if isinstance(damp_factor, (tuple, list, numpy.ndarray)):
        if mf.dm_elec.ndim > 2:
            dampa, dampb, dampp = damp_factor
        else:
            dampa, dampp = damp_factor
    else:
        dampa = dampb = dampp = damp_factor

    # damping only when not CNEO
    if not isinstance(mf, neo.CDFT):
        if 0 <= cycle < diis_start_cycle-1:
            if mf.dm_elec.ndim > 2:
                if abs(dampa)+abs(dampb) > 1e-4:
                    f[0] = scf.hf.damping(s1e[0], dm[0], f[0], dampa)
                    f[1] = scf.hf.damping(s1e[1], dm[1], f[1], dampb)
                start = 2
            else:
                if abs(dampa) > 1e-4:
                    f[0] = scf.hf.damping(s1e[0], dm[0]*.5, f[0], dampa)
                start = 1
            if abs(dampp) > 1e-4:
                for i in range(mol.nuc_num):
                    f[start + i] = scf.hf.damping(s1e[start + i], dm[start + i],
                                                  f[start + i], dampp)

    if diis and cycle >= diis_start_cycle:
        if isinstance(mf, neo.CDFT):
            # if CNEO, needs to manually use lib.diis and pack/unpack
            sizes = [0]
            shapes = []
            for a in f:
                sizes.append(sizes[-1] + a.size)
                shapes.append(a.shape)
            f_ravel = numpy.concatenate(f, axis=None)
            if diis_type == 1:
                f0_ravel = numpy.concatenate(f0, axis=None)
                f_ravel = diis.update(f0_ravel, scf.diis.get_err_vec(s1e, dm, f))
            elif diis_type == 2:
                f_ravel = diis.update(f_ravel)
            elif diis_type == 3:
                f_ravel = diis.update(f_ravel, scf.diis.get_err_vec(s1e, dm, f))
            else:
                print("\nWARN: Unknow CDFT DIIS type, NO DIIS IS USED!!!\n")
            f = []
            for i in range(len(shapes)):
                f.append(f_ravel[sizes[i] : sizes[i+1]].reshape(shapes[i]))
            if diis_type == 1:
                for i in range(mol.nuc_num):
                    f[start + i] += fock_add[i]
        else:
            # if not CNEO, directly use the scf.diis object provided
            f = diis.update(s1e, dm, f)
            # WARNING: CDIIS only. Using EDIIS or ADIIS will cause errors

    if isinstance(mf, neo.CDFT) and (diis_pos == 'post' or diis_pos == 'both'):
        # notice that we redefine f0 as the extrapolated value, as f got extrapolated
        f0 = copy.deepcopy(f)
        for i in range(mol.nuc_num):
            f0[start + i] = f[start + i] - fock_add[i]
        # optimize f in cNEO
        for i in range(mol.nuc_num):
            ia = mf.mf_nuc[i].mol.atom_index
            opt = scipy.optimize.root(mf.position_analysis, mf.f[ia],
                                      args=(mf.mf_nuc[i], f0[start + i],
                                            s1e[start + i]), method='hybr')
            logger.debug(mf, 'Lagrange multiplier of %s(%i) atom: %s' %
                         (mf.mol.atom_symbol(ia), ia, mf.f[ia]))
            logger.debug(mf, 'Position deviation: %s', opt.fun)
        fock_add = mf.get_fock_add_cdft()
        for i in range(mol.nuc_num):
            f[start + i] = f0[start + i] + fock_add[i]

    # level shift only when not CNEO
    if not isinstance(mf, neo.CDFT):
        if mf.dm_elec.ndim > 2:
            if abs(shifta)+abs(shiftb) > 1e-4:
                f[0] = scf.hf.level_shift(s1e[0], dm[0], f[0], shifta)
                f[1] = scf.hf.level_shift(s1e[1], dm[1], f[1], shiftb)
            start = 2
        else:
            if abs(shifta) > 1e-4:
                f[0] = scf.hf.level_shift(s1e[0], dm[0]*.5, f[0], shifta)
            start = 1
        if abs(shiftp) > 1e-4:
            for i in range(mol.nuc_num):
                f[start + i] = scf.hf.level_shift(s1e[start + i], dm[start + i],
                                                  f[start + i], shiftp)
    return f

def energy_qmnuc(mf, h1n, dm_nuc, veff_n=None):
    '''Energy of the quantum nucleus'''
    ia = mf.mol.atom_index
    n1 = numpy.einsum('ij,ji', h1n, dm_nuc)
    logger.debug(mf, 'Energy of %s (%3d): %s', mf.mol.super_mol.atom_symbol(ia), ia, n1)
    return n1

def energy_tot(mf_elec, mf_nuc, dm_elec=None, dm_nuc=None, h1e=None, vhf_e=None,
               h1n=None, veff_n=None):
    '''Total energy of NEO-HF'''
    E_tot = 0
    # add the energy of electrons
    if dm_elec is None:
        dm_elec = mf_elec.make_rdm1()
    if dm_nuc is None:
        dm_nuc = []
        for i in range(len(mf_nuc)):
            dm_nuc.append(mf_nuc[i].make_rdm1())
    if h1e is None:
        h1e = mf_elec.get_hcore()
    if vhf_e is None:
        vhf_e = mf_elec.get_veff(mf_elec.mol, dm_elec)
    E_tot += mf_elec.energy_elec(dm=dm_elec, h1e=h1e, vhf=vhf_e)[0]
    # add the energy of quantum nuclei
    if h1n is None:
        h1n = []
        for i in range(len(mf_nuc)):
            h1n.append(mf_nuc[i].get_hcore())
    for i in range(len(mf_nuc)):
        E_tot += mf_nuc[i].energy_qmnuc(mf_nuc[i], h1n[i], dm_nuc[i], veff_n=veff_n[i])
    # substract double-counted terms and add classical nuclear repulsion
    super_mol = mf_elec.mol.super_mol
    ne_U = 0.0
    nn_U = 0.0
    for i in range(len(mf_nuc)):
        ne_U += h1n[i].ne_U
        nn_U += 0.5 * h1n[i].nn_U
    logger.debug(super_mol, 'Energy of e-n Coulomb interactions: %s', ne_U)
    logger.debug(super_mol, 'Energy of n-n Coulomb interactions: %s', nn_U)
    E_tot = E_tot - ne_U - nn_U + mf_elec.energy_nuc()
    return E_tot

def init_guess_mixed(mol, mixing_parameter = numpy.pi/4):
    ''' Copy from pyscf/examples/scf/56-h2_symm_breaking.py

    Generate density matrix with broken spatial and spin symmetry by mixing
    HOMO and LUMO orbitals following ansatz in Szabo and Ostlund, Sec 3.8.7.

    psi_1a = numpy.cos(q)*psi_homo + numpy.sin(q)*psi_lumo
    psi_1b = numpy.cos(q)*psi_homo - numpy.sin(q)*psi_lumo

    psi_2a = -numpy.sin(q)*psi_homo + numpy.cos(q)*psi_lumo
    psi_2b =  numpy.sin(q)*psi_homo + numpy.cos(q)*psi_lumo

    Returns:
        Density matrices, a list of 2D ndarrays for alpha and beta spins
    '''
    # opt: q, mixing parameter 0 < q < 2 pi

    # based on init_guess_by_1e
    h1e = scf.hf.get_hcore(mol)
    s1e = scf.hf.get_ovlp(mol)
    mo_energy, mo_coeff = scf.hf.eig(h1e, s1e)
    mf = scf.HF(mol)
    mo_occ = mf.get_occ(mo_energy=mo_energy, mo_coeff=mo_coeff)

    homo_idx=0
    lumo_idx=1

    for i in range(len(mo_occ)-1):
        if mo_occ[i]>0 and mo_occ[i+1]<0:
            homo_idx=i
            lumo_idx=i+1

    psi_homo=mo_coeff[:, homo_idx]
    psi_lumo=mo_coeff[:, lumo_idx]

    Ca=numpy.zeros_like(mo_coeff)
    Cb=numpy.zeros_like(mo_coeff)

    # mix homo and lumo of alpha and beta coefficients
    q=mixing_parameter

    for k in range(mo_coeff.shape[0]):
        if k == homo_idx:
            Ca[:,k] = numpy.cos(q)*psi_homo + numpy.sin(q)*psi_lumo
            Cb[:,k] = numpy.cos(q)*psi_homo - numpy.sin(q)*psi_lumo
            continue
        if k==lumo_idx:
            Ca[:,k] = -numpy.sin(q)*psi_homo + numpy.cos(q)*psi_lumo
            Cb[:,k] =  numpy.sin(q)*psi_homo + numpy.cos(q)*psi_lumo
            continue
        Ca[:,k]=mo_coeff[:,k]
        Cb[:,k]=mo_coeff[:,k]

    dm =scf.UHF(mol).make_rdm1( (Ca,Cb), (mo_occ,mo_occ) )
    return dm

def init_guess_elec(mol, unrestricted, init_guess):
    """Generate initial dm"""
    # get electronic initial guess, which uses default minao initial gues
    dm_elec = None
    if unrestricted:
        dm_elec = scf.UHF(mol).get_init_guess(key=init_guess)
        # alternatively, try the mixed initial guess
        #dm_elec = init_guess_mixed(mol)
    else:
        dm_elec = scf.RHF(mol).get_init_guess(key=init_guess)
    return dm_elec

def init_guess_nuc_by_calculation(mf_nuc, mol):
    """Generate initial dm"""
    dm_nuc = [None] * mol.nuc_num
    for i in range(mol.nuc_num):
        ia = mol.nuc[i].atom_index
        mol_tmp = neo.Mole()
        # do not invoke possibly expensive QMMM during init guess
        mol_tmp.build(quantum_nuc=[ia], nuc_basis=mol.nuc_basis, mm_mol=None,
                      dump_input=False, parse_arg=False, verbose=mol.verbose,
                      output=mol.output, max_memory=mol.max_memory,
                      atom=mol.atom, unit=mol.unit, nucmod=mol.nucmod,
                      ecp=mol.ecp, charge=mol.charge, spin=mol.spin,
                      symmetry=mol.symmetry, symmetry_subgroup=mol.symmetry_subgroup,
                      cart=mol.cart, magmom=mol.magmom)
        dm_nuc[i] = get_init_guess_nuc(mf_nuc[i], mol_tmp.nuc[0])
        mf_nuc[i].hcore_static = None # clear the true hcore cache
    return dm_nuc

def kernel(mf, conv_tol=1e-10, conv_tol_grad=None,
           dump_chk=False, dm0=None, callback=None, conv_check=True, **kwargs):
    cput0 = (logger.process_clock(), logger.perf_counter())
    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(conv_tol)
        logger.info(mf, 'Set gradient conv threshold to %g', conv_tol_grad)
    mol = mf.mol
    if dm0 is None:
        # note that mol is used instead of mol.elec, because mol.elec
        # will have zero charges for quantum nuclei, but we want a
        # classical HF initial guess here
        mf.dm_elec = mf.get_init_guess_elec(mol, mf.init_guess)
    elif isinstance(dm0, (tuple, list)):
        mf.dm_elec = dm0[0]
    elif isinstance(dm0, numpy.ndarray):
        mf.dm_elec = dm0
    # mf.init_guess only affects the electronic part
    if dm0 is None:
        mf.dm_nuc = mf.get_init_guess_nuc(mol)
    elif isinstance(dm0, (tuple, list)) and len(dm0) >= mol.nuc_num + 1:
        mf.dm_nuc = dm0[1 : 1+mol.nuc_num]
    else:
        mf.dm_nuc = mf.get_init_guess_nuc(mol)

    if mf.dm_elec.ndim > 2:
        dm = [mf.dm_elec[0], mf.dm_elec[1]]
    else:
        dm = [mf.dm_elec]
    dm += mf.dm_nuc
    h1e = [mf.mf_elec.get_hcore()]
    vhf = [mf.mf_elec.get_veff(mol.elec, mf.dm_elec)]
    for i in range(mol.nuc_num):
        h1e.append(mf.mf_nuc[i].get_hcore())
        vhf.append(mf.mf_nuc[i].get_veff(mol.nuc[i], mf.dm_nuc[i]))
    e_tot = mf.energy_tot(mf.dm_elec, mf.dm_nuc, h1e[0], vhf[0], h1e[1:], vhf[1:])
    logger.info(mf, 'init E= %.15g', e_tot)

    scf_conv = False
    mo_energy_e = mo_coeff_e = mo_occ_e = None
    mo_energy_n = [None] * mol.nuc_num
    mo_coeff_n = [None] * mol.nuc_num
    mo_occ_n = [None] * mol.nuc_num

    s1e_e = mf.mf_elec.get_ovlp()
    if mf.dm_elec.ndim > 2:
        s1e = [s1e_e, s1e_e]
    else:
        s1e = [s1e_e]
    cond = lib.cond(s1e[0])
    logger.debug(mf, 'cond(S) = %s', cond)
    if numpy.max(cond) * 1e-17 > conv_tol:
        logger.warn(mf, 'Singularity detected in overlap matrix (condition number = %4.3g). '
                    'SCF may be inaccurate and hard to converge.', numpy.max(cond))
    for i in range(mol.nuc_num):
        s1e.append(mf.mf_nuc[i].get_ovlp())

    # Skip SCF iterations. Compute only the total energy of the initial density
    if mf.max_cycle <= 0:
        fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
        if mf.dm_elec.ndim > 2:
            fock_e = numpy.asarray((fock[0], fock[1]))
            start = 2
        else:
            fock_e = fock[0]
            start = 1
        mo_energy_e, mo_coeff_e = mf.mf_elec.eig(fock_e, s1e[0])
        mf.mf_elec.mo_energy, mf.mf_elec.mo_coeff = mo_energy_e, mo_coeff_e
        mo_occ_e = mf.mf_elec.get_occ(mo_energy_e, mo_coeff_e)
        mf.mf_elec.mo_occ = mo_occ_e
        for i in range(mol.nuc_num):
            fock_n = fock[start + i]
            mo_energy_n[i], mo_coeff_n[i] = mf.mf_nuc[i].eig(fock_n, s1e[start + i])
            mf.mf_nuc[i].mo_energy, mf.mf_nuc[i].mo_coeff = mo_energy_n[i], mo_coeff_n[i]
            mo_occ_n[i] = mf.mf_nuc[i].get_occ(mo_energy_n[i], mo_coeff_n[i])
            mf.mf_nuc[i].mo_occ = mo_occ_n[i]
        if mf.dm_elec.ndim > 2:
            mf.dm_elec = mf.dm_elec[0] + mf.dm_elec[1]
        return scf_conv, e_tot, mo_energy_e, mo_coeff_e, mo_occ_e, \
               mo_energy_n, mo_coeff_n, mo_occ_n

    if isinstance(mf.diis, lib.diis.DIIS):
        mf_diis = mf.diis
    elif mf.diis:
        assert issubclass(mf.DIIS, lib.diis.DIIS)
        mf_diis = mf.DIIS(mf, mf.diis_file)
        mf_diis.space = mf.diis_space
        mf_diis.rollback = mf.diis_space_rollback
    else:
        mf_diis = None

    if dump_chk and mf.chkfile:
        # dump electronic part mol
        chkfile.save_mol(mol.elec, mf.chkfile)

    # A preprocessing hook before the SCF iteration
    mf.pre_kernel(locals())

    if isinstance(mf, neo.CDFT):
        # mf_diis needs to be the raw lib.diis.DIIS() for CNEO
        mf_diis = lib.diis.DIIS()
        mf_diis.space = 8

    cput1 = logger.timer(mf, 'initialize scf', *cput0)
    for cycle in range(mf.max_cycle):
        dm_elec_last = numpy.copy(mf.dm_elec) # why didn't pyscf.scf.hf use copy?
        dm_nuc_last = copy.deepcopy(mf.dm_nuc)
        last_e = e_tot

        fock = mf.get_fock(h1e, s1e, vhf, dm, cycle, mf_diis)
        if mf.dm_elec.ndim > 2:
            fock_e = numpy.asarray((fock[0], fock[1]))
            start = 2
        else:
            fock_e = fock[0]
            start = 1
        mo_energy_e, mo_coeff_e = mf.mf_elec.eig(fock_e, s1e[0])
        mf.mf_elec.mo_energy, mf.mf_elec.mo_coeff = mo_energy_e, mo_coeff_e
        mo_occ_e = mf.mf_elec.get_occ(mo_energy_e, mo_coeff_e)
        mf.mf_elec.mo_occ = mo_occ_e
        mf.dm_elec = mf.mf_elec.make_rdm1(mo_coeff_e, mo_occ_e)
        # attach mo_coeff and mo_occ to dm to improve DFT get_veff efficiency
        mf.dm_elec = lib.tag_array(mf.dm_elec, mo_coeff=mo_coeff_e, mo_occ=mo_occ_e)

        for i in range(mol.nuc_num):
            fock_n = fock[start + i]
            mo_energy_n[i], mo_coeff_n[i] = mf.mf_nuc[i].eig(fock_n, s1e[start + i])
            mf.mf_nuc[i].mo_energy, mf.mf_nuc[i].mo_coeff = mo_energy_n[i], mo_coeff_n[i]
            mo_occ_n[i] = mf.mf_nuc[i].get_occ(mo_energy_n[i], mo_coeff_n[i])
            mf.mf_nuc[i].mo_occ = mo_occ_n[i]
            mf.dm_nuc[i] = mf.mf_nuc[i].make_rdm1(mo_coeff_n[i], mo_occ_n[i])

        if mf.dm_elec.ndim > 2:
            dm = [mf.dm_elec[0], mf.dm_elec[1]]
        else:
            dm = [mf.dm_elec]
        dm += mf.dm_nuc

        # update the so-called "core" Hamiltonian and veff after the density is updated
        h1e = [mf.mf_elec.get_hcore()]
        vhf_last = vhf
        vhf = [mf.mf_elec.get_veff(mol.elec, mf.dm_elec, dm_elec_last, vhf_last[0])]
        for i in range(mol.nuc_num):
            h1e.append(mf.mf_nuc[i].get_hcore())
            vhf.append(mf.mf_nuc[i].get_veff(mol.nuc[i], mf.dm_nuc[i], dm_nuc_last[i], vhf_last[1 + i]))
        vhf_last = None

        # Here Fock matrix is h1e + vhf, without DIIS.  Calling get_fock
        # instead of the statement "fock = h1e + vhf" because Fock matrix may
        # be modified in some methods.
        fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
        if mf.dm_elec.ndim > 2:
            fock_e = numpy.asarray((fock[0], fock[1]))
            start = 2
        else:
            fock_e = fock[0]
            start = 1
        norm_gorb_e = numpy.linalg.norm(mf.mf_elec.get_grad(mo_coeff_e, mo_occ_e, fock_e))
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb_e = norm_gorb_e / numpy.sqrt(norm_gorb_e.size)
        norm_ddm_e = numpy.linalg.norm(mf.dm_elec - dm_elec_last)

        grad_n = []
        for i in range(mol.nuc_num):
            fock_n = fock[start + i]
            grad_n.append(mf.mf_nuc[i].get_grad(mo_coeff_n[i], mo_occ_n[i], fock_n))
        norm_gorb_n = numpy.linalg.norm(numpy.concatenate(grad_n, axis=None))
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb_n = norm_gorb_n / numpy.sqrt(norm_gorb_n.size)
        norm_ddm_n = numpy.linalg.norm(numpy.concatenate(mf.dm_nuc, axis=None)
                                       - numpy.concatenate(dm_nuc_last, axis=None))

        e_tot = mf.energy_tot(mf.dm_elec, mf.dm_nuc, h1e[0], vhf[0], h1e[1:], vhf[1:])
        logger.info(mf, 'cycle= %d E= %.15g  delta_E= %4.3g  |g_e|= %4.3g  |ddm_e|= %4.3g  |g_n|= %4.3g  |ddm_n|= %4.3g',
                    cycle + 1, e_tot, e_tot - last_e, norm_gorb_e, norm_ddm_e, norm_gorb_n, norm_ddm_n)

        if abs(e_tot - last_e) < conv_tol and norm_gorb_e < conv_tol_grad and norm_gorb_n < conv_tol_grad:
            scf_conv = True

        if dump_chk:
            mf.dump_chk(locals())

        if callable(callback):
            callback(locals())

        cput1 = logger.timer(mf, 'cycle= %d' % (cycle + 1), *cput1)

        if scf_conv:
            break

    if scf_conv and conv_check:
        # An extra diagonalization, to remove level shift
        #fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf
        mo_energy_e, mo_coeff_e = mf.mf_elec.eig(fock_e, s1e[0])
        mf.mf_elec.mo_energy, mf.mf_elec.mo_coeff = mo_energy_e, mo_coeff_e
        mo_occ_e = mf.mf_elec.get_occ(mo_energy_e, mo_coeff_e)
        mf.mf_elec.mo_occ = mo_occ_e
        mf.dm_elec, dm_elec_last = mf.mf_elec.make_rdm1(mo_coeff_e, mo_occ_e), mf.dm_elec
        mf.dm_elec = lib.tag_array(mf.dm_elec, mo_coeff=mo_coeff_e, mo_occ=mo_occ_e)

        for i in range(mol.nuc_num):
            fock_n = fock[start + i]
            mo_energy_n[i], mo_coeff_n[i] = mf.mf_nuc[i].eig(fock_n, s1e[start + i])
            mf.mf_nuc[i].mo_energy, mf.mf_nuc[i].mo_coeff = mo_energy_n[i], mo_coeff_n[i]
            mo_occ_n[i] = mf.mf_nuc[i].get_occ(mo_energy_n[i], mo_coeff_n[i])
            mf.mf_nuc[i].mo_occ = mo_occ_n[i]
            mf.dm_nuc[i], dm_nuc_last[i] = mf.mf_nuc[i].make_rdm1(mo_coeff_n[i], mo_occ_n[i]), mf.dm_nuc[i]

        if mf.dm_elec.ndim > 2:
            dm = [mf.dm_elec[0], mf.dm_elec[1]]
        else:
            dm = [mf.dm_elec]
        dm += mf.dm_nuc

        # update the so-called "core" Hamiltonian and veff after the density is updated
        h1e = [mf.mf_elec.get_hcore()]
        vhf_last = vhf
        vhf = [mf.mf_elec.get_veff(mol.elec, mf.dm_elec, dm_elec_last, vhf_last[0])]
        for i in range(mol.nuc_num):
            h1e.append(mf.mf_nuc[i].get_hcore())
            vhf.append(mf.mf_nuc[i].get_veff(mol.nuc[i], mf.dm_nuc[i], dm_nuc_last[i], vhf_last[1 + i]))
        vhf_last = None

        e_tot, last_e = mf.energy_tot(mf.dm_elec, mf.dm_nuc, h1e[0], vhf[0], h1e[1:], vhf[1:]), e_tot

        fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf, no DIIS
        if mf.dm_elec.ndim > 2:
            fock_e = numpy.asarray((fock[0], fock[1]))
            start = 2
        else:
            fock_e = fock[0]
            start = 1
        norm_gorb_e = numpy.linalg.norm(mf.mf_elec.get_grad(mo_coeff_e, mo_occ_e, fock_e))
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb_e = norm_gorb_e / numpy.sqrt(norm_gorb_e.size)
        norm_ddm_e = numpy.linalg.norm(mf.dm_elec - dm_elec_last)

        grad_n = []
        for i in range(mol.nuc_num):
            fock_n = fock[start + i]
            grad_n.append(mf.mf_nuc[i].get_grad(mo_coeff_n[i], mo_occ_n[i], fock_n))
        norm_gorb_n = numpy.linalg.norm(numpy.concatenate(grad_n, axis=None))
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb_n = norm_gorb_n / numpy.sqrt(norm_gorb_n.size)
        norm_ddm_n = numpy.linalg.norm(numpy.concatenate(mf.dm_nuc, axis=None)
                                       - numpy.concatenate(dm_nuc_last, axis=None))

        conv_tol = conv_tol * 10
        conv_tol_grad = conv_tol_grad * 3
        if abs(e_tot - last_e) < conv_tol or (norm_gorb_e < conv_tol_grad and norm_gorb_n < conv_tol_grad):
            scf_conv = True
        logger.info(mf, 'Extra cycle  E= %.15g  delta_E= %4.3g  |g_e|= %4.3g  |ddm_e|= %4.3g  |g_n|=%4.3g  |ddm_n|= %4.3g',
                    e_tot, e_tot - last_e, norm_gorb_e, norm_ddm_e, norm_gorb_n, norm_ddm_n)
        if dump_chk:
            mf.dump_chk(locals())

    logger.timer(mf, 'scf_cycle', *cput0)
    # A post-processing hook before return
    mf.post_kernel(locals())

    if mf.dm_elec.ndim > 2:
        mf.dm_elec = mf.dm_elec[0] + mf.dm_elec[1]
    return scf_conv, e_tot, mo_energy_e, mo_coeff_e, mo_occ_e, \
           mo_energy_n, mo_coeff_n, mo_occ_n

def as_scanner(mf):
    '''Generating a scanner/solver for (C)NEO PES.
    Copied from scf.hf.as_scanner
    '''
    if isinstance(mf, lib.SinglePointScanner):
        return mf

    logger.info(mf, 'Create scanner for %s', mf.__class__)

    class CNEO_Scanner(mf.__class__, lib.SinglePointScanner):
        def __init__(self, mf_obj):
            self.__dict__.update(mf_obj.__dict__)

        def __call__(self, mol_or_geom, **kwargs):
            if isinstance(mol_or_geom, neo.Mole):
                mol = mol_or_geom
            else:
                mol = self.mol.set_geom_(mol_or_geom, inplace=False)

            # Cleanup intermediates associated to the pervious mol object
            self.reset(mol)

            # Only electronic dm0 is of interest
            if 'dm0' in kwargs:
                dm0 = kwargs.pop('dm0')
            elif self.mf_elec.mo_coeff is None:
                dm0 = None
            elif self.chkfile and h5py.is_hdf5(self.chkfile):
                dm0 = self.mf_elec.from_chk(self.chkfile)
            else:
                dm0 = self.mf_elec.make_rdm1()
                if dm0.shape[-1] != mol.elec.nao:
                    dm0 = None
            self.mf_elec.mo_coeff = None
            if dm0 is not None:
                dm0n = []
                for i in range(mol.nuc_num):
                    dm0n.append(self.mf_nuc[i].make_rdm1())
                    if dm0n[-1].shape[-1] != mol.nuc[i].nao:
                        dm0n = None
                        break
                if dm0n is not None and len(dm0n) == mol.nuc_num:
                    dm0 = [dm0] + dm0n
            for i in range(mol.nuc_num):
                self.mf_nuc[i].mo_coeff = None
            e_tot = self.kernel(dm0=dm0, **kwargs)
            return e_tot

    return CNEO_Scanner(mf)


class HF(scf.hf.SCF):
    '''Hartree Fock for NEO

    Examples::

    >>> from pyscf import neo
    >>> mol = neo.Mole()
    >>> mol.build(atom='H 0 0 0; F 0 0 0.917', quantum_nuc=[0], basis='ccpvdz',
    >>>           nuc_basis='pb4d')
    >>> mf = neo.HF(mol)
    >>> mf.scf()
    -99.98104139461894
    '''

    def __init__(self, mol, unrestricted=False):
        scf.hf.SCF.__init__(self, mol)
        if mol.elec.nhomo is not None or mol.spin != 0:
            unrestricted = True
        self.unrestricted = unrestricted
        self.mf_elec = None
        # dm_elec will be the total density after SCF, but can be spin
        # densities during the SCF procedure
        self.dm_elec = None
        self.mf_nuc = []
        self.dm_nuc = []
        # The verbosity flag is passed now instead of when creating those mol
        # objects because users need time to change mol's verbose level after
        # the mol object is created.
        self.mol.elec.verbose = self.mol.verbose
        for i in range(mol.nuc_num):
            self.mol.nuc[i].verbose = self.mol.verbose
        # placeholder for ERIs
        self._eri_ne = []
        self._eri_nn = []
        for i in range(mol.nuc_num):
            self._eri_ne.append(None)
            self._eri_nn.append([None] * mol.nuc_num)

        # initialize sub mf objects for electrons and quantum nuclei
        # electronic part:
        if unrestricted:
            self.mf_elec = scf.UHF(mol.elec)
        else:
            self.mf_elec = scf.RHF(mol.elec)
        if self.mol.mm_mol is not None:
            self.mf_elec = qmmm_for_scf(self.mf_elec, self.mol.mm_mol)
        self.mf_elec.get_hcore = self.get_hcore_elec
        self.mf_elec.hcore_static = None # cache true hcore
        if mol.elec.nhomo is not None:
            self.mf_elec.get_occ = self.get_occ_elec(self.mf_elec)

        # nuclear part
        for i in range(mol.nuc_num):
            self.mf_nuc.append(scf.RHF(mol.nuc[i]))
            mf_nuc = self.mf_nuc[-1]
            mf_nuc.occ_state = 0 # for Delta-SCF
            mf_nuc.get_occ = self.get_occ_nuc(mf_nuc)
            mf_nuc.get_hcore = self.get_hcore_nuc(mf_nuc)
            mf_nuc.hcore_static = None # cache true hcore
            mf_nuc.get_veff = self.get_veff_nuc_bare
            mf_nuc.energy_qmnuc = self.energy_qmnuc
            self.dm_nuc.append(None)

    def dump_chk(self, envs):
        if self.chkfile:
            chkfile.dump_scf(self.mol.elec, self.chkfile,
                             envs['e_tot'], envs['mo_energy_e'],
                             envs['mo_coeff_e'], envs['mo_occ_e'],
                             overwrite_mol=False)
        return self

    def get_init_guess_elec(self, mol, init_guess):
        return init_guess_elec(mol, self.unrestricted, init_guess)

    def get_init_guess_nuc(self, mol):
        return init_guess_nuc_by_calculation(self.mf_nuc, mol)

    def get_j_e_dm_n(self, idx_nuc, dm_n, mol_elec=None, mol_nuc=None, eri_ne=None):
        if mol_elec is None:
            mol_elec = self.mol.elec
        if mol_nuc is None:
            mol_nuc = self.mol.nuc[idx_nuc]
        if eri_ne is None:
            eri_ne = self._eri_ne
        return get_j_e_dm_n(idx_nuc, dm_n, mol_elec=mol_elec,
                            mol_nuc=mol_nuc, eri_ne=eri_ne)

    def get_j_n_dm_e(self, idx_nuc, dm_e, mol_elec=None, mol_nuc=None, eri_ne=None):
        if mol_elec is None:
            mol_elec = self.mol.elec
        if mol_nuc is None:
            mol_nuc = self.mol.nuc[idx_nuc]
        if eri_ne is None:
            eri_ne = self._eri_ne
        return get_j_n_dm_e(idx_nuc, dm_e, mol_elec=mol_elec,
                            mol_nuc=mol_nuc, eri_ne=eri_ne)

    def get_j_nn(self, idx1, idx2, dm_n2, mol_nuc1=None, mol_nuc2=None, eri_nn=None):
        if mol_nuc1 is None:
            mol_nuc1 = self.mol.nuc[idx1]
        if mol_nuc2 is None:
            mol_nuc2 = self.mol.nuc[idx2]
        if eri_nn is None:
            eri_nn = self._eri_nn
        return get_j_nn(idx1, idx2, dm_n2, mol_nuc1=mol_nuc1,
                        mol_nuc2=mol_nuc2, eri_nn=eri_nn)

    def get_hcore_elec(self, mol=None):
        if mol is None: mol = self.mol.elec
        return get_hcore_elec(mol, self.mf_elec, self.dm_nuc,
                              self.mol.nuc, eri_ne=self._eri_ne)

    def get_occ_nuc(self, mf):
        return get_occ_nuc(mf)

    def get_occ_elec(self, mf):
        return get_occ_elec(mf)

    def get_hcore_nuc(self, mf):
        def get_hcore(mol=None):
            if mol is None: mol = mf.mol
            return get_hcore_nuc(mol, mf, self.dm_elec, self.dm_nuc,
                                 self.mol.elec, self.mol.nuc,
                                 eri_ne=self._eri_ne, eri_nn=self._eri_nn)
        return get_hcore

    def get_veff_nuc_bare(self, mol, dm, dm_last=0, vhf_last=0, hermi=1):
        return get_veff_nuc_bare(mol)

    get_fock = get_fock

    def energy_qmnuc(self, mf, h1n, dm_nuc, veff_n=None):
        return energy_qmnuc(mf, h1n, dm_nuc, veff_n=veff_n)

    def energy_tot(self, dm_elec, dm_nuc, h1e, vhf_e, h1n, veff_n=None):
        return energy_tot(self.mf_elec, self.mf_nuc, dm_elec, dm_nuc, h1e,
                          vhf_e, h1n, veff_n=veff_n)

    def scf(self, dm0=None, **kwargs):
        cput0 = (logger.process_clock(), logger.perf_counter())

        self.dump_flags()
        # QMMM dump_flags:
        if self.mol.mm_mol is not None:
            logger.info(self, '** Add background charges for %s **',
                        self.__class__)
            if self.verbose >= logger.DEBUG:
                logger.debug(self, 'Charge      Location')
                coords = self.mol.mm_mol.atom_coords()
                charges = self.mol.mm_mol.atom_charges()
                for i, z in enumerate(charges):
                    logger.debug(self, '%.9g    %s', z, coords[i])

        if self.max_cycle > 0 or self.mo_coeff is None:
            # Note that all mo_*_[e,n] have already been passed to
            # mf_elec and mf_nuc inside kernel. Here pass them again
            # just to make it look like scf function in scf.hf
            self.converged, self.e_tot, self.mf_elec.mo_energy, \
                self.mf_elec.mo_coeff, self.mf_elec.mo_occ = \
                    kernel(self, self.conv_tol, self.conv_tol_grad,
                           dm0=dm0, callback=self.callback,
                           conv_check=self.conv_check, **kwargs)[0:5]
            self.mf_elec.converged = self.converged
        else:
            self.e_tot = kernel(self, self.conv_tol, self.conv_tol_grad,
                                dm0=dm0, callback=self.callback,
                                conv_check=self.conv_check, **kwargs)[1]

        logger.timer(self, '(C)NEO-SCF', *cput0)
        self._finalize()
        return self.e_tot
    kernel = lib.alias(scf, alias_name='kernel')

    def reset(self, mol=None):
        '''Reset mol and relevant attributes associated to the old mol object'''
        super().reset(mol=mol)
        self.mf_elec.reset(self.mol.elec)
        self.dm_elec = None
        for i in range(self.mol.nuc_num):
            self.mf_nuc[i].reset(self.mol.nuc[i])
            self.dm_nuc[i] = None
            self._eri_ne[i] = None
            self._eri_nn[i] = [None] * self.mol.nuc_num

        # point to correct ``self'' for overriden functions
        self.mf_elec.get_hcore = self.get_hcore_elec
        self.mf_elec.hcore_static = None
        if mol.elec.nhomo is not None:
            self.mf_elec.get_occ = self.get_occ_elec(self.mf_elec)
        for i in range(mol.nuc_num):
            mf_nuc = self.mf_nuc[i]
            mf_nuc.get_occ = self.get_occ_nuc(mf_nuc)
            mf_nuc.get_hcore = self.get_hcore_nuc(mf_nuc)
            mf_nuc.hcore_static = None
            mf_nuc.get_veff = self.get_veff_nuc_bare
            mf_nuc.energy_qmnuc = self.energy_qmnuc
        return self

    as_scanner = as_scanner
