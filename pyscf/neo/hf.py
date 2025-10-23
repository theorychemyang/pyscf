#!/usr/bin/env python

'''
Nuclear Electronic Orbital Hartree-Fock (NEO-HF)
'''

import copy
import ctypes
import h5py
import numpy
import warnings
from pyscf import df, gto, lib, neo, scf
from pyscf.data import nist
from pyscf.lib import logger
from pyscf.scf import _vhf, chkfile
from pyscf.scf.hf import TIGHT_GRAD_CONV_TOL
from pyscf.lib.exceptions import PointGroupSymmetryError
from pyscf import __config__

WITH_META_LOWDIN = getattr(__config__, 'scf_analyze_with_meta_lowdin', True)
PRE_ORTH_METHOD = getattr(__config__, 'scf_analyze_pre_orth_method', 'ANO')


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
    if len(dms_shape) == 2:
        vj = vj.reshape((nao_v,nao_v))
    else:
        vj = vj.reshape((n_dm,nao_v,nao_v)) # should preserve the shape even when n_dm=1
    return vj

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

    homo_idx = 0
    lumo_idx = 1

    for i in range(len(mo_occ)-1):
        if mo_occ[i] > 0 and mo_occ[i+1] < 0:
            homo_idx = i
            lumo_idx = i+1

    psi_homo = mo_coeff[:, homo_idx]
    psi_lumo = mo_coeff[:, lumo_idx]

    Ca = numpy.zeros_like(mo_coeff)
    Cb = numpy.zeros_like(mo_coeff)

    # mix homo and lumo of alpha and beta coefficients
    q = mixing_parameter

    for k in range(mo_coeff.shape[0]):
        if k == homo_idx:
            Ca[:,k] = numpy.cos(q)*psi_homo + numpy.sin(q)*psi_lumo
            Cb[:,k] = numpy.cos(q)*psi_homo - numpy.sin(q)*psi_lumo
            continue
        if k == lumo_idx:
            Ca[:,k] = -numpy.sin(q)*psi_homo + numpy.cos(q)*psi_lumo
            Cb[:,k] =  numpy.sin(q)*psi_homo + numpy.cos(q)*psi_lumo
            continue
        Ca[:,k] = mo_coeff[:,k]
        Cb[:,k] = mo_coeff[:,k]

    dm = scf.UHF(mol).make_rdm1( (Ca,Cb), (mo_occ,mo_occ) )
    return dm

def as_scanner(mf):
    '''Generating a scanner/solver for (C)NEO PES.
    Copied from scf.hf.as_scanner
    '''
    if isinstance(mf, lib.SinglePointScanner):
        return mf

    logger.info(mf, 'Create scanner for %s', mf.__class__)
    name = mf.__class__.__name__ + CNEO_Scanner.__name_mixin__
    return lib.set_class(CNEO_Scanner(mf), (CNEO_Scanner, mf.__class__), name)

class CNEO_Scanner(lib.SinglePointScanner):
    def __init__(self, mf_obj):
        self.__dict__.update(mf_obj.__dict__)
        self._last_mol_fp = mf_obj.mol.ao_loc

    def __call__(self, mol_or_geom, **kwargs):
        if isinstance(mol_or_geom, neo.Mole):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)

        # Cleanup intermediates associated to the pervious mol object
        self.reset(mol)

        if 'dm0' in kwargs:
            dm0 = kwargs.pop('dm0') # this can be electronic only or full
        elif self.mo_coeff is None:
            dm0 = None
        else:
            dm0 = None
            # dm0 form last calculation cannot be used in the current
            # calculation if a completely different system is given.
            # Obviously, the systems are very different if the number of
            # basis functions are different.
            # TODO: A robust check should include more comparison on
            # various attributes between current `mol` and the `mol` in
            # last calculation.
            if numpy.array_equal(self._last_mol_fp, mol.ao_loc):
                dm0 = self.make_rdm1()
            # TODO: chkfile support
            # Currently mo_coeff dumped is a dict, can't go through from_chk
            #elif self.chkfile and h5py.is_hdf5(self.chkfile):
            #    dm0 = self.from_chk(self.chkfile)
        for _, comp in self.components.items():
            comp.mo_coeff = None
        e_tot = self.kernel(dm0=dm0, **kwargs)
        self._last_mol_fp = mol.ao_loc
        return e_tot

def hcore_qmmm(mol, mm_mol):
    '''From qmmm/itrf'''
    coords = mm_mol.atom_coords()
    charges = mm_mol.atom_charges()
    nao = mol.nao
    max_memory = mol.super_mol.max_memory - lib.current_memory()[0]
    blksize = int(min(max_memory*1e6/8/nao**2, 200))
    blksize = max(blksize, 1)
    v = 0
    if mm_mol.charge_model == 'gaussian':
        expnts = mm_mol.get_zetas()

        if mol.cart:
            intor = 'int3c2e_cart'
        else:
            intor = 'int3c2e_sph'
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas,
                                             mol._env, intor)
        for i0, i1 in lib.prange(0, charges.size, blksize):
            fakemol = gto.fakemol_for_charges(coords[i0:i1], expnts[i0:i1])
            j3c = df.incore.aux_e2(mol, fakemol, intor=intor,
                                   aosym='s2ij', cintopt=cintopt)
            v += numpy.einsum('xk,k->x', j3c, -charges[i0:i1])
        v = lib.unpack_tril(v)
    else: # point-charge model
        for i0, i1 in lib.prange(0, charges.size, blksize):
            j3c = mol.intor('int1e_grids', hermi=1, grids=coords[i0:i1])
            v += numpy.einsum('kpq,k->pq', j3c, -charges[i0:i1])
    return v

def general_scf(method, charge=1, mass=1, is_nucleus=False, nuc_occ_state=0):
    '''Modify SCF (HF and DFT) method to support for general charge
    and general mass, such that positrons and nuclei can be calculated.

    Args:
        charge : float
            Charge of the particle. 1 means electron, -1 means positron.
        mass : float
            Mass of the particle in a.u. Nuclei will have high mass
        is_nucleus : bool
            If the particle is nucleus. Nucleus won't see PP and is
            considered a distinguishable single particle
        nuc_occ_state : int
            Select the nuclear orbital that is occupied. For Delta-SCF.
    '''
    assert (isinstance(method, scf.hf.SCF))
    if isinstance(method, Component):
        method.charge = charge
        method.mass = mass
        method.is_nucleus = is_nucleus
        method.nuc_occ_state = nuc_occ_state
        return method
    return lib.set_class(ComponentSCF(method, charge, mass, is_nucleus, nuc_occ_state),
                         (ComponentSCF, method.__class__))

class Component:
    __name_mixin__ = 'Component'

class ComponentSCF(Component):
    _keys = {'charge', 'mass', 'is_nucleus', 'nuc_occ_state'}
    def __init__(self, method, charge=1, mass=1, is_nucleus=False, nuc_occ_state=0):
        self.__dict__.update(method.__dict__)
        self.charge = charge
        self.mass = mass
        self.is_nucleus = is_nucleus
        self.nuc_occ_state = nuc_occ_state
        self._vint = None

    def undo_component(self):
        obj = lib.view(self, lib.drop_class(self.__class__, Component))
        return obj

    def get_hcore(self, mol=None):
        '''Core Hamiltonian'''
        if mol is None: mol = self.mol
        h = mol.intor_symmetric('int1e_kin') / self.mass

        if mol._pseudo and not self.is_nucleus:
            # Although mol._pseudo for GTH PP is only available in Cell, GTH PP
            # may exist if mol is converted from cell object.
            from pyscf.gto import pp_int
            h += pp_int.get_gth_pp(mol) * self.charge
        else:
            h += mol.intor_symmetric('int1e_nuc') * self.charge

        if len(mol._ecpbas) > 0 and not self.is_nucleus:
            h += mol.intor_symmetric('ECPscalar') * self.charge

        if hasattr(mol, 'super_mol'):
            if mol.super_mol.mm_mol is not None:
                h += hcore_qmmm(mol, mol.super_mol.mm_mol) * self.charge
        elif hasattr(mol, 'mm_mol'): # elec guess directly uses super_mol
            if mol.mm_mol is not None:
                h += hcore_qmmm(mol, mol.mm_mol) * self.charge
        return h

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        if self.is_nucleus: # Nucleus does not have self-type interaction
            if mol is None:
                mol = self.mol
            veff = numpy.zeros((mol.nao, mol.nao))
            if isinstance(self, scf.hf.KohnShamDFT):
                veff = lib.tag_array(veff, ecoul=0, exc=0, vj=veff.copy(), vk=veff.copy())
        else:
            if abs(self.charge) != 1.:
                raise NotImplementedError('General charge J/K with tag_array')
            if hasattr(vhf_last, 'vhf'):
                veff = super().get_veff(mol, dm, dm_last, vhf_last.vhf, hermi)
            else:
                veff = super().get_veff(mol, dm, dm_last, vhf_last, hermi)

        # Cached Coulomb/epc interaction from other components
        if self._vint is not None:
            vhf = veff.view(numpy.ndarray).copy()
            if hasattr(veff, '__dict__'):
                vhf = lib.tag_array(vhf, **veff.__dict__)

            if hasattr(veff, 'vj'): # KS
                exc = veff.exc
                if hasattr(self._vint, 'vj'): # epc is ON
                    vj = self._vint.vj
                    assert hasattr(self._vint, 'exc') and hasattr(veff, 'exc')
                    exc += self._vint.exc # pass epc correlation energy to exc
                else: # epc is OFF
                    vj = self._vint

                # update ecoul
                if hasattr(veff, 'ecoul') and veff.ecoul is not None:
                    ecoul = veff.ecoul
                    if isinstance(self, scf.uhf.UHF):
                        if not isinstance(dm, numpy.ndarray):
                            dm = numpy.asarray(dm)
                        if dm.ndim == 2:  # RHF DM
                            dm = numpy.repeat(dm[None]*.5, 2, axis=0)
                        ground_state = (dm.ndim == 3 and dm.shape[0] == 2)
                        if ground_state:
                            ecoul += numpy.einsum('ij,ji', dm[0]+dm[1], vj).real * .5
                    else:
                        ground_state = (isinstance(dm, numpy.ndarray) and dm.ndim == 2)
                        if ground_state:
                            ecoul += numpy.einsum('ij,ji', dm, vj).real * .5
                # Update overall veff, note that veff is a tag_array!
                # The update of vj is a bit useless, because ecoul has already been evaluated.
                veff = lib.tag_array(veff + self._vint, ecoul=ecoul, exc=exc,
                                     vj=veff.vj+vj, vk=veff.vk, vhf=vhf)
            else: # HF
                assert not hasattr(self._vint, 'vj') # Must be KS when epc is enabled
                veff = lib.tag_array(veff + self._vint, vhf=vhf)
        return veff

    def get_occ(self, mo_energy=None, mo_coeff=None):
        '''Support fractional occupation. For nucleus, make sure it is a single particle'''
        if mo_energy is None: mo_energy = self.mo_energy
        if self.is_nucleus:
            mo_occ = numpy.zeros_like(mo_energy)
            if self.mol.symmetry:
                if self.nuc_occ_state != 0:
                    raise NotImplementedError('With symmetry the nucleus must occupy ground state.')
                # MO's are not sorted, but grouped according to symmetry
                # copied from scf.hf_symm
                mol = self.mol
                orbsym = self.get_orbsym(mo_coeff)
                rest_idx = numpy.ones(mo_occ.size, dtype=bool)
                nelec_fix = 0
                for i, ir in enumerate(mol.irrep_id):
                    irname = mol.irrep_name[i]
                    if irname in self.irrep_nelec:
                        ir_idx = numpy.where(orbsym == ir)[0]
                        n = self.irrep_nelec[irname]
                        occ_sort = numpy.argsort(mo_energy[ir_idx].round(9), kind='stable')
                        occ_idx  = ir_idx[occ_sort[:n//2]]
                        mo_occ[occ_idx] = self.mol.nnuc
                        nelec_fix += n
                        rest_idx[ir_idx] = False
                nelec_float = mol.nelectron - nelec_fix
                assert (nelec_float >= 0)
                if nelec_float > 0:
                    rest_idx = numpy.where(rest_idx)[0]
                    occ_sort = numpy.argsort(mo_energy[rest_idx].round(9), kind='stable')
                    occ_idx  = rest_idx[occ_sort[:nelec_float//2]]
                    mo_occ[occ_idx] = self.mol.nnuc

                assert numpy.sum(mo_occ > 0) == 1 # ensure singly occupied

                vir_idx = (mo_occ==0)
                if self.verbose >= logger.INFO and numpy.count_nonzero(vir_idx) > 0:
                    ehomo = max(mo_energy[~vir_idx])
                    elumo = min(mo_energy[ vir_idx])
                    noccs = []
                    for i, ir in enumerate(mol.irrep_id):
                        irname = mol.irrep_name[i]
                        ir_idx = (orbsym == ir)

                        noccs.append(int(mo_occ[ir_idx].sum()))
                        if ehomo in mo_energy[ir_idx]:
                            irhomo = irname
                        if elumo in mo_energy[ir_idx]:
                            irlumo = irname
                    logger.info(self, 'CNEO NUC HOMO (%s) = %.15g  LUMO (%s) = %.15g',
                                irhomo, ehomo, irlumo, elumo)

                    logger.debug(self, 'CNEO NUC irrep_nelec = %s', noccs)
                    scf.hf_symm._dump_mo_energy(mol, mo_energy, mo_occ, ehomo, elumo, orbsym,
                                                title='CNEO NUC ', verbose=self.verbose)
            else:
                e_idx = numpy.argsort(mo_energy)
                e_sort = mo_energy[e_idx]
                nmo = mo_energy.size
                nocc = 1 # singly occupied nuclear orbital
                mo_occ[e_idx[self.nuc_occ_state]] = self.mol.nnuc # 1 or fractional
                if self.verbose >= logger.INFO and nocc < nmo:
                    if e_sort[nocc-1]+1e-3 > e_sort[nocc]:
                        logger.warn(self, 'CNEO NUC HOMO %.15g == LUMO %.15g',
                                    e_sort[nocc-1], e_sort[nocc])
                    else:
                        logger.info(self, '  CNEO NUC HOMO = %.15g  LUMO = %.15g',
                                    e_sort[nocc-1], e_sort[nocc])

                if self.verbose >= logger.DEBUG:
                    numpy.set_printoptions(threshold=nmo)
                    logger.debug(self, '  CNEO NUC mo_energy =\n%s', mo_energy)
                    numpy.set_printoptions(threshold=1000)
            return mo_occ
        else:
            if self.mol.nhomo is None:
                return super().get_occ(mo_energy, mo_coeff)
            else:
                assert isinstance(self, scf.uhf.UHF)
                if self.mol.symmetry:
                    raise NotImplementedError('Fractional does not support symmetry yet.')
                e_idx_a = numpy.argsort(mo_energy[0])
                e_idx_b = numpy.argsort(mo_energy[1])
                e_sort_a = mo_energy[0][e_idx_a]
                e_sort_b = mo_energy[1][e_idx_b]
                nmo = mo_energy[0].size
                n_a, n_b = self.nelec
                mo_occ = numpy.zeros_like(mo_energy)
                # change the homo occupation to fractional
                if n_a > n_b or n_b <= 0:
                    mo_occ[0,e_idx_a[:n_a - 1]] = 1
                    mo_occ[0,e_idx_a[n_a - 1]] = self.mol.nhomo
                    mo_occ[1,e_idx_b[:n_b]] = 1
                else:
                    mo_occ[1,e_idx_b[:n_b - 1]] = 1
                    mo_occ[1,e_idx_b[n_b - 1]] = self.mol.nhomo
                    mo_occ[0,e_idx_a[:n_a]] = 1
                if self.verbose >= logger.INFO and n_a < nmo and n_b > 0 and n_b < nmo:
                    if e_sort_a[n_a-1]+1e-3 > e_sort_a[n_a]:
                        logger.warn(self, 'alpha nocc = %d  HOMO %.15g >= LUMO %.15g',
                                    n_a, e_sort_a[n_a-1], e_sort_a[n_a])
                    else:
                        logger.info(self, '  alpha nocc = %d  HOMO = %.15g  LUMO = %.15g',
                                    n_a, e_sort_a[n_a-1], e_sort_a[n_a])

                    if e_sort_b[n_b-1]+1e-3 > e_sort_b[n_b]:
                        logger.warn(self, 'beta  nocc = %d  HOMO %.15g >= LUMO %.15g',
                                    n_b, e_sort_b[n_b-1], e_sort_b[n_b])
                    else:
                        logger.info(self, '  beta  nocc = %d  HOMO = %.15g  LUMO = %.15g',
                                    n_b, e_sort_b[n_b-1], e_sort_b[n_b])

                    if e_sort_a[n_a-1]+1e-3 > e_sort_b[n_b]:
                        logger.warn(self, 'system HOMO %.15g >= system LUMO %.15g',
                                    e_sort_b[n_a-1], e_sort_b[n_b])

                    numpy.set_printoptions(threshold=nmo)
                    logger.debug(self, '  alpha mo_energy =\n%s', mo_energy[0])
                    logger.debug(self, '  beta  mo_energy =\n%s', mo_energy[1])
                    numpy.set_printoptions(threshold=1000)

                if mo_coeff is not None and self.verbose >= logger.DEBUG:
                    ss, s = self.spin_square((mo_coeff[0][:,mo_occ[0]>0],
                                              mo_coeff[1][:,mo_occ[1]>0]), self.get_ovlp())
                    logger.debug(self, 'multiplicity <S^2> = %.8g  2S+1 = %.8g', ss, s)
                return mo_occ

    def get_init_guess(self, mol=None, key='minao', **kwargs):
        if self.is_nucleus:
            # Nuclear initial DM cannot be generated here. It needs electronic DM
            # for the initial diagonalization.
            return 0
        else:
            # For electrons, use super_mol to generate the guess, because e_mol
            # will have zero charges for quantum nuclei, but we want a classical
            # HF initial guess here.
            # For positrons, just use the electron guess as a bad guess, but
            # better than nothing.
            if mol is None:
                mol = self.mol
            charge = self.charge
            self.charge = abs(charge)
            dm = super().get_init_guess(mol.super_mol, key, **kwargs)
            self.charge = charge
            return dm

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        # Compatibility for mp.mp2.get_e_hf
        warnings.warn('Calling energy_tot from a single component of multi-component SCF might be wrong!')
        return super().energy_tot(dm, h1e, vhf)

    def scf(self, dm0=None, **kwargs):
        raise AttributeError('scf should not be called from ComponentSCF')

    def mulliken_pop(self, mol=None, dm=None, s=None, verbose=logger.DEBUG):
        if self.is_nucleus:
            return None
        return super().mulliken_pop(mol, dm, s, verbose)

    def mulliken_meta(self, mol=None, dm=None, verbose=logger.DEBUG,
                      pre_orth_method=PRE_ORTH_METHOD, s=None):
        if self.is_nucleus:
            return None
        return super().mulliken_meta(mol, dm, verbose, pre_orth_method, s)

    def dip_moment(self, mol=None, dm=None, unit='Debye', origin=None, verbose=logger.NOTE,
                   **kwargs):
        if self.is_nucleus:
            return None
        # Temporarily modify charge to supress warning about nonzero charge for a neutral molecule
        charge = self.mol.charge
        self.mol.charge = self.mol.super_mol.charge
        dip = super().dip_moment(mol, dm, unit, origin=origin, verbose=verbose, **kwargs)
        self.mol.charge = charge
        return dip

    def to_gpu(self):
        obj = self.undo_component().to_gpu()
        obj = general_scf(obj, self.charge, self.mass, self.is_nucleus, self.nuc_occ_state)
        return lib.to_gpu(self, obj)

def _build_eri(mol1, mol2, cart):
    atm, bas, env = gto.conc_env(mol1._atm, mol1._bas, mol1._env,
                                 mol2._atm, mol2._bas, mol2._env)
    intor_name = 'int2e_sph'
    if cart:
        intor_name = 'int2e_cart'
    size1 = mol1._bas.shape[0]
    size2 = mol2._bas.shape[0]
    return gto.moleintor.getints(intor_name, atm, bas, env,
                                 shls_slice=(0, size1, 0, size1,
                                             size1, size1 + size2,
                                             size1, size1 + size2),
                                 aosym='s4')

class InteractionCoulomb:
    '''Inter-component Coulomb interactions'''
    def __init__(self, mf1_type, mf1, mf2_type, mf2, max_memory):
        self.mf1_type = mf1_type
        self.mf1 = mf1
        self.mf1_unrestricted = isinstance(self.mf1, scf.uhf.UHF)
        self.mf2_type = mf2_type
        self.mf2 = mf2
        self.mf2_unrestricted = isinstance(self.mf2, scf.uhf.UHF)
        self.max_memory = max_memory
        self._eri = None # mol1: left; mol2: right

    def _is_mem_enough(self):
        nao1 = self.mf1.mol.nao_nr()
        nao2 = self.mf2.mol.nao_nr()
        return nao1**2*nao2**2*2/1e6+lib.current_memory()[0] < self.max_memory*.95

    def get_vint(self, dm):
        '''Obtain vj for both components'''
        assert isinstance(dm, dict)
        assert self.mf1_type in dm or self.mf2_type in dm
        # Spin-insensitive interactions: sum over spin in dm first
        dm1 = dm.get(self.mf1_type)
        dm2 = dm.get(self.mf2_type)
        # Get total densities if the dm has two spin channels
        if dm1 is not None and self.mf1_unrestricted:
            assert dm1.ndim > 2 and dm1.shape[0] == 2
            dm1 = dm1[0] + dm1[1]
        if dm2 is not None and self.mf2_unrestricted:
            assert dm2.ndim > 2 and dm2.shape[0] == 2
            dm2 = dm2[0] + dm2[1]
        mol1 = self.mf1.mol
        mol2 = self.mf2.mol
        mol = mol1.super_mol
        assert mol == mol2.super_mol
        vj = {}
        if (not mol.direct_vee and
            (self._eri is not None or mol.incore_anyway or self._is_mem_enough())):
            if self._eri is None:
                if mol.verbose >= logger.DEBUG:
                    cput0 = (logger.process_clock(), logger.perf_counter())
                self._eri = _build_eri(mol1, mol2, mol.cart)
                if mol.verbose >= logger.DEBUG and self._eri is not None:
                    logger.timer(mol,
                                 f'Incore ERI between {self.mf1_type} and {self.mf2_type}',
                                 *cput0)
                    logger.debug(mol, f'    Memory usage: {self._eri.nbytes/1024**2:.3f} MB')
            if dm2 is not None:
                vj[self.mf1_type] = dot_eri_dm(self._eri, dm2,
                                               nao_v=mol1.nao, eri_dot_dm=True)
            if dm1 is not None:
                vj[self.mf2_type] = dot_eri_dm(self._eri, dm1,
                                               nao_v=mol2.nao, eri_dot_dm=False)
        else:
            if not mol.direct_vee:
                warnings.warn(f'Direct Vee is used for {self.mf1_type}-{self.mf2_type} ERIs, '
                              +'might be slow. '
                              +f'PYSCF_MAX_MEMORY is set to {mol.max_memory} MB, '
                              +f'required memory: {mol1.nao**2*mol2.nao**2*2/1e6=:.2f} MB')
            if dm1 is not None and dm2 is not None:
                vj[self.mf1_type], vj[self.mf2_type] = \
                        scf.jk.get_jk((mol1, mol1, mol2, mol2),
                                      (dm2, dm1),
                                      scripts=('ijkl,lk->ij', 'ijkl,ji->kl'),
                                      intor='int2e', aosym='s4')
            elif dm1 is not None:
                vj[self.mf2_type] = \
                        scf.jk.get_jk((mol1, mol1, mol2, mol2),
                                      dm1,
                                      scripts='ijkl,ji->kl',
                                      intor='int2e', aosym='s4')
            else:
                vj[self.mf1_type] = \
                        scf.jk.get_jk((mol1, mol1, mol2, mol2),
                                      dm2,
                                      scripts='ijkl,lk->ij',
                                      intor='int2e', aosym='s4')
        charge_product = self.mf1.charge * self.mf2.charge
        if self.mf1_type in vj:
            vj[self.mf1_type] *= charge_product
        if self.mf2_type in vj:
            vj[self.mf2_type] *= charge_product
        return vj

def generate_interactions(components, interaction_class, max_memory, **kwagrs):
    keys = sorted(components.keys())

    interactions = {}
    for i, p1 in enumerate(keys):
        for p2 in keys[i+1:]:
            if p1 != p2:
                interaction = interaction_class(p1, components[p1],
                                                p2, components[p2],
                                                max_memory, **kwagrs)
                interactions[(p1, p2)] = interaction

    return interactions

def get_fock(mf, h1e=None, s1e=None, vhf=None, vint=None, dm=None, cycle=-1,
             diis=None, diis_start_cycle=None, level_shift_factor=None,
             damp_factor=None, fock_last=None, diis_pos='both', diis_type=3):
    if h1e is None: h1e = mf.get_hcore()
    if dm is None: dm = mf.make_rdm1()
    if vint is None:
        vint = mf.get_vint(mf.mol, dm)
        vhf = mf.get_veff(mf.mol, dm) # update vhf because vhf relies on _vint
    if vhf is None: vhf = mf.get_veff(mf.mol, dm)
    f = {}
    for t, comp in mf.components.items():
        f[t] = numpy.asarray(h1e[t]) + vhf[t] + numpy.asarray(vint[t])
        if not t.startswith('n') and isinstance(comp, scf.uhf.UHF) and f[t].ndim == 2:
            f[t] = numpy.asarray((f[t],) * 2)

    # CNEO constraint term
    # NOTE: even if not using DIIS, we still optimize f.
    # This helps with final extra cycle convergence.
    f0 = None
    if isinstance(mf, neo.CDFT):
        if diis_pos == 'pre' or diis_pos == 'both' or (cycle < 0 and diis is None):
            # optimize the Lagrange multiplier in CNEO
            for t, comp in mf.components.items():
                if t.startswith('n'):
                    ia = comp.mol.atom_index
                    opt = neo.cdft.solve_constraint(comp, f[t], s1e[t], mf.f[ia])
                    mf.f[ia] = opt.x
                    if opt.success:
                        logger.debug(mf, 'CNEO NUC constraint optimization succeeded.')
                        logger.debug(mf, 'Lagrange multiplier of %s(%i) atom: %s' %
                                     (mf.mol.atom_symbol(ia), ia, mf.f[ia]))
                        logger.debug(mf, 'Position deviation: %s', opt.fun)
                    else:
                        logger.warn(mf, 'CNEO NUC constraint optimization failed!')
                        logger.warn(mf, f'scipy.optimize.least_squares message: {opt.message}')
                        logger.warn(mf, 'Lagrange multiplier of %s(%i) atom: %s' %
                                    (mf.mol.atom_symbol(ia), ia, mf.f[ia]))
                        logger.warn(mf, 'Position deviation: %s', opt.fun)

        # For DIIS type 1, preserve original matrices
        if diis_type == 1:
            f0 = f.copy()

        fock_add = mf.get_fock_add_cdft()
        for t in fock_add:
            f[t] += fock_add[t]

    if cycle < 0 and diis is None:  # Not inside the SCF iteration
        return f

    if diis_start_cycle is None:
        diis_start_cycle = mf.diis_start_cycle
    if level_shift_factor is None:
        level_shift_factor = mf.level_shift
    if damp_factor is None:
        damp_factor = mf.damp
    if s1e is None: s1e = mf.get_ovlp()

    # TODO: unpack level_shift_factor and damp_factor

    for t, comp in mf.components.items():
        if not t.startswith('n') and isinstance(comp, scf.uhf.UHF) \
                and isinstance(dm[t], numpy.ndarray) and dm[t].ndim == 2:
            dm[t] = numpy.asarray((dm[t]*0.5,) * 2)

    if damp_factor is not None and abs(damp_factor) > 1e-4:
        warnings.warn('Damping for multi-component SCF is not yet implemented.')
    # if 0 <= cycle < diis_start_cycle-1 and abs(damp_factor) > 1e-4 and fock_last is not None:

    if diis is not None and cycle >= diis_start_cycle:
        if isinstance(mf, neo.CDFT):
            # if CNEO, needs to manually use lib.diis and pack/unpack
            keys = sorted(f.keys())
            shapes = {k: f[k].shape for k in keys}
            if diis_type != 1:
                f_flat = numpy.concatenate([f[k].ravel() for k in keys])

            if diis_type == 1:
                f0_flat = numpy.concatenate([f0[k].ravel() for k in keys])
                f_flat = diis.update(f0_flat, scf.diis.get_err_vec(s1e, dm, f))
            elif diis_type == 2:
                f_flat = diis.update(f_flat)
            elif diis_type == 3:
                f_flat = diis.update(f_flat, scf.diis.get_err_vec(s1e, dm, f))
            else:
                print("\nWARN: Unknow CDFT DIIS type, NO DIIS IS USED!!!\n")

            # Reconstruct dictionary
            offset = 0
            f_new = {}
            for k in keys:
                size = numpy.prod(shapes[k])
                f_new[k] = f_flat[offset:offset+size].reshape(shapes[k])
                offset += size
            f = f_new

            if diis_type == 1:
                for t in fock_add:
                    f[t] += fock_add[t]
        else:
            # if not CNEO, directly use the scf.diis object provided
            f = diis.update(s1e, dm, f)
            # WARNING: CDIIS only. Using EDIIS or ADIIS will cause errors

    if level_shift_factor is not None and abs(level_shift_factor) > 1e-4:
        warnings.warn('Level shift for multi-component SCF is not yet implemented.')
    # if abs(level_shift_factor) > 1e-4:

    # Post-DIIS CDFT optimization
    if isinstance(mf, neo.CDFT) and (diis_pos == 'post' or diis_pos == 'both'):
        f0 = {}
        for t in f:
            if t.startswith('n'):
                f0[t] = f[t] - fock_add[t]
            else:
                f0[t] = f[t]

        for t, comp in mf.components.items():
            if t.startswith('n'):
                ia = comp.mol.atom_index
                opt = neo.cdft.solve_constraint(comp, f0[t], s1e[t], mf.f[ia])
                mf.f[ia] = opt.x
                if opt.success:
                    logger.debug(mf, 'CNEO NUC constraint optimization succeeded.')
                    logger.debug(mf, 'Lagrange multiplier of %s(%i) atom: %s' %
                                 (mf.mol.atom_symbol(ia), ia, mf.f[ia]))
                    logger.debug(mf, 'Position deviation: %s', opt.fun)
                else:
                    logger.warn(mf, 'CNEO NUC constraint optimization failed!')
                    logger.warn(mf, f'scipy.optimize.least_squares message: {opt.message}')
                    logger.warn(mf, 'Lagrange multiplier of %s(%i) atom: %s' %
                                (mf.mol.atom_symbol(ia), ia, mf.f[ia]))
                    logger.warn(mf, 'Position deviation: %s', opt.fun)

        fock_add = mf.get_fock_add_cdft()
        for t in fock_add:
            f[t] = f0[t] + fock_add[t]
    return f

def kernel(mf, conv_tol=1e-10, conv_tol_grad=None,
           dump_chk=True, dm0=None, callback=None, conv_check=True, **kwargs):
    cput0 = (logger.process_clock(), logger.perf_counter())
    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(conv_tol)
        logger.info(mf, 'Set gradient conv threshold to %g', conv_tol_grad)

    mol = mf.mol
    s1e = mf.get_ovlp(mol)

    if dm0 is None:
        dm = mf.get_init_guess(mol, mf.init_guess, **kwargs)
    else:
        dm = mf.get_init_guess(mol, dm0, **kwargs)

    h1e = mf.get_hcore(mol)
    vint = mf.get_vint(mol, dm)
    vhf = mf.get_veff(mol, dm)
    e_tot = mf.energy_tot(dm, h1e, vhf, vint)
    logger.info(mf, 'init E= %.15g', e_tot)

    scf_conv = False
    mo_energy = mo_coeff = mo_occ = None

    # Skip SCF iterations. Compute only the total energy of the initial density
    if mf.max_cycle <= 0:
        fock = mf.get_fock(h1e, s1e, vhf, vint, dm)  # = h1e + vhf + vint, no DIIS
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

    if isinstance(mf.diis, lib.diis.DIIS):
        mf_diis = mf.diis
    elif mf.diis:
        assert issubclass(mf.DIIS, lib.diis.DIIS)
        mf_diis = mf.DIIS(mf, mf.diis_file)
        mf_diis.space = mf.diis_space
        mf_diis.rollback = mf.diis_space_rollback
        mf_diis.damp = mf.diis_damp

        # We get the used orthonormalized AO basis from any old eigendecomposition.
        # Since the ingredients for the Fock matrix has already been built, we can
        # just go ahead and use it to determine the orthonormal basis vectors.
        fock = mf.get_fock(h1e, s1e, vhf, vint, dm)
        _, mf_diis.Corth = mf.eig(fock, s1e)
    else:
        mf_diis = None

    if isinstance(mf, neo.CDFT):
        # mf_diis needs to be the raw lib.diis.DIIS() for CNEO
        mf_diis = lib.diis.DIIS()
        mf_diis.space = 8

    if dump_chk and mf.chkfile:
        # Explicit overwrite the mol object in chkfile
        # Note in pbc.scf, mf.mol == mf.cell, cell is saved under key "mol"
        chkfile.save_mol(mol, mf.chkfile)

    # A preprocessing hook before the SCF iteration
    mf.pre_kernel(locals())

    fock_last = None
    cput1 = logger.timer(mf, 'initialize scf', *cput0)
    mf.cycles = 0
    for cycle in range(mf.max_cycle):
        dm_last = dm
        last_hf_e = e_tot

        fock = mf.get_fock(h1e, s1e, vhf, vint, dm, cycle, mf_diis, fock_last=fock_last)
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        vint = mf.get_vint(mol, dm)
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        e_tot = mf.energy_tot(dm, h1e, vhf, vint)

        # Here Fock matrix is h1e + vhf + vint, without DIIS.  Calling get_fock
        # instead of the statement "fock = h1e + vhf" because Fock matrix may
        # be modified in some methods.
        fock_last = fock
        fock = mf.get_fock(h1e, s1e, vhf, vint, dm)  # = h1e + vhf + vint, no DIIS
        grad = mf.get_grad(mo_coeff, mo_occ, fock)
        norm_gorb = {}
        for t in grad.keys():
            norm_gorb[t] = numpy.linalg.norm(grad[t])
            if not TIGHT_GRAD_CONV_TOL:
                norm_gorb[t] = norm_gorb[t] / numpy.sqrt(norm_gorb[t].size)
        norm_ddm = {}
        for t in dm.keys():
            norm_ddm[t] = numpy.linalg.norm(dm[t]-dm_last[t])
        logger.info(mf, 'cycle= %d E= %.15g  delta_E= %4.3g  |g_e|= %4.3g  |ddm_e|= %4.3g',
                    cycle+1, e_tot, e_tot-last_hf_e, norm_gorb['e'], norm_ddm['e'])
        for t in grad.keys():
            if not t.startswith('e'):
                logger.info(mf, f'    |g_{t}|= %4.3g  |ddm_{t}|= %4.3g',
                            norm_gorb[t], norm_ddm[t])

        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot-last_hf_e) < conv_tol and norm_gorb['e'] < conv_tol_grad:
            scf_conv = True

        if dump_chk and mf.chkfile:
            mf.dump_chk(locals())

        if callable(callback):
            callback(locals())

        cput1 = logger.timer(mf, 'cycle= %d'%(cycle+1), *cput1)

        if scf_conv:
            break

    mf.cycles = cycle + 1
    if scf_conv and conv_check:
        # An extra diagonalization, to remove level shift
        #fock = mf.get_fock(h1e, s1e, vhf, dm)  # = h1e + vhf
        mo_energy, mo_coeff = mf.eig(fock, s1e)
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
        dm, dm_last = mf.make_rdm1(mo_coeff, mo_occ), dm
        vint = mf.get_vint(mol, dm)
        vhf = mf.get_veff(mol, dm, dm_last, vhf)
        e_tot, last_hf_e = mf.energy_tot(dm, h1e, vhf, vint), e_tot

        fock = mf.get_fock(h1e, s1e, vhf, vint, dm)
        grad = mf.get_grad(mo_coeff, mo_occ, fock)
        norm_gorb = {}
        for t in grad.keys():
            norm_gorb[t] = numpy.linalg.norm(grad[t])
            if not TIGHT_GRAD_CONV_TOL:
                norm_gorb[t] = norm_gorb[t] / numpy.sqrt(norm_gorb[t].size)
        norm_ddm = {}
        for t in dm.keys():
            norm_ddm[t] = numpy.linalg.norm(dm[t]-dm_last[t])

        conv_tol = conv_tol * 10
        conv_tol_grad = conv_tol_grad * 3
        if callable(mf.check_convergence):
            scf_conv = mf.check_convergence(locals())
        elif abs(e_tot-last_hf_e) < conv_tol or norm_gorb['e'] < conv_tol_grad:
            scf_conv = True
        logger.info(mf, 'Extra cycle  E= %.15g  delta_E= %4.3g  |g_e|= %4.3g  |ddm_e|= %4.3g',
                    e_tot, e_tot-last_hf_e, norm_gorb['e'], norm_ddm['e'])
        for t in grad.keys():
            if not t.startswith('e'):
                logger.info(mf, f'    |g_{t}|= %4.3g  |ddm_{t}|= %4.3g',
                            norm_gorb[t], norm_ddm[t])
        if dump_chk and mf.chkfile:
            mf.dump_chk(locals())

    logger.timer(mf, 'scf_cycle', *cput0)
    # A post-processing hook before return
    mf.post_kernel(locals())
    return scf_conv, e_tot, mo_energy, mo_coeff, mo_occ

class HF(scf.hf.SCF):
    '''Multicomponent Hartree-Fock

    Examples::

    >>> from pyscf import neo
    >>> mol = neo.M(atom='H 0 0 0; F 0 0 0.917', quantum_nuc=[0], basis='ccpvdz',
    >>>             nuc_basis='pb4d')
    >>> mf = neo.HF(mol)
    >>> mf.scf()
    -99.98104139461894
    '''
    def __init__(self, mol, unrestricted=False):
        super().__init__(mol)
        # NOTE: unrestricted should be understood as "force unrestricted".
        # With unrestricted=False, each component will still be RHF/UHF depending on the spin
        self.unrestricted = unrestricted
        self.components = {}
        for t, comp in self.mol.components.items():
            if t.startswith('n'):
                self.components[t] = general_scf(scf.RHF(comp),
                                                 charge=-1. * self.mol.atom_charge(comp.atom_index),
                                                 mass=self.mol.mass[comp.atom_index]
                                                      * nist.ATOMIC_MASS / nist.E_MASS,
                                                 is_nucleus=True,
                                                 nuc_occ_state=0)
            else:
                if self.unrestricted:
                    mf = scf.UHF(comp)
                else:
                    if getattr(comp, 'nhomo', None) is not None or comp.spin != 0:
                        mf = scf.UHF(comp)
                    else:
                        mf = scf.RHF(comp)
                charge = 1.
                if t.startswith('p'):
                    charge = -1.
                self.components[t] = general_scf(mf, charge=charge)
        self.interactions = generate_interactions(self.components, InteractionCoulomb,
                                                  self.max_memory)

    # mf_elec and mf_nuc for backward compatibility
    @property
    def mf_elec(self):
        warnings.warn(f"{self.__class__}.mf_elec will be removed in the future. Change to components['e']")
        return self.components.get('e')

    @property
    def mf_positron(self):
        warnings.warn(f"{self.__class__}.mf_positron will be removed in the future. Change to components['p']")
        return self.components.get('p')

    @property
    def mf_nuc(self):
        # NOTE: sorted sort in a way that: n0, n1, n10, n11, ..., n2, ..., i.e, 1x is front of 2
        warnings.warn(f"{self.__class__}.mf_nuc will be removed in the future. Change to components[f'n{{i}}']")
        return [self.components[key] for key in
                sorted(key for key in self.components if key.startswith('n'))]

    # _eri_ne and _eri_nn for backward compatibility
    @property
    def _eri_ne(self):
        warnings.warn(f"{self.__class__}._eri_ne will be removed in the future.")
        class ModifiableERINEList:
            def __init__(self, parent_obj):
                self.parent = parent_obj

            def _get_sorted_n_keys(self):
                return sorted(key for key in self.parent.components if key.startswith('n'))

            def __getitem__(self, index):
                sorted_n_keys = self._get_sorted_n_keys()
                key = sorted_n_keys[index]
                if self.parent.interactions[('e', key)]._eri is not None:
                    return self.parent.interactions[('e', key)]._eri.T
                else:
                    return None

            def __setitem__(self, index, value):
                sorted_n_keys = self._get_sorted_n_keys()
                key = sorted_n_keys[index]
                if value is not None:
                    self.parent.interactions[('e', key)]._eri = value.T
                else:
                    self.parent.interactions[('e', key)]._eri = None

            def __len__(self):
                return len(self._get_sorted_n_keys())

            def __iter__(self):
                for i in range(len(self)):
                    yield self[i]

        return ModifiableERINEList(self)

    @property
    def _eri_nn(self):
        warnings.warn(f"{self.__class__}._eri_nn will be removed in the future.")
        class ModifiableERINNRow:
            def __init__(self, parent_obj, row_index):
                self.parent = parent_obj
                self.row_index = row_index

            def _get_sorted_n_keys(self):
                return sorted(key for key in self.parent.components if key.startswith('n'))

            def __getitem__(self, col_index):
                sorted_n_keys = self._get_sorted_n_keys()
                if self.row_index >= col_index:
                    return None
                k1 = sorted_n_keys[self.row_index]
                k2 = sorted_n_keys[col_index]
                return self.parent.interactions[(k1, k2)]._eri

            def __setitem__(self, col_index, value):
                sorted_n_keys = self._get_sorted_n_keys()
                if self.row_index >= col_index:
                    raise ValueError(f"Cannot set element [{self.row_index}][{col_index}]."
                                     +"Only upper triangle elements (i < j) can be set.")
                k1 = sorted_n_keys[self.row_index]
                k2 = sorted_n_keys[col_index]
                self.parent.interactions[(k1, k2)]._eri = value

            def __len__(self):
                return len(self._get_sorted_n_keys())

        class ModifiableERINNList:
            def __init__(self, parent_obj):
                self.parent = parent_obj

            def _get_sorted_n_keys(self):
                return sorted(key for key in self.parent.components if key.startswith('n'))

            def __getitem__(self, index):
                return ModifiableERINNRow(self.parent, index)

            def __len__(self):
                return len(self._get_sorted_n_keys())

        return ModifiableERINNList(self)

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        if self.mol.mm_mol is not None:
            logger.info(self, '** Add background charges for %s **',
                        self.__class__.__name__)
            if self.verbose >= logger.DEBUG:
                logger.debug(self, 'Charge      Location')
                coords = self.mol.mm_mol.atom_coords()
                charges = self.mol.mm_mol.atom_charges()
                for i, z in enumerate(charges):
                    logger.debug(self, '%.9g    %s', z, coords[i])
        return self

    def check_sanity(self):
        return self.components['e'].check_sanity()

    def build(self, mol=None):
        if mol is None: mol = self.mol
        return self.components['e'].build(mol=mol.components['e'])

    def eig(self, h, s):
        e = {}
        c = {}
        for t, comp in self.components.items():
            e[t], c[t] = comp.eig(h[t], s[t])
        return e, c

    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        hcore = {}
        for t, comp in mol.components.items():
            hcore[t] = self.components[t].get_hcore(mol=comp)
        return hcore

    def get_ovlp(self, mol=None):
        if mol is None: mol = self.mol
        ovlp = {}
        for t, comp in mol.components.items():
            ovlp[t] = self.components[t].get_ovlp(mol=comp)
        return ovlp

    get_fock = get_fock

    def get_occ(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None: mo_energy = self.mo_energy
        mo_occ = {}
        for t, comp in self.components.items():
            coeff = mo_coeff.get(t) if mo_coeff is not None and \
                    isinstance(mo_coeff, dict) else None
            mo_occ[t] = comp.get_occ(mo_energy[t], coeff)
        return mo_occ

    def get_grad(self, mo_coeff, mo_occ, fock):
        grad = {}
        for t, comp in self.components.items():
            grad[t] = comp.get_grad(mo_coeff[t], mo_occ[t], fock[t])
        return grad

    def get_init_guess(self, mol=None, key='minao', **kwargs):
        dm_guess = {}
        if not isinstance(key, str):
            if isinstance(key, dict): # several components are given
                dm_guess = key
            else: # numpy.ndarray
                dm_guess['e'] = key   # only e_guess is provided
            key = 'minao' # for remaining components, use default minao guess
        if mol is None: mol = self.mol
        if 'e' not in dm_guess:
            # Note that later in the code, super_mol is used instead of mol.components['e'],
            # because e_comp will have zero charges for quantum nuclei, but we want a
            # classical HF initial guess here
            e_guess = self.components['e'].get_init_guess(mol.components['e'], key, **kwargs)
            # alternatively, try the mixed initial guess
            # e_guess = init_guess_mixed(mol)
            dm_guess['e'] = e_guess

        if 'p' in self.components and 'p' not in dm_guess:
            p_guess = self.components['p'].get_init_guess(mol.components['p'], key, **kwargs)
            dm_guess['p'] = p_guess

        # Nuclear guess is obtained from the electron guess from a pure classical nuclei
        # calculation, then only set one nucleus to be quantum and obtain its solution
        for t, comp in self.components.items():
            if t.startswith('n') and t not in dm_guess:
                mol_tmp = neo.Mole()
                # Do not invoke possibly expensive QMMM during init guess
                mol_tmp.build(quantum_nuc=[comp.mol.atom_index],
                              nuc_basis=mol.nuclear_basis,
                              mm_mol=None, dump_input=False, parse_arg=False,
                              verbose=mol.verbose, output=mol.output,
                              max_memory=mol.max_memory, atom=mol.atom, unit=mol.unit,
                              nucmod=mol.nucmod, ecp=mol.ecp, pseudo=mol.pseudo,
                              charge=mol.charge, spin=mol.spin, symmetry=mol.symmetry,
                              symmetry_subgroup=mol.symmetry_subgroup, cart=mol.cart,
                              magmom=mol.magmom)
                if hasattr(mol.components[t], 'intor_symmetric_original'):
                    assert t in mol_tmp.components
                    mol_tmp.components[t].nao = mol.components[t].nao
                    mol_tmp.components[t].intor_symmetric_original = \
                            mol.components[t].intor_symmetric_original
                    mol_tmp.components[t].intor_symmetric = \
                            mol.components[t].intor_symmetric
                h_core = comp.get_hcore(mol_tmp.components[t])
                s = comp.get_ovlp(mol_tmp.components[t])
                vint = 0
                for t_pair, interaction in self.interactions.items():
                    # Only get e Coulomb on n
                    if t in t_pair and 'e' in t_pair:
                        vint += interaction.get_vint(dm_guess)[t]
                mo_energy, mo_coeff = comp.eig(h_core + vint, s)
                mo_occ = comp.get_occ(mo_energy, mo_coeff)
                dm_guess[t] = comp.make_rdm1(mo_coeff, mo_occ)
        return dm_guess

    def make_rdm1(self, mo_coeff=None, mo_occ=None, **kwargs):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if mo_occ is None: mo_occ = self.mo_occ
        dm = {}
        for t, comp in self.components.items():
            coeff = comp.mo_coeff if mo_coeff is None else mo_coeff[t]
            occ = comp.mo_occ if mo_occ is None else mo_occ[t]
            dm[t] = comp.make_rdm1(coeff, occ, **kwargs)
        return dm

    def make_rdm2(self, mo_coeff=None, mo_occ=None, **kwargs):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if mo_occ is None: mo_occ = self.mo_occ
        dm2 = {}
        for t, comp in self.components.items():
            coeff = comp.mo_coeff if mo_coeff is None else mo_coeff[t]
            occ = comp.mo_occ if mo_occ is None else mo_occ[t]
            dm2[t] = comp.make_rdm2(coeff, occ, **kwargs)
        return dm2

    def energy_elec(self, dm=None, h1e=None, vhf=None, vint=None):
        if dm is None: dm = self.make_rdm1()
        if h1e is None: h1e = self.get_hcore()
        if vint is None:
            vint = self.get_vint(self.mol, dm) # call this to update _vint in vhf
            vhf = self.get_veff(self.mol, dm)
        if vhf is None: vhf = self.get_veff(self.mol, dm)
        self.scf_summary['e1'] = 0
        self.scf_summary['e2'] = 0
        e_elec = 0
        e_coul = 0
        for t, comp in self.components.items():
            logger.debug(self, f'Component: {t}')
            # vint is already in vhf
            e_elec_t, e_coul_t = comp.energy_elec(dm[t], h1e[t], vhf[t])
            e_elec += e_elec_t
            e_coul += e_coul_t
            self.scf_summary['e1'] += comp.scf_summary['e1']
            self.scf_summary['e2'] += comp.scf_summary['e2']
        return e_elec, e_coul

    def energy_tot(self, dm=None, h1e=None, vhf=None, vint=None):
        nuc = self.energy_nuc()
        self.scf_summary['nuc'] = nuc.real

        e_tot = self.energy_elec(dm, h1e, vhf, vint)[0] + nuc
        if self.disp is not None:
            self.components['e'].disp = self.disp
        if self.components['e'].do_disp():
            if 'dispersion' in self.components['e'].scf_summary:
                e_tot += self.components['e'].scf_summary['dispersion']
            else:
                e_disp = self.components['e'].get_dispersion()
                self.components['e'].scf_summary['dispersion'] = e_disp
                e_tot += e_disp
            self.scf_summary['dispersion'] = self.components['e'].scf_summary['dispersion']

        return e_tot

    def energy_nuc(self):
        nuc = self.components['e'].energy_nuc()
        if self.mol.mm_mol is not None:
            # interactions between QM nuclei and MM particles
            mm_mol = self.mol.mm_mol
            coords = mm_mol.atom_coords()
            charges = mm_mol.atom_charges()
            mol_e = self.components['e'].mol
            for j in range(mol_e.natm):
                q2, r2 = mol_e.atom_charge(j), mol_e.atom_coord(j)
                r = lib.norm(r2-coords, axis=1)
                nuc += q2*(charges/r).sum()
        return nuc

    def scf(self, dm0=None, **kwargs):
        cput0 = (logger.process_clock(), logger.perf_counter())

        self.dump_flags()
        self.build(self.mol)

        if dm0 is None and self.mo_coeff is not None and self.mo_occ is not None:
            # Initial guess from existing wavefunction
            dm0 = self.make_rdm1()

        if self.max_cycle > 0 or self.mo_coeff is None:
            self.converged, self.e_tot, \
                    self.mo_energy, self.mo_coeff, self.mo_occ = \
                    kernel(self, self.conv_tol, self.conv_tol_grad,
                           dm0=dm0, callback=self.callback,
                           conv_check=self.conv_check, **kwargs)
            # Distribute mo_energy, mo_coeff and mo_occ
            for t, comp in self.components.items():
                comp.mo_energy = self.mo_energy[t]
                comp.mo_coeff = self.mo_coeff[t]
                comp.mo_occ = self.mo_occ[t]
                comp.converged = self.converged # MP2 needs to know HF is converged
        else:
            self.e_tot = kernel(self, self.conv_tol, self.conv_tol_grad,
                                dm0=dm0, callback=self.callback,
                                conv_check=self.conv_check, **kwargs)[1]

        logger.timer(self, 'Multicomponent-SCF', *cput0)
        self._finalize()
        return self.e_tot
    kernel = lib.alias(scf, alias_name='kernel')

    def _finalize(self):
        super()._finalize()
        # scf.hf_symm and scf.uhf_symm
        if self.mol.symmetry:
            for t, comp in self.components.items():
                if isinstance(comp, scf.uhf.UHF):
                    ea = numpy.hstack(comp.mo_energy[0])
                    eb = numpy.hstack(comp.mo_energy[1])
                    # Using mergesort because it is stable. We don't want to change the
                    # ordering of the symmetry labels when two orbitals are degenerated.
                    oa_sort = numpy.argsort(ea[comp.mo_occ[0]>0 ].round(9), kind='stable')
                    va_sort = numpy.argsort(ea[comp.mo_occ[0]==0].round(9), kind='stable')
                    ob_sort = numpy.argsort(eb[comp.mo_occ[1]>0 ].round(9), kind='stable')
                    vb_sort = numpy.argsort(eb[comp.mo_occ[1]==0].round(9), kind='stable')
                    idxa = numpy.arange(ea.size)
                    idxa = numpy.hstack((idxa[comp.mo_occ[0]> 0][oa_sort],
                                         idxa[comp.mo_occ[0]==0][va_sort]))
                    idxb = numpy.arange(eb.size)
                    idxb = numpy.hstack((idxb[comp.mo_occ[1]> 0][ob_sort],
                                         idxb[comp.mo_occ[1]==0][vb_sort]))
                    self.mo_energy[t] = comp.mo_energy = (ea[idxa], eb[idxb])
                    orbsyma, orbsymb = comp.get_orbsym(comp.mo_coeff)
                    orbsyma = orbsyma[idxa]
                    orbsymb = orbsymb[idxb]
                    degen_a = degen_b = None
                    if self.mol.components[t].groupname in ('Dooh', 'Coov'):
                        try:
                            degen_a = scf.hf_symm.map_degeneracy(comp.mo_energy[0], orbsyma)
                            degen_b = scf.hf_symm.map_degeneracy(comp.mo_energy[1], orbsymb)
                        except PointGroupSymmetryError:
                            logger.warn(self, 'Orbital degeneracy broken')
                    if degen_a is None or degen_b is None:
                        mo_a = lib.tag_array(comp.mo_coeff[0][:,idxa], orbsym=orbsyma)
                        mo_b = lib.tag_array(comp.mo_coeff[1][:,idxb], orbsym=orbsymb)
                    else:
                        mo_a = lib.tag_array(comp.mo_coeff[0][:,idxa], orbsym=orbsyma,
                                             degen_mapping=degen_a)
                        mo_b = lib.tag_array(comp.mo_coeff[1][:,idxb], orbsym=orbsymb,
                                             degen_mapping=degen_b)
                    self.mo_coeff[t] = comp.mo_coeff = (mo_a, mo_b)
                    self.mo_occ[t] = comp.mo_occ = numpy.asarray([comp.mo_occ[0][idxa], comp.mo_occ[1][idxb]])
                else:
                    # sort MOs wrt orbital energies, it should be done last.
                    # Using mergesort because it is stable. We don't want to change the
                    # ordering of the symmetry labels when two orbitals are degenerated.
                    o_sort = numpy.argsort(comp.mo_energy[comp.mo_occ> 0].round(9), kind='stable')
                    v_sort = numpy.argsort(comp.mo_energy[comp.mo_occ==0].round(9), kind='stable')
                    idx = numpy.arange(comp.mo_energy.size)
                    idx = numpy.hstack((idx[comp.mo_occ> 0][o_sort],
                                        idx[comp.mo_occ==0][v_sort]))
                    self.mo_energy[t] = comp.mo_energy = comp.mo_energy[idx]
                    self.mo_occ[t] = comp.mo_occ = comp.mo_occ[idx]
                    orbsym = comp.get_orbsym(comp.mo_coeff)
                    orbsym = orbsym[idx]
                    degen_mapping = None
                    if self.mol.components[t].groupname in ('Dooh', 'Coov'):
                        try:
                            degen_mapping = scf.hf_symm.map_degeneracy(comp.mo_energy, orbsym)
                        except PointGroupSymmetryError:
                            logger.warn(self, 'Orbital degeneracy broken')
                    if degen_mapping is None:
                        self.mo_coeff[t] = comp.mo_coeff = lib.tag_array(comp.mo_coeff[:,idx], orbsym=orbsym)
                    else:
                        self.mo_coeff[t] = comp.mo_coeff = lib.tag_array(
                            comp.mo_coeff[:,idx], orbsym=orbsym, degen_mapping=degen_mapping)
            if self.chkfile:
                chkfile.dump_scf(self.mol, self.chkfile, self.e_tot, self.mo_energy,
                                 self.mo_coeff, self.mo_occ, overwrite_mol=False)
            return self

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        raise AttributeError('get_jk should not be called from multi-component SCF')

    def get_j(self, mol=None, dm=None, hermi=1, omega=None):
        raise AttributeError('get_j should not be called from multi-component SCF')

    def get_k(self, mol=None, dm=None, hermi=1, omega=None):
        raise AttributeError('get_k should not be called from multi-component SCF')

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        '''Self-type JK'''
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        vhf = {}
        for t, comp in self.components.items():
            dm_last_t = dm_last.get(t, 0) if isinstance(dm_last, dict) else 0
            vhf_last_t = vhf_last.get(t, 0) if isinstance(vhf_last, dict) else 0
            vhf[t] = comp.get_veff(mol.components[t], dm[t], dm_last_t, vhf_last_t, hermi)
        return vhf

    def get_vint(self, mol=None, dm=None, **kwargs):
        '''Inter-type Coulomb'''
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        vint = {}
        for t in self.components.keys():
            vint[t] = 0
        for t_pair, interaction in self.interactions.items():
            v = interaction.get_vint(dm, **kwargs)
            for t in t_pair:
                vint[t] += v[t]
        # Transfer vj to vint cache
        for t in self.components.keys():
            self.components[t]._vint = vint[t]
            vint[t] = 0
        return vint

    def mulliken_pop(self, mol=None, dm=None, s=None, verbose=logger.DEBUG):
        # Electronic only
        if hasattr(mol, 'components'):
            mol = mol.components['e']
        if isinstance(dm, dict) and 'e' in dm:
            dm = dm['e']
        if isinstance(s, dict) and 'e' in s:
            s = s['e']
        logger.note(self, 'Mulliken for electronic part only!\n'
                    +'Quantum nuclear charges are not considered.')
        return self.components['e'].mulliken_pop(mol, dm, s, verbose)

    def mulliken_meta(self, mol=None, dm=None, verbose=logger.DEBUG,
                      pre_orth_method=PRE_ORTH_METHOD, s=None):
        # Electronic only
        if hasattr(mol, 'components'):
            mol = mol.components['e']
        if isinstance(dm, dict) and 'e' in dm:
            dm = dm['e']
        if isinstance(s, dict) and 'e' in s:
            s = s['e']
        logger.note(self, 'Mulliken for electronic part only!\n'
                    +'Quantum nuclear charges are not considered.')
        return self.components['e'].mulliken_meta(mol, dm, verbose, pre_orth_method, s)

    def analyze(self, verbose=None, with_meta_lowdin=WITH_META_LOWDIN,
                **kwargs):
        if verbose is None: verbose = self.verbose
        log = logger.new_logger(self, verbose)
        pop_and_charge = {}
        dip = {}
        for t, comp in self.components.items():
            log.note(f'\nAnalyze for {t}:')
            pop_and_charge[t], dip[t] = comp.analyze(verbose, with_meta_lowdin, **kwargs)
        # NOTE: dipole moment for each component reported here are incomplete
        # and should not be used. Use dip_moment from cdft.
        logger.note(self, 'Mulliken and dipole moment for electronic part only!\n'
                    +'Contributions from quantum nuclear charges are not considered.')
        return pop_and_charge, dip

    def canonicalize(self, mo_coeff, mo_occ, fock=None):
        raise NotImplementedError('Please perform canonicalization for each component individually.')

    def dip_moment(self, mol=None, dm=None, unit='Debye', origin=None,
                   verbose=logger.NOTE, **kwargs):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        log = logger.new_logger(mol, verbose)

        # Suppress warning about nonzero charge (if neutral)
        charge = self.components['e'].mol.charge
        self.components['e'].mol.charge = self.mol.charge
        el_dip = self.components['e'].dip_moment(mol.components['e'],
                                                 dm['e'], unit=unit,
                                                 origin=origin, verbose=verbose-1)
        self.components['e'].mol.charge = charge

        # Quantum nuclei
        if origin is None:
            origin = numpy.zeros(3)
        else:
            origin = numpy.asarray(origin, dtype=numpy.float64)
        assert origin.shape == (3,)
        nucl_dip = 0
        for t, comp in self.components.items():
            if t.startswith('n'):
                nucl_dip -= comp.charge * (lib.einsum('xij,ji->x', comp.mol.intor_symmetric('int1e_r', comp=3), dm[t]) - origin)
        if unit.upper() == 'DEBYE':
            nucl_dip *= nist.AU2DEBYE
            mol_dip = nucl_dip + el_dip
            log.note('Dipole moment(X, Y, Z, Debye): %8.5f, %8.5f, %8.5f', *mol_dip)
        else:
            mol_dip = nucl_dip + el_dip
            log.note('Dipole moment(X, Y, Z, A.U.): %8.5f, %8.5f, %8.5f', *mol_dip)
        return mol_dip

    def quad_moment(self, mol=None, dm=None, unit='DebyeAngstrom', origin=None,
                    verbose=logger.NOTE, **kwargs):
        raise NotImplementedError

    def _is_mem_enough(self):
        raise NotImplementedError

    def density_fit(self, auxbasis=None, ee_only_dfj=False, df_ne=False):
        import pyscf.neo.df
        return pyscf.neo.df.density_fit(self, auxbasis=auxbasis,
                                        ee_only_dfj=ee_only_dfj, df_ne=df_ne)

    def sfx2c1e(self):
        raise NotImplementedError

    def newton(self):
        raise NotImplementedError

    def remove_soscf(self):
        raise NotImplementedError

    def stability(self):
        raise NotImplementedError

    def nuc_grad_method(self):
        from pyscf.neo import grad
        return grad.Gradients(self)

    def update_(self, chkfile=None):
        raise NotImplementedError

    as_scanner = as_scanner

    def reset(self, mol=None):
        '''Reset mol and relevant attributes associated to the old mol object'''
        if mol is not None:
            self.mol = mol
        super().reset(mol=mol)
        if sorted(self.components.keys()) == sorted(self.mol.components.keys()):
            # quantum nuc is the same, reset each component
            for t, comp in self.components.items():
                comp.reset(self.mol.components[t])
                comp._vint = None
            for t, comp in self.interactions.items():
                comp._eri = None
        else:
            # quantum nuc is different, need to rebuild
            self.components.clear()
            for t, comp in self.mol.components.items():
                if t.startswith('n'):
                    self.components[t] = general_scf(scf.RHF(comp),
                                                     charge=-1. * self.mol.atom_charge(comp.atom_index),
                                                     mass=self.mol.mass[comp.atom_index] * nist.ATOMIC_MASS
                                                          / nist.E_MASS,
                                                     is_nucleus=True,
                                                     nuc_occ_state=0)
                else:
                    if self.unrestricted:
                        mf = scf.UHF(comp)
                    else:
                        if getattr(comp, 'nhomo', None) is not None or comp.spin != 0:
                            mf = scf.UHF(comp)
                        else:
                            mf = scf.RHF(comp)
                    charge = 1.
                    if t.startswith('p'):
                        charge = -1.
                    self.components[t] = general_scf(mf, charge=charge)
            self.interactions.clear()
            self.interactions.update(generate_interactions(self.components, InteractionCoulomb,
                                                           self.max_memory))

        return self

    def apply(self, fn, *args, **kwargs):
        raise NotImplementedError

    def to_ks(self, xc='HF'):
        raise NotImplementedError

    def to_gpu(self):
        raise NotImplementedError

if __name__ == '__main__':
    mol = neo.M(atom='H 0 0 0', basis='ccpvdz', nuc_basis='pb4d', verbose=5, spin=1)
    mf = neo.HF(mol)
    mf.scf()
