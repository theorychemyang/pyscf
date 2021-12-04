#!/usr/bin/env python

'''
Nuclear Electronic Orbital Hartree-Fock (NEO-HF)
'''

import copy
import numpy
import scipy
from pyscf import scf
from pyscf import neo
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import chkfile
from pyscf.data import nist
from pyscf import __config__

TIGHT_GRAD_CONV_TOL = getattr(__config__, 'scf_hf_kernel_tight_grad_conv_tol', True)

def get_hcore_nuc(mol, dm_elec, dm_nuc, mol_elec=None, mol_nuc=None):
    '''Get the core Hamiltonian for quantum nucleus.'''
    super_mol = mol.super_mol
    if mol_elec is None: mol_elec = super_mol.elec
    if mol_nuc is None: mol_nuc = super_mol.nuc
    ia = mol.atom_index
    mass = super_mol.mass[ia] * nist.ATOMIC_MASS / nist.E_MASS # the mass of the quantum nucleus in a.u.
    charge = super_mol.atom_charge(ia)
    # nuclear kinetic energy and Coulomb interactions with classical nuclei
    h = mol.intor_symmetric('int1e_kin') / mass
    h -= mol.intor_symmetric('int1e_nuc') * charge
    # Coulomb interaction between the quantum nucleus and electrons
    if dm_elec.ndim > 2:
        h -= scf.jk.get_jk((mol, mol, mol_elec, mol_elec),
                           dm_elec[0] + dm_elec[1], scripts='ijkl,lk->ij',
                           intor='int2e', aosym ='s4') * charge
    else:
        h -= scf.jk.get_jk((mol, mol, mol_elec, mol_elec),
                           dm_elec, scripts='ijkl,lk->ij',
                           intor='int2e', aosym ='s4') * charge
    # Coulomb interactions between quantum nuclei
    if len(dm_nuc) == len(mol_nuc) == super_mol.nuc_num:
        for j in range(super_mol.nuc_num):
            ja = mol_nuc[j].atom_index
            if ja != ia and isinstance(dm_nuc[j], numpy.ndarray):
                h += scf.jk.get_jk((mol, mol, mol_nuc[j], mol_nuc[j]),
                                   dm_nuc[j], scripts='ijkl,lk->ij', intor='int2e') * charge \
                     * super_mol.atom_charge(ja)
    return h

def get_occ_nuc(mf):
    def get_occ(mo_energy=mf.mo_energy, mo_coeff=mf.mo_coeff):
        '''Label the occupation for each orbital of the quantum nucleus'''
        e_idx = numpy.argsort(mo_energy)
        mo_occ = numpy.zeros(mo_energy.size)
        mo_occ[e_idx[mf.occ_state]] = 1
        return mo_occ
    return get_occ

def get_init_guess_nuc(mf, mol):
    '''Generate initial guess density matrix for the quantum nucleus

        Returns:
        Density matrix, 2D ndarray
    '''
    h1n = mf.get_hcore(mol)
    s1n = mol.intor_symmetric('int1e_ovlp')
    mo_energy, mo_coeff = mf.eig(h1n, s1n)
    mo_occ = mf.get_occ(mo_energy, mo_coeff)
    return mf.make_rdm1(mo_coeff, mo_occ)

def get_hcore_elec(mol, dm_nuc, mol_nuc=None):
    '''Get the core Hamiltonian for electrons in NEO'''
    super_mol = mol.super_mol
    if mol_nuc is None: mol_nuc = super_mol.nuc
    j = 0
    # Coulomb interactions between electrons and all quantum nuclei
    for i in range(super_mol.nuc_num):
        ia = mol_nuc[i].atom_index
        charge = super_mol.atom_charge(ia)
        if isinstance(dm_nuc[i], numpy.ndarray):
            j -= scf.jk.get_jk((mol, mol, mol_nuc[i], mol_nuc[i]),
                               dm_nuc[i], scripts='ijkl,lk->ij', intor='int2e', aosym='s4') * charge
    return scf.hf.get_hcore(mol) + j

def get_veff_nuc_bare(mol, dm):
    return numpy.zeros((mol.nao_nr(), mol.nao_nr()))

def elec_nuc_coulomb(super_mol, dm_elec, dm_nuc):
    '''Coulomb energy between electrons and quantum nuclei'''
    E = 0
    if super_mol.nuc_num > 0:
        mol_elec = super_mol.elec
        mol_nuc = super_mol.nuc
        jcross = 0
        for i in range(super_mol.nuc_num):
            ia = mol_nuc[i].atom_index
            charge = super_mol.atom_charge(ia)
            jcross -= scf.jk.get_jk((mol_elec, mol_elec, mol_nuc[i], mol_nuc[i]),
                                    dm_nuc[i], scripts='ijkl,lk->ij', intor='int2e', aosym='s4') * charge
        if dm_elec.ndim > 2:
            E = numpy.einsum('ij,ji', jcross, dm_elec[0] + dm_elec[1])
        else:
            E = numpy.einsum('ij,ji', jcross, dm_elec)
    logger.debug(super_mol, 'Energy of e-n Coulomb interactions: %s', E)
    return E

def nuc_nuc_coulomb(super_mol, dm_nuc):
    '''Coulomb energy between quantum nuclei'''
    E = 0
    mol_nuc = super_mol.nuc
    for i in range(super_mol.nuc_num - 1):
        ia = mol_nuc[i].atom_index
        for j in range(i + 1, super_mol.nuc_num):
            ja = mol_nuc[j].atom_index
            jcross = scf.jk.get_jk((mol_nuc[i], mol_nuc[i], mol_nuc[j], mol_nuc[j]),
                                   dm_nuc[j], scripts='ijkl,lk->ij', intor='int2e', aosym='s4') \
                     * super_mol.atom_charge(ia) * super_mol.atom_charge(ja)
            E += numpy.einsum('ij,ji', jcross, dm_nuc[i])
    logger.debug(super_mol, 'Energy of n-n Comlomb interactions: %s', E)
    return E

def energy_qmnuc(mf, h1n, dm_nuc, h_ep=None):
    '''Energy of the quantum nucleus'''
    ia = mf.mol.atom_index
    n1 = numpy.einsum('ij,ji', h1n, dm_nuc)
    logger.debug(mf, 'Energy of %s (%3d): %s', mf.mol.atom_symbol(ia), ia, n1)
    return n1

def energy_tot(mf_elec, mf_nuc, dm_elec=None, dm_nuc=None, h1e=None, vhf_e=None,
               h1n=None, h_ep=None):
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
        h1e = mf_elec.get_hcore(mf_elec.mol)
    if vhf_e is None:
        vhf_e = mf_elec.get_veff(mf_elec.mol, dm_elec)
    E_tot += mf_elec.energy_elec(dm=dm_elec, h1e=h1e, vhf=vhf_e)[0]
    # add the energy of quantum nuclei
    if h1n is None:
        h1n = []
        for i in range(len(mf_nuc)):
            h1n.append(mf_nuc[i].get_hcore(mf_nuc[i].mol))
    if h_ep is None:
        h_ep = [None] * len(mf_nuc)
    for i in range(len(mf_nuc)):
        E_tot += mf_nuc[i].energy_qmnuc(mf_nuc[i], h1n[i], dm_nuc[i], h_ep=h_ep[i])
    # substract double-counted terms and add classical nuclear repulsion
    super_mol = mf_elec.mol.super_mol
    E_tot = E_tot - elec_nuc_coulomb(super_mol, dm_elec, dm_nuc) \
            - nuc_nuc_coulomb(super_mol, dm_nuc) + mf_elec.energy_nuc()
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

def init_guess_elec_by_calculation(mol, unrestricted, init_guess):
    """Generate initial dm"""
    # get electronic initial guess, which uses default minao initial gues
    dm_elec = None
    if unrestricted == True:
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
        mol_tmp.build(atom=mol.atom, charge=mol.charge, spin=mol.spin, quantum_nuc=[ia])
        dm_nuc[i] = get_init_guess_nuc(mf_nuc[i], mol_tmp.nuc[0])
    return dm_nuc

def init_guess_elec_by_chkfile(mol, chkfile_name):
    dm_elec = None
    return dm_elec

def init_guess_nuc_by_chkfile(mol, chkfile_name):
    dm_nuc = None
    return dm_nuc

def kernel(mf, conv_tol=1e-10, conv_tol_grad=None,
           dump_chk=False, dm0e=None, dm0n=[], callback=None, conv_check=True, **kwargs):
    cput0 = (logger.process_clock(), logger.perf_counter())
    if conv_tol_grad is None:
        conv_tol_grad = numpy.sqrt(conv_tol)
        logger.info(mf, 'Set gradient conv threshold to %g', conv_tol_grad)
    mol = mf.mol
    if dm0e is None:
        mf.dm_elec = mf.get_init_guess_elec(mol, mf.init_guess)
    else:
        mf.dm_elec = dm0e
    if len(dm0n) < mol.nuc_num:
        mf.dm_nuc = mf.get_init_guess_nuc(mol, mf.init_guess)
        # if mf.init_guess is not 'chkfile', then it only affects the electronic part
    else:
        mf.dm_nuc = dm0n

    h1e = mf.mf_elec.get_hcore(mol.elec)
    vhf_e = mf.mf_elec.get_veff(mol.elec, mf.dm_elec)
    h_ep_e = mf.mf_elec.get_h_ep(mol.elec, mf.dm_elec)
    h1n = []
    veff_n = []
    h_ep_n = []
    for i in range(mol.nuc_num):
        h1n.append(mf.mf_nuc[i].get_hcore(mol.nuc[i]))
        veff_n.append(mf.mf_nuc[i].get_veff(mol.nuc[i], mf.dm_nuc[i]))
        h_ep_n.append(mf.mf_nuc[i].get_h_ep(mol.nuc[i], mf.dm_nuc[i]))
    e_tot = mf.energy_tot(mf.dm_elec, mf.dm_nuc, h1e, vhf_e, h1n, h_ep_n)
    logger.info(mf, 'init E= %.15g', e_tot)

    scf_conv = False
    mo_energy_e = mo_coeff_e = mo_occ_e = None
    mo_energy_n = [None] * mol.nuc_num
    mo_coeff_n = [None] * mol.nuc_num
    mo_occ_n = [None] * mol.nuc_num

    s1e = mf.mf_elec.get_ovlp(mol.elec)
    cond = lib.cond(s1e)
    logger.debug(mf, 'cond(S) = %s', cond)
    if numpy.max(cond) * 1e-17 > conv_tol:
        logger.warn(mf, 'Singularity detected in overlap matrix (condition number = %4.3g). '
                    'SCF may be inaccurate and hard to converge.', numpy.max(cond))
    s1n = []
    for i in range(mol.nuc_num):
        s1n.append(mf.mf_nuc[i].get_ovlp(mol.nuc[i]))

    # Skip SCF iterations. Compute only the total energy of the initial density
    if mf.max_cycle <= 0:
        fock_e = mf.mf_elec.get_fock(h1e + h_ep_e, s1e, vhf_e, mf.dm_elec)  # = h1e + vhf, no DIIS
        mo_energy_e, mo_coeff_e = mf.mf_elec.eig(fock_e, s1e)
        mo_occ_e = mf.mf_elec.get_occ(mo_energy_e, mo_coeff_e)
        mf.mf_elec.mo_energy = mo_energy_e
        mf.mf_elec.mo_coeff = mo_coeff_e
        mf.mf_elec.mo_occ = mo_occ_e
        for i in range(mol.nuc_num):
            mo_energy_n[i], mo_coeff_n[i] = mf.mf_nuc[i].eig(h1n[i] + h_ep_n[i] + veff_n[i], s1n[i])
            mo_occ_n[i] = mf.mf_nuc[i].get_occ(mo_energy_n[i], mo_coeff_e[i])
            mf.mf_nuc[i].mo_energy = mo_energy_n[i]
            mf.mf_nuc[i].mo_coeff = mo_coeff_n[i]
            mf.mf_nuc[i].mo_occ = mo_occ_n[i]
        if mf.dm_elec.ndim > 2:
            mf.dm_elec = mf.dm_elec[0] + mf.dm_elec[1]
        return scf_conv, e_tot, mo_energy_e, mo_coeff_e, mo_occ_e, \
               mo_energy_n, mo_coeff_n, mo_occ_n

    # Only electrons need DIIS
    if isinstance(mf.mf_elec.diis, lib.diis.DIIS):
        mf_diis = mf.mf_elec.diis
    elif mf.mf_elec.diis:
        assert issubclass(mf.mf_elec.DIIS, lib.diis.DIIS)
        mf_diis = mf.mf_elec.DIIS(mf.mf_elec, mf.mf_elec.diis_file)
        mf_diis.space = mf.mf_elec.diis_space
        mf_diis.rollback = mf.mf_elec.diis_space_rollback
    else:
        mf_diis = None

    if dump_chk and mf.chkfile:
        # Explicit overwrite the mol object in chkfile
        # Note in pbc.scf, mf.mol == mf.cell, cell is saved under key "mol"
        chkfile.save_mol(mol, mf.chkfile)

    # A preprocessing hook before the SCF iteration
    mf.pre_kernel(locals())

    if isinstance(mf, neo.CDFT):
        int1e_r = []
        for i in range(mol.nuc_num):
            int1e_r.append(mf.mf_nuc[i].mol.intor_symmetric('int1e_r', comp=3))

    cput1 = logger.timer(mf, 'initialize scf', *cput0)
    for cycle in range(mf.max_cycle):
        dm_elec_last = copy.copy(mf.dm_elec) # why didn't pyscf.scf.hf use copy?
        dm_nuc_last = copy.copy(mf.dm_nuc)
        last_e = e_tot

        # set up the electronic Hamiltonian and diagonalize it
        fock_e = mf.mf_elec.get_fock(h1e + h_ep_e, s1e, vhf_e, mf.dm_elec, cycle, mf_diis)
        mo_energy_e, mo_coeff_e = mf.mf_elec.eig(fock_e, s1e)
        mo_occ_e = mf.mf_elec.get_occ(mo_energy_e, mo_coeff_e)
        mf.dm_elec = mf.mf_elec.make_rdm1(mo_coeff_e, mo_occ_e)
        # attach mo_coeff and mo_occ to dm to improve DFT get_veff efficiency
        mf.dm_elec = lib.tag_array(mf.dm_elec, mo_coeff=mo_coeff_e, mo_occ=mo_occ_e)
        vhf_e = mf.mf_elec.get_veff(mol.elec, mf.dm_elec, dm_elec_last, vhf_e)

        # set up the nuclear Hamiltonian and diagonalize it
        for i in range(mol.nuc_num):
            # update nuclear core Hamiltonian after the electron density is updated
            h1n[i] = mf.mf_nuc[i].get_hcore(mf.mf_nuc[i].mol)
            h_ep_n[i] = mf.mf_nuc[i].get_h_ep(mf.mf_nuc[i].mol, mf.dm_nuc[i])
            # optimize f in cNEO
            if isinstance(mf, neo.CDFT):
                ia = mf.mf_nuc[i].mol.atom_index
                fx = numpy.einsum('xij,x->ij', int1e_r[i], mf.f[ia])
                opt = scipy.optimize.root(mf.first_order_de, mf.f[ia],
                                          args=(mf.mf_nuc[i], h1n[i] + h_ep_n[i] - fx,
                                                veff_n[i], s1n[i], int1e_r[i]), method='hybr')
                logger.debug(mf, 'f of %s(%i) atom: %s' %(mf.mf_nuc[i].mol.atom_symbol(ia), ia, mf.f[ia]))
                logger.debug(mf, '1st de of L: %s', opt.fun)
            else:
                mo_energy_n[i], mo_coeff_n[i] = mf.mf_nuc[i].eig(h1n[i] + h_ep_n[i] + veff_n[i], s1n[i])
                mf.mf_nuc[i].mo_energy, mf.mf_nuc[i].mo_coeff = mo_energy_n[i], mo_coeff_n[i]
                mo_occ_n[i] = mf.mf_nuc[i].get_occ(mo_energy_n[i], mo_coeff_n[i])
                mf.mf_nuc[i].mo_occ = mo_occ_n[i]
                mf.dm_nuc[i] = mf.mf_nuc[i].make_rdm1(mo_coeff_n[i], mo_occ_n[i])
            # in principle, update nuclear veff after the diagonalization, but in fact
            # this matrix is just zero
            veff_n[i] = mf.mf_nuc[i].get_veff(mf.mf_nuc[i].mol, mf.dm_nuc[i])
        norm_ddm_n = numpy.linalg.norm(numpy.array(mf.dm_nuc) - numpy.array(dm_nuc_last))

        # update electronic core Hamiltonian after the nuclear density is updated
        h1e = mf.mf_elec.get_hcore(mol.elec)
        h_ep_e = mf.mf_elec.get_h_ep(mol.elec, mf.dm_elec)

        # Here Fock matrix is h1e + vhf, without DIIS.  Calling get_fock
        # instead of the statement "fock = h1e + vhf" because Fock matrix may
        # be modified in some methods.
        fock_e = mf.mf_elec.get_fock(h1e + h_ep_e, s1e, vhf_e, mf.dm_elec)  # = h1e + vhf, no DIIS
        norm_gorb_e = numpy.linalg.norm(mf.mf_elec.get_grad(mo_coeff_e, mo_occ_e, fock_e))
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb_e = norm_gorb_e / numpy.sqrt(norm_gorb_e.size)
        norm_ddm_e = numpy.linalg.norm(mf.dm_elec - dm_elec_last)

        e_tot = mf.energy_tot(mf.dm_elec, mf.dm_nuc, h1e, vhf_e, h1n, h_ep_n)
        logger.info(mf, 'cycle= %d E= %.15g  delta_E= %4.3g  |g_e|= %4.3g  |ddm_e|= %4.3g  |ddm_n|= %4.3g',
                    cycle + 1, e_tot, e_tot - last_e, norm_gorb_e, norm_ddm_e, norm_ddm_n)

        if abs(e_tot - last_e) < conv_tol and norm_gorb_e < conv_tol_grad:
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
        mo_energy_e, mo_coeff_e = mf.mf_elec.eig(fock_e, s1e)
        mo_occ_e = mf.mf_elec.get_occ(mo_energy_e, mo_coeff_e)
        mf.dm_elec, dm_elec_last = mf.mf_elec.make_rdm1(mo_coeff_e, mo_occ_e), mf.dm_elec
        mf.dm_elec = lib.tag_array(mf.dm_elec, mo_coeff=mo_coeff_e, mo_occ=mo_occ_e)
        vhf_e = mf.mf_elec.get_veff(mol.elec, mf.dm_elec, dm_elec_last, vhf_e)

        for i in range(mol.nuc_num):
            h1n[i] = mf.mf_nuc[i].get_hcore(mf.mf_nuc[i].mol)
            h_ep_n[i] = mf.mf_nuc[i].get_h_ep(mf.mf_nuc[i].mol, mf.dm_nuc[i])
            mo_energy_n[i], mo_coeff_n[i] = mf.mf_nuc[i].eig(h1n[i] + h_ep_n[i] + veff_n[i], s1n[i])
            mf.mf_nuc[i].mo_energy, mf.mf_nuc[i].mo_coeff = mo_energy_n[i], mo_coeff_n[i]
            mo_occ_n[i] = mf.mf_nuc[i].get_occ(mo_energy_n[i], mo_coeff_n[i])
            mf.mf_nuc[i].mo_occ = mo_occ_n[i]
            mf.dm_nuc[i], dm_nuc_last[i] = mf.mf_nuc[i].make_rdm1(mo_coeff_n[i], mo_occ_n[i]), mf.dm_nuc[i]
            veff_n[i] = mf.mf_nuc[i].get_veff(mf.mf_nuc[i].mol, mf.dm_nuc[i])
        norm_ddm_n = numpy.linalg.norm(numpy.array(mf.dm_nuc) - numpy.array(dm_nuc_last))

        h1e = mf.mf_elec.get_hcore(mol.elec)
        h_ep_e = mf.mf_elec.get_h_ep(mol.elec, mf.dm_elec)
        fock_e = mf.mf_elec.get_fock(h1e + h_ep_e, s1e, vhf_e, mf.dm_elec)
        norm_gorb_e = numpy.linalg.norm(mf.mf_elec.get_grad(mo_coeff_e, mo_occ_e, fock_e))
        if not TIGHT_GRAD_CONV_TOL:
            norm_gorb_e = norm_gorb_e / numpy.sqrt(norm_gorb_e.size)
        norm_ddm_e = numpy.linalg.norm(mf.dm_elec - dm_elec_last)

        e_tot, last_e = mf.energy_tot(mf.dm_elec, mf.dm_nuc, h1e, vhf_e, h1n, h_ep_n), e_tot
        conv_tol = conv_tol * 10
        conv_tol_grad = conv_tol_grad * 3
        if abs(e_tot - last_e) < conv_tol or norm_gorb_e < conv_tol_grad:
            scf_conv = True
        logger.info(mf, 'Extra cycle  E= %.15g  delta_E= %4.3g  |g_e|= %4.3g  |ddm_e|= %4.3g  |ddm_n|= %4.3g',
                    e_tot, e_tot - last_e, norm_gorb_e, norm_ddm_e, norm_ddm_n)
        if dump_chk:
            mf.dump_chk(locals())

    logger.timer(mf, 'scf_cycle', *cput0)
    # A post-processing hook before return
    mf.post_kernel(locals())

    if mf.dm_elec.ndim > 2:
        mf.dm_elec = mf.dm_elec[0] + mf.dm_elec[1]
    return scf_conv, e_tot, mo_energy_e, mo_coeff_e, mo_occ_e, \
           mo_energy_n, mo_coeff_n, mo_occ_n


class HF(scf.hf.SCF):
    '''Hartree Fock for NEO

    Example:

    >>> from pyscf import neo
    >>> mol = neo.Mole()
    >>> mol.build(atom = 'H 0 0 0; F 0 0 0.917', basis = 'ccpvdz')
    >>> mf = neo.HF(mol)
    >>> mf.scf()

    '''

    def __init__(self, mol, unrestricted=False):
        scf.hf.SCF.__init__(self, mol)
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

        # initialize sub mf objects for electrons and quantum nuclei
        # electronic part:
        if unrestricted:
            self.mf_elec = scf.UHF(mol.elec)
        else:
            self.mf_elec = scf.RHF(mol.elec)
        self.mf_elec.get_hcore = self.get_hcore_elec
        self.mf_elec.get_h_ep = self.get_h_ep_elec
        self.mf_elec.super_mf = self

        # nuclear part
        for i in range(mol.nuc_num):
            self.mf_nuc.append(scf.RHF(mol.nuc[i]))
            mf_nuc = self.mf_nuc[-1]
            mf_nuc.occ_state = 0 # for Delta-SCF
            mf_nuc.get_occ = self.get_occ_nuc(mf_nuc)
            mf_nuc.get_hcore = self.get_hcore_nuc
            mf_nuc.get_h_ep = self.get_h_ep_nuc
            mf_nuc.get_veff = self.get_veff_nuc_bare
            mf_nuc.energy_qmnuc = self.energy_qmnuc
            mf_nuc.super_mf = self
            self.dm_nuc.append(None)

    #def dump_chk():

    def get_init_guess_elec(self, mol, init_guess):
        if init_guess != 'chkfile':
            return init_guess_elec_by_calculation(mol, self.unrestricted, init_guess)
        else:
            return init_guess_elec_by_chkfile(mol, self.chkfile)

    def get_init_guess_nuc(self, mol, init_guess):
        if init_guess != 'chkfile':
            return init_guess_nuc_by_calculation(self.mf_nuc, mol)
        else:
            return init_guess_nuc_by_chkfile(mol, self.chkfile)

    def get_hcore_elec(self, mol):
        return get_hcore_elec(mol, self.dm_nuc, self.mol.nuc)

    def get_occ_nuc(self, mf):
        return get_occ_nuc(mf)

    def get_hcore_nuc(self, mol):
        return get_hcore_nuc(mol, self.dm_elec, self.dm_nuc, self.mol.elec, self.mol.nuc)

    def get_veff_nuc_bare(self, mol, dm):
        return get_veff_nuc_bare(mol, dm)

    def energy_qmnuc(self, mf, h1n, dm_nuc, h_ep=None):
        return energy_qmnuc(mf, h1n, dm_nuc, h_ep=h_ep)

    def energy_tot(self, dm_elec, dm_nuc, h1e, vhf_e, h1n, h_ep=None):
        return energy_tot(self.mf_elec, self.mf_nuc, dm_elec, dm_nuc, h1e, vhf_e, h1n, h_ep=h_ep)

    def get_h_ep_nuc(self, mol, dm):
        '''EP contribution to nuclear part'''
        nao = mol.nao_nr()
        excsum = 0
        vmat = numpy.zeros((nao, nao))
        vmat = lib.tag_array(vmat, exc=excsum, ecoul=0, vj=0, vk=0)
        return vmat

    def get_h_ep_elec(self, mol, dm):
        '''EP contribution to electronic part'''
        nao = mol.nao_nr()
        return numpy.zeros((nao, nao))

    def scf(self, dm0e=None, dm0n=[], **kwargs):
        cput0 = (logger.process_clock(), logger.perf_counter())

        self.dump_flags()

        if self.max_cycle > 0 or self.mo_coeff is None:
            self.converged, self.e_tot = \
                    kernel(self, self.conv_tol, self.conv_tol_grad,
                           dm0e=dm0e, dm0n=dm0n, callback=self.callback,
                           conv_check=self.conv_check, **kwargs)[0 : 2]
        else:
            self.e_tot = kernel(self, self.conv_tol, self.conv_tol_grad,
                                dm0e=dm0e, dm0n=dm0n, callback=self.callback,
                                conv_check=self.conv_check, **kwargs)[1]

        logger.timer(self, 'SCF', *cput0)
        self._finalize()
        return self.e_tot
    kernel = lib.alias(scf, alias_name='kernel')
