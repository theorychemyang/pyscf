#!/usr/bin/env python

'''
Nuclear Electronic Orbital Hartree-Fock (NEO-HF)
'''

import numpy
import scipy
from pyscf import scf
from pyscf import neo
from pyscf.lib import logger
from pyscf.data import nist

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



class HF(scf.hf.SCF):
    '''Hartree Fock for NEO

    Example:

    >>> from pyscf import neo
    >>> mol = neo.Mole()
    >>> mol.build(atom = 'H 0 0 0; F 0 0 0.917', basis = 'ccpvdz')
    >>> mf = neo.HF(mol)
    >>> mf.scf()

    '''

    def __init__(self, mol, unrestricted = False, verbose=4):
        'build up the Hamiltonian and inital density matrix for NEO-HF'
        scf.hf.SCF.__init__(self, mol)

        self.verbose = verbose
        self.mol = mol
        self.unrestricted = unrestricted

        # get electronic initial guess
        if self.unrestricted == True:
            self.dm_elec = scf.UHF(self.mol.elec).get_init_guess()
        else:
            self.dm_elec = scf.RHF(self.mol.elec).get_init_guess()
        # set quantum nuclei to zero charges
        quantum_nuclear_charge = 0
        for i in range(self.mol.natm):
            if self.mol.quantum_nuc[i] is True:
                quantum_nuclear_charge -= self.mol.elec._atm[i,0]
                self.mol.elec._atm[i,0] = 0 # set the nuclear charge of quantum nuclei to be 0
        self.mol.elec.charge += quantum_nuclear_charge # charge determines the number of electrons
        # set up the Hamiltonian for electrons
        if self.unrestricted == True:
            self.mf_elec = scf.UHF(self.mol.elec)
            self.mf_elec.verbose = self.verbose
        else:
            self.mf_elec = scf.RHF(self.mol.elec)
            self.mf_elec.verbose = self.verbose
        self.mf_elec.get_hcore = self.get_hcore_elec

        # set up the Hamiltonian for quantum nuclei
        self.mf_nuc = [None] * self.mol.nuc_num
        self.dm_nuc = [None] * self.mol.nuc_num

        for i in range(len(self.mol.nuc)):
            # To get proper initial guess, other quantum nuclei are treated as
            # classical ones for now, and only the quantum nucleus being processed
            # gets zero charge before this step. See mole.py
            self.dm_nuc[i] = self.get_init_guess_nuc(scf.RHF(self.mol.nuc[i]))
            # set all quantum nuclei to have zero charges
            quantum_nuclear_charge = 0
            for j in range(self.mol.natm):
                if self.mol.quantum_nuc[j] is True:
                    quantum_nuclear_charge -= self.mol.nuc[i]._atm[j,0]
                    self.mol.nuc[i]._atm[j,0] = 0 # set the nuclear charge of quantum nuclei to be 0
            self.mol.nuc[i].charge += quantum_nuclear_charge
            # set up
            self.mf_nuc[i] = scf.RHF(self.mol.nuc[i])
            self.mf_nuc[i].verbose = self.verbose
            self.mf_nuc[i].occ_state = 0 # for delta-SCF
            self.mf_nuc[i].get_occ = self.get_occ_nuc(self.mf_nuc[i])
            self.mf_nuc[i].get_init_guess = self.get_init_guess_nuc
            self.mf_nuc[i].get_hcore = self.get_hcore_nuc
            self.mf_nuc[i].get_veff = self.get_veff_nuc_bare

    def get_hcore_nuc(self, mole):
        'get the core Hamiltonian for quantum nucleus.'

        ia = mole.atom_index
        mass = self.mol.mass[ia] * nist.ATOMIC_MASS/nist.E_MASS # the mass of quantum nucleus in a.u.
        charge = self.mol.atom_charge(ia)

        # nuclear kinetic energy and Coulomb interactions with classical nuclei
        h = mole.intor_symmetric('int1e_kin')/mass
        h -= mole.intor_symmetric('int1e_nuc')*charge

        # Coulomb interactions between quantum nucleus and electrons
        if self.unrestricted == True:
            h -= scf.jk.get_jk((mole, mole, self.mol.elec, self.mol.elec),
                               self.dm_elec[0], scripts='ijkl,lk->ij', intor='int2e', aosym ='s4') * charge
            h -= scf.jk.get_jk((mole, mole, self.mol.elec, self.mol.elec),
                               self.dm_elec[1], scripts='ijkl,lk->ij', intor='int2e', aosym ='s4') * charge
        else:
            h -= scf.jk.get_jk((mole, mole, self.mol.elec, self.mol.elec),
                               self.dm_elec, scripts='ijkl,lk->ij', intor='int2e', aosym ='s4') * charge

        # Coulomb interactions between quantum nuclei
        for j in range(len(self.dm_nuc)):
            ja = self.mol.nuc[j].atom_index
            if ja != ia and isinstance(self.dm_nuc[j], numpy.ndarray):
                h += scf.jk.get_jk((mole, mole, self.mol.nuc[j], self.mol.nuc[j]),
                                   self.dm_nuc[j], scripts='ijkl,lk->ij', intor='int2e')*charge*self.mol.atom_charge(ja)

        return h

    def get_occ_nuc(self, mf_nuc):
        def get_occ(nuc_energy, nuc_coeff):
            'label the occupation for quantum nucleus'

            e_idx = numpy.argsort(nuc_energy)
            nuc_occ = numpy.zeros(nuc_energy.size)
            nuc_occ[e_idx[mf_nuc.occ_state]] = 1

            return nuc_occ
        return get_occ

    def get_init_guess_nuc(self, mf_nuc):
        '''Generate initial guess density matrix for quantum nuclei

           Returns:
            Density matrix, 2D ndarray
        '''
        mol = mf_nuc.mol
        h1n = self.get_hcore_nuc(mol)
        s1n = mol.intor_symmetric('int1e_ovlp')
        nuc_energy, nuc_coeff = scf.hf.eig(h1n, s1n)
        nuc_occ = mf_nuc.get_occ(nuc_energy, nuc_coeff)

        return scf.hf.make_rdm1(nuc_coeff, nuc_occ)

    def get_hcore_elec(self, mole=None):
        'Get the core Hamiltonian for electrons in NEO'
        if mole == None:
            mole = self.mol.elec # the Mole object for electrons in NEO

        j = 0
        # Coulomb interactions between electrons and all quantum nuclei
        for i in range(len(self.dm_nuc)):
            ia = self.mol.nuc[i].atom_index
            charge = self.mol.atom_charge(ia)
            if isinstance(self.dm_nuc[i], numpy.ndarray):
                j -= scf.jk.get_jk((mole, mole, self.mol.nuc[i], self.mol.nuc[i]),
                                   self.dm_nuc[i], scripts='ijkl,lk->ij', intor='int2e', aosym='s4') * charge

        return scf.hf.get_hcore(mole) + j

    def get_veff_nuc_bare(self, mol, dm, dm_last=None, vhf_last=None, hermi=1, vhfopt=None):
        'NOTE: Only for single quantum proton system.'
        return numpy.zeros((mol.nao_nr(), mol.nao_nr()))

    def get_veff_nuc(self, mol, dm, dm_last=None, vhf_last=None, hermi=1, vhfopt=None):
        '(not used) get the HF effective potential for quantum nuclei in NEO'

        Z2 = self.mol._atm[mol.atom_index, 0]**2

        if dm_last is None:
            vj, vk = scf.jk.get_jk(mol, (dm, dm), ('ijkl,ji->kl','ijkl,jk->il'), aosym='s8')
            return Z2*(vj - vk)
        else:
            ddm = numpy.asarray(dm) - numpy.asarray(dm_last)
            vj, vk = scf.jk.get_jk(mol, (ddm, ddm), ('ijkl,ji->kl','ijkl,jk->il'), aosym='s8')
            return Z2*(vj - vk)  + numpy.asarray(vhf_last)

    def elec_nuc_coulomb(self, dm_elec, dm_nuc):
        'the energy of Coulomb interactions between electrons and quantum nuclei'
        mol = self.mol
        jcross = 0
        for i in range(len(dm_nuc)):
            ia = mol.nuc[i].atom_index
            charge = mol.atom_charge(ia)

            jcross -= scf.jk.get_jk((mol.elec, mol.elec, mol.nuc[i], mol.nuc[i]),
                                    dm_nuc[i], scripts='ijkl,lk->ij', intor='int2e', aosym = 's4') * charge

        if self.unrestricted == True:
            E = numpy.einsum('ij,ji', jcross, dm_elec[0] + dm_elec[1])
        else:
            E = numpy.einsum('ij,ji', jcross, dm_elec)
        logger.debug(self, 'Energy of e-n Coulomb interactions: %s', E)
        return E

    def nuc_nuc_coulomb(self, dm_nuc):
        'the energy of Coulomb interactions between quantum nuclei'
        mol = self.mol
        E = 0
        for i in range(len(dm_nuc)):
            ia = mol.nuc[i].atom_index
            for j in range(len(dm_nuc)):
                if j != i:
                    ja = mol.nuc[j].atom_index
                    jcross = scf.jk.get_jk((mol.nuc[i], mol.nuc[i], mol.nuc[j], mol.nuc[j]),
                                           dm_nuc[j], scripts='ijkl,lk->ij', intor='int2e', aosym='s4') \
                            * mol.atom_charge(ia) * mol.atom_charge(ja)
                    E += numpy.einsum('ij,ji', jcross, dm_nuc[i])

        logger.debug(self, 'Energy of n-n Comlomb interactions: %s', E*.5) # double counted
        return E*.5

    def energy_qmnuc(self, mf_nuc, h1n, dm_nuc):
        'the energy of quantum nucleus'

        ia = mf_nuc.mol.atom_index
        n1 = numpy.einsum('ij,ji', h1n, dm_nuc)
        logger.debug(self, 'Energy of %s: %s', self.mol.atom_symbol(ia), n1)

        return n1

    def energy_tot(self, dm_elec=None, dm_nuc=None, h1e=None, vhf=None, h1n=None):
        'Total energy of NEO-HF'
        E_tot = 0

        # add the energy of electrons
        if dm_elec is None:
            dm_elec = self.mf_elec.make_rdm1()
        if dm_nuc is None:
            dm_nuc = [None] * len(self.mf_nuc)
            for i in range(len(self.mf_nuc)):
                dm_nuc[i] = self.mf_nuc[i].make_rdm1()

        if h1e is None:
            h1e = self.mf_elec.get_hcore(self.mf_elec.mol)

        if vhf is None:
            vhf = self.mf_elec.get_veff(self.mf_elec.mol, dm_elec)

        E_tot += self.mf_elec.energy_elec(dm = dm_elec, h1e = h1e, vhf = vhf)[0]

        # add the energy of quantum nuclei
        if h1n is None:
            h1n = [None] * self.mol.nuc_num
            for i in range(len(self.mf_nuc)):
                h1n[i] = self.mf_nuc[i].get_hcore(self.mf_nuc[i].mol)

        for i in range(len(self.mf_nuc)):
            E_tot += self.energy_qmnuc(self.mf_nuc[i], h1n[i], dm_nuc[i])

        # substract repeatedly counted terms and add classical nuclear replusion
        E_tot =  E_tot - self.elec_nuc_coulomb(dm_elec, dm_nuc) - self.nuc_nuc_coulomb(dm_nuc) \
            + self.mf_elec.energy_nuc()

        return E_tot

    def scf(self, conv_tol = 1e-9, max_cycle = 60):
        '''self-consistent field driver for NEO
        electrons and quantum nuclei are self-consistent in turn
        '''

        cput0 = (logger.process_clock(), logger.perf_counter())

        E_tot = self.energy_tot(self.dm_elec, self.dm_nuc)
        logger.info(self, 'Initial total Energy of NEO: %.15g\n' %(E_tot))

        cycle = 0
        cput1 = logger.timer(self, 'initialize scf', *cput0)

        while not self.converged:
            cycle += 1

            if cycle > max_cycle:
                raise RuntimeError('SCF is not convergent within %i cycles' %(max_cycle))

            E_last = E_tot

            self.mf_elec.scf(self.dm_elec)
            self.dm_elec = self.mf_elec.make_rdm1()
            for i in range(len(self.mf_nuc)):
                self.mf_nuc[i].scf(self.dm_nuc[i], dump_chk=False)
                self.dm_nuc[i] = self.mf_nuc[i].make_rdm1()

                # optimize f in cNEO
                if isinstance(self, neo.CDFT):
                    ia = self.mf_nuc[i].mol.atom_index
                    opt = scipy.optimize.root(self.first_order_de, self.f[ia], args=self.mf_nuc[i], method='hybr')
                    self.f[ia] = opt.x

                    logger.info(self, 'f of %s(%i) atom: %s' %(self.mf_nuc[i].mol.atom_symbol(ia), ia, self.f[ia]))
                    logger.info(self, '1st de of L: %s', opt.fun)

            E_tot = self.energy_tot(self.dm_elec, self.dm_nuc)
            logger.info(self, 'Cycle %i Total Energy of NEO: %s\n' %(cycle, E_tot))
            cput1 = logger.timer(self, 'cycle= %d'%(cycle), *cput1)

            if abs(E_tot - E_last) < conv_tol:
                self.converged = True
                logger.note(self, 'converged NEO energy = %.15g', E_tot)
                logger.timer(self, 'scf_cycle', *cput0)
                return E_tot

    def scf2(self, conv_tol = 1e-9, max_cycle = 60):
        '(beta) scf cycle'

        cput0 = (logger.process_clock(), logger.perf_counter())

        E_tot = self.energy_tot(self.dm_elec, self.dm_nuc)
        logger.info(self, 'Initial total Energy of NEO: %.15g\n' %(E_tot))

        # DIIS for electrons
        mf_diis = self.mf_elec.DIIS(self.mf_elec, self.mf_elec.diis_file)
        mf_diis.space = self.mf_elec.diis_space
        mf_diis.rollback = self.mf_elec.diis_space_rollback

        # DIIS for quantum nuclei
        #mf_diis_n = [None] * self.mol.nuc_num
        # for i in range(self.mol.nuc_num):
        #    mf_diis_n[i] = self.mf_nuc[i].DIIS(self.mf_nuc[i], self.mf_nuc[i].diis_file)
        #    mf_diis_n[i].space = self.mf_nuc[i].diis_space
        #    mf_diis_n[i].rollback = self.mf_nuc[i].diis_space_rollback

        s1e = self.mf_elec.get_ovlp()

        h1n = [None] * self.mol.nuc_num
        s1n = [None] * self.mol.nuc_num
        for i in range(len(self.mf_nuc)):
            s1n[i] = self.mf_nuc[i].get_ovlp()

        cycle = 0

        cput1 = logger.timer(self, 'initialize scf', *cput0)

        while not self.converged:
            cycle += 1
            if cycle > max_cycle:
                raise RuntimeError('SCF is not convergent within %i cycles' %(max_cycle))

            E_last = E_tot

            # set up electronic Hamiltonian and diagonalize it
            h1e = self.mf_elec.get_hcore()
            vhf = self.mf_elec.get_veff(self.mf_elec.mol, dm=self.dm_elec)
            fock_e = self.mf_elec.get_fock(h1e, s1e, vhf,
                                           self.dm_elec, cycle, mf_diis)
            mo_energy_e, mo_coeff_e = scf.hf.eig(fock_e, s1e)
            mo_occ_e = self.mf_elec.get_occ(mo_energy_e, mo_coeff_e)
            self.dm_elec = self.mf_elec.make_rdm1(mo_coeff_e, mo_occ_e)

            # set up nuclear Hamiltonian and diagonalize it
            for i in range(len(self.mf_nuc)):
                # optimize f in cNEO
                if isinstance(self, neo.CDFT):
                    ia = self.mf_nuc[i].mol.atom_index
                    opt = scipy.optimize.root(self.first_order_de, self.f[ia], args=self.mf_nuc[i], method='hybr')
                    self.f[ia] = opt.x

                    logger.info(self, 'f of %s(%i) atom: %s' %(self.mf_nuc[i].mol.atom_symbol(ia), ia, self.f[ia]))
                    logger.info(self, '1st de of L: %s', opt.fun)

                h1n[i] = self.mf_nuc[i].get_hcore(self.mf_nuc[i].mol)
                veff_n = self.mf_nuc[i].get_veff(self.mf_nuc[i].mol, self.dm_nuc[i])

                # fock_n = self.mf_nuc[i].get_fock(h1n[i], s1n[i], veff_n,
                #        self.dm_nuc[i], cycle, mf_diis_n[i])
                mo_energy_n, mo_coeff_n = scf.hf.eig(h1n[i] + veff_n, s1n[i])
                mo_occ_n = self.mf_nuc[i].get_occ(mo_energy_n, mo_coeff_n)
                self.dm_nuc[i] = self.mf_nuc[i].make_rdm1(mo_coeff_n, mo_occ_n)

            E_tot = self.energy_tot(self.dm_elec, self.dm_nuc, h1e, vhf, h1n)
            logger.info(self, 'Cycle %i Total Energy of NEO: %s\n' %(cycle, E_tot))
            cput1 = logger.timer(self, 'cycle= %d'%(cycle), *cput1)

            if abs(E_tot - E_last) < conv_tol:
                self.converged = True
                logger.note(self, 'converged NEO energy = %.15g', E_tot)
                logger.timer(self, 'scf_cycle', *cput0)
                return E_tot
