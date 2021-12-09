#!/usr/bin/env python

import os
import numpy
from pyscf import gto
from pyscf.lib import logger

class Mole(gto.mole.Mole):
    '''A subclass of gto.mole.Mole to handle quantum nuclei in NEO.
    By default, all atoms would be treated quantum mechanically.

    Example:

    >>> from pyscf import neo
    >>> mol = neo.Mole()
    >>> mol.build(atom = 'H 0 0 0; C 0 0 1.1; N 0 0 2.2', quantum_nuc = [0,1], basis = 'ccpvdz')
    # H and C would be treated quantum mechanically
    >>> mol.build(atom = 'H 0 0 0; C 0 0 1.1; N 0 0 2.2', basis = 'ccpvdz')
    # All atoms are treated quantum mechanically by default

    '''

    def __init__(self, **kwargs):
        gto.mole.Mole.__init__(self, **kwargs)

        self.quantum_nuc = [] # a list to assign which nuclei are treated quantum mechanically
        self.nuc_num = 0 # the number of quantum nuclei
        self.mass = [] # masses of nuclei
        self.elec = None # a Mole object for NEO-electron and classical nuclei
        self.nuc = [] # a list of Mole objects for quantum nuclei

    def build_nuc_mole(self, atom_index, frac=None):
        '''
        Return a Mole object for specified quantum nuclei.

        Nuclear basis:

        H: PB4-D  J. Chem. Phys. 152, 244123 (2020)
        D: scaled PB4-D
        other atoms: 12s12p12d, alpha=2*sqrt(2)*mass, beta=sqrt(3)
        '''

        nuc = gto.Mole() # a Mole object for quantum nuclei
        nuc.atom_index = atom_index
        nuc.super_mol = self

        dirnow = os.path.realpath(os.path.join(__file__, '..'))
        if self.atom_symbol(atom_index) == 'H@2':
            basis = gto.basis.parse(open(os.path.join(dirnow, 'basis/s-pb4d.dat')).read())
        elif self.atom_pure_symbol(atom_index) == 'H':
            basis = gto.basis.parse(open(os.path.join(dirnow, 'basis/pb4d.dat')).read())
            #alpha = 2 * numpy.sqrt(2) * self.mass[atom_index]
            #beta = numpy.sqrt(2)
            #n = 8
            #basis = gto.expand_etbs([(0, n, alpha, beta), (1, n, alpha, beta), (2, n, alpha, beta)])
        else:
            # even-tempered basis
            alpha = 2 * numpy.sqrt(2) * self.mass[atom_index]
            beta = numpy.sqrt(3)
            n = 12
            basis = gto.expand_etbs([(0, n, alpha, beta), (1, n, alpha, beta), (2, n, alpha, beta)])
            #logger.info(self, 'Nuclear basis for %s: n %s alpha %s beta %s' %(self.atom_symbol(atom_index), n, alpha, beta))
        nuc.build(atom = self.atom, basis={self.atom_symbol(atom_index): basis},
                  charge = self.charge, cart = self.cart, spin = self.spin)

        # set all quantum nuclei to have zero charges
        quantum_nuclear_charge = 0
        for i in range(self.natm):
            if self.quantum_nuc[i] is True:
                quantum_nuclear_charge -= nuc._atm[i, 0]
                nuc._atm[i, 0] = 0 # set nuclear charges of quantum nuclei to 0
        nuc.charge += quantum_nuclear_charge

        # avoid UHF
        nuc.spin = 0
        nuc.nelectron = 2

        # fractional
        nuc.nnuc = frac

        return nuc

    def build(self, quantum_nuc='all', q_nuc_occ=None, **kwargs):
        '''assign which nuclei are treated quantum mechanically by quantum_nuc (list)'''
        super().build(self, **kwargs)

        self.quantum_nuc = [False] * self.natm

        if quantum_nuc == 'all':
            self.quantum_nuc = [True] * self.natm
            logger.info(self, 'All atoms are treated quantum-mechanically by default.')
        elif isinstance(quantum_nuc, list):
            for i in quantum_nuc:
                self.quantum_nuc[i] = True
                logger.info(self, 'The %s(%i) atom is treated quantum-mechanically' %(self.atom_symbol(i), i))
        else:
            raise TypeError('Unsupported parameter %s' %(quantum_nuc))

        self.nuc_num = len([i for i in self.quantum_nuc if i == True])

        self.mass = self.atom_mass_list(isotope_avg=True)
        for i in range(self.natm):
            if self.atom_symbol(i) == 'H@2': # Deuterium (from Wikipedia)
                self.mass[i] = 2.01410177811
            elif self.atom_symbol(i) == 'H@0': # Muonium (TODO: precise mass)
                self.mass[i] = 0.114
            elif self.atom_pure_symbol(i) == 'H': # Proton (from Wikipedia)
                self.mass[i] = 1.007276466621

        # build the Mole object for electrons and classical nuclei
        self.elec = gto.Mole()
        self.elec.super_mol = self
        self.elec.build(**kwargs)

        # deal with fractional number of nuclei
        if q_nuc_occ is not None:
            q_nuc_occ = numpy.array(q_nuc_occ)
            if q_nuc_occ.size != self.nuc_num:
                raise ValueError('q_nuc_occ must match the dimension of quantum_nuc')
            unocc = numpy.ones_like(q_nuc_occ) - q_nuc_occ
            unocc_Z = 0
            idx = 0
        # set all quantum nuclei to have zero charges
        quantum_nuclear_charge = 0
        for i in range(self.natm):
            if self.quantum_nuc[i] is True:
                quantum_nuclear_charge -= self.elec._atm[i, 0]
                if q_nuc_occ is not None:
                    unocc_Z += unocc[idx] * self.elec._atm[i, 0]
                    idx += 1
                self.elec._atm[i, 0] = 0 # set nuclear charges of quantum nuclei to 0
        self.elec.charge += quantum_nuclear_charge # charge determines the number of electrons
        if q_nuc_occ is not None:
            # remove excessive electrons to make the system neutral
            self.elec.charge += numpy.floor(unocc_Z)
            self.elec.nhomo = 1.0 - (unocc_Z - numpy.floor(unocc_Z))
        else:
            self.elec.nhomo = None

        # build a list of Mole objects for quantum nuclei
        if q_nuc_occ is None:
            q_nuc_occ = [None] * self.nuc_num
        idx = 0
        for i in range(self.natm):
            if self.quantum_nuc[i] == True:
                self.nuc.append(self.build_nuc_mole(i, frac=q_nuc_occ[idx]))
                idx += 1
