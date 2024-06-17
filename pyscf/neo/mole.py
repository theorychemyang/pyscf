#!/usr/bin/env python

import os, sys, re
import numpy
import contextlib
from pyscf import gto
from pyscf.data import nist
from pyscf.lib import logger, param


def M(**kwargs):
    r'''This is a shortcut to build up Mole object.
    '''
    mol = Mole()
    mol.build(**kwargs)
    return mol

def copy(mol, deep=True):
    '''Deepcopy of the given :class:`Mole` object
    '''
    newmol = mol.view(mol.__class__)
    if not deep:
        return newmol

    import copy
    newmol = gto.mole.Mole.copy(mol)

    # extra things for neo.Mole
    newmol.quantum_nuc = copy.deepcopy(mol.quantum_nuc)
    newmol.mass = copy.deepcopy(mol.mass)

    # inner mole's
    newmol.elec = mol.elec.copy()
    newmol.elec.super_mol = newmol
    if mol.positron is not None:
        newmol.positron = mol.positron.copy()
        newmol.positron.super_mol = newmol

    newmol.nuc = [None] * mol.nuc_num
    for i in range(mol.nuc_num):
        newmol.nuc[i] = mol.nuc[i].copy()
        newmol.nuc[i].super_mol = newmol

    return newmol

def build_nuc_mole(mol, index, atom_index, nuc_basis, frac=None):
    '''
    Return a Mole object for specified quantum nuclei.

    Nuclear basis:

    H: PB4-D  J. Chem. Phys. 152, 244123 (2020)
    D: scaled PB4-D
    other atoms: 12s12p12d, alpha=2*sqrt(2)*mass, beta=sqrt(3)
    '''

    nuc = gto.Mole() # a Mole object for quantum nuclei
    nuc.super_mol = mol
    nuc.atom_index = atom_index
    nuc.index = index

    nuc_basis = nuc_basis.lower() # e.g., PB4D will be converted to pb4d
    dirnow = os.path.realpath(os.path.join(__file__, '..'))
    if 'H+' in mol.atom_symbol(atom_index): # H+ for deuterium
        try:
            with open(os.path.join(dirnow, 'basis/'+nuc_basis+'.dat'), 'r') as f:
                basis = gto.basis.parse(f.read())
                # read in H basis, but scale the exponents by sqrt(mass_D/mass_H)
                for x in basis:
                    x[1][0] *= numpy.sqrt((2.01410177811 - nist.E_MASS / nist.ATOMIC_MASS)
                                        / (1.007825  - nist.E_MASS / nist.ATOMIC_MASS))
        except FileNotFoundError:
            try:
                # try basis name without dashes/underscores
                # e.g., PB4-D, PB4_D and pb4-_-d will be converted to pb4d
                nuc_basis_without_dash_underscore = nuc_basis.replace('-', '').replace('_', '')
                with open(os.path.join(dirnow, 'basis/'+nuc_basis_without_dash_underscore+'.dat'), 'r') as f:
                    basis = gto.basis.parse(f.read())
                    for x in basis:
                        x[1][0] *= numpy.sqrt((2.01410177811 - nist.E_MASS / nist.ATOMIC_MASS)
                                            / (1.007825  - nist.E_MASS / nist.ATOMIC_MASS))
            except FileNotFoundError:
                m = re.search("(\d+)s(\d+)p(\d+)d(\d+)?f?", nuc_basis)
                if m:
                # even-tempered basis for D
                    alpha = 4
                    beta = numpy.sqrt(2)
                    if m.group(4) is None:
                        basis = gto.expand_etbs([(0, int(m.group(1)), alpha, beta),
                                                 (1, int(m.group(2)), alpha, beta),
                                                 (2, int(m.group(3)), alpha, beta)])
                    else:
                        basis = gto.expand_etbs([(0, int(m.group(1)), alpha, beta),
                                                 (1, int(m.group(2)), alpha, beta),
                                                 (2, int(m.group(3)), alpha, beta),
                                                 (3, int(m.group(4)), alpha, beta)])
                else:
                    raise ValueError('Unsupported nuclear basis %s', nuc_basis)
    elif 'H*' in mol.atom_symbol(atom_index): # H* for muonium
        try:
            with open(os.path.join(dirnow, 'basis/'+nuc_basis+'.dat'), 'r') as f:
                basis = gto.basis.parse(f.read())
                # read in H basis, but scale the exponents by sqrt(mass_mu/mass_H)
                for x in basis:
                    x[1][0] *= numpy.sqrt(0.1134289259 / (1.007825  - nist.E_MASS / nist.ATOMIC_MASS))
        except FileNotFoundError:
            nuc_basis_without_dash_underscore = nuc_basis.replace('-', '').replace('_', '')
            with open(os.path.join(dirnow, 'basis/'+nuc_basis_without_dash_underscore+'.dat'), 'r') as f:
                basis = gto.basis.parse(f.read())
                for x in basis:
                    x[1][0] *= numpy.sqrt(0.1134289259 / (1.007825  - nist.E_MASS / nist.ATOMIC_MASS))
    elif 'H#' in mol.atom_symbol(atom_index): # H# for HeMu
        try:
            with open(os.path.join(dirnow, 'basis/'+nuc_basis+'.dat'), 'r') as f:
                basis = gto.basis.parse(f.read())
                # read in H basis, but scale the exponents by sqrt(mass_HeMu/mass_H)
                for x in basis:
                    x[1][0] *= numpy.sqrt((4.002603254 - 2 * nist.E_MASS / nist.ATOMIC_MASS + 0.1134289259)
                                          / (1.007825  - nist.E_MASS / nist.ATOMIC_MASS))
        except FileNotFoundError:
            nuc_basis_without_dash_underscore = nuc_basis.replace('-', '').replace('_', '')
            with open(os.path.join(dirnow, 'basis/'+nuc_basis_without_dash_underscore+'.dat'), 'r') as f:
                basis = gto.basis.parse(f.read())
                for x in basis:
                    x[1][0] *= numpy.sqrt((4.002603254 - 2 * nist.E_MASS / nist.ATOMIC_MASS + 0.1134289259)
                                          / (1.007825  - nist.E_MASS / nist.ATOMIC_MASS))
    elif mol.atom_pure_symbol(atom_index) == 'H':
        try:
            with open(os.path.join(dirnow, 'basis/'+nuc_basis+'.dat'), 'r') as f:
                basis = gto.basis.parse(f.read())
        except FileNotFoundError:
            try:
                nuc_basis_without_dash_underscore = nuc_basis.replace('-', '').replace('_', '')
                with open(os.path.join(dirnow, 'basis/'+nuc_basis_without_dash_underscore+'.dat'), 'r') as f:
                    basis = gto.basis.parse(f.read())
            except FileNotFoundError:
                m = re.search("(\d+)s(\d+)p(\d+)d(\d+)?f?", nuc_basis)
                if m:
                # even-tempered basis for H
                    alpha = 2 * numpy.sqrt(2)
                    beta = numpy.sqrt(2)
                    if m.group(4) is None:
                        basis = gto.expand_etbs([(0, int(m.group(1)), alpha, beta),
                                                 (1, int(m.group(2)), alpha, beta),
                                                 (2, int(m.group(3)), alpha, beta)])
                    else:
                        basis = gto.expand_etbs([(0, int(m.group(1)), alpha, beta),
                                                 (1, int(m.group(2)), alpha, beta),
                                                 (2, int(m.group(3)), alpha, beta),
                                                 (3, int(m.group(4)), alpha, beta)])
                else:
                    raise ValueError('Unsupported nuclear basis %s', nuc_basis)

    else:
        # even-tempered basis
        alpha = 2 * numpy.sqrt(2) * mol.mass[atom_index]
        beta = numpy.sqrt(3)
        n = 12
        basis = gto.expand_etbs([(0, n, alpha, beta), (1, n, alpha, beta),
                                 (2, n, alpha, beta)])
        #logger.info(mol, 'Nuclear basis for %s: n %s alpha %s beta %s'
        #            %(mol.atom_symbol(atom_index), n, alpha, beta))

    # Automatically label quantum nuclei to prevent spawning multiple basis functions
    # at different positions
    modified_symbol = mol.atom_symbol(atom_index) + str(atom_index)
    modified_atom = mol._atom.copy()
    modified_atom[atom_index] = list(modified_atom[atom_index])
    modified_atom[atom_index][0] = modified_symbol
    modified_atom[atom_index] = tuple(modified_atom[atom_index])
    # suppress "Warning: Basis not found for atom" in line 921 of gto/mole.py
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stderr(devnull):
            nuc.build(basis={modified_symbol: basis},
                      dump_input=False, parse_arg=False, verbose=mol.verbose,
                      output=mol.output, max_memory=mol.max_memory,
                      atom=modified_atom, unit='bohr', nucmod=mol.nucmod,
                      ecp=mol.ecp, charge=mol.charge, spin=mol.spin,
                      symmetry=mol.symmetry,
                      symmetry_subgroup=mol.symmetry_subgroup,
                      cart=mol.cart, magmom=mol.magmom)

    # set all quantum nuclei to have zero charges
    quantum_nuclear_charge = 0
    for i in range(mol.natm):
        if mol.quantum_nuc[i]:
            quantum_nuclear_charge -= nuc._atm[i, gto.CHARGE_OF]
            nuc._atm[i, gto.CHARGE_OF] = 0 # set nuclear charges of quantum nuclei to 0
    nuc.charge += quantum_nuclear_charge

    # avoid UHF
    nuc.nelec = (1,1) # this calls nelec.setter, which modifies _nelectron and spin

    # fractional
    if frac is not None:
        nuc.nnuc = frac
    else:
        nuc.nnuc = 1

    nuc._keys.update(['super_mol', 'index', 'atom_index', 'nnuc'])

    return nuc


class Mole(gto.mole.Mole):
    '''A class similar to gto.mole.Mole to handle quantum nuclei in (C)NEO.
    It has an inner layer of mole's that are gto.mole.Mole for electrons and
    quantum nuclei.

    Examples::

    >>> from pyscf import neo
    >>> mol = neo.Mole()
    >>> mol.build(atom='H 0.00 0.76 -0.48; H 0.00 -0.76 -0.48; O 0.00 0.00 0.00',
    >>>           basis='ccpvdz')
    # All hydrogen atoms are treated quantum mechanically by default
    >>> mol = neo.Mole()
    >>> mol.build(atom='H 0.00 0.76 -0.48; H 0.00 -0.76 -0.48; O 0.00 0.00 0.00',
    >>>           quantum_nuc=[0,1], basis='ccpvdz')
    # Explictly assign the first two H atoms to be treated quantum mechanically
    >>> mol = neo.Mole()
    >>> mol.build(atom='H 0.00 0.76 -0.48; H 0.00 -0.76 -0.48; O 0.00 0.00 0.00',
    >>>           quantum_nuc=['H'], basis='ccpvdz')
    # All hydrogen atoms are treated quantum mechanically
    >>> mol = neo.Mole()
    >>> mol.build(atom='H0 0.00 0.76 -0.48; H1 0.00 -0.76 -0.48; O 0.00 0.00 0.00',
    >>>           quantum_nuc=['H'], basis='ccpvdz')
    # Avoid repeated nuclear basis by labelling atoms of the same type
    >>> mol = neo.Mole()
    >>> mol.build(atom='H 0 0 0; C 0 0 1.1; N 0 0 2.2', quantum_nuc=[0],
    >>>           basis='ccpvdz', nuc_basis='pb4d')
    # Pick the nuclear basis for protons
    '''

    def __init__(self, **kwargs):
        gto.mole.Mole.__init__(self, **kwargs)

        self.quantum_nuc = None # a list to assign which nuclei are treated quantum mechanically
        self.nuc_num = 0 # the number of quantum nuclei
        self.mass = None # masses of nuclei
        self.elec = None # a Mole object for NEO-electron and classical nuclei
        self.nuc = None # a list of Mole objects for quantum nuclei
        self.nuc_basis = None # name of nuclear basis
        self.mm_mol = None # QMMM support
        self.positron = None
        self.positron_charge = None
        self.positron_spin = None
        self._keys.update(['quantum_nuc', 'nuc_num', 'mass', 'elec',
                           'nuc', 'nuc_basis', 'mm_mol',
                           'positron', 'positron_charge', 'positron_spin'])

    def build(self, quantum_nuc=None, nuc_basis=None, q_nuc_occ=None,
              mm_mol=None, positron_charge=None, positron_spin=None, **kwargs):
        '''assign which nuclei are treated quantum mechanically by quantum_nuc (list)'''
        super().build(**kwargs)

        # By default, all H (including isotopes) are quantum
        if self.quantum_nuc is None and quantum_nuc is None:
            quantum_nuc = ['H']

        # By default, use pb4d basis
        if self.nuc_basis is None and nuc_basis is None:
            nuc_basis = 'pb4d'

        # QMMM mm mole from pyscf.qmmm.mm_mole.create_mm_mol
        if mm_mol is not None:
            self.mm_mol = mm_mol

        # NOTE: positron_charge should be understood as, with nuclei and
        # the same amount of electrons as positrons, how much charge the
        # molecule has.
        # For example, for proton, 2e-, 1e+, you will need to build
        # a mole with one H atom, set charge=-1, spin=0 to get H- (proton and 2e-),
        # and set positron_charge=0 because proton and 1e- has 0 charge
        # (also positron_spin=1 becaue of unpaired positron).
        if positron_charge is not None:
            self.positron_charge = int(numpy.floor(positron_charge+1e-10))
            self.positron_spin = 0
        if positron_spin is not None:
            self.positron_spin = int(numpy.floor(positron_spin+1e-10))
            if self.positron_charge is None:
                self.positron_charge = 0

        if nuc_basis is not None: self.nuc_basis = nuc_basis

        if quantum_nuc is not None:
            self.quantum_nuc = [False] * self.natm
            for i in quantum_nuc:
                if isinstance(i, int):
                    self.quantum_nuc[i] = True
                    logger.info(self, 'The %s(%i) atom is treated quantum-mechanically'
                                %(self.atom_symbol(i), i))
                elif isinstance(i, str):
                    for j in range(self.natm):
                        if self.atom_pure_symbol(j) == i:
                            # NOTE: isotopes are labelled with '+' or '*', e.g.,
                            # 'H+' stands for 'D', thus both 'H+' and 'H' are
                            # treated by q.m. even quantum_nuc=['H']
                            self.quantum_nuc[j] = True
                    logger.info(self, 'All %s atoms are treated quantum-mechanically.' %i)
            self.nuc_num = len([i for i in self.quantum_nuc if i])

        if self.mass is None:
            # Use the most common isotope mass, not isotope_avg mass for quantum nuclei
            # NOTE: the definition of gto.mole.atom_mass_list is modified.
            # originally it returns elements.ISOTOPE_MAIN, now I change it
            # to elements.COMMON_ISOTOPE_MASSES, which I think makes more sense
            mass_list_not_avg = self.atom_mass_list(isotope_avg=False)
            self.mass = self.atom_mass_list(isotope_avg=True)
            for i in range(self.natm):
                if 'H+' in self.atom_symbol(i): # Deuterium (from Wikipedia)
                    self.mass[i] = 2.01410177811
                elif 'H*' in self.atom_symbol(i): # antimuon (Muonium without electron) (from Wikipedia)
                    self.mass[i] = 0.1134289259 + nist.E_MASS / nist.ATOMIC_MASS
                elif 'H#' in self.atom_symbol(i): # Muonic Helium without electron = He4 nucleus + Muon
                    # He4 atom mass from Wikipedia
                    self.mass[i] = 4.002603254 - nist.E_MASS / nist.ATOMIC_MASS + 0.1134289259
                elif self.quantum_nuc[i]:
                    # use the most common isotope mass. For H, it is 1.007825
                    self.mass[i] = mass_list_not_avg[i]
                # else: use originally provided isotope_avg mass for classical nuclei
                # this is mainly for harmonic normal mode analysis
                if self.quantum_nuc[i]:
                    # subtract electron mass to get nuclear mass
                    # the biggest error is from isotope_avg, though
                    self.mass[i] -= self.atom_charge(i) * nist.E_MASS / nist.ATOMIC_MASS

        # build the Mole object for electrons and classical nuclei
        if self.elec is None:
            self.elec = gto.Mole()
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
                if self.quantum_nuc[i]:
                    quantum_nuclear_charge -= self.elec._atm[i, gto.CHARGE_OF]
                    if q_nuc_occ is not None:
                        unocc_Z += unocc[idx] * self.elec._atm[i, gto.CHARGE_OF]
                        idx += 1
                    # set nuclear charges of quantum nuclei to 0
                    self.elec._atm[i, gto.CHARGE_OF] = 0
            # charge determines the number of electrons
            self.elec.charge += quantum_nuclear_charge
            if q_nuc_occ is not None:
                # remove excessive electrons to make the system neutral
                self.elec.charge += numpy.floor(unocc_Z)
                self.elec.nhomo = 1.0 - (unocc_Z - numpy.floor(unocc_Z))
            else:
                self.elec.nhomo = None
            self.elec.super_mol = self # proper super_mol linking
            self.elec._keys.update(['super_mol', 'nhomo'])

        # build the Mole object for positrons and classical nuclei
        # by default uses the same basis as electronic basis
        if self.positron is None and self.positron_charge is not None:
            self.positron = gto.Mole()
            kwargs['charge'] = self.positron_charge
            kwargs['spin'] = self.positron_spin
            self.positron.build(**kwargs)

            if q_nuc_occ is not None:
                raise NotImplementedError
            # set all quantum nuclei to have zero charges
            quantum_nuclear_charge = 0
            for i in range(self.natm):
                if self.quantum_nuc[i]:
                    quantum_nuclear_charge -= self.positron._atm[i, gto.CHARGE_OF]
                    if q_nuc_occ is not None:
                        unocc_Z += unocc[idx] * self.positron._atm[i, gto.CHARGE_OF]
                        idx += 1
                    # set nuclear charges of quantum nuclei to 0
                    self.positron._atm[i, gto.CHARGE_OF] = 0
            # charge determines the number of electrons/positrons
            self.positron.charge += quantum_nuclear_charge
            self.positron.nhomo = None
            self.positron.super_mol = self # proper super_mol linking
            self.positron._keys.update(['super_mol', 'nhomo'])

        # build a list of Mole objects for quantum nuclei
        if self.nuc is None:
            if q_nuc_occ is None:
                q_nuc_occ = [None] * self.nuc_num
            idx = 0
            self.nuc = []
            for i in range(self.natm):
                if self.quantum_nuc[i]:
                    self.nuc.append(build_nuc_mole(self, idx, i,
                                                   nuc_basis=self.nuc_basis,
                                                   frac=q_nuc_occ[idx]))
                    idx += 1
        else:
            if len(self.nuc) != self.nuc_num:
                raise ValueError('Number of quantum nuc changed from '+
                                 f'{len(self.nuc)} to {self.nuc_num}')

        return self

    copy = copy

    def set_geom_(self, atoms_or_coords, unit=None, symmetry=None,
                  inplace=True):
        '''Update geometry
        '''
        if self.positron is not None:
            raise NotImplementedError
        if inplace:
            mol = self
        else:
            mol = self.copy(deep=False)
            mol._env = mol._env.copy()
            mol.nuc = [None] * mol.nuc_num

        # first set_geom_ for inner mole's
        # use default charge, as gto.mole.Mole.build may complain about spin
        charge = self.elec.charge
        self.elec.charge = mol.charge
        mol.elec = self.elec.set_geom_(atoms_or_coords, unit=unit,
                                       symmetry=symmetry, inplace=inplace)
        mol.elec.charge = self.elec.charge = charge

        # must relink back to mol in case inplace=False, otherwise
        # it will point back to ``self'' here
        mol.elec.super_mol = mol
        # ensure correct core charge in case got elec mole rebuild
        for i in range(mol.natm):
            if mol.quantum_nuc[i]:
                mol.elec._atm[i, gto.CHARGE_OF] = 0

        for i in range(mol.nuc_num):
            atom_index = self.nuc[i].atom_index
            charge = self.nuc[i].charge
            self.nuc[i].charge = mol.charge
            modified_symbol = mol.elec.atom_symbol(atom_index) + str(atom_index)
            modified_atom = mol.elec._atom.copy()
            modified_atom[atom_index] = list(modified_atom[atom_index])
            modified_atom[atom_index][0] = modified_symbol
            modified_atom[atom_index] = tuple(modified_atom[atom_index])
            # In this way, nuc mole must get rebuilt.
            # It is possible to pass a numpy.ndarray such that no rebuild
            # is needed, but in rare cases even numpy.ndarray can trigger
            # a rebuild. (because of symmetry flag)
            # In that case, nuc mole will again lose basis information.
            # Therefore, here we choose a way to ensure nuclear basis is
            # correctly assigned. (and no duplication)
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stderr(devnull):
                    mol.nuc[i] = self.nuc[i].set_geom_(modified_atom, unit='bohr',
                                                       symmetry=symmetry,
                                                       inplace=inplace)
            mol.nuc[i].charge = self.nuc[i].charge = charge

            # must relink back to mol in case inplace=False, otherwise
            # it will point back to ``self'' here
            mol.nuc[i].super_mol = mol
            for j in range(mol.natm):
                # ensure correct core charge, because got rebuilt
                if mol.quantum_nuc[j]:
                    mol.nuc[i]._atm[j, gto.CHARGE_OF] = 0
            mol.nuc[i].nelec = (1,1)

        # then set_geom_ for the base mole
        # copied from gto.mole.Mole.set_geom_
        if unit is None:
            unit = mol.unit
        else:
            mol.unit = unit
        if symmetry is None:
            symmetry = mol.symmetry

        if isinstance(atoms_or_coords, numpy.ndarray):
            mol.atom = list(zip([x[0] for x in mol._atom],
                                atoms_or_coords.tolist()))
        else:
            mol.atom = atoms_or_coords

        if isinstance(atoms_or_coords, numpy.ndarray) and not symmetry:
            if isinstance(unit, str):
                if unit.upper().startswith(('B', 'AU')):
                    unit = 1.
                else: #unit[:3].upper() == 'ANG':
                    unit = 1./param.BOHR
            else:
                unit = 1./unit

            mol._atom = list(zip([x[0] for x in mol._atom],
                                 (atoms_or_coords * unit).tolist()))
            ptr = mol._atm[:, gto.PTR_COORD]
            mol._env[ptr+0] = unit * atoms_or_coords[:,0]
            mol._env[ptr+1] = unit * atoms_or_coords[:,1]
            mol._env[ptr+2] = unit * atoms_or_coords[:,2]
        else:
            mol.symmetry = symmetry
            mol.build(dump_input=False, parse_arg=False)

        if mol.verbose >= logger.INFO:
            logger.info(mol, 'New geometry')
            for ia, atom in enumerate(mol._atom):
                coorda = tuple([x * param.BOHR for x in atom[1]])
                coordb = tuple([x for x in atom[1]])
                coords = coorda + coordb
                logger.info(mol, ' %3d %-4s %16.12f %16.12f %16.12f AA  '
                            '%16.12f %16.12f %16.12f Bohr\n',
                            ia+1, mol.atom_symbol(ia), *coords)
        return mol
