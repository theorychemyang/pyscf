#!/usr/bin/env python
'''
Multi-component molecular structure to handle quantum nuclei, electrons, and positrons.
'''

import contextlib
import copy as cp
import numpy
import os
import re
import sys
import warnings
from pyscf import gto
from pyscf.gto.mole import DUMPINPUT, ARGPARSE, _length_in_au
from pyscf.data import nist
from pyscf.lib import logger, param
from pyscf.lib.exceptions import PointGroupSymmetryError


def M(*args, **kwargs):
    r'''This is a shortcut to build up Mole object.
    '''
    mol = Mole()
    mol.build(*args, **kwargs)
    return mol

def copy(mol, deep=True):
    '''Deepcopy of the given :class:`Mole` object
    '''
    newmol = mol.view(mol.__class__)
    if not deep:
        return newmol

    newmol = gto.Mole.copy(mol)

    newmol.mass = numpy.copy(mol.mass)
    # Extra things for neo.Mole
    newmol._quantum_nuc = cp.deepcopy(mol._quantum_nuc)

    # Components
    newmol.components = {}
    for comp_id, comp in mol.components.items():
        newmol.components[comp_id] = comp.copy()
        newmol.components[comp_id].super_mol = newmol

    return newmol

# Patch the verbose attribute because many like to change it after the initialization,
# though the correct way is something like neo.M(atom=atom, verbose=5)
gto.Mole.verbose = property(
    lambda self: (
        self.super_mol.verbose
        if hasattr(self, 'super_mol') and hasattr(self.super_mol, 'verbose')
        else (
            self.__dict__.get('_verbose')
            if '_verbose' in self.__dict__
            else next(
                (getattr(cls, 'verbose') for cls in type(self).__mro__
                 if hasattr(cls, 'verbose') and not isinstance(getattr(cls, 'verbose'), property)),
                None
            )
        )
    ),
    lambda self, value: setattr(self, '_verbose', value)
)

# Let direct_vee respect the global setting too
gto.Mole.direct_vee = property(
    lambda self: (
        self.super_mol.direct_vee
        if hasattr(self, 'super_mol') and hasattr(self.super_mol, 'direct_vee')
        else (
            self.__dict__.get('_direct_vee')
            if '_direct_vee' in self.__dict__
            else next(
                (getattr(cls, 'direct_vee') for cls in type(self).__mro__
                 if hasattr(cls, 'direct_vee') and not isinstance(getattr(cls, 'direct_vee'), property)),
                None
            )
        )
    ),
    lambda self, value: setattr(self, '_direct_vee', value)
)

class Mole(gto.Mole):
    '''A class similar to gto.Mole to handle quantum nuclei in (C)NEO.
    It has an inner layer of mole's that are gto.Mole for electrons and
    quantum nuclei.

    Examples::

    >>> from pyscf import neo
    # All hydrogen atoms are treated quantum mechanically by default
    >>> mol = neo.Mole()
    >>> mol.build(atom='H 0.00 0.76 -0.48; H 0.00 -0.76 -0.48; O 0.00 0.00 0.00',
    >>>           basis='ccpvdz', nuc_basis='pb4d')
    # Can specify the symbol for quantum treatment
    >>> mol = neo.Mole()
    >>> mol.build(atom='H 0.00 0.76 -0.48; H 0.00 -0.76 -0.48; O 0.00 0.00 0.00',
    >>>           quantum_nuc=['H'], basis='ccpvdz', nuc_basis='pb4d')
    # Explictly assign the first two H atoms to be treated quantum mechanically
    >>> mol = neo.Mole()
    >>> mol.build(atom='H 0.00 0.76 -0.48; H 0.00 -0.76 -0.48; O 0.00 0.00 0.00',
    >>>           quantum_nuc=[0,1], basis='ccpvdz', nuc_basis='pb4d')
    '''

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.components = {} # Dictionary to store all components

        self._quantum_nuc = None # List to assign which nuclei are treated quantum mechanically
        self._n_quantum_nuc = 0 # Number of quantum nuclei
        self.mass = None # Masses of nuclei
        self.nuclear_basis = 'pb4d' # Name of nuclear basis
        self.mm_mol = None # QMMM support
        self.positron_charge = None
        self.positron_spin = None
        self._keys.update(['components', '_quantum_nuc', '_n_quantum_nuc',
                           'mass','nuclear_basis', 'mm_mol',
                           'positron_charge', 'positron_spin'])

    # elec and nuc for backward compatibility
    @property
    def elec(self):
        warnings.warn(f"{self.__class__}.elec will be removed in the future. Change to components['e']")
        return self.components.get('e')

    @property
    def positron(self):
        warnings.warn(f"{self.__class__}.positron will be removed in the future. Change to components['p']")
        return self.components.get('p')

    @property
    def nuc(self):
        warnings.warn(f"{self.__class__}.mf_nuc will be removed in the future. Change to components[f'n{{i}}']")
        # NOTE: sorted sort in a way that: n0, n1, n10, n11, ..., n2, ..., i.e, 1x is front of 2
        return [self.components[key] for key in
                sorted(key for key in self.components if key.startswith('n'))]

    @property
    def nuc_num(self):
        '''Number of quantum nuclei'''
        return self._n_quantum_nuc

    def build(self, dump_input=DUMPINPUT, parse_arg=ARGPARSE,
              quantum_nuc=None, nuc_basis=None, q_nuc_occ=None,
              mm_mol=None, positron_charge=None, positron_spin=None, **kwargs):
        '''assign which nuclei are treated quantum mechanically by quantum_nuc (list)

        Args:
            quantum_nuc : list
                Which nuclei to treat quantum mechanically
            nuc_basis : str
                Basis set for quantum nuclei
            q_nuc_occ : list
                Fractional occupations for quantum nuclei
            mm_mol : Mole
                MM molecule for QM/MM
            positron_charge : int
                Charge for positron component
            positron_spin : int
                Spin for positron component
            **kwargs :
                Arguments passed to parent Mole.build()
        '''
        if isinstance(dump_input, str):
            sys.stderr.write('Assigning the first argument %s to mol.atom\n' %
                             dump_input)
            dump_input, kwargs['atom'] = True, dump_input

        # Do not yet build the symmetry
        symmetry = kwargs.get('symmetry', None)
        if symmetry is not None:
            kwargs['symmetry'] = False
        # Delay dump_input
        super().build(False, parse_arg, **kwargs)

        # Do not dump_input or parse_arg for components
        kwargs['dump_input'] = False
        kwargs['parse_arg'] = False

        # By default, all H (including isotopes) are quantum
        if quantum_nuc is None:
            quantum_nuc = ['H']

        # By default, use pb4d basis
        if nuc_basis is not None:
            self.nuclear_basis = nuc_basis

        # QMMM mm mole from pyscf.qmmm.mm_mole.create_mm_mol
        if mm_mol is not None:
            self.mm_mol = mm_mol

        # Handle positron parameters
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

        # Setup quantum nuclei
        if quantum_nuc is not None:
            try:
                None in quantum_nuc
            except TypeError:
                raise RuntimeError('quantum_nuc should be a list')
            self._setup_quantum_nuclei(quantum_nuc)

        # Intialize masses
        self._init_masses()

        # Build components
        self._build_components(q_nuc_occ, **kwargs)

        # Build symmetry now
        if symmetry is not None:
            self.symmetry = symmetry
            if self.symmetry:
                self._build_symmetry_mc()

        # Dump input now
        if dump_input and self.verbose > logger.NOTE:
            self.dump_input()

        return self

    def _build_symmetry_mc(self):
        if 'e' not in self.components:
            raise RuntimeError('Should not build symmetry when components are not ready.')
        if 'p' in self.components:
            raise NotImplementedError('No symmetry for positron yet.')
        # This symmetry detection is a bit stricter than the original one.
        # Two atoms are considered equivalent if they have: the same charge/symbol,
        # the same electronic basis, the same mass, and the same nuclear basis.
        # To achieve this distinction:
        # 1. Label _atom
        # 2. Add mass and nuclear basis information to _basis
        _basis_bak = cp.deepcopy(self._basis)
        _atom_bak = cp.deepcopy(self._atom)
        for i in range(self.natm):
            if self._quantum_nuc[i]:
                modified_symbol = self.atom_symbol(i) + str(i)
                if self.atom_symbol(i) in _basis_bak:
                    self._basis[modified_symbol] = cp.deepcopy(_basis_bak[self.atom_symbol(i)])
                else:
                    self._basis[modified_symbol] = cp.deepcopy(_basis_bak[self.atom_pure_symbol(i)])
                self._basis[modified_symbol].append(self.mass[i].item())
                n_mol = self.components[f'n{i}']
                self._basis[modified_symbol].append(n_mol._basis[modified_symbol])
                self._atom[i] = list(self._atom[i])
                self._atom[i][0] = modified_symbol
                self._atom[i] = tuple(self._atom[i])
        self._build_symmetry()
        self._basis, _basis_bak = _basis_bak, self._basis
        self._atom, _atom_bak = _atom_bak, self._atom
        for t, comp in self.components.items():
            if t == 'e':
                comp.symmetry = self.symmetry
                # copy results from super_mol
                comp.topgroup = self.topgroup
                comp._symm_orig = self._symm_orig
                comp._symm_axes = self._symm_axes
                comp.groupname = self.groupname
                comp.symm_orb = self.symm_orb
                comp.irrep_id = self.irrep_id
                comp.irrep_name = self.irrep_name
            elif t.startswith('n'):
                comp.symmetry = True
                # detect symmetry and build
                _basis_bak2 = cp.deepcopy(comp._basis)
                comp._basis = cp.deepcopy(_basis_bak)
                comp._atom, _atom_bak = _atom_bak, comp._atom
                modified_symbol = comp.atom_symbol(comp.atom_index)
                # Denote that nuclear basis at the specific atom is actually
                # being used for this particular nuclear component, while others
                # are in _basis merely for symmetry detection purpose
                comp._basis[modified_symbol].append(9999)
                comp._build_symmetry()
                comp._basis = _basis_bak2
                comp._atom, _atom_bak = _atom_bak, comp._atom
                _basis_bak2 = None
            else:
                raise RuntimeError(f'Unrecognized component name {t}')
        _atom_bak = None
        _basis_bak = None
        return self

    def _setup_quantum_nuclei(self, quantum_nuc):
        '''Setup which nuclei are quantum mechanical'''
        self._quantum_nuc = [False] * self.natm
        for i in quantum_nuc:
            if isinstance(i, int):
                self._quantum_nuc[i] = True
                logger.info(self, 'The %s(%i) atom is treated quantum-mechanically'
                            % (self.atom_symbol(i), i))
            elif isinstance(i, str):
                for j in range(self.natm):
                    if self.atom_pure_symbol(j) == i:
                        # NOTE: isotopes are labelled with '+' or '*', e.g.,
                        # 'H+' stands for 'D', thus both 'H+' and 'H' are
                        # treated by q.m. even quantum_nuc=['H']
                        self._quantum_nuc[j] = True
                logger.info(self, 'All %s atoms are treated quantum-mechanically.' % i)
        self._n_quantum_nuc = sum(self._quantum_nuc)
        if self._n_quantum_nuc <= 0:
            raise RuntimeError('No quantum nucleus detected! Check your quantum_nuc input.')

    def _init_masses(self):
        '''Initialize nuclear masses including special isotopes'''
        if self.mass is None:
            # Use the most common isotope mass, not isotope_avg mass for quantum nuclei
            mass_commom = self.atom_mass_list(common=True)
            self.mass = self.atom_mass_list(isotope_avg=True)
            for i in range(self.natm):
                if 'H+' in self.atom_symbol(i): # Deuterium (from Wikipedia)
                    self.mass[i] = 2.01410177811
                elif 'H*' in self.atom_symbol(i): # antimuon (Muonium without electron) (from Wikipedia)
                    self.mass[i] = 0.1134289259 + nist.E_MASS / nist.ATOMIC_MASS
                elif 'H#' in self.atom_symbol(i): # Muonic Helium without electron = He4 nucleus + Muon
                    # He4 atom mass from Wikipedia
                    self.mass[i] = 4.002603254 - nist.E_MASS / nist.ATOMIC_MASS + 0.1134289259
                elif self._quantum_nuc[i]:
                    # use the most common isotope mass. For H, it is 1.007825
                    self.mass[i] = mass_commom[i]
                # else: use originally provided isotope_avg mass for classical nuclei
                # this is mainly for harmonic normal mode analysis
                if self._quantum_nuc[i]:
                    # subtract electron mass to get nuclear mass
                    # the biggest error is from isotope_avg, though
                    self.mass[i] -= self.atom_charge(i) * nist.E_MASS / nist.ATOMIC_MASS

    def _build_components(self, q_nuc_occ=None, **kwargs):
        '''Build all quantum components'''
        self.components.clear()

        # Build electronic component
        e_mol = self._build_electronic_mol(q_nuc_occ, **kwargs)
        self.components['e'] = e_mol

        # Build positron component if needed
        if self.positron_charge is not None and self.positron_spin is not None:
            p_mol = self._build_positron_mol(q_nuc_occ, **kwargs)
            self.components['p'] = p_mol

        # Build nuclear components
        if q_nuc_occ is None:
            q_nuc_occ = [None] * self._n_quantum_nuc

        idx = 0
        for i in range(self.natm):
            if self._quantum_nuc[i]:
                n_mol = self._build_nuclear_mol(i, q_nuc_occ[idx])
                self.components[f'n{i}'] = n_mol
                idx += 1

                # modified_symbol is the symbol with atom index
                modified_symbol = n_mol.atom_symbol(i)
                # When there are ghost atoms, hack the electronic mole, to remove the
                # 'GHOST-' or 'X-' prefix of the quantum nuclear ghost atom, such that
                # the DFT grid code will be tricked into generating grids with radius
                # corresponding to that actual element, but not for charge 0 ghost atom.
                # FIXME: A dirty hack. Need to let the upstream have a nicer interface
                # to change the grid radius instead of this brutal force hack.
                for j in range(self.natm):
                    # Remove ghost prefix but do not remove digit because we need
                    # the atom index from user manual input
                    symb = self.components['e'].atom_symbol(j)
                    is_ghost = False
                    if len(symb) > 2 and symb[0].upper() == 'X' and symb[:2].upper() != 'XE':
                        symb = symb[2:]  # Remove the prefix 'X-'
                        is_ghost = True
                    elif len(symb) > 6 and symb[:5].upper() == 'GHOST':
                        symb = symb[6:]  # Remove the prefix 'GHOST-'
                        is_ghost = True
                    if is_ghost and modified_symbol.upper() == symb.upper():
                        # This ghost atom is indeed specifically for this quantum nucleus
                        modified_atom = list(self.components['e']._atom[j])
                        modified_atom[0] = modified_symbol
                        self.components['e']._atom[j] = tuple(modified_atom)

    def _build_electronic_mol(self, q_nuc_occ=None, **kwargs):
        '''Build electronic component including fractional occupations'''
        e_mol = gto.Mole()
        e_mol.build(**kwargs)
        quantum_nuc_charge = 0

        # Handle fractional occupations
        if q_nuc_occ is not None:
            q_nuc_occ = numpy.asarray(q_nuc_occ)
            if q_nuc_occ.size != self._n_quantum_nuc:
                raise ValueError('q_nuc_occ must match the dimension of quantum_nuc')
            unocc = numpy.ones_like(q_nuc_occ) - q_nuc_occ
            unocc_Z = 0
            idx = 0

            # Set all quantum nuclei to have zero charges
            for i in range(self.natm):
                if self._quantum_nuc[i]:
                    charge = e_mol._atm[i, gto.CHARGE_OF]
                    quantum_nuc_charge -= charge
                    unocc_Z += unocc[idx] * charge
                    idx += 1
                    # Set nuclear charges of quantum nuclei to 0
                    e_mol._atm[i, gto.CHARGE_OF] = 0
                    # Also label this nucleus as CNEO quantum nucleus
                    e_mol._atm[i, gto.NUC_MOD_OF] = gto.NUC_CNEO

            # if e_mol._enuc is already evaluated, need to update it
            # because quantum nuclei just get changed to zero charges
            if e_mol._enuc is not None:
                e_mol._enuc = e_mol.energy_nuc()

            # Charge determines the number of electrons
            # Remove excessive electrons to make the system neutral
            e_mol.charge += quantum_nuc_charge + numpy.floor(unocc_Z)
            e_mol.nhomo = 1.0 - (unocc_Z - numpy.floor(unocc_Z))
        else:
            # Set all quantum nuclei to have zero charges
            for i in range(self.natm):
                if self._quantum_nuc[i]:
                    charge = e_mol._atm[i, gto.CHARGE_OF]
                    quantum_nuc_charge -= charge
                    # Set nuclear charges of quantum nuclei to 0
                    e_mol._atm[i, gto.CHARGE_OF] = 0
                    # Also label this nucleus as CNEO quantum nucleus
                    e_mol._atm[i, gto.NUC_MOD_OF] = gto.NUC_CNEO
            # Charge determines the number of electrons
            e_mol.charge += quantum_nuc_charge
            e_mol.nhomo = None

        e_mol.super_mol = self # proper super_mol linking
        e_mol._keys.update(['super_mol', 'nhomo'])
        return e_mol

    def _build_positron_mol(self, q_nuc_occ=None, **kwargs):
        '''Build positron component Mole'''
        p_mol = gto.Mole()
        p_mol.build(**kwargs)
        p_mol.charge = self.positron_charge
        p_mol.spin = self.positron_spin

        quantum_nuc_charge = 0
        if q_nuc_occ is not None:
            raise NotImplementedError
        else:
            # Set all quantum nuclei to have zero charges
            for i in range(self.natm):
                if self._quantum_nuc[i]:
                    charge = p_mol._atm[i, gto.CHARGE_OF]
                    quantum_nuc_charge -= charge
                    # Set nuclear charges of quantum nuclei to 0
                    p_mol._atm[i, gto.CHARGE_OF] = 0
                    # Also label this nucleus as CNEO quantum nucleus
                    p_mol._atm[i, gto.NUC_MOD_OF] = gto.NUC_CNEO
            # Charge determines the number of electrons
            p_mol.charge += quantum_nuc_charge
            p_mol.nhomo = None

        p_mol.super_mol = self # proper super_mol linking
        p_mol._keys.update(['super_mol', 'nhomo'])
        return p_mol

    def _build_nuclear_mol(self, atom_id, frac=None):
        '''Build nuclear component Mole'''
        n_mol = gto.Mole()

        # Automatically label quantum nuclei to prevent spawning multiple basis
        # functions at different positions
        modified_symbol = self.atom_symbol(atom_id) + str(atom_id)
        modified_atom = cp.deepcopy(self._atom)
        modified_atom[atom_id] = list(modified_atom[atom_id])
        modified_atom[atom_id][0] = modified_symbol
        modified_atom[atom_id] = tuple(modified_atom[atom_id])

        basis = self._get_nuclear_basis(atom_id)

        # suppress "Warning: Basis not found for atom" in gto/mole.py
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stderr(devnull):
                n_mol.build(basis={modified_symbol: basis},
                            dump_input=False, parse_arg=False, verbose=self.verbose,
                            output=self.output, max_memory=self.max_memory,
                            atom=modified_atom, unit='bohr', nucmod=self.nucmod,
                            ecp=self.ecp, charge=self.charge, spin=self.spin,
                            symmetry=self.symmetry,
                            symmetry_subgroup=self.symmetry_subgroup,
                            cart=self.cart, magmom=self.magmom)

        # Set all quantum nuclei to have zero charges
        for i in range(self.natm):
            if self._quantum_nuc[i]:
                # Set nuclear charges of quantum nuclei to 0
                n_mol._atm[i, gto.CHARGE_OF] = 0
                # Also label this nucleus as CNEO quantum nucleus
                n_mol._atm[i, gto.NUC_MOD_OF] = gto.NUC_CNEO

        # Avoid UHF. This calls nelec.setter, which modifies _nelectron and spin
        n_mol.nelec = (1,1)

        # Set fractional occupation if needed
        if frac is not None:
            n_mol.nnuc = frac
        else:
            n_mol.nnuc = 1.0

        n_mol.super_mol = self
        n_mol.atom_index = atom_id
        n_mol._keys.update(['super_mol', 'atom_index', 'nnuc'])
        return n_mol

    def _get_nuclear_basis(self, atom_id):
        '''Get basis set for quantum nucleus

        Nuclear basis:

        H: PB4-D  J. Chem. Phys. 152, 244123 (2020)
        D: scaled PB4-D
        other atoms: 12s12p12d, alpha=2*sqrt(2)*mass, beta=sqrt(3)
        '''
        # For hydrogen and its isotopes, try reading from basis file
        if self.atom_pure_symbol(atom_id) == 'H':
            dirnow = os.path.realpath(os.path.join(__file__, '..'))
            basis = self._try_read_nuclear_basis(dirnow, atom_id)
            if basis is not None:
                return basis

        # If basis not found for hydrogen, and for heavier elements,
        # generate even-tempered basis
        return self._build_even_tempered_nuclear_basis(atom_id)

    def _try_read_nuclear_basis(self, dirnow, atom_id):
        '''Try reading nuclear basis from file with proper isotope scaling'''
        # Try original basis name
        try:
            basis = self._read_basis_file(dirnow, self.nuclear_basis, atom_id)
            if basis is not None:
                return basis
        except FileNotFoundError:
            pass

        # Try lower case basis name
        try:
            basis = self._read_basis_file(dirnow, self.nuclear_basis.lower(), atom_id)
            if basis is not None:
                return basis
        except FileNotFoundError:
            pass

        # Try basis name without dashes/underscores
        try:
            clean_name = self.nuclear_basis.replace('-','').replace('_','')
            basis = self._read_basis_file(dirnow, clean_name, atom_id)
            if basis is not None:
                return basis
        except FileNotFoundError:
            pass

        # Try lower case basis name without dashes/underscores
        try:
            clean_name = self.nuclear_basis.lower().replace('-','').replace('_','')
            basis = self._read_basis_file(dirnow, clean_name, atom_id)
            if basis is not None:
                return basis
        except FileNotFoundError:
            return None

    def _read_basis_file(self, dirnow, basis_name, atom_id):
        '''Read and scale nuclear basis from file'''
        with open(os.path.join(dirnow, f'basis/{basis_name}.dat'), 'r') as f:
            basis = gto.basis.parse(f.read())
            ratio = 1.0
            if 'H+' in self.atom_symbol(atom_id): # H+ for deuterium
                ratio = numpy.sqrt((2.01410177811 - nist.E_MASS/nist.ATOMIC_MASS)
                                   / (1.007825 - nist.E_MASS/nist.ATOMIC_MASS)).item()
            elif 'H*' in self.atom_symbol(atom_id): # H* for muonium
                ratio = numpy.sqrt(0.1134289259 / (1.007825  - nist.E_MASS / nist.ATOMIC_MASS)).item()
            elif 'H#' in self.atom_symbol(atom_id): # H# for HeMu
                ratio = numpy.sqrt((4.002603254 - 2 * nist.E_MASS / nist.ATOMIC_MASS + 0.1134289259)
                                   / (1.007825  - nist.E_MASS / nist.ATOMIC_MASS)).item()
            if ratio != 1.0:
                for x in basis:
                    x[1][0] *= ratio
            return basis

    def _build_even_tempered_nuclear_basis(self, atom_id):
        '''Build default even-tempered basis for nucleus'''
        if self.atom_pure_symbol(atom_id) == 'H':
            if 'H+' in self.atom_symbol(atom_id):
                alpha = 4.0
            else:
                alpha = 2 * numpy.sqrt(2)
            beta = numpy.sqrt(2)
            n = -1
        else:
            # Probably should have been sqrt(mass) but this is in the paper
            alpha = 2 * numpy.sqrt(2) * self.mass[atom_id]
            beta = numpy.sqrt(3)
            n = 12
        m = re.search(r"(\d+)s(\d+)p(\d+)d(\d+)?f?", self.nuclear_basis)
        if m:
            if m.group(4) is None:
                basis = gto.expand_etbs([(0, int(m.group(1)), alpha, beta),
                                         (1, int(m.group(2)), alpha, beta),
                                         (2, int(m.group(3)), alpha, beta)])
            else:
                basis = gto.expand_etbs([(0, int(m.group(1)), alpha, beta),
                                         (1, int(m.group(2)), alpha, beta),
                                         (2, int(m.group(3)), alpha, beta),
                                         (3, int(m.group(4)), alpha, beta)])
            return basis
        elif n > 0:
            return gto.expand_etbs([(0, n, alpha, beta),
                                    (1, n, alpha, beta),
                                    (2, n, alpha, beta)])
        else:
            raise ValueError(f'Unsupported nuclear basis {self.nuclear_basis}')

    copy = copy

    def set_geom_(self, atoms_or_coords, unit=None, symmetry=None,
                  inplace=True):
        '''Update geometry
        '''
        if self.components.get('p') is not None:
            raise NotImplementedError

        if inplace:
            mol = self
        else:
            mol = self.copy(deep=False)
            mol._env = mol._env.copy()
            mol.components = {}

        # First set_geom_ for e component
        # Use default charge, as gto.Mole.build may complain about spin
        charge = self.components['e'].charge
        self.components['e'].charge = mol.charge
        mol.components['e'] = self.components['e'].set_geom_(atoms_or_coords, unit=unit,
                                                             symmetry=False,
                                                             inplace=inplace)
        mol.components['e'].charge = self.components['e'].charge = charge

        # Must relink back to mol in case inplace=False, otherwise
        # it will point back to ``self'' here
        mol.components['e'].super_mol = mol
        # ensure correct core charge in case got elec mole rebuilt
        for i in range(mol.natm):
            if mol._quantum_nuc[i]:
                mol.components['e']._atm[i, gto.CHARGE_OF] = 0
                mol.components['e']._atm[i, gto.NUC_MOD_OF] = gto.NUC_CNEO

        for i in range(mol.natm):
            if mol._quantum_nuc[i]:
                n_mol = self.components[f'n{i}']
                charge = n_mol.charge
                n_mol.charge = mol.charge
                modified_symbol = mol.components['e'].atom_symbol(i) + str(i)
                modified_atom = cp.deepcopy(mol.components['e']._atom)
                modified_atom[i] = list(modified_atom[i])
                modified_atom[i][0] = modified_symbol
                modified_atom[i] = tuple(modified_atom[i])
                # In this way, nuc mole must get rebuilt.
                # It is possible to pass a numpy.ndarray such that no rebuild
                # is needed, but in rare cases even numpy.ndarray can trigger
                # a rebuild.
                # In that case, nuc mole will again lose basis information.
                # Therefore, here we choose a way to ensure nuclear basis is
                # correctly assigned (and no duplication).
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stderr(devnull):
                        mol.components[f'n{i}'] = n_mol.set_geom_(modified_atom,
                                                                  unit='bohr',
                                                                  symmetry=False,
                                                                  inplace=inplace)
                mol.components[f'n{i}'].charge = n_mol.charge = charge

                # must relink back to mol in case inplace=False, otherwise
                # it will point back to ``self'' here
                mol.components[f'n{i}'].super_mol = mol
                for j in range(mol.natm):
                    # ensure correct core charge, because got rebuilt
                    if mol._quantum_nuc[j]:
                        mol.components[f'n{i}']._atm[j, gto.CHARGE_OF] = 0
                        mol.components[f'n{i}']._atm[i, gto.NUC_MOD_OF] = gto.NUC_CNEO
                mol.components[f'n{i}'].nelec = (1,1)

        # When there are ghost atoms, hack the electronic mole, to remove the
        # 'GHOST-' or 'X-' prefix of the quantum nuclear ghost atom, such that
        # the DFT grid code will be tricked into generating grids with radius
        # corresponding to that actual element, but not for charge 0 ghost atom.
        # FIXME: A dirty hack. Need to let the upstream have a nicer interface
        # to change the grid radius instead of this brutal force hack.
        for j in range(mol.natm):
            # Remove ghost prefix but do not remove digit because we need
            # the atom index from user manual input
            symb = mol.components['e'].atom_symbol(j)
            is_ghost = False
            if len(symb) > 2 and symb[0].upper() == 'X' and symb[:2].upper() != 'XE':
                symb = symb[2:]  # Remove the prefix 'X-'
                is_ghost = True
            elif len(symb) > 6 and symb[:5].upper() == 'GHOST':
                symb = symb[6:]  # Remove the prefix 'GHOST-'
                is_ghost = True
            if is_ghost:
                for i in range(mol.natm):
                    if mol._quantum_nuc[i]:
                        modified_symbol = mol.components[f'n{i}'].atom_symbol(i)
                        if modified_symbol.upper() == symb.upper():
                            # This ghost atom is indeed specifically for this quantum nucleus
                            modified_atom = list(mol.components['e']._atom[j])
                            modified_atom[0] = modified_symbol
                            mol.components['e']._atom[j] = tuple(modified_atom)

        # then set_geom_ for the base mole
        # copied from gto.mole.Mole.set_geom_
        if unit is None:
            _unit = mol.unit
        else:
            _unit = _length_in_au(unit)
            if _unit != _length_in_au(self.unit):
                logger.warn(mol, 'Mole.unit (%s) is changed to %s', self.unit, unit)
                mol.unit = unit
        if symmetry is None:
            symmetry = mol.symmetry

        if isinstance(atoms_or_coords, numpy.ndarray):
            mol.atom = list(zip([x[0] for x in mol._atom],
                                atoms_or_coords.tolist()))
        else:
            mol.atom = atoms_or_coords

        if isinstance(atoms_or_coords, numpy.ndarray) and not symmetry:
            _unit = _length_in_au(mol.unit)
            mol._atom = list(zip([x[0] for x in mol._atom],
                                 (atoms_or_coords * _unit).tolist()))
            ptr = mol._atm[:, gto.PTR_COORD]
            mol._env[ptr+0] = _unit * atoms_or_coords[:,0]
            mol._env[ptr+1] = _unit * atoms_or_coords[:,1]
            mol._env[ptr+2] = _unit * atoms_or_coords[:,2]
        else:
            mol.symmetry = False
            gto.Mole.build(mol, dump_input=False, parse_arg=False)
            mol.symmetry = symmetry
            if mol.symmetry:
                mol._build_symmetry_mc()

        if mol.verbose >= logger.INFO:
            logger.info(mol, 'New geometry')
            for ia, atom in enumerate(mol._atom):
                coorda = tuple([x * param.BOHR for x in atom[1]])
                coordb = tuple([x for x in atom[1]])
                coords = coorda + coordb
                logger.info(mol, ' %3d %-4s %16.12f %16.12f %16.12f AA  '
                            '%16.12f %16.12f %16.12f Bohr',
                            ia+1, mol.atom_symbol(ia), *coords)
        return mol
