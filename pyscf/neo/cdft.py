#!/usr/bin/env python

'''
Constrained nuclear-electronic orbital density functional theory
'''

import numpy
from pyscf.data import nist
from pyscf.lib import logger
from pyscf.neo import ks

class CDFT(ks.KS):
    '''
    Examples::

    >>> from pyscf import neo
    >>> mol = neo.Mole()
    >>> mol.build(atom='H 0.0 0.0 0.0; C 0.0 0.0 1.064; N 0.0 0.0 2.220',
    >>>           quantum_nuc=[0], basis='ccpvdz', nuc_basis='pb4d')
    >>> mf = neo.CDFT(mol, xc='b3lyp5')
    >>> mf.scf()
    -93.33840234527442
    '''

    def __init__(self, mol, *args, **kwargs):
        super().__init__(mol, *args, **kwargs)
        self.f = numpy.zeros((mol.natm, 3))
        self._setup_position_matrices()

    def _setup_position_matrices(self):
        '''Set up position matrices for each quantum nucleus for constraint'''
        for t, comp in self.components.items():
            if t.startswith('n'):
                comp.nuclear_expect_position = comp.mol.atom_coord(comp.mol.atom_index)
                # Position matrix with origin shifted to nuclear expectation position
                s1e = comp.get_ovlp()
                comp.int1e_r = comp.mol.intor_symmetric('int1e_r', comp=3) \
                             - numpy.asarray([comp.nuclear_expect_position[i] * s1e for i in range(3)])

    def get_fock_add_cdft(self):
        '''Get additional Fock terms from constraints'''
        f_add = {}
        for t, comp in self.components.items():
            if t.startswith('n'):
                ia = comp.mol.atom_index
                f_add[t] = numpy.einsum('xij,x->ij', comp.int1e_r, self.f[ia])
        return f_add

    def position_analysis(self, f, comp, fock0, s1e=None):
        '''Analyze nuclear position constraints for optimization

        Args:
            f : ndarray
                Current Lagrange multipliers for the nucleus
            comp : Component object
                Nuclear component being analyzed
            fock0 : ndarray
                Base Fock matrix without constraint terms
            s1e : ndarray, optional
                Overlap matrix. Will be computed if not provided.

        Returns:
            ndarray : Position deviation from expectation value
        '''
        ia = comp.mol.atom_index
        self.f[ia] = f
        if s1e is None:
            s1e = comp.get_ovlp()

        # Get Fock matrix with constraint
        fock = fock0 + numpy.einsum('xij,x->ij', comp.int1e_r, f)

        # Calculate expectation value
        mo_energy, mo_coeff = comp.eig(fock, s1e)
        mo_occ = comp.get_occ(mo_energy, mo_coeff)
        dm = comp.make_rdm1(mo_coeff, mo_occ)

        # Return deviation from expected position
        return numpy.einsum('xij,ji->x', comp.int1e_r, dm)

    def dip_moment(self, mol=None, dm=None, unit='Debye', origin=None,
                   verbose=logger.NOTE, **kwargs):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        log = logger.new_logger(mol, verbose)

        el_dip = self.components['e'].dip_moment(mol.components['e'],
                                                 dm['e'], unit=unit,
                                                 origin=origin, verbose=verbose-1)
        # Quantum nuclei
        if origin is None:
            origin = numpy.zeros(3)
        else:
            origin = numpy.asarray(origin, dtype=numpy.float64)
        assert origin.shape == (3,)
        nucl_dip = 0
        for t, comp in self.components.items():
            if t.startswith('n'):
                nucl_dip -= comp.charge * (comp.nuclear_expect_position - origin)
        if unit.upper() == 'DEBYE':
            nucl_dip *= nist.AU2DEBYE
            mol_dip = nucl_dip + el_dip
            log.note('Dipole moment(X, Y, Z, Debye): %8.5f, %8.5f, %8.5f', *mol_dip)
        else:
            mol_dip = nucl_dip + el_dip
            log.note('Dipole moment(X, Y, Z, A.U.): %8.5f, %8.5f, %8.5f', *mol_dip)
        return mol_dip

    def reset(self, mol=None):
        '''Reset mol and relevant attributes associated to the old mol object'''
        super().reset(mol=mol)
        self.f = numpy.zeros((self.mol.natm, 3))
        self._setup_position_matrices()
        return self

    def nuc_grad_method(self):
        from pyscf.neo import grad
        return grad.Gradients(self)

if __name__ == '__main__':
    from pyscf import neo
    mol = neo.M(atom='H 0 0 0; H 0 0 0.74', basis='ccpvdz', nuc_basis='pb4d', verbose=5)
    mf = neo.CDFT(mol, xc='PBE', epc='17-2')
    mf.scf()
