#!/usr/bin/env python

'''
Constrained nuclear-electronic orbital density functional theory
'''

import numpy
import scipy.optimize
from pyscf import symm
from pyscf.data import nist
from pyscf.lib import logger
from pyscf.neo import ks

def _get_mo_coeff_occ(mf, fock, s1e):
    mo_energy, mo_coeff = mf.eig(fock, s1e)
    # Temporarily disable the verbose output in get_occ
    verbose = mf.verbose
    mf.verbose = 0
    mo_occ = mf.get_occ(mo_energy, mo_coeff)
    mf.verbose = verbose
    return mo_coeff, mo_occ

def solve_constraint(mf, fock0, s1e=None, f_lagrange_guess=None):
    '''Solve the Kohn-Sham equation with position constraint
        [H + f_lagrange * (r - R)] y = e y, <y|r - R|y> = 0.
    '''
    if s1e is None:
        s1e = mf.get_ovlp()
    if f_lagrange_guess is None:
        f_lagrange_guess = numpy.zeros(mf.int1e_r.shape[0])

    if mf.int1e_r_symm is not None:
        # Detect ground state symmetry with fock0 and f guess.
        # This symmetry detection mostly relies on the zero guess of f, then
        # the symmetry is unlikely to be changed in following steps.
        # It is possible that the true ground state has a different symmetry,
        # but how to detect that? This is a global optimization problem.
        # TODO: may result in wrong symmetry with bad f guess, how to improve?
        fock = fock0 + numpy.einsum('xij,x->ij', mf.int1e_r, f_lagrange_guess)
        mo_coeff, mo_occ = _get_mo_coeff_occ(mf, fock, s1e)
        mocc = mo_coeff[:,mo_occ>0]
        assert mocc.shape[1] == 1 # singly occupied
        orbsym = mo_coeff.orbsym[mo_occ>0][0]
        # For this symmetry, test SO matrix
        symm_orb = mf.mol.symm_orb
        irrep_id = mf.mol.irrep_id
        nirrep = symm_orb.__len__()
        important_axes = []
        for idx, int1e_x in enumerate(mf.int1e_r_symm):
            int1e_x_so = symm.symmetrize_matrix(int1e_x, symm_orb)
            for ir in range(nirrep):
                if irrep_id[ir] == orbsym:
                    if numpy.abs(int1e_x_so[ir]).max() > 1e-12: # NOTE: can adjust 1e-12
                        important_axes.append(idx)
        if len(important_axes) == 0:
            logger.warn(mf, 'No important symmetry axes found! Fallback to no symm')
            important_axes = [x for x in range(mf.int1e_r.shape[0])]
        # Transform to along symmetry axes
        f_lagrange_guess = mf.mol._symm_axes @ f_lagrange_guess
        # Only keep the axes with non-trivial contributions
        f_lagrange_guess = f_lagrange_guess[important_axes]

    def position_deviation(f_lagrange):
        '''Calculate position deviation from the Kohn-Sham orbital with
        frozen unconstrained NEO Fock and provided Lagrange multiplier'''
        # Get Fock matrix with constraint
        if mf.int1e_r_symm is not None:
            fock = fock0 + numpy.einsum('xij,x->ij', mf.int1e_r_symm[important_axes], f_lagrange)
        else:
            fock = fock0 + numpy.einsum('xij,x->ij', mf.int1e_r, f_lagrange)

        # Calculate expectation position deviation
        mo_coeff, mo_occ = _get_mo_coeff_occ(mf, fock, s1e)
        dm = mf.make_rdm1(mo_coeff, mo_occ)
        if mf.int1e_r_symm is not None:
            deviation = numpy.einsum('xij,ji->x', mf.int1e_r_symm[important_axes], dm)
        else:
            deviation = numpy.einsum('xij,ji->x', mf.int1e_r, dm)
        return deviation

    opt = scipy.optimize.root(position_deviation, f_lagrange_guess, method='hybr')

    if mf.int1e_r_symm is not None:
        # Recover the full dimensional f_lagrange
        f_lagrange_full = numpy.zeros(mf.int1e_r.shape[0])
        fun_full = numpy.zeros(mf.int1e_r.shape[0])
        for i, idx in enumerate(important_axes):
            f_lagrange_full[idx] = opt.x[i]
            fun_full[idx] = opt.fun[i]
        # Transform back to original Cartesian coordinate
        opt.x = mf.mol._symm_axes.T @ f_lagrange_full
        opt.fun = mf.mol._symm_axes.T @ fun_full
    return opt

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
                comp.int1e_r_symm = None
                if comp.mol.symmetry and comp.mol._symm_axes is not None:
                    # Transform to along symmetry axes
                    comp.int1e_r_symm = numpy.einsum('xy,yij->xij', comp.mol._symm_axes, comp.int1e_r)


    def get_fock_add_cdft(self):
        '''Get additional Fock terms from constraints'''
        f_add = {}
        for t, comp in self.components.items():
            if t.startswith('n'):
                ia = comp.mol.atom_index
                f_add[t] = numpy.einsum('xij,x->ij', comp.int1e_r, self.f[ia])
        return f_add

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
