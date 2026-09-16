#!/usr/bin/env python

'''
Constrained nuclear-electronic orbital density functional theory
'''

import numpy
import scipy.optimize
from pyscf.data import nist
from pyscf.lib import logger
from pyscf.neo import ks

def _get_mo_energy_coeff_occ(mf, fock, s1e):
    mo_energy, mo_coeff = mf.eig(fock, s1e)
    verbose = mf.verbose
    nnuc = mf.mol.nnuc
    try:
        # Temporarily disable the verbose output in get_occ
        mf.verbose = 0
        # Temporarily set nnuc=1 for expectation position constraint
        # and restore nnuc before returning to the physical SCF density.
        mf.mol.nnuc = 1.0
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
    finally:
        mf.mol.nnuc = nnuc
        mf.verbose = verbose
    return mo_energy, mo_coeff, mo_occ

def analytic_position_jacobian(mo_energy, mo_coeff, mo_occ, int1e_r,
                               gap_tol=1e-14):
    '''
    Frozen-Fock derivative of the position constraint with respect to f.
    '''
    mo_energy = numpy.asarray(mo_energy)
    mo_coeff = numpy.asarray(mo_coeff)
    mo_occ = numpy.asarray(mo_occ)
    int1e_r = numpy.asarray(int1e_r)

    occidx = mo_occ > 0
    viridx = mo_occ == 0
    nocc = numpy.count_nonzero(occidx)
    if nocc != 1:
        raise RuntimeError(
            'Analytic CNEO position Jacobian requires exactly one occupied '
            f'nuclear orbital; found {nocc}.'
        )

    e_a = mo_energy[viridx]
    e_i = mo_energy[occidx]
    e_ai = e_a[:,None] - e_i
    if numpy.any(numpy.abs(e_ai) <= gap_tol):
        raise numpy.linalg.LinAlgError(
            'The occupied nuclear orbital is degenerate or nearly degenerate '
            'with another orbital; the analytic position Jacobian is not '
            'valid.'
        )
    e_ai = 1 / e_ai

    orbo = mo_coeff[:,occidx]
    orbv = mo_coeff[:,viridx]
    coupling = numpy.matmul(numpy.matmul(orbo.conj().T, int1e_r), orbv)[:,0]
    occupation = mo_occ[occidx][0]
    jacobian = -2.0 * occupation * numpy.real((coupling * e_ai.T) @ coupling.conj().T)
    return 0.5 * (jacobian + jacobian.T)

def _position_deviation(mf, mo_coeff, mo_occ, position_matrices=None):
    if position_matrices is None:
        position_matrices = mf.int1e_r
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    return numpy.einsum('xij,ji->x', position_matrices, dm).real

def _get_constraint_symmetry(mf, mo_coeff, mo_occ):
    mocc = mo_coeff[:,mo_occ>0]
    assert mocc.shape[1] == 1 # singly occupied
    orbsym = mo_coeff.orbsym[mo_occ>0][0]
    # For this symmetry, test SO matrix
    symm_orb = mf.mol.symm_orb
    irrep_id = mf.mol.irrep_id
    nirrep = symm_orb.__len__()
    occupied_symm_orb = None
    for ir in range(nirrep):
        if irrep_id[ir] == orbsym:
            occupied_symm_orb = symm_orb[ir]
    important_axes = []
    if occupied_symm_orb is not None:
        for idx, int1e_x in enumerate(mf.int1e_r_symm):
            int1e_x_so = numpy.matmul(numpy.matmul(occupied_symm_orb.conj().T, int1e_x), occupied_symm_orb)
            if numpy.abs(int1e_x_so).max() > 1e-12: # NOTE: can adjust 1e-12
                important_axes.append(idx)
    if len(important_axes) == 0:
        logger.warn(mf, 'No important symmetry axes found! Fallback to no symm')
        important_axes = [x for x in range(mf.int1e_r.shape[0])]
    return important_axes, mf.int1e_r_symm[important_axes]

def get_position_error(mf, fock, s1e):
    '''Return concatenated position-constraint errors for quantum nuclei.'''
    deviations = []
    for t in sorted(mf.components):
        if not t.startswith('n'):
            continue

        comp = mf.components[t]

        _, mo_coeff, mo_occ = _get_mo_energy_coeff_occ(comp, fock[t], s1e[t])

        if comp.int1e_r_symm is not None:
            important_axes, position_matrices = _get_constraint_symmetry(comp, mo_coeff, mo_occ)
            deviation = _position_deviation(comp, mo_coeff, mo_occ, position_matrices)
            deviation_full = numpy.zeros(comp.int1e_r.shape[0])
            for i, idx in enumerate(important_axes):
                deviation_full[idx] = deviation[i]
            deviation = comp.mol._symm_axes.T @ deviation_full
        else:
            deviation = _position_deviation(comp, mo_coeff, mo_occ)

        deviations.append(deviation)

    return numpy.concatenate(deviations)

def update_lagrange_multipliers(mf, fock0, s1e, one_step=False, tol=1e-15,
                                gap_tol=1e-14, minimum_step=0.01):
    '''Update the CNEO Lagrange multipliers.'''
    if not one_step:
        for t, comp in mf.components.items():
            if t.startswith('n'):
                ia = comp.mol.atom_index
                opt = solve_constraint(comp, fock0[t], s1e[t], mf.f[ia])
                mf.f[ia] = opt.x
                if opt.success:
                    logger.debug(mf, 'CNEO NUC constraint optimization succeeded.')
                    logger.debug(mf, 'Lagrange multiplier of %s(%i) atom: %s' %
                                 (mf.mol.atom_symbol(ia), ia, mf.f[ia]))
                    logger.debug(mf, 'Position deviation: %s', opt.fun)
                else:
                    logger.warn(mf, 'CNEO NUC constraint optimization failed!')
                    logger.warn(mf, 'scipy.optimize.least_squares message: %s',
                                opt.message)
                    logger.warn(mf, 'Lagrange multiplier of %s(%i) atom: %s' %
                                (mf.mol.atom_symbol(ia), ia, mf.f[ia]))
                    logger.warn(mf, 'Position deviation: %s', opt.fun)
        return

    deviations = []
    for t in sorted(mf.components):
        if not t.startswith('n'):
            continue

        comp = mf.components[t]
        ia = comp.mol.atom_index

        f_lagrange = numpy.asarray(mf.f[ia], dtype=float).copy()
        initial_orbitals = None

        if comp.int1e_r_symm is not None:
            # Detect the occupied nuclear orbital symmetry using the current f.
            fock = fock0[t] + numpy.einsum('xij,x->ij', comp.int1e_r, f_lagrange)

            mo_energy, mo_coeff, mo_occ = _get_mo_energy_coeff_occ(comp, fock, s1e[t])

            important_axes, position_matrices = _get_constraint_symmetry(comp, mo_coeff, mo_occ)

            # Cartesian -> symmetry-axis coordinates, then remove forbidden axes.
            f_lagrange = comp.mol._symm_axes @ f_lagrange
            f_lagrange = f_lagrange[important_axes]

            initial_fock = fock0[t] + numpy.einsum('xij,x->ij', position_matrices, f_lagrange)
            if numpy.allclose(fock, initial_fock, rtol=0.0, atol=1e-12):
                initial_orbitals = mo_energy, mo_coeff, mo_occ

        else:
            important_axes = None
            position_matrices = comp.int1e_r

        def residual(f_lagrange):
            fock = fock0[t] + numpy.einsum('xij,x->ij', position_matrices,
                                           f_lagrange)

            _, mo_coeff, mo_occ = _get_mo_energy_coeff_occ(comp, fock, s1e[t])

            return _position_deviation(comp, mo_coeff, mo_occ, position_matrices)

        def evaluate(f_lagrange):
            fock = fock0[t] + numpy.einsum('xij,x->ij', position_matrices, f_lagrange)

            mo_energy, mo_coeff, mo_occ = _get_mo_energy_coeff_occ(comp, fock, s1e[t])

            deviation = _position_deviation(comp, mo_coeff, mo_occ, position_matrices)

            return deviation, mo_energy, mo_coeff, mo_occ

        if initial_orbitals is None:
            deviation, mo_energy, mo_coeff, mo_occ = evaluate(f_lagrange)
        else:
            mo_energy, mo_coeff, mo_occ = initial_orbitals
            deviation = _position_deviation(comp, mo_coeff, mo_occ, position_matrices)

        if numpy.max(numpy.abs(deviation)) >= tol:

            try:
                jacobian = analytic_position_jacobian(mo_energy, mo_coeff, mo_occ, position_matrices, gap_tol)
            except (RuntimeError, numpy.linalg.LinAlgError) as err:
                logger.warn(mf, '%s Falling back to numerical position Jacobian.', err)
                displacements = numpy.eye(f_lagrange.size) * 1e-3
                jacobian = numpy.column_stack([
                    (residual(f_lagrange + displacement) - deviation) / 1e-3
                    for displacement in displacements
                ])

            deviation_norm = numpy.linalg.norm(deviation)
            try:
                step_direction = numpy.linalg.solve(jacobian, -deviation)
            except numpy.linalg.LinAlgError:
                step_direction = numpy.linalg.lstsq(jacobian, -deviation, rcond=gap_tol)[0]

            direction_norm = numpy.linalg.norm(step_direction)
            if direction_norm == 0.0 or not numpy.isfinite(direction_norm):
                logger.warn(mf, 'Invalid CNEO constraint Newton step for %s; '
                            'keeping the previous Lagrange multiplier', t)
            else:
                step_size = 1.0
                f_trial = f_lagrange + step_direction
                trial_deviation = residual(f_trial)
                trial_norm = numpy.linalg.norm(trial_deviation)

                while trial_norm >= deviation_norm:
                    if step_size < minimum_step:
                        break

                    slope = -deviation_norm / (step_size * direction_norm)
                    denominator = 2.0 * (trial_norm - deviation_norm - slope)
                    if denominator == 0.0 or not numpy.isfinite(denominator):
                        step_size *= 0.5
                    else:
                        step_size *= max(-slope / denominator, 0.1)

                    f_trial = f_lagrange + step_size * step_direction
                    trial_deviation = residual(f_trial)
                    trial_norm = numpy.linalg.norm(trial_deviation)

                if trial_norm < deviation_norm:
                    f_lagrange = f_trial
                    deviation = trial_deviation
                else:
                    logger.debug(mf, 'CNEO constraint line search failed for %s; '
                                 'keeping the previous Lagrange multiplier', t)

        if important_axes is not None:
            f_lagrange_full = numpy.zeros(comp.int1e_r.shape[0])
            deviation_full = numpy.zeros(comp.int1e_r.shape[0])

            for i, idx in enumerate(important_axes):
                f_lagrange_full[idx] = f_lagrange[i]
                deviation_full[idx] = deviation[i]

            mf.f[ia] = comp.mol._symm_axes.T @ f_lagrange_full
            deviation = comp.mol._symm_axes.T @ deviation_full

        else:
            mf.f[ia] = f_lagrange

        deviations.append(deviation)
    return numpy.concatenate(deviations)


def solve_constraint(mf, fock0, s1e=None, f_lagrange_guess=None,
                     jacobian_gap_tol=1e-14):
    '''Solve the Kohn-Sham equation with position constraint
        [H + f_lagrange * (r - R)] y = e y, <y|r - R|y> = 0.
    '''
    if s1e is None:
        s1e = mf.get_ovlp()
    if f_lagrange_guess is None:
        f_lagrange_guess = numpy.zeros(mf.int1e_r.shape[0])

    initial_orbitals = None
    if mf.int1e_r_symm is not None:
        # Detect ground state symmetry with fock0 and f guess.
        # This symmetry detection mostly relies on the zero guess of f, then
        # the symmetry is unlikely to be changed in following steps.
        # It is possible that the true ground state has a different symmetry,
        # but how to detect that? This is a global optimization problem.
        # TODO: may result in wrong symmetry with bad f guess, how to improve?
        fock = fock0 + numpy.einsum('xij,x->ij', mf.int1e_r, f_lagrange_guess)
        mo_energy, mo_coeff, mo_occ = _get_mo_energy_coeff_occ(mf, fock, s1e)
        important_axes, position_matrices = _get_constraint_symmetry(mf, mo_coeff, mo_occ)

        # Transform to along symmetry axes
        f_lagrange_guess = mf.mol._symm_axes @ f_lagrange_guess
        # Only keep the axes with non-trivial contributions
        f_lagrange_guess = f_lagrange_guess[important_axes]

        initial_fock = fock0 + numpy.einsum('xij,x->ij', position_matrices, f_lagrange_guess)
        if numpy.allclose(fock, initial_fock, rtol=0.0, atol=1e-12):
            initial_orbitals = mo_energy, mo_coeff, mo_occ

    else:
        important_axes = None
        position_matrices = mf.int1e_r

    cache = {'f_lagrange': None, 'deviation': None, 'jacobian': None,
             'orbitals': None}
    if initial_orbitals is not None:
        mo_energy, mo_coeff, mo_occ = initial_orbitals
        cache['f_lagrange'] = numpy.asarray(f_lagrange_guess).copy()
        cache['deviation'] = _position_deviation(mf, mo_coeff, mo_occ, position_matrices)
        cache['orbitals'] = initial_orbitals

    jacobian_failure_point = None

    def evaluate(f_lagrange):
        f_lagrange = numpy.asarray(f_lagrange)
        if (cache['f_lagrange'] is not None and
                numpy.array_equal(f_lagrange, cache['f_lagrange'])):
            return cache['deviation']
        # Get Fock matrix with constraint
        fock = fock0 + numpy.einsum('xij,x->ij', position_matrices, f_lagrange)
        # Calculate expectation position deviation
        mo_energy, mo_coeff, mo_occ = _get_mo_energy_coeff_occ(mf, fock, s1e)
        deviation = _position_deviation(mf, mo_coeff, mo_occ, position_matrices)
        cache['f_lagrange'] = f_lagrange.copy()
        cache['deviation'] = deviation
        cache['jacobian'] = None
        cache['orbitals'] = mo_energy, mo_coeff, mo_occ
        return deviation

    def position_deviation(f_lagrange):
        '''Calculate position deviation from the Kohn-Sham orbital with
        frozen unconstrained NEO Fock and provided Lagrange multiplier'''
        return evaluate(f_lagrange)

    def position_jacobian(f_lagrange):
        nonlocal jacobian_failure_point
        jacobian_failure_point = None
        deviation = evaluate(f_lagrange)
        # SciPy requests Jacobians only for accepted trials. Reuse their orbitals.
        if cache['jacobian'] is None:
            mo_energy, mo_coeff, mo_occ = cache['orbitals']
            try:
                cache['jacobian'] = analytic_position_jacobian(
                    mo_energy, mo_coeff, mo_occ, position_matrices, jacobian_gap_tol)
            except (RuntimeError, numpy.linalg.LinAlgError):
                f_lagrange = numpy.asarray(f_lagrange)
                if numpy.all(numpy.isfinite(f_lagrange)) and numpy.all(numpy.isfinite(deviation)):
                    jacobian_failure_point = f_lagrange.copy()
                raise
        return cache['jacobian']

    def position_deviation_numeric(f_lagrange):
        fock = fock0 + numpy.einsum('xij,x->ij', position_matrices, f_lagrange)
        _, mo_coeff, mo_occ = _get_mo_energy_coeff_occ(mf, fock, s1e)
        return _position_deviation(mf, mo_coeff, mo_occ, position_matrices)

    #opt = scipy.optimize.root(position_deviation, f_lagrange_guess, method='hybr')
    try:
        opt = scipy.optimize.least_squares(position_deviation, f_lagrange_guess,
                                           jac=position_jacobian, gtol=1e-15)
    except (RuntimeError, numpy.linalg.LinAlgError) as err:
        logger.warn(mf, '%s Falling back to numerical position Jacobian.', err)
        if jacobian_failure_point is None:
            jacobian_failure_point = f_lagrange_guess
        opt = scipy.optimize.least_squares(position_deviation_numeric, jacobian_failure_point, gtol=1e-15)

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
