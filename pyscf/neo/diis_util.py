"""Position-constraint helpers for simultaneous CNEO DIIS.

This version replaces the finite-difference Jacobian in ``update_f`` with the
analytic frozen-Fock orbital-response Jacobian.  The public interface is kept
compatible with the original implementation on the ``yw732/pyscf`` DIIS
branch, so no changes to the existing ``neo.hf.get_fock`` call sites are
required.
"""

import numpy

from pyscf.lib import logger


def _get_mo_energy_coeff_occ(mf, fock, s1e):
    """Diagonalize once and obtain the occupation without verbose printing."""
    mo_energy, mo_coeff = mf.eig(fock, s1e)
    verbose = mf.verbose
    mf.verbose = 0
    try:
        mo_occ = mf.get_occ(mo_energy, mo_coeff)
    finally:
        mf.verbose = verbose
    return mo_energy, mo_coeff, mo_occ


def _deviation_from_mos(mf, mo_coeff, mo_occ):
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    return numpy.einsum("xij,ji->x", mf.int1e_r, dm).real


def analytic_position_jacobian(
        mo_energy, mo_coeff, mo_occ, int1e_r, gap_tol=1e-10):
    r"""Return the frozen-Fock derivative of position with respect to ``f``.

    For ``H(f) = H0 + sum_y f_y R_y`` and one occupied nuclear orbital ``i``,

    .. math::

        J_{xy} = 2 n_i \operatorname{Re} \sum_{a \ne i}
                 \frac{(R_x)_{ia}(R_y)_{ai}}{\epsilon_i-\epsilon_a}.

    The AO basis and overlap do not depend on the Lagrange multiplier.  PySCF
    eigenvectors are overlap-orthonormal, so transforming ``R`` with ``C`` is
    sufficient for the generalized eigenvalue problem.

    The expression is also valid for a selected non-degenerate excited nuclear
    orbital, although in that case the Jacobian need not be negative definite.
    """
    mo_energy = numpy.asarray(mo_energy)
    mo_coeff = numpy.asarray(mo_coeff)
    mo_occ = numpy.asarray(mo_occ)
    int1e_r = numpy.asarray(int1e_r)

    occupied = numpy.flatnonzero(mo_occ > 0)
    if occupied.size != 1:
        raise RuntimeError(
            "Analytic CNEO position Jacobian requires exactly one occupied "
            f"nuclear orbital; found {occupied.size}."
        )

    occ = int(occupied[0])
    other = numpy.arange(mo_energy.size) != occ
    denominator = mo_energy[occ] - mo_energy[other]
    valid = numpy.abs(denominator) > gap_tol
    if not numpy.all(valid):
        raise numpy.linalg.LinAlgError(
            "The occupied nuclear orbital is degenerate or nearly degenerate "
            "with another orbital; the non-degenerate analytic position "
            "Jacobian is not valid. Reduce f_jacobian_gap_tol only if the "
            "small energy gap is physically safe."
        )

    # R_mo[x,p,q] = C[:,p]^H R[x] C[:,q]
    r_mo = numpy.einsum(
        "pi,xpq,qj->xij",
        mo_coeff.conj(),
        int1e_r,
        mo_coeff,
        optimize=True,
    )
    coupling = r_mo[:, occ, other]
    inverse_gap = 1.0 / denominator
    occupation = float(numpy.real(mo_occ[occ]))

    jacobian = 2.0 * occupation * numpy.real(
        numpy.einsum(
            "xa,ya,a->xy",
            coupling,
            coupling.conj(),
            inverse_gap,
            optimize=True,
        )
    )
    return 0.5 * (jacobian + jacobian.T)


def get_deviation(mf, fock, s1e):
    _, mo_coeff, mo_occ = _get_mo_energy_coeff_occ(mf, fock, s1e)
    return _deviation_from_mos(mf, mo_coeff, mo_occ)


def get_deviation_and_jacobian(mf, fock, s1e, gap_tol=1e-10):
    """Get the position residual and analytic Jacobian from one eig call."""
    mo_energy, mo_coeff, mo_occ = _get_mo_energy_coeff_occ(mf, fock, s1e)
    deviation = _deviation_from_mos(mf, mo_coeff, mo_occ)
    jacobian = analytic_position_jacobian(
        mo_energy,
        mo_coeff,
        mo_occ,
        mf.int1e_r,
        gap_tol=gap_tol,
    )
    return deviation, jacobian


def get_pos_err(mf, fock, s1e):
    deviations = []
    for component_name, comp in mf.components.items():
        if component_name.startswith("n"):
            deviations.append(
                get_deviation(comp, fock[component_name], s1e[component_name])
            )
    return numpy.concatenate(deviations)


def _central_difference_jacobian(residual, f0, step=1e-4):
    """Reference Jacobian used only by the optional validation mode."""
    f0 = numpy.asarray(f0, dtype=float)
    r0 = numpy.asarray(residual(f0))
    jacobian = numpy.empty((r0.size, f0.size))
    for axis in range(f0.size):
        h = step * max(1.0, abs(f0[axis]))
        f_plus = f0.copy()
        f_minus = f0.copy()
        f_plus[axis] += h
        f_minus[axis] -= h
        jacobian[:, axis] = (
            residual(f_plus) - residual(f_minus)
        ) / (2.0 * h)
    return jacobian


def _validate_analytic_jacobian_once(
        mf, component_name, residual, f0, analytic_jacobian):
    """Optionally compare one analytic Jacobian per component against FD."""
    if not getattr(mf, "f_jacobian_check", False):
        return

    checked = getattr(mf, "_f_jacobian_checked_components", None)
    if checked is None:
        checked = set()
        mf._f_jacobian_checked_components = checked
    if component_name in checked:
        return

    step = getattr(mf, "f_jacobian_check_step", 1e-4)
    reference = _central_difference_jacobian(residual, f0, step=step)
    denominator = max(numpy.linalg.norm(reference), 1e-16)
    relative_error = numpy.linalg.norm(
        analytic_jacobian - reference
    ) / denominator
    checked.add(component_name)

    logger.info(
        mf,
        "Analytic position Jacobian check for %s: relative error %.6g",
        component_name,
        relative_error,
    )
    warning_tol = getattr(mf, "f_jacobian_check_tol", 1e-4)
    if relative_error > warning_tol:
        logger.warn(
            mf,
            "Analytic position Jacobian for %s differs from central finite "
            "difference by %.6g (threshold %.6g).",
            component_name,
            relative_error,
            warning_tol,
        )


def update_f(mf, fock0, s1e, nsteps, tol=1e-15):
    """Update CNEO Lagrange multipliers with an analytic Newton Jacobian.

    With no line-search retry, the first Newton step for each nucleus requires
    two eigendecompositions: one for the current residual/Jacobian and one for
    the trial residual/Jacobian.  Each additional Newton step reuses the trial
    Jacobian and therefore requires one additional eigendecomposition.
    """
    gap_tol = getattr(mf, "f_jacobian_gap_tol", 1e-10)
    minimum_step = getattr(mf, "f_line_search_min_step", 0.01)

    for component_name, comp in mf.components.items():
        if not component_name.startswith("n"):
            continue

        atom_index = comp.mol.atom_index

        def evaluate(f_lagrange):
            fock = fock0[component_name] + numpy.einsum(
                "xij,x->ij", comp.int1e_r, f_lagrange
            )
            return get_deviation_and_jacobian(
                comp, fock, s1e[component_name], gap_tol=gap_tol
            )

        def residual(f_lagrange):
            fock = fock0[component_name] + numpy.einsum(
                "xij,x->ij", comp.int1e_r, f_lagrange
            )
            return get_deviation(comp, fock, s1e[component_name])

        f_current = numpy.asarray(mf.f[atom_index], dtype=float).copy()
        r_current, jacobian = evaluate(f_current)
        _validate_analytic_jacobian_once(
            mf,
            component_name,
            residual,
            f_current,
            jacobian,
        )

        for istep in range(nsteps):
            if numpy.max(numpy.abs(r_current)) < tol:
                logger.debug(
                    mf,
                    "DIIS type 4 analytic Newton step converged at step %d",
                    istep,
                )
                break

            residual_norm = numpy.linalg.norm(r_current)
            try:
                delta_f = numpy.linalg.solve(jacobian, -r_current)
            except numpy.linalg.LinAlgError:
                delta_f = numpy.linalg.lstsq(
                    jacobian, -r_current, rcond=gap_tol
                )[0]

            delta_norm = numpy.linalg.norm(delta_f)
            if delta_norm == 0.0 or not numpy.isfinite(delta_norm):
                logger.warn(
                    mf,
                    "Invalid analytic Newton constraint step for %s; "
                    "keeping the previous Lagrange multiplier.",
                    component_name,
                )
                break

            step = 1.0
            f_trial = f_current + delta_f
            r_trial, trial_jacobian = evaluate(f_trial)
            trial_norm = numpy.linalg.norm(r_trial)

            # Retain the original quadratic-interpolation line search.  Every
            # retry costs one eig call, but the analytic Jacobian itself adds
            # no extra diagonalizations.
            while trial_norm >= residual_norm:
                if step < minimum_step:
                    break

                directional_slope = -residual_norm / (step * delta_norm)
                denominator = 2.0 * (
                    trial_norm - residual_norm - directional_slope
                )
                if denominator == 0.0 or not numpy.isfinite(denominator):
                    step *= 0.5
                else:
                    step *= max(-directional_slope / denominator, 0.1)

                f_trial = f_current + step * delta_f
                r_trial, trial_jacobian = evaluate(f_trial)
                trial_norm = numpy.linalg.norm(r_trial)

            f_current = f_trial
            r_current = r_trial
            jacobian = trial_jacobian

        mf.f[atom_index] = f_current

    return
