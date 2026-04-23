from pyscf.neo.cdft import _get_mo_coeff_occ
import numpy
import scipy
from pyscf.lib import logger


def get_deviation(mf, fock, s1e):
    mo_coeff, mo_occ = _get_mo_coeff_occ(mf, fock, s1e)
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    return numpy.einsum('xij,ji->x', mf.int1e_r, dm)

def get_pos_err(mf, fock, s1e):

    deviations = []

    for t, comp in mf.components.items():
        if t.startswith('n'):
            deviations.append(get_deviation(comp, fock[t], s1e[t]))

    return numpy.concatenate(deviations)


def update_f(mf, fock0, s1e, nsteps, tol=1e-15):
    for t, comp in mf.components.items():
        if t.startswith('n'):
            ia = comp.mol.atom_index
            def residual(f_lagrange):
                fock = fock0[t] + numpy.einsum('xij,x->ij', comp.int1e_r, f_lagrange)
                return get_deviation(comp, fock, s1e[t])

            def residual_vectorize(x):
                x = numpy.asarray(x)

                if x.ndim == 1:
                    return residual(x)

                return numpy.apply_along_axis(residual, axis=0, arr=x)

            f0 = mf.f[ia]
            r0 = residual(f0)

            for istep in range(nsteps):

                if abs(r0).max() < tol:
                    logger.debug(mf, f'DIIS type 4: Newton step converged at step {istep}')
                    break

                res = scipy.differentiate.jacobian(residual_vectorize, f0, maxiter=1,
                                                   order=2, initial_step=1e-3)
                J = res.df

                g0 = numpy.sqrt(numpy.dot(r0, r0))

                try:
                    df = numpy.linalg.solve(J, -r0)
                except numpy.linalg.LinAlgError:
                    df = numpy.linalg.lstsq(J, -r0, rcond=None)[0]

                df_norm = numpy.linalg.norm(df)


                step = 1.0
                f_trial = f0 + step * df
                r1 = residual(f_trial)
                g1 = numpy.sqrt(numpy.dot(r1, r1))

                # Quadratic interpolation line search
                while g1 >= g0:

                    if step < 0.01:
                        break

                    g0p = -g0 / (step * df_norm)

                    step = step*max(-g0p/(2.0*(g1-g0-g0p)), 0.1)

                    f_trial = f0 + step * df
                    r1 = residual(f_trial)
                    g1 = numpy.sqrt(numpy.dot(r1, r1))

                # Final update
                f0 = f_trial
                r0 = r1

            mf.f[ia] = f0

    return
