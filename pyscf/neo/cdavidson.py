import numpy
import scipy.linalg
import scipy.optimize
from pyscf.lib import logger
from pyscf.lib import misc
from pyscf.lib.linalg_helper import (DAVIDSON_LINDEP, MAX_MEMORY,
                                     FOLLOW_STATE,
                                     SORT_EIG_BY_SIMILARITY,
                                     make_diag_precond,
                                     LinearDependenceError,
                                     _fill_heff_hermitian,
                                     _gen_x0, _normalize_xt_,
                                     _qr, _sort_by_similarity,
                                     _sort_elast, _Xlist)

def davidson(a_and_r_op, x0, f0, adiag, rdiag,
             tol=1e-12, max_cycle=200, max_space=24,
             lindep=DAVIDSON_LINDEP, max_memory=MAX_MEMORY,
             dot=numpy.dot, callback=None,
             nroots=1, lessio=False, pick=None, verbose=logger.WARN,
             follow_state=FOLLOW_STATE):
    e, x, f = davidson1(lambda xs: [list(t) for t in zip(*[a_and_r_op(x) for x in xs])],
                        x0, f0, adiag, rdiag, tol, max_cycle, max_space, lindep,
                        max_memory, dot, callback, nroots, lessio, pick, verbose,
                        follow_state)[1:]
    if nroots == 1:
        return e[0], x[0], f
    else:
        return e, x, f

def davidson1(a_and_r_op, x0, f0, adiag, rdiag,
              tol=1e-12, max_cycle=200, max_space=24,
              lindep=DAVIDSON_LINDEP, max_memory=MAX_MEMORY,
              dot=numpy.dot, callback=None,
              nroots=1, lessio=False, pick=None, verbose=logger.WARN,
              follow_state=FOLLOW_STATE, tol_residual=None,
              fill_heff=_fill_heff_hermitian):
    if isinstance(verbose, logger.Logger):
        log = verbose
    else:
        log = logger.Logger(misc.StreamObject.stdout, verbose)

    if tol_residual is None:
        toloose = numpy.sqrt(tol)
    else:
        toloose = tol_residual
    log.debug1('tol %g  toloose %g', tol, toloose)

    if callable(adiag) or callable(rdiag):
        raise RuntimeError("Because of constraint, precond should not be made callable.")

    if callable(x0):  # lazy initialization to reduce memory footprint
        x0 = x0()
    if isinstance(x0, numpy.ndarray) and x0.ndim == 1:
        x0 = [x0]
    #max_cycle = min(max_cycle, x0[0].size)
    max_space = max_space + (nroots-1) * 4
    # max_space*2 for holding ax and xs, nroots*2 for holding axt and xt
    #_incore = max_memory*1e6/x0[0].nbytes > max_space*2+nroots*3
    if rdiag.ndim == 2:
        rdim = rdiag.shape[0]
    else:
        assert rdiag.ndim == 1
        rdim = 1
        rdiag = rdiag.reshape(1,rdiag.size)
    # additional rx and rxt
    _incore = max_memory*1e6/x0[0].nbytes > max_space*(2+rdim)+nroots*(3+rdim)
    lessio = lessio and not _incore
    log.debug1('max_cycle %d  max_space %d  max_memory %d  incore %s',
               max_cycle, max_space, max_memory, _incore)
    dtype = None
    heff = None
    reff = None
    fresh_start = True
    e = None
    v = None
    conv = numpy.zeros(nroots, dtype=bool)
    emin = None
    prefer_lessio_in_a_and_r_x0 = False

    bounds = []
    zero_bound = (-1.0, 1.0)
    lstol = 1e-12
    for x in f0:
        if abs(x) < 0.1:
            bounds.append(zero_bound)
        else:
            bounds.append((x - 10.0*abs(x), x + 10.0*abs(x)))
    lb, ub = zip(*bounds)
    bounds = (lb, ub)

    for icyc in range(max_cycle):
        if fresh_start:
            if _incore:
                xs = []
                ax = []
                rx = []
                for i in range(rdim):
                    rx.append([])
            else:
                xs = _Xlist()
                ax = _Xlist()
                rx = _Xlist()
                for i in range(rdim):
                    rx.append(_Xlist())
            space = 0
# Orthogonalize xt space because the basis of subspace xs must be orthogonal
# but the eigenvectors x0 might not be strictly orthogonal
            xt = None
            x0len = len(x0)
            xt = _qr(x0, dot, lindep)[0]
            if len(xt) != x0len:
                log.warn('QR decomposition removed %d vectors.', x0len - len(xt))
                if callable(pick):
                    log.warn('Check to see if `pick` function %s is providing '
                             'linear dependent vectors', pick.__name__)
                if len(xt) == 0:
                    if icyc == 0:
                        msg = 'Initial guess is empty or zero'
                    else:
                        msg = ('No more linearly independent basis were found. '
                               'Unless loosen the lindep tolerance (current value '
                               f'{lindep}), the diagonalization solver is not able '
                               'to find eigenvectors.')
                    raise LinearDependenceError(msg)
            x0 = None
            max_dx_last = 1e9
            if SORT_EIG_BY_SIMILARITY:
                conv = numpy.zeros(nroots, dtype=bool)
        elif len(xt) > 1:
            xt = _qr(xt, dot, lindep)[0]
            xt = xt[:40]  # 40 trial vectors at most

        t0 = logger.perf_counter()
        axt, rxt = a_and_r_op(xt)
        ar_time = logger.perf_counter() - t0
        for k, xi in enumerate(xt):
            xs.append(xt[k])
            ax.append(axt[k])
            for i in range(rdim):
                rx[i].append(rxt[k][i])
        rnow = len(xt)
        head, space = space, space+rnow

        if dtype is None:
            try:
                dtype = numpy.result_type(axt[0], rxt[0][0], xt[0])
            except IndexError:
                raise LinearDependenceError('No linearly independent basis found '
                                            'by the diagonalization solver.')
        if heff is None:  # Lazy initialize heff to determine the dtype
            heff = numpy.empty((max_space+nroots,max_space+nroots), dtype=dtype)
        else:
            heff = numpy.asarray(heff, dtype=dtype)
        if reff is None:
            reff = numpy.empty((rdim,max_space+nroots,max_space+nroots), dtype=dtype)
        else:
            reff = numpy.asarray(reff, dtype=dtype)

        elast = e
        vlast = v
        conv_last = conv

        fill_heff(heff, xs, ax, xt, axt, dot)
        assert nroots == 1 # NOTE: support lowest eigenvalue state constraint only!
        for i in range(rdim):
            fill_heff(reff[i], xs, rx[i], xt, [rxt[0][i]], dot)
        xt = axt = rxt = None
        def constraint_fn(f):
            heff_mod = heff[:space,:space] + numpy.einsum('i,ijk->jk', f, reff[:,:space,:space])
            w, v = scipy.linalg.eigh(heff_mod)
            return v[:,0].T @ reff[:,:space,:space] @ v[:,0]

        assert max_space >= 20
        f0_opt_on = False
        if space >= 4:
            f0_opt_on = True
            #result = scipy.optimize.root(constraint_fn, f0, method='hybr')
            # Using root may cause divergent f when the subspace is small
            #result = scipy.optimize.minimize(lambda f: numpy.linalg.norm(constraint_fn(f))**2 * 1e8,
            #                                 f0, method='L-BFGS-B')
            result = scipy.optimize.least_squares(constraint_fn, f0, bounds=bounds, ftol=lstol,
                                                  xtol=lstol, gtol=lstol)
            f0 = result.x
            log.debug(f'    Lagrange multiplier optimized: {f0}')
        else:
            log.debug(f'        Lagrange multiplier fixed: {f0}')
        heff_final = heff[:space,:space] + numpy.einsum('i,ijk->jk', f0, reff[:,:space,:space])
        w, v = scipy.linalg.eigh(heff_final)
        r = v[:,0].T @ reff[:,:space,:space] @ v[:,0]
        log.debug(f'             Constraint deviation: {r}')
        precond = make_diag_precond(adiag + f0 @ rdiag)
        if callable(pick):
            w, v, idx = pick(w, v, nroots, locals())
            if len(w) == 0:
                raise RuntimeError(f'Not enough eigenvalues found by {pick}')

        if SORT_EIG_BY_SIMILARITY:
            e, v = _sort_by_similarity(w, v, nroots, conv, vlast, emin)
        else:
            e = w[:nroots]
            v = v[:,:nroots]
            conv = numpy.zeros(e.size, dtype=bool)
            if not fresh_start:
                elast, conv_last = _sort_elast(elast, conv_last, vlast, v, log)

        if elast is None:
            de = e
        elif elast.size != e.size:
            log.debug('Number of roots different from the previous step (%d,%d)',
                      e.size, elast.size)
            de = e
        else:
            de = e - elast

        x0 = None
        x0 = _gen_x0(v, xs)
        if fresh_start:
            if icyc > 0 and prefer_lessio_in_a_and_r_x0:
                log.debug('    Fresh start, go back to _gen_x0')
            prefer_lessio_in_a_and_r_x0 = False
        if lessio or prefer_lessio_in_a_and_r_x0:
            ax0, rx0 = a_and_r_op(x0)
        else:
            t0 = logger.perf_counter()
            ax0 = _gen_x0(v, ax)
            assert nroots == 1 # NOTE: support lowest eigenvalue state constraint only!
            assert v.shape[1] == 1
            rx0 = [[]]
            for i in range(rdim):
                rx0[0].append(_gen_x0(v, rx[i])[0])
            gen_ar0_time = logger.perf_counter() - t0
            if gen_ar0_time > ar_time:
                prefer_lessio_in_a_and_r_x0 = True
                log.debug('    _gen_x0 takes longer, lessio starting from next cycle')

        dx_norm = numpy.zeros(e.size)
        xt = [None] * nroots
        for k, ek in enumerate(e):
            xt[k] = ax0[k] - ek * x0[k]
            for i in range(rdim):
                xt[k] += f0[i] * rx0[k][i]
            dx_norm[k] = numpy.sqrt(dot(xt[k].conj(), xt[k]).real)
            conv[k] = abs(de[k]) < tol and dx_norm[k] < toloose \
                      and f0_opt_on and all(abs(r) < 1e-12) #and numpy.linalg.norm(r) < 1e-8
            if conv[k] and not conv_last[k]:
                log.debug('root %d converged  |r|= %4.3g  e= %s  max|de|= %4.3g',
                          k, dx_norm[k], ek, de[k])
        ax0 = rx0 = None
        max_dx_norm = max(dx_norm)
        ide = numpy.argmax(abs(de))
        if all(conv):
            log.debug('converged %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g',
                      icyc, space, max_dx_norm, e, de[ide])
            break
        elif (follow_state and max_dx_norm > 1 and
              max_dx_norm/max_dx_last > 3 and space > nroots+2):
            log.debug('davidson %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g',
                      icyc, space, max_dx_norm, e, de[ide])
            log.debug('Large |r| detected, restore to previous x0')
            x0 = _gen_x0(vlast, xs)
            fresh_start = True
            continue

        if SORT_EIG_BY_SIMILARITY:
            if any(conv) and e.dtype == numpy.double:
                emin = min(e)

        # remove subspace linear dependency
        for k, ek in enumerate(e):
            if (not conv[k]) and dx_norm[k]**2 > lindep:
                xt[k] = precond(xt[k], e[0], x0[k])
                xt[k] *= dot(xt[k].conj(), xt[k]).real ** -.5
            elif not conv[k]:
                # Remove linearly dependent vector
                xt[k] = None
                log.debug1('Drop eigenvector %d, norm=%4.3g', k, dx_norm[k])
            else:
                xt[k] = None

        xt, norm_min = _normalize_xt_(xt, xs, lindep, dot)
        log.debug('davidson %d %d  |r|= %4.3g  e= %s  max|de|= %4.3g  lindep= %4.3g',
                  icyc, space, max_dx_norm, e, de[ide], norm_min)
        if len(xt) == 0:
            log.debug('Linear dependency in trial subspace. |r| for each state %s',
                      dx_norm)
            conv = dx_norm < toloose
            for k, convk in enumerate(conv):
                conv[k] = convk and f0_opt_on and numpy.linalg.norm(r) < 1e-8
            break

        max_dx_last = max_dx_norm
        fresh_start = space+nroots > max_space

        if callable(callback):
            callback(locals())

    x0 = list(x0)  # nparray -> list

    # Check whether the solver finds enough eigenvectors.
    h_dim = x0[0].size
    if len(x0) < min(h_dim, nroots):
        # Two possible reasons:
        # 1. All the initial guess are the eigenvectors. No more trial vectors
        # can be generated.
        # 2. The initial guess sits in the subspace which is smaller than the
        # required number of roots.
        log.warn(f'Not enough eigenvectors (len(x0)={len(x0)}, nroots={nroots})')

    # Remove f0*r in e
    e[0] -= numpy.dot(f0, r)

    return numpy.asarray(conv), e, x0, f0

if __name__ == '__main__':
    n = 500  # Matrix dimension
    m = 6   # Number of constraints (first dimension of r)

    numpy.random.seed(42)
    A = numpy.random.randn(n, n)
    A = (A + A.T) / 2  # Make symmetric

    r = numpy.random.randn(m, n, n)
    for i in range(m):
        r[i] = (r[i] + r[i].T) / 2  # Make each slice symmetric

    def a_and_r_op(x):
        result = numpy.zeros((m, n))
        for i in range(m):
            result[i] = r[i] @ x
        return A @ x, result

    adiag = numpy.diag(A)
    rdiag = numpy.zeros((m, n))
    for i in range(m):
        rdiag[i] = numpy.diag(r[i])

    x0 = numpy.random.randn(n)
    x0 = x0 / numpy.linalg.norm(x0)  # Normalize
    f0 = numpy.zeros(m)  # Initial Lagrange multipliers

    def constraint(x, f):
        return x.T @ r @ x

    def verify_solution(x, f, eigenvalue):
        Ax = A @ x
        for i in range(m):
            Ax += f[i] * (r[i] @ x)

        residual = Ax - eigenvalue * x
        eigenvalue_error = numpy.linalg.norm(residual)

        constraints = constraint(x, f)
        constraint_error = numpy.linalg.norm(constraints)

        print(f"Eigenvalue equation error: {eigenvalue_error:.2e}")
        print(f"Constraint error: {constraint_error:.2e}")
        print(f"Eigenvalue: {eigenvalue:.6f}")
        print(f"Lagrange multipliers: {f}")

        return eigenvalue_error, constraint_error

    def solve_full_problem():
        def objective(f_vals):
            H_eff = A.copy()
            for i in range(m):
                H_eff += f_vals[i] * r[i]

            eigvals, eigvecs = scipy.linalg.eigh(H_eff)
            x = eigvecs[:, 0]

            return constraint(x, f_vals)

        result = scipy.optimize.root(objective, f0, method='hybr')
        f_opt = result.x

        H_eff = A.copy()
        for i in range(m):
            H_eff += f_opt[i] * r[i]
        eigvals, eigvecs = scipy.linalg.eigh(H_eff)

        return eigvals[0], eigvecs[:, 0], f_opt

    print("Test problem dimensions:")
    print(f"Matrix A: {A.shape}")
    print(f"Tensor r: {r.shape}")
    print(f"Initial x0: {x0.shape}")
    print(f"Initial f0: {f0.shape}")
    print(f"adiag: {adiag.shape}")
    print(f"rdiag: {rdiag.shape}")

    eigenvalue, eigenvector, f = davidson(a_and_r_op, x0, f0, adiag, rdiag, verbose=10, max_cycle=200, max_space=24)
    print(f"Lowest eigenvalue without f*r: {eigenvalue}")
    print("\nVerifying Davidson solution:")
    verify_solution(eigenvector, f, eigenvalue)

    print("\nSolving with full matrix approach for comparison...")
    ref_eigenvalue, ref_eigenvector, ref_f = solve_full_problem()
    print(f"Reference eigenvalue: {ref_eigenvalue:.6f}")
    print(f"Reference Lagrange multipliers: {ref_f}")

    print("\nVerifying reference solution:")
    verify_solution(ref_eigenvector, ref_f, ref_eigenvalue)
