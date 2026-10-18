'''Initial density guesses from the history of converged molecular dynamics steps.

Prediction methods
------------------
ASPC (scheme='aspc'):
    Extrapolate historical occupied-space projectors with Kolafa's weights.
    Apply the extrapolated operator U to the latest occupied orbitals, then
    normalize the predicted orbitals in the current overlap metric.
    References: Kolafa, Mol. Simul. 18, 193 (1996); J. Comput. Chem. 25,
    335 (2004), doi:10.1002/jcc.10385. Electronic adaptation: Kuhne et al.,
    PRL 98, 066401 (2007), Eq. 2, doi:10.1103/PhysRevLett.98.066401.
    Normalization follows CP2K qs_wf_history_methods.F, wfi_extrapolate.

    The optional density trial (method='density') forms U P_previous,
    using the expression printed in Dickinson et al.,
    JCP 164, 234108 (2026), Eqs. 18-19, doi:10.1063/5.0332811.
    Apply McWeeny purification: P <- 3 P S P - 2 P S P S P.
    Reference: McWeeny, Rev. Mod. Phys. 32, 335 (1960),
    doi:10.1103/RevModPhys.32.335.
    The one-sided trial can be non-Hermitian. Explicit averaging with its
    adjoint supplies Hermitian SCF input. hermitize='before' averages before
    purification; 'after' averages its output and can change idempotency.
    These averaging options are local adaptations of the printed predictor.

Polynomial projector prediction (scheme='polynomial'):
    Use equally spaced polynomial weights in the same projector action.
    Two snapshots give linear prediction. This follows CP2K's
    wfi_extrapolate, wfi_ps_method_nr branch.

Fock prediction (scheme='fock'):
    Fit a polynomial to historical AO Fock matrices. Diagonalize the
    predicted Fock matrix to construct occupied orbitals.
    Reference: Herbert and Head-Gordon, PCCP 7, 3269 (2005), Sec. 2,
    doi:10.1039/B509494A.

Grassmann prediction (scheme='grassmann'):
    Map occupied subspaces to a fixed-reference tangent space, fit the
    current molecular descriptor using historical descriptors, and map the
    predicted tangent back to an occupied subspace.
    Reference: Polack et al., JCTC 17, 6965 (2021), Algorithm 1 and
    Supporting Information, doi:10.1021/acs.jctc.1c00751.
    scheme='grassmann_qtr' uses the quasi-time-reversible fit of Pes et al.,
    JPCL 14, 9720 (2023), Eqs. 3-5, doi:10.1021/acs.jpclett.3c02098.
    The density-approximation framework also appears in Polack et al.,
    Mol. Phys. 118, e1779834 (2020), doi:10.1080/00268976.2020.1779834.
    Grassmann geometry itself predates these BOMD algorithms; see Edelman,
    Arias and Smith (1998), doi:10.1137/S0895479895290954.

Projected Grassmann prediction (scheme='gext_proj', the default):
    Fit historical overlap matrices to the current overlap and use the
    fitted weights in the occupied-projector action. The QTR variant is
    scheme='gext_proj_qtr'. These follow CP2K's wfi_gext_proj_nr and
    wfi_gext_proj_qtr_nr implementations.
    Wilhelm et al., CP2K dynamics/transport review (2026), Sec. X.2.1,
    Eq. 100, doi:10.48550/arXiv.2607.22916, describes this projected
    formulation and its difference-fit and QTR coefficient choices.

Source reference for polynomial/projected prediction and normalization:
    CP2K src/qs_wf_history_methods.F, wfi_extrapolate, diff_fitting,
    tr_fitting, revision dfe8787b4f0a22da8e73e9abe2b384fa021d2f53:
    https://github.com/cp2k/cp2k/blob/dfe8787b4f0a22da8e73e9abe2b384fa021d2f53/src/qs_wf_history_methods.F

Extrapolator is a factory for ASPC, Polynomial, Fock, Grassmann and
ProjectedGrassmann. Each predictor inherits lib.StreamObject and shares
in-memory history, component packing and metric normalization. kernel
(also named predict) returns dm0 and stores it as the predictor's dm0.
Prediction and overlap evaluation use NumPy on CPU for both SCF backends.
GPU orbitals/Fock matrices are downloaded once when storing an accepted
point; GPU SCF converts the returned tagged NumPy guess at initialization.
Use kernel before SCF and update after convergence. Each history entry
represents one fixed-timestep MD point. Geometry-optimization integration
requires a separate policy for variable steps and trial geometries.
Occupied-orbital normalization follows CP2K's Cholesky procedure.
Numerical prediction failures emit RuntimeWarning and return the preceding
converged density unchanged, allowing SCF to continue with density reuse.
'''

from contextlib import contextmanager
import hashlib
import json
from math import comb
import sys
import warnings

import numpy
from pyscf import lib


# Mean SCF cycles for CNEO PBE0, EPC=None, all-Coulomb DF,
# def2-TZVP/JKFIT/PB4D/ETB and electronic grid level 3.
# The initial geometry is excluded; history startup is included.
#
# Columns: Water10 tight (1e-10/1e-7), default (1e-9/sqrt(1e-9)),
# loose (1e-8/1e-4); ethanol NVE initialized at 300/600 K and NVT at 600 K.
# Length is the maximum snapshot count. A dash is unmeasured.
#
# Method                     Length W10tight W10default W10loose Et300 Et600 EtNVT
# Fresh initial guess             -        -      10.92        -     -     -     -
# Previous density                1    12.67       8.58     7.50  7.38  7.58  7.62
# ASPC orbitals                   4    10.92       6.92     5.83  5.25  5.58  5.92
# ASPC orbitals                   6        -       6.83        -     -     -     -
# ASPC density, before            4    11.92       7.08     6.83  6.62  6.79  6.83
# ASPC density, before            6        -       7.08        -     -     -     -
# ASPC density, after             4        -       7.08        -     -     -     -
# ASPC density, after             6        -       7.08        -     -     -     -
# Polynomial projectors           4        -       5.67        -     -     -     -
# Polynomial projectors           6     9.50       5.17     4.92  4.21  4.25  6.62
# Fock, degree 3                  4        -       6.50        -     -     -     -
# Fock, degree 3                  6        -       7.17        -     -     -     -
# Grassmann                       4        -       6.33        -     -     -     -
# Grassmann                       6        -       6.33        -     -     -     -
# Grassmann QTR                   4        -       5.83        -     -     -     -
# Grassmann QTR                   6        -       6.33        -     -     -     -
# Projected Grassmann             1        -       7.92        -     -     -     -
# Projected Grassmann             2        -       7.08        -     -     -     -
# Projected Grassmann             3        -       5.50        -     -     -     -
# Projected Grassmann             4        -       4.75        -     -     -     -
# Projected Grassmann             6     8.92       4.75     4.42  3.88  4.12  5.17
# Projected Grassmann             8        -       4.75        -     -     -     -
# Projected Grassmann QTR         4        -       5.42        -     -     -     -
# Projected Grassmann QTR         6        -       5.17        -     -     -     -
#
# Water10 uses 12 common geometries at 0.5 fs spacing. Threshold pairs are
# conv_tol / conv_tol_grad. All 20 H nuclei are quantum mechanical.
# Ethanol uses 24 MD steps at 0.5 fs, six quantum H, native SCF thresholds,
# grid response, common initial momenta and common thermostat random numbers.
# Langevin friction is 0.01/fs. NVE temperatures specify initialization.
# All SCFs converged and every predictor retained sufficient occupied rank.
#
# Ordinary projected histories 4, 6 and 8 give identical cycle sequences.
# Defaults use four snapshots for Fock, Grassmann and ordinary projected
# prediction; six for ASPC, polynomial and projected QTR. The four/six
# sweep includes startup. These choices are local Water10-based defaults.
# At 600 K and a 1 fs timestep, ethanol NVE averages 7.92 cycles with density reuse
# and 5.38 with projected G-Ext. Water10 density prediction averages 7.08
# cycles for either before- or after-purification Hermitization.
#
# History one renormalizes the previous occupied orbitals in the current AO
# metric, accounting for its difference from previous-density reuse.


def _overlap_eigh(s):
    '''Diagonalize overlaps and select the AO directions retained by SCF.

    scf.hf.check_linear_dependency defines the eigenvalue cutoff. Applying
    it to each component keeps prediction in the AO subspace used by SCF.
    '''
    from pyscf.scf import hf
    # The electronic block follows vanilla SCF's two-dimensional eigensolve;
    # equal-sized nuclear overlaps retain their batched solve.
    if s.shape[0] == 1:
        w, v = numpy.linalg.eigh(s[0])
        w, v = w[None], v[None]
    else:
        w, v = numpy.linalg.eigh(s)
    if hf.remove_overlap_zero_eigenvalue:
        retained = w > hf.overlap_zero_eigenvalue_threshold
    else:
        if bool(numpy.any(w <= 0)):
            raise numpy.linalg.LinAlgError('Enable SCF linear-dependency removal for singular overlaps')
        retained = numpy.ones_like(w, dtype=bool)
    return w, v, retained


def _overlap_factors(s):
    '''Lowdin frame for Grassmann prediction, doi:10.1021/acs.jctc.1c00751.

    Polack et al. Sec. 2.4 uses S**(1/2) C to represent occupied spaces.
    SCF's linear-dependency removal changes its orthogonalizer and MO rank;
    the supplied AO overlap retains its small or negative eigenvalues.
    Apply the CPU SCF cutoff here and form the symmetric pseudoinverse
    on the retained eigenspace. This extends the paper's full-rank frame to
    the rank-reduced AO representation used by PySCF.

    With S = V diag(w) V^H, use sqrt(w) and 1/sqrt(w) for retained
    eigenvalues and zero for discarded directions. Their product is the
    retained-space projector. This is spectral square-root/pseudoinverse
    algebra; the rank cutoff follows scf.hf.check_linear_dependency.
    '''
    w, v, retained = _overlap_eigh(s)
    # Filling discarded eigenvalues with one keeps both square-root and
    # division operations finite; their final spectral weights are zero.
    scale = numpy.sqrt(numpy.where(retained, w, 1.))
    root = (v * numpy.where(retained, scale, 0.)[...,None,:]) @ v.swapaxes(-1, -2).conj()
    inverse = (v * numpy.where(retained, 1/scale, 0.)[...,None,:]) @ v.swapaxes(-1, -2).conj()
    return root, inverse


def _grassmann_log(reference, occupied):
    '''Grassmann log map, Polack SI, doi:10.1021/acs.jctc.1c00751.

    For orthonormal occupied columns R and C, form
    H = C (R^H C)^(-1) - R. If H = U diag(sigma) V^H,
    the tangent is U diag(arctan(sigma)) V^H. The arctangents are
    principal angles: tangent-space linear combinations extrapolate
    subspace rotations independently of occupied-orbital gauge choices.
    The chart requires invertible R^H C. Equal-sized nuclear components
    share a batched solve and thin SVD.
    '''
    cross = reference.swapaxes(-1, -2).conj() @ occupied
    horizontal = numpy.linalg.solve(cross.swapaxes(-1, -2),
                                    occupied.swapaxes(-1, -2)).swapaxes(-1, -2) - reference
    u, singular, vh = numpy.linalg.svd(horizontal, full_matrices=False)
    return (u * numpy.arctan(singular)[...,None,:]) @ vh


def _grassmann_exp(reference, tangent):
    '''Return tangent-space prediction to orthonormal occupied columns.

    Polack SI, doi:10.1021/acs.jctc.1c00751: for tangent T = U theta V^H
    at R, return (R V cos(theta) + U sin(theta)) V^H. This rotates the
    reference occupied subspace through the predicted principal angles,
    preserving orthonormality in the Euclidean frame S**(1/2) C.
    '''
    u, singular, vh = numpy.linalg.svd(tangent, full_matrices=False)
    return ((reference @ vh.swapaxes(-1, -2).conj()) * numpy.cos(singular)[...,None,:]
            + u * numpy.sin(singular)[...,None,:]) @ vh


def _fit_weights(history, target, regularization, reversible=False, difference=False,
                 normal_equations=False):
    '''Fit prediction weights for Grassmann and ProjectedGrassmann.

    Callers and settings:
      Grassmann._weights: molecular Coulomb descriptors, augmented least
      squares; reversible=True selects Grassmann QTR history pairing.
      ProjectedGrassmann._weights: electronic AO overlaps, normal equations;
      difference=True selects projected G-Ext, reversible=True its QTR form.
    ASPC and Polynomial compute their weights analytically; Fock fits time
    polynomials separately.

    history is ordered from the newest snapshot to the oldest. The augmented
    least-squares formulation follows Pes et al. Eqs. 3-5. The difference
    and normal-equation forms follow CP2K diff_fitting/tr_fitting in
    src/qs_wf_history_methods.F at revision
    dfe8787b4f0a22da8e73e9abe2b384fa021d2f53 (source URL in module docstring).

    Grassmann fits molecular Coulomb descriptors and applies the resulting
    weights to tangent matrices. ProjectedGrassmann fits AO overlaps and
    applies its weights to occupied projectors. Each design column is a
    flattened historical descriptor/overlap; solve min ||D w-target||^2
    + regularization^2 ||w||^2. Difference fitting uses the newest snapshot
    as the reference, enforcing sum(weights)=1. Reversible fitting combines
    history pairs symmetric about the prediction interval and subtracts
    the oldest snapshot from the predicted combination.
    '''
    length = len(history)
    if length == 1:
        return numpy.ones(1)
    data = numpy.stack(history).reshape(length, -1)
    target = target.reshape(-1)
    if reversible:
        pairs = [data[i-1] + data[length-i-1] for i in range(1, length//2+1)]
        design = numpy.stack(pairs, axis=1)
        rhs = target + data[-1]
    elif difference:
        design = (data[1:] - data[0]).T
        rhs = target - data[0]
    else:
        design, rhs = data.T, target
    count = design.shape[1]
    if normal_equations:
        # CP2K diff_fitting/tr_fitting reduce overlap-matrix traces to this
        # small regularized system of overlap inner products.
        gram = design.conj().T @ design + regularization**2*numpy.eye(count)
        coefficients = numpy.linalg.solve(gram, design.conj().T @ rhs)
    else:
        design = numpy.concatenate((design, regularization*numpy.eye(count)), axis=0)
        rhs = numpy.concatenate((rhs, numpy.zeros(count)))
        coefficients = numpy.linalg.lstsq(design, rhs, rcond=None)[0]
    if reversible:
        weights = numpy.zeros(length)
        weights[-1] = -1
        for i in range(1, length//2+1):
            weights[i-1] += coefficients[i-1]
            weights[length-i-1] += coefficients[i-1]
        return weights
    if difference:
        return numpy.concatenate(((1-coefficients.sum()).reshape(1), coefficients))
    return coefficients


class _Extrapolator(lib.StreamObject):
    '''Predict an SCF initial density from converged, fixed-timestep MD history.

    Method selection
    ----------------
    scheme : str
        'gext_proj' (default), 'gext_proj_qtr', 'grassmann', 'grassmann_qtr',
        'aspc', 'polynomial', or 'fock'. The module documentation describes
        each algorithm and its reference.
    method : str
        'orbital' (default) constructs a density from normalized occupied
        orbitals. 'density' applies Hermitization and McWeeny purification
        to Dickinson's one-sided density trial with scheme='aspc'.
    hermitize : str
        PySCF's SCF builders expect a Hermitian density. The one-sided
        density trial can violate this, so average it with its adjoint.
        'before' (default) performs this averaging before purification,
        giving a Hermitian occupied-space projector after purification.
        'after' averages the purified result; the averaged density can have
        fractional occupations. It is returned without mo_coeff/mo_occ tags,
        because DF exchange builders use those tags to reconstruct a density
        from occupied orbitals. The returned matrix therefore uses DF-K's
        general-density path: CPU DF-K contracts the full AO density, while
        GPU DF-K decomposes it into factors with fractional occupations.
        This adds work to the initial exchange build compared with the
        occupied-orbital shortcut available to 'before'. Subsequent SCF
        iterations obtain fresh orbital tags from their own diagonalizations.

    History and fitting
    -------------------
    history_length : int
        Maximum number of snapshots, including the latest converged point.
        Defaults are four for Fock, Grassmann and ordinary projected
        prediction; six for ASPC, Polynomial and projected QTR, selected
        from the Water10 comparisons above. The factory uses the selected
        class's default when history_length is omitted.
        Startup uses the available history.
        CP2K's ASPC EXTRAPOLATION_ORDER counts two fewer than this length.
        CP2K's PS order counts one fewer; its projected G-Ext order equals
        this length. Dickinson's K also counts the stored snapshots.
    degree : int
        Fock polynomial degree, reduced during startup as needed.
        Herbert/Head-Gordon use a (6,3) fit; this implementation defaults
        to four snapshots and degree three from the Water10 comparison.
        Using fewer snapshots during startup is a local extension;
        their simulations use SAD guesses until the full history is ready.
        Capture Fock matrices with record_fock around SCF or pass them
        explicitly to update.
    regularization : float
        Fit regularization: 0.01 for Grassmann, 0.001 for Grassmann QTR,
        and 1e-4 for projected fits. CP2K uses 1e-4 for its projected fits.
        Polack sets epsilon=1000*r_scf; 0.01 corresponds to r_scf=1e-5.
        Pes selects epsilon and history jointly for each SCF threshold;
        0.001 is one of the choices for r_scf=1e-7. These fixed defaults
        leave epsilon independent of PySCF's energy convergence threshold.
        Original Grassmann uses a QM Coulomb descriptor in Bohr and a
        fixed-reference tangent space; projected fits use AO overlaps.

    Components and execution
    ------------------------
    extrapolate_nuc : bool
        Predict both electrons and quantum nuclei when True (default).
        False predicts only the electronic density.
        Electronic spin blocks and distinguishable nuclei retain separate
        occupied projectors. Rank-reduced overlaps use the CPU SCF
        linear-dependency cutoff. Occupations must be integer projectors.
        History, overlaps and prediction use NumPy; returned densities carry
        occupied-orbital tags for CPU/GPU SCF initialization.

    Call predict before SCF and update after convergence. Advance the step
    index once per accepted MD point. Supplying an extrapolator explicitly
    enables prediction in the ASE calculators; their default is disabled.
    '''

    _keys = {'history_length', 'extrapolate_nuc', 'dm0'}

    def __init__(self, history_length=6, extrapolate_nuc=True):
        self.history_length = history_length
        self.extrapolate_nuc = extrapolate_nuc
        self.verbose = 0
        self.stdout = sys.stdout
        self.clear()

    @property
    def history_length(self):
        return self._history_length

    @history_length.setter
    def history_length(self, value):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('history_length must be a positive integer')
        self._history_length = value
        if hasattr(self, 'history'):
            self.history = self.history[-value:]
        if hasattr(self, 'descriptors'):
            self.descriptors = self.descriptors[-value:]

    def clear(self):
        self.history = []
        self.signature = None
        self.last_step = None
        self.last_coords = None
        self.effective_history = 0
        self.fallbacks = 0
        self.dm0 = None

    @contextmanager
    def record_fock(self, mf):
        '''Allow calculators to use one SCF context for every predictor.

        Fock overrides this context to capture the matrices it extrapolates.
        The other predictors store orbitals and overlaps in update.
        '''
        yield

    @staticmethod
    def _asarray(value):
        # GPU history enters CPU prediction through one download per stored
        # array at an accepted MD point, using the hf._to_cpu conversion.
        if hasattr(value, 'get'):
            return value.get()
        return numpy.asarray(value)

    def _check(self, mf, step, mol=None):
        if mol is None:
            mol = mf.mol
        if step is None:
            # Repeated properties at one geometry share a history entry.
            step = 0 if self.last_step is None else self.last_step + int(
                not numpy.array_equal(mol.atom_coords(), self.last_coords))
        if isinstance(step, bool) or not isinstance(step, int):
            raise ValueError('step must be an integer MD point index')
        mols = mol.components if hasattr(mf, 'components') else {'e': mol}
        # History requires matching basis, isotope, component identity and
        # electronic state. Coordinates vary between accepted MD points and
        # are excluded from this compatibility key.
        data = [(t, m._basis, m._ecp, m.cart, m.charge, m.spin,
                 [m.atom_symbol(i) for i in range(m.natm)],
                 getattr(m, 'mass', None), getattr(m, 'nuc_occ_state', None))
                for t, m in mols.items()]
        electronic = mf.components['e'] if hasattr(mf, 'components') else mf
        state = (mol.charge, mol.spin, getattr(electronic, 'xc', None),
                 getattr(mf, 'epc', None), data)
        # PySCF's basis dictionaries can have different insertion orders in
        # independent SCF objects. Canonical JSON keeps the history key stable.
        encoded = json.dumps(state, sort_keys=True, default=lambda value: value.tolist())
        signature = hashlib.sha256(encoded.encode()).hexdigest()
        if self.signature is not None and self.signature != signature:
            self.clear()
        if self.last_step is not None and step not in (self.last_step, self.last_step+1):
            self.clear()
        self.signature = signature
        if step == self.last_step and not numpy.array_equal(mol.atom_coords(), self.last_coords):
            raise ValueError('One MD step index refers to two geometries')
        return step

    def _overlaps(self, mf, mol):
        if not hasattr(mf, 'components'):
            return {'e': mol.intor_symmetric('int1e_ovlp')}
        overlaps = {}
        for t in mf.components:
            if t.startswith('n') and not self.extrapolate_nuc:
                continue
            cmol = mol.components[t]
            # A rigidly translated one-center nuclear basis has invariant S.
            if t.startswith('n') and self.history and len(set(cmol._bas[:,0])) == 1:
                for group in self.history[-1]:
                    for i, key in enumerate(group['keys']):
                        if key[0] == t:
                            overlaps[t] = group['s'][i]
                            break
                    if t in overlaps:
                        break
            if t not in overlaps:
                # CPU SCF.hf.get_ovlp uses this same AO integral. Evaluate
                # it on the host for prediction even when SCF runs on GPU.
                overlaps[t] = cmol.intor_symmetric('int1e_ovlp')
        return overlaps

    def update(self, mf, step=None, fock=None):
        '''Store converged orbitals; omitted step counts successive geometries.

        Updating the same point again leaves its stored snapshot unchanged.
        '''
        if not mf.converged:
            return
        step = self._check(mf, step)
        if step == self.last_step:
            return
        coeff = mf.mo_coeff if hasattr(mf, 'components') else {'e': mf.mo_coeff}
        occ = mf.mo_occ if hasattr(mf, 'components') else {'e': mf.mo_occ}
        overlaps = self._overlaps(mf, mf.mol)
        groups = {}
        for t in overlaps:
            c = coeff[t]
            o = self._asarray(occ[t])
            blocks = [(c, o)] if c.ndim == 2 else list(zip(c, o))
            for spin, (ci, oi) in enumerate(blocks):
                idx = numpy.flatnonzero(oi > 0)
                factor = float(oi[idx[0]]) if len(idx) else 1.
                if not numpy.all(numpy.isin(oi, (0., 1., 2.))) or not numpy.all(oi[idx] == factor):
                    raise ValueError('Extrapolation requires uniform integer occupations within each spin block')
                key = (ci.shape[0], len(idx), factor, t.startswith('n'))
                group = groups.setdefault(key, dict(keys=[], c=[], s=[], occupied=[], factor=factor))
                group['keys'].append((t, spin if c.ndim == 3 else -1))
                # Select occupied columns before downloading GPU orbitals.
                # Advanced indexing creates an independent history snapshot.
                group['c'].append(self._asarray(ci[:,idx]))
                group['s'].append(overlaps[t])
                group['occupied'].append(idx)
        snapshot = []
        for group in groups.values():
            group['c'] = numpy.stack(group['c'])
            group['s'] = numpy.stack(group['s'])
            group['occupied'] = numpy.stack(group['occupied'])
            group['cs'] = group['c'].swapaxes(-1, -2).conj() @ group['s']
            snapshot.append(group)
        if self.history and [(g['keys'], g['c'].shape) for g in self.history[-1]] != [
                (g['keys'], g['c'].shape) for g in snapshot]:
            self.history = []
            self._clear_method()
        self._prepare_snapshot(snapshot, mf, fock)
        self.history.append(snapshot)
        self.history = self.history[-self.history_length:]
        self.last_step = step
        self.last_coords = mf.mol.atom_coords().copy()

    def _clear_method(self):
        pass

    def _prepare_snapshot(self, snapshot, mf, fock):
        pass

    def _predict_orbitals(self, index, latest, snapshots, weights, s):
        '''Shared projector action for ASPC, Polynomial and ProjectedGrassmann.'''
        # Kuhne Eq. 2, doi:10.1103/PhysRevLett.98.066401, and CP2K
        # wfi_extrapolate: U C_previous = sum_j w_j C_j C_j^H S_j C_previous.
        # Occupied factors give the same projector action with smaller products.
        trial = numpy.zeros_like(latest['c'])
        for weight, snapshot in zip(weights, snapshots):
            group = snapshot[index]
            trial += weight * (group['c'] @ (group['cs'] @ latest['c']))
        return trial

    def _purify(self, trial, cprev, s):
        return None, trial

    def kernel(self, mf, mol, step=None):
        '''Return dm0, or None at startup; omitted step counts new geometries.'''
        self.dm0 = None
        # Validate the target geometry while the scanner retains its current mol.
        step = self._check(mf, step, mol)
        if not self.history or step == self.last_step:
            return None
        self.effective_history = min(self.history_length, len(self.history))
        try:
            with numpy.errstate(over='raise', invalid='raise', divide='raise'):
                self.dm0 = self._predict_density(mf, mol)
        except (numpy.linalg.LinAlgError, FloatingPointError) as err:
            # Prediction supplies an SCF guess. Preserve the last converged
            # density when its linear algebra or purification fails, so SCF
            # can proceed from the same guess as ordinary density reuse.
            warnings.warn('SCF guess extrapolation failed (%s); using the previous density' % err,
                          RuntimeWarning, stacklevel=2)
            self.fallbacks += 1
            result = {}
            for group in self.history[-1]:
                for i, (t, spin) in enumerate(group['keys']):
                    c = group['c'][i]
                    value = lib.tag_array(group['factor'] * (c @ c.conj().T), mo_coeff=c,
                                          mo_occ=numpy.full(c.shape[-1], group['factor']))
                    if spin < 0:
                        result[t] = value
                    else:
                        result.setdefault(t, [None, None])[spin] = value
            for t, value in result.items():
                if isinstance(value, list):
                    result[t] = numpy.stack(value)
            self.dm0 = result if hasattr(mf, 'components') else result['e']
        return self.dm0

    def _predict_density(self, mf, mol):
        '''Build the predicted density inside kernel's numerical-error boundary.'''
        snapshots = self.history[-self.effective_history:][::-1]
        overlaps = self._overlaps(mf, mol)
        weights = self._weights(snapshots, overlaps, mol)
        result = {}
        for index, latest in enumerate(snapshots[0]):
            cprev = latest['c']
            if cprev.shape[-1] == 0:
                for t, spin in latest['keys']:
                    value = numpy.zeros((cprev.shape[-2], cprev.shape[-2]))
                    result.setdefault(t, [None, None])[spin] = value
                continue
            s = numpy.stack([overlaps[t] for t, spin in latest['keys']])
            trial = self._predict_orbitals(index, latest, snapshots, weights, s)
            purified_dm, trial = self._purify(trial, cprev, s)
            gram = trial.swapaxes(-1, -2).conj() @ s @ trial
            # CP2K qs_wf_history_methods.F:reorthogonalize_vectors calls
            # qs_mo_methods.F:make_basis_sm for ordinary occupied orbitals.
            # Factor C^H S C = L L^H and apply C <- C L^(-H), preserving
            # the occupied span. Both files use revision
            # dfe8787b4f0a22da8e73e9abe2b384fa021d2f53 cited above.
            chol = numpy.linalg.cholesky(gram)
            cguess = numpy.linalg.solve(chol, trial.swapaxes(-1, -2).conj()).swapaxes(-1, -2).conj()
            density = purified_dm if purified_dm is not None else cguess @ cguess.swapaxes(-1, -2).conj()
            if not numpy.isfinite(density).all():
                raise FloatingPointError('Predicted density contains nonfinite values')
            for i, (t, spin) in enumerate(latest['keys']):
                value = latest['factor'] * density[i]
                # DF-K uses mo_coeff/mo_occ tags to contract occupied factors
                # and reconstructs P = C occ C^H from them. Orbital prediction
                # and successful 'before' purification give an occupied-space
                # projector, represented by cguess and integer occupations.
                # In 'after' mode, averaging the purified one-sided density
                # can produce fractional occupations. Integer-occupation tags
                # would replace that density by cguess's projector inside K.
                # CPU df.df_jk.get_jk then contracts the full AO density;
                # GPU df.df_jk.decompose_rdm1 diagonalizes the density to
                # obtain factors, retaining fractional/negative occupations.
                # Both incur extra initial-K work relative to reusing the
                # occupied factors supplied by 'before'. Later SCF iterations
                # construct tagged densities from their updated orbitals.
                if purified_dm is None or self.hermitize == 'before':
                    value = lib.tag_array(value, mo_coeff=cguess[i],
                                          mo_occ=numpy.full(cguess.shape[-1], latest['factor']))
                if spin < 0:
                    result[t] = value
                else:
                    result.setdefault(t, [None, None])[spin] = value
        for t, value in result.items():
            if isinstance(value, list):
                result[t] = numpy.stack(value)
        return result if hasattr(mf, 'components') else result['e']

    predict = kernel


class ASPC(_Extrapolator):
    '''Kolafa/Kuhne occupied-projector prediction with optional density trial.

    ASPC weights: doi:10.1002/jcc.10385.
    Electronic orbital prediction: doi:10.1103/PhysRevLett.98.066401, Eq. 2.
    method='density' selects Dickinson's trial and McWeeny purification;
    hermitize selects averaging before or after that polynomial.
    '''
    scheme = 'aspc'
    _keys = {'method', 'hermitize'}

    def __init__(self, history_length=6, extrapolate_nuc=True,
                 method='orbital', hermitize='before'):
        if method not in ('orbital', 'density'):
            raise ValueError("method must be 'orbital' or 'density'")
        if hermitize not in ('before', 'after'):
            raise ValueError("hermitize must be 'before' or 'after'")
        self.method = method
        self.hermitize = hermitize
        super().__init__(history_length, extrapolate_nuc)

    @staticmethod
    def weights(length):
        # Kolafa's ASPC prediction coefficients, doi:10.1002/jcc.10385;
        # electronic projector application: Kuhne Eq. 2,
        # doi:10.1103/PhysRevLett.98.066401. j=1 is the latest snapshot.
        return [(-1)**(j+1) * j * comb(2*length, length-j) / comb(2*length-2, length-1)
                for j in range(1, length+1)]

    def _weights(self, snapshots, overlaps, mol):
        return self.weights(len(snapshots))

    def _purify(self, trial, cprev, s):
        if self.method == 'orbital':
            return None, trial
        if self.method != 'density':
            raise ValueError("method must be 'orbital' or 'density'")
        # Dickinson Eqs. 18-19, doi:10.1063/5.0332811, form the ASPC
        # density trial U P_previous = (U C_previous) C_previous^H.
        # U is the weighted sum of historical P_j S_j projectors, with
        # Kolafa's ASPC weights. McWeeny, doi:10.1103/RevModPhys.32.335,
        # supplies the purification polynomial cited beside Dickinson Eq. 22.
        #
        # PySCF scf.hf.get_jk/get_veff default to hermi=1, so the predicted
        # density must be Hermitian. Dickinson's one-sided U P_previous can
        # violate this even when every stored density is Hermitian. McWeeny
        # purification alone cannot repair it: a non-Hermitian idempotent
        # matrix is already a fixed point of the polynomial.
        #
        # To supply Hermitian SCF input, average P with its adjoint:
        # P <- (P + P^H)/2. 'before' supplies a Hermitian trial to
        # purification; 'after' preserves the printed trial
        # during purification and averages its output for PySCF. Averaging
        # commutes with the Hermitian congruence S^(1/2) P S^(1/2), so the AO
        # operation also gives the averaged density in the orthogonal frame.
        # Averaging the purified output can change its idempotency. kernel
        # therefore returns that matrix without integer-occupation MO tags;
        # DF-K then uses its more expensive general-density path for the
        # initial exchange build (the tag-assignment comment in kernel details
        # the CPU/GPU contractions). Failed idempotency/particle-count checks
        # return control to kernel's warning and previous-density fallback.
        #
        # Reconstructing a density from the trial's column space would mean
        # orthonormalizing B = U C_previous and returning
        # B (B^H S B)^(-1) B^H. This is exactly the orbital predictor's
        # density when B has full occupied rank. Such reconstruction would
        # collapse the density trial to orbital prediction. Averaging retains
        # the density trial's fractional occupations in 'after' mode.
        trial_dm = trial @ cprev.swapaxes(-1, -2).conj()
        if self.hermitize == 'before':
            trial_dm = (trial_dm + trial_dm.swapaxes(-1, -2).conj()) * .5
        # Dickinson Eq. 22 in AO coordinates: P <- 3 P S P - 2 P S P S P.
        # Eq. 23 supplies the idempotency test; trace checks particle count.
        candidate = trial_dm
        for iteration in range(50):
            sp = s @ candidate
            psp = candidate @ sp
            err = float(numpy.max(numpy.linalg.norm(psp - candidate, axis=(-2, -1))))
            if err < 1e-12 or not numpy.isfinite(err):
                break
            candidate = 3*psp - 2*(psp @ sp)
        trace_error = numpy.max(abs(numpy.trace(sp, axis1=-2, axis2=-1) - cprev.shape[-1]))
        if err < 1e-12 and float(trace_error) < 1e-8:
            # 'after' applies its Hermitian averaging here. In 'before' mode
            # this removes roundoff asymmetry accumulated during purification.
            density = (candidate + candidate.swapaxes(-1, -2).conj()) * .5
            return density, candidate @ s @ cprev
        raise numpy.linalg.LinAlgError('Density purification failed its idempotency or particle-count check')


class Polynomial(_Extrapolator):
    '''Polynomial occupied-projector prediction, CP2K's PS method.

    Source: wfi_extrapolate, wfi_ps_method_nr branch, in the pinned
    src/qs_wf_history_methods.F linked in the module documentation.
    '''
    scheme = 'polynomial'

    @staticmethod
    def weights(length):
        # CP2K PS uses alpha_j=-alpha_(j-1)*(K-j+1)/j, beginning at K.
        # The resulting binomial coefficients extrapolate equally spaced
        # snapshots at -1,...,-K to the next point at zero.
        return [(-1)**(j+1) * comb(length, j) for j in range(1, length+1)]

    def _weights(self, snapshots, overlaps, mol):
        return self.weights(len(snapshots))


class Fock(_Extrapolator):
    '''AO Fock polynomial prediction, doi:10.1039/B509494A, Eqs. 2.1-2.3.

    Observe existing SCF Fock calls with record_fock, or pass a converged
    raw Fock explicitly to update. Prediction diagonalizes the fitted Fock
    in the current metric, retaining each block's occupied state indices.
    '''
    scheme = 'fock'
    _keys = {'degree'}

    def __init__(self, history_length=4, extrapolate_nuc=True, degree=3):
        if isinstance(degree, bool) or not isinstance(degree, int) or degree < 0:
            raise ValueError('degree must be a nonnegative integer')
        self.degree = degree
        super().__init__(history_length, extrapolate_nuc)

    def clear(self):
        super().clear()
        self._last_fock = None

    @contextmanager
    def record_fock(self, mf):
        '''Retain the last Fock returned by the existing SCF call sequence.'''
        self._last_fock = None
        original = mf.get_fock
        instance_method = mf.__dict__.get('get_fock')

        def get_fock(*args, **kwargs):
            value = original(*args, **kwargs)
            self._last_fock = value
            return value

        # CPU/GPU SCF kernels already call get_fock with the final h1e/vhf/dm
        # for orbital-gradient evaluation, and again during an extra iteration.
        # This observer retains the returned raw Fock and its constraint terms.
        mf.get_fock = get_fock
        try:
            yield
        finally:
            if instance_method is None:
                del mf.get_fock
            else:
                mf.get_fock = instance_method

    @staticmethod
    def weights(length, degree=3):
        # Herbert/Head-Gordon Eqs. 2.1-2.3, doi:10.1039/B509494A.
        # Rescaled time improves Vandermonde conditioning while preserving
        # the fitted polynomial evaluated one step beyond the latest point.
        scale = max(length-1, 1)
        times = -numpy.arange(length)/scale
        design = numpy.vander(times, N=min(degree, length-1)+1, increasing=True)
        target = (1/scale)**numpy.arange(design.shape[1])
        return numpy.linalg.lstsq(design.T, target, rcond=None)[0]

    def _weights(self, snapshots, overlaps, mol):
        return self.weights(len(snapshots), self.degree)

    def _prepare_snapshot(self, snapshot, mf, fock):
        fock = self._last_fock if fock is None else fock
        if fock is None:
            raise ValueError('Use record_fock around SCF or supply fock to update')
        if not hasattr(mf, 'components'):
            fock = {'e': fock}
        for group in snapshot:
            matrices = []
            for t, spin in group['keys']:
                matrix = fock[t] if spin < 0 else fock[t][spin]
                matrices.append(self._asarray(matrix))
            group['fock'] = numpy.stack(matrices)
        self._last_fock = None

    def _predict_orbitals(self, index, latest, snapshots, weights, s):
        predicted = numpy.zeros_like(latest['fock'])
        for weight, snapshot in zip(weights, snapshots):
            predicted += weight * snapshot[index]['fock']
        trial = numpy.empty_like(latest['c'])
        w, v, retained = _overlap_eigh(s)
        ranks = retained.sum(axis=-1)
        # SCF.check_linear_dependency uses sorted eigenvectors above its
        # cutoff. Group equal retained ranks, as GPU NEO _eig_batch groups
        # equal matrix shapes, to preserve nuclear batching after reduction.
        for rank in numpy.unique(ranks):
            if rank < latest['c'].shape[-1]:
                raise numpy.linalg.LinAlgError('Retained overlap rank is smaller than the occupied space')
            members = numpy.flatnonzero(ranks == rank)
            x = v[members, :, -rank:] / numpy.sqrt(w[members, None, -rank:])
            reduced = x.swapaxes(-1, -2).conj() @ predicted[members] @ x
            if len(members) == 1:
                _, orbitals = numpy.linalg.eigh(reduced[0])
                orbitals = orbitals[None]
            else:
                _, orbitals = numpy.linalg.eigh(reduced)
            orbitals = x @ orbitals
            columns = numpy.broadcast_to(latest['occupied'][members, None, :],
                                         (len(members), trial.shape[-2], trial.shape[-1]))
            trial[members] = numpy.take_along_axis(orbitals, columns, axis=-1)
        return trial


class Grassmann(_Extrapolator):
    '''Descriptor-based tangent-space extrapolation and its QTR variant.

    Polack et al., doi:10.1021/acs.jctc.1c00751, Algorithm 1/SI, fits
    molecular Coulomb descriptors and extrapolates occupied subspaces in
    a fixed-reference tangent space. Pes et al., doi:10.1021/acs.jpclett.3c02098,
    Eqs. 3-5, pairs the historical descriptors to construct QTR weights.
    '''
    _keys = {'regularization', 'reversible'}

    def __init__(self, history_length=4, extrapolate_nuc=True,
                 regularization=None, reversible=False):
        if regularization is None:
            regularization = .001 if reversible else .01
        if regularization < 0 or not numpy.isfinite(regularization):
            raise ValueError('regularization must be finite and nonnegative')
        self.regularization = regularization
        self.reversible = reversible
        super().__init__(history_length, extrapolate_nuc)

    @property
    def scheme(self):
        return 'grassmann_qtr' if self.reversible else 'grassmann'

    def clear(self):
        super().clear()
        self._clear_method()

    def _clear_method(self):
        self.descriptors = []
        self.reference = []

    @staticmethod
    def _descriptor(mol):
        # Polack Eq. 6, doi:10.1021/acs.jctc.1c00751, and Pes Eq. 2,
        # doi:10.1021/acs.jpclett.3c02098: Coulomb-matrix descriptor with
        # fixed atom order, distances in Bohr and physical nuclear charges.
        # The NEO parent molecule retains those charges for quantum nuclei.
        # Store the symmetric Coulomb matrix. The QTR fit below vectorizes
        # its lower triangle as specified beside Pes Eq. 4.
        charges = mol.atom_charges()
        coords = mol.atom_coords()
        distance = numpy.linalg.norm(coords[:,None,:]-coords[None,:,:], axis=-1)
        numpy.fill_diagonal(distance, 1.)
        descriptor = charges[:,None]*charges[None,:] / distance
        numpy.fill_diagonal(descriptor, .5*charges**2.4)
        return descriptor

    def _prepare_snapshot(self, snapshot, mf, fock):
        for index, group in enumerate(snapshot):
            root, _ = _overlap_factors(group['s'])
            group['orth'] = root @ group['c']
            if len(self.reference) <= index:
                self.reference.append(group['orth'].copy())
        try:
            for index, group in enumerate(snapshot):
                if group['c'].shape[-1]:
                    group['tangent'] = _grassmann_log(self.reference[index], group['orth'])
                else:
                    group['tangent'] = numpy.zeros_like(group['orth'])
        except numpy.linalg.LinAlgError:
            # Polack Sec. 2.4, doi:10.1021/acs.jctc.1c00751: choose a new
            # reference chart when the occupied-space logarithm loses its
            # inverse. Restart bounded history at the accepted SCF solution.
            self.history = []
            self.descriptors = []
            self.reference = [group['orth'].copy() for group in snapshot]
            for group in snapshot:
                group['tangent'] = numpy.zeros_like(group['orth'])
            self.fallbacks += 1
        self.descriptors.append(self._descriptor(mf.mol))
        self.descriptors = self.descriptors[-self.history_length:]

    def _weights(self, snapshots, overlaps, mol):
        descriptors = self.descriptors[-len(snapshots):][::-1]
        target = self._descriptor(mol)
        if self.reversible:
            # Pes Eq. 4 uses each independent Coulomb-matrix entry once.
            # Duplicating off-diagonal entries changes the regularized fit.
            indices = numpy.tril_indices(mol.natm)
            descriptors = [d[indices] for d in descriptors]
            target = target[indices]
        return _fit_weights(descriptors, target, self.regularization,
                            reversible=self.reversible)

    def _predict_orbitals(self, index, latest, snapshots, weights, s):
        tangent = numpy.zeros_like(latest['c'])
        for weight, snapshot in zip(weights, snapshots):
            tangent += weight * snapshot[index]['tangent']
        _, inverse = _overlap_factors(s)
        reference = self.reference[index]
        return inverse @ _grassmann_exp(reference, tangent)


class ProjectedGrassmann(_Extrapolator):
    '''CP2K overlap-fitted projector prediction, including QTR fitting.

    Wilhelm et al., doi:10.48550/arXiv.2607.22916, Sec. X.2.1, Eq. 100,
    fits overlap descriptors and applies weighted historical P_j S_j
    operators to the latest occupied orbitals before reorthonormalization.
    CP2K's wfi_gext_proj_nr/wfi_gext_proj_qtr_nr branches implement this
    action; diff_fitting/tr_fitting construct its difference-fit/QTR weights.
    The pinned src/qs_wf_history_methods.F URL is in the module docstring.
    '''
    _keys = {'regularization', 'reversible'}

    def __init__(self, history_length=None, extrapolate_nuc=True,
                 regularization=1e-4, reversible=False):
        if regularization < 0 or not numpy.isfinite(regularization):
            raise ValueError('regularization must be finite and nonnegative')
        self.regularization = regularization
        self.reversible = reversible
        if history_length is None:
            history_length = 6 if reversible else 4
        super().__init__(history_length, extrapolate_nuc)

    @property
    def scheme(self):
        return 'gext_proj_qtr' if self.reversible else 'gext_proj'

    def _weights(self, snapshots, overlaps, mol):
        # CP2K diff_fitting/tr_fitting fit the electronic overlap. NEO shares
        # those coefficients across quantum nuclei: their rigid one-center
        # overlaps carry no trajectory information for a separate fit.
        electronic = []
        for snapshot in snapshots:
            for group in snapshot:
                match = next((i for i, key in enumerate(group['keys']) if key[0] == 'e'), None)
                if match is not None:
                    electronic.append(group['s'][match])
                    break
        return _fit_weights(electronic, overlaps['e'],
                            self.regularization, reversible=self.reversible,
                            difference=not self.reversible, normal_equations=True)


def Extrapolator(history_length=None, method='orbital', extrapolate_nuc=True,
                 scheme='gext_proj', degree=3, regularization=None,
                 hermitize='before'):
    '''Construct a method-specific StreamObject predictor.

    Defaults to ProjectedGrassmann with four snapshots. Direct constructors
    ASPC, Polynomial, Fock, Grassmann and ProjectedGrassmann expose each
    method's own options. History stays in memory; ASE enables prediction
    only when an extrapolator is supplied explicitly.
    '''
    if method not in ('orbital', 'density'):
        raise ValueError("method must be 'orbital' or 'density'")
    if method == 'density' and scheme != 'aspc':
        raise ValueError("Density purification requires scheme='aspc'")
    common = dict(extrapolate_nuc=extrapolate_nuc)
    if history_length is not None:
        common['history_length'] = history_length
    if scheme == 'aspc':
        return ASPC(**common, method=method, hermitize=hermitize)
    if scheme == 'polynomial':
        return Polynomial(**common)
    if scheme == 'fock':
        return Fock(**common, degree=degree)
    if scheme in ('grassmann', 'grassmann_qtr'):
        return Grassmann(**common, regularization=regularization, reversible=scheme == 'grassmann_qtr')
    if scheme in ('gext_proj', 'gext_proj_qtr'):
        if regularization is None:
            regularization = 1e-4
        return ProjectedGrassmann(**common, regularization=regularization,
                                  reversible=scheme == 'gext_proj_qtr')
    raise ValueError('Unknown extrapolation scheme: %s' % scheme)
