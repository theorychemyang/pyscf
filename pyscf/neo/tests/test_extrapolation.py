import copy
import unittest
from unittest.mock import patch

import numpy
from pyscf import gto, neo, scf
from pyscf.neo.extrapolation import (Fock, Grassmann, Polynomial, Extrapolator, _fit_weights,
                                     _grassmann_exp, _grassmann_log)


class KnownValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mol = neo.M(atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587',
                    basis='sto3g', nuc_basis='pb4d', quantum_nuc=['H'], verbose=0)
        cls.mf = neo.CDFT(mol, xc='PBE', epc=None).density_fit(df_ne=True, df_nn=True)
        cls.mf.components['e'].grids.level = 1
        cls.mf.conv_tol = 1e-10
        cls.mf.kernel()
        assert cls.mf.converged

    def test_history_lifecycle(self):
        predictor = Extrapolator(2, scheme='aspc')
        predictor.update(self.mf, 0)
        predictor.update(self.mf, 0)
        self.assertEqual(len(predictor.history), 1)
        predictor.predict(self.mf, self.mf.mol, 1)
        self.assertEqual(predictor.effective_history, 1)
        predictor.update(self.mf, 1)
        predictor.update(self.mf, 2)
        self.assertEqual(len(predictor.history), 2)
        failed = copy.copy(self.mf)
        failed.converged = False
        predictor.update(failed, 3)
        self.assertEqual(predictor.last_step, 2)
        predictor.predict(self.mf, self.mf.mol, 7)
        self.assertEqual(len(predictor.history), 0)
        predictor.update(self.mf, 7)
        changed = self.mf.mol.copy()
        changed.spin = 2
        predictor.predict(self.mf, changed, 8)
        self.assertEqual(len(predictor.history), 0)

    def test_occupied_gauge(self):
        predictor = Extrapolator(4, scheme='aspc')
        predictor.update(self.mf, 0)
        original = predictor.predict(self.mf, self.mf.mol, 1)
        rotated = copy.deepcopy(predictor)
        rng = numpy.random.default_rng(12)
        for group in rotated.history[0]:
            rank = group['c'].shape[-1]
            q, _ = numpy.linalg.qr(rng.normal(size=(rank, rank)))
            group['c'] = group['c'] @ q
            group['cs'] = q.T @ group['cs']
        other = rotated.predict(self.mf, self.mf.mol, 1)
        for t in original:
            numpy.testing.assert_allclose(original[t], other[t], atol=1e-8)

    def test_polynomial_weights(self):
        for length in (1, 2, 4, 6):
            weights = Polynomial.weights(length)
            times = -numpy.arange(1, length+1)
            for power in range(length):
                self.assertAlmostEqual(numpy.dot(weights, times**power),
                                       1 if power == 0 else 0)

    def test_fock_weights(self):
        for length, degree in ((1, 0), (4, 2), (6, 3), (12, 6)):
            weights = Fock.weights(length, degree)
            times = -numpy.arange(length)/max(length-1, 1)
            for power in range(degree+1):
                numpy.testing.assert_allclose(numpy.dot(weights, times**power),
                                              (1/max(length-1, 1))**power, atol=1e-10)

    def test_grassmann_maps(self):
        rng = numpy.random.default_rng(14)
        reference, _ = numpy.linalg.qr(rng.normal(size=(2, 8, 3)))
        tangent = .03*rng.normal(size=reference.shape)
        tangent -= reference @ (reference.swapaxes(-1, -2) @ tangent)
        occupied = _grassmann_exp(reference, tangent)
        restored = _grassmann_log(reference, occupied)
        numpy.testing.assert_allclose(restored, tangent, atol=1e-10)
        rotation, _ = numpy.linalg.qr(rng.normal(size=(2, 3, 3)))
        rotated = _grassmann_log(reference, occupied @ rotation)
        numpy.testing.assert_allclose(rotated, tangent, atol=1e-10)
        numpy.testing.assert_allclose(occupied.swapaxes(-1, -2) @ occupied,
                                      numpy.broadcast_to(numpy.eye(3), (2, 3, 3)), atol=1e-10)

    def test_descriptor_fits(self):
        mol = self.mf.mol
        descriptor = Grassmann._descriptor(mol)
        self.assertEqual(descriptor.shape, (mol.natm, mol.natm))
        numpy.testing.assert_allclose(descriptor, descriptor.T)
        numpy.testing.assert_allclose(descriptor.diagonal(), .5*mol.atom_charges()**2.4)
        distance = numpy.linalg.norm(mol.atom_coord(0)-mol.atom_coord(1))
        self.assertAlmostEqual(descriptor[0,1], 8/distance)
        qtr = Grassmann(reversible=True, regularization=.001)
        for displacement in (0., .01, .03, .06):
            displaced = mol.copy()
            coords = mol.atom_coords()
            coords[1,0] += displacement
            displaced.set_geom_(coords, unit='Bohr')
            qtr.descriptors.append(qtr._descriptor(displaced))
        indices = numpy.tril_indices(mol.natm)
        expected = _fit_weights([d[indices] for d in qtr.descriptors[::-1]],
                                descriptor[indices], .001, reversible=True)
        numpy.testing.assert_allclose(qtr._weights([None]*4, None, mol), expected)
        times = -numpy.arange(1, 5)
        history = numpy.stack([times**i for i in range(3)], axis=1)
        target = numpy.array([1., 0., 0.])
        for reversible, difference in ((False, False), (False, True), (True, False)):
            weights = _fit_weights(history, target, 0,
                                   reversible=reversible, difference=difference)
            numpy.testing.assert_allclose(weights @ history, target, atol=1e-10)
            if reversible:
                self.assertAlmostEqual(weights[-1], -1)

    def test_added_schemes(self):
        for scheme in ('fock', 'grassmann', 'grassmann_qtr', 'gext_proj', 'gext_proj_qtr'):
            predictor = Extrapolator(4, scheme=scheme, degree=2)
            original = self.mf.get_fock.__func__
            for step in range(4):
                with predictor.record_fock(self.mf):
                    if scheme == 'fock':
                        self.mf.get_fock(s1e=self.mf.get_ovlp())
                self.assertIs(self.mf.get_fock.__func__, original)
                predictor.update(self.mf, step)
            mol = self.mf.mol.copy()
            coords = mol.atom_coords()
            coords[1,0] += .0001
            mol.set_geom_(coords, unit='Bohr')
            dm = predictor.predict(self.mf, mol, 4)
            overlaps = self.mf.get_ovlp(mol)
            for t, value in dm.items():
                factor = 2 if t == 'e' else 1
                numpy.testing.assert_allclose(value, value.T, atol=1e-8)
                numpy.testing.assert_allclose(value @ overlaps[t] @ value,
                                              factor*value, atol=1e-8)
                self.assertAlmostEqual(numpy.trace(value @ overlaps[t]),
                                       10 if t == 'e' else 1, 8)
            predictor.clear()
            self.assertEqual(predictor.history, [])
            if scheme.startswith('grassmann'):
                self.assertEqual(predictor.reference, [])
                self.assertEqual(predictor.descriptors, [])

    def test_polynomial_ps(self):
        nocc = numpy.count_nonzero(self.mf.mo_occ['e'])
        s = self.mf.get_ovlp()['e']
        for scheme, length in (('polynomial', 2), ('polynomial', 4), ('aspc', 2), ('aspc', 4)):
            predictor = Extrapolator(length, scheme=scheme)
            orbitals = []
            for step in range(length):
                changed = copy.copy(self.mf)
                changed.mo_coeff = {t: c.copy() for t, c in self.mf.mo_coeff.items()}
                c = changed.mo_coeff['e']
                occupied, virtual = c[:,0].copy(), c[:,nocc].copy()
                angle = .02*step**2
                c[:,0] = numpy.cos(angle)*occupied + numpy.sin(angle)*virtual
                c[:,nocc] = -numpy.sin(angle)*occupied + numpy.cos(angle)*virtual
                predictor.update(changed, step)
                orbitals.append(c[:,:nocc])
            dm = predictor.predict(changed, changed.mol, length)
            cprev = orbitals[-1]
            # CP2K PS applies the extrapolated historical projectors to C_latest.
            trial = numpy.zeros_like(cprev)
            for weight, c in zip(predictor.weights(length), orbitals[::-1]):
                trial += weight * (c @ c.T @ s @ cprev)
            w, v = numpy.linalg.eigh(trial.T @ s @ trial)
            normalized = trial @ ((v / numpy.sqrt(w)) @ v.T)
            numpy.testing.assert_allclose(dm['e'], 2*normalized @ normalized.T, atol=1e-10)
            if scheme == 'polynomial':
                continue
            purified = copy.deepcopy(predictor)
            purified.method = 'density'
            density = purified.predict(changed, changed.mol, length)
            overlaps = changed.get_ovlp()
            # Compare against the explicit orthogonal-basis polynomial.
            w, v = numpy.linalg.eigh(s)
            root = (v * numpy.sqrt(w)) @ v.T
            inverse = (v / numpy.sqrt(w)) @ v.T
            candidate = trial @ cprev.T
            candidate = root @ ((candidate + candidate.T)*.5) @ root
            for iteration in range(50):
                square = candidate @ candidate
                if numpy.linalg.norm(square-candidate) < 1e-12:
                    break
                candidate = 3*square - 2*square @ candidate
            expected = 2*inverse @ candidate @ inverse
            numpy.testing.assert_allclose(density['e'], expected, atol=1e-9)
            self.assertGreater(numpy.linalg.norm(density['e']-dm['e']), 1e-5)
            after = copy.deepcopy(purified)
            after.hermitize = 'after'
            after_density = after.predict(changed, changed.mol, length)
            candidate = root @ (trial @ cprev.T) @ root
            for iteration in range(50):
                square = candidate @ candidate
                if numpy.linalg.norm(square-candidate) < 1e-12:
                    break
                candidate = 3*square - 2*square @ candidate
            expected = inverse @ (candidate + candidate.T) @ inverse
            numpy.testing.assert_allclose(after_density['e'], expected, atol=1e-9)
            self.assertFalse(hasattr(after_density['e'], 'mo_coeff'))
            self.assertGreater(numpy.linalg.norm(after_density['e']-density['e']), 1e-7)
            self.assertGreater(numpy.linalg.norm(
                after_density['e'] @ s @ after_density['e'] - 2*after_density['e']), 1e-7)
            for t, value in density.items():
                numpy.testing.assert_allclose(value, value.T, atol=1e-8)
                reconstructed = (value.mo_coeff * value.mo_occ) @ value.mo_coeff.T
                numpy.testing.assert_allclose(value, reconstructed, atol=1e-8)
                factor = 2 if t == 'e' else 1
                numpy.testing.assert_allclose(value @ overlaps[t] @ value,
                                              factor*value, atol=1e-8)
                self.assertAlmostEqual(numpy.trace(value @ overlaps[t]),
                                       10 if t == 'e' else 1, 8)

    def test_empty_unrestricted_spin(self):
        mol = gto.M(atom='H 0 0 0', basis='sto3g', spin=1, verbose=0)
        mf = scf.UHF(mol).run()
        for method in ('orbital', 'density'):
            predictor = Extrapolator(4, method=method, scheme='aspc')
            predictor.update(mf, 0)
            dm = predictor.predict(mf, mol, 1)
            numpy.testing.assert_allclose(dm, mf.make_rdm1(), atol=1e-8)

    def test_linear_dependency(self):
        # Duplicate s functions exercise SCF's reduced MO rank while the
        # AO overlap retains a null direction on each hydrogen center.
        mol = gto.M(atom='H 0 0 0; H 0 0 1.4', unit='Bohr', verbose=0,
                    basis={'H': [[0, [1., 1.]], [0, [1., 1.]], [0, [.3, 1.]]]})
        mf = scf.RHF(mol).run()
        self.assertTrue(mf.converged)
        self.assertLess(mf.mo_coeff.shape[1], mol.nao)
        for scheme in ('aspc', 'polynomial', 'fock', 'grassmann', 'gext_proj'):
            predictor = Extrapolator(scheme=scheme)
            predictor.update(mf, 0, fock=mf.get_fock())
            dm = predictor.kernel(mf, mol, 1)
            s = mf.get_ovlp()
            numpy.testing.assert_allclose(dm, dm.T, atol=1e-8)
            numpy.testing.assert_allclose(dm @ s @ dm, 2*dm, atol=1e-8)
            self.assertAlmostEqual(numpy.trace(dm @ s), mol.nelectron, 8)

    def test_rank_loss_fallback(self):
        for method in ('orbital', 'density'):
            predictor = Extrapolator(4, method, scheme='aspc')
            for step in range(4):
                predictor.update(self.mf, step)
            # A zero projector combination loses occupied rank exactly.
            # Even at a changed geometry, failure must return the preceding
            # density unchanged and leave its converged history available.
            mol = self.mf.mol.copy()
            coords = mol.atom_coords()
            coords[1,0] += .01
            mol.set_geom_(coords, unit='Bohr')
            with patch.object(predictor, '_weights', return_value=[0.]*4), \
                    self.assertWarnsRegex(RuntimeWarning, 'using the previous density'):
                dm = predictor.predict(self.mf, mol, 4)
            self.assertEqual(predictor.fallbacks, 1)
            self.assertEqual(len(predictor.history), 4)
            reference = self.mf.make_rdm1()
            for t in dm:
                numpy.testing.assert_allclose(dm[t], reference[t], atol=1e-8)


if __name__ == '__main__':
    unittest.main()
