#!/usr/bin/env python
# Copyright 2014-2023 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy
import pyscf
from pyscf import lib
from pyscf import neo
from pyscf import gto, dft, grad
from pyscf.data import nist
from pyscf.qmmm import itrf
from pyscf.qmmm.mm_mole import create_mm_mol


def setUpModule():
    global mol, mol_1e6, mol_dft, mm_coords, mm_charges, mm_radii, mm_mol, \
           mm_mol_point, mm_mol_small_radius, mol_point, mol_small_radius
    mm_coords = [(1.369, 0.146,-0.395),
                 (1.894, 0.486, 0.335),
                 (0.451, 0.165,-0.083)]
    mm_charges = [-1.040, 0.520, 0.520]
    mm_radii = [0.63, 0.32, 0.32]
    mm_mol = create_mm_mol(mm_coords, mm_charges, mm_radii)
    mm_mol_point = create_mm_mol(mm_coords, mm_charges)
    mm_mol_small_radius = create_mm_mol(mm_coords, mm_charges, [1e-8]*3)
    atom='''O       -1.464   0.099   0.300
            H       -1.956   0.624  -0.340
            H       -1.797  -0.799   0.206'''
    mol = neo.M(atom=atom, basis='631G', nuc_basis='pb4d',
                quantum_nuc=['H'], mm_mol=mm_mol)
    mol_1e6 = neo.M(atom=atom, basis='631G', nuc_basis='1e6',
                    quantum_nuc=['H'], mm_mol=mm_mol)
    mol_dft = gto.M(atom=atom, basis='631G')
    mol_point = neo.M(atom=atom, basis='631G', nuc_basis='pb4d',
                      quantum_nuc=['H'], mm_mol=mm_mol_point)
    mol_small_radius = neo.M(atom=atom, basis='631G', nuc_basis='pb4d',
                             quantum_nuc=['H'], mm_mol=mm_mol_small_radius)

def tearDownModule():
    global mol, mol_1e6, mol_dft, mm_coords, mm_charges, mm_radii, mm_mol, \
           mm_mol_point, mm_mol_small_radius, mol_point, mol_small_radius

class KnowValues(unittest.TestCase):
    def test(self):
        # energy
        mf = neo.CDFT(mol, xc='PBE0')
        e0 = mf.kernel()
        self.assertAlmostEqual(e0, -76.23177544660282, 8)
        # gradient
        # qm gradient
        g = mf.Gradients()
        g_qm = g.grad()
        self.assertAlmostEqual(numpy.linalg.norm(g_qm), 0.05246990993385856, 6)
        # mm gradient
        g_mm = g.grad_mm()
        self.assertAlmostEqual(numpy.linalg.norm(g_mm), 0.02085800635723028, 6)

        # finite difference gradient
        # qm gradient
        mfs = mf.as_scanner() # as_scanner fixes MM atoms
        # change O atom
        e1 = mfs('''O       -1.464   0.099   0.299
                    H       -1.956   0.624  -0.340
                    H       -1.797  -0.799   0.206''')
        e2 = mfs('''O       -1.464   0.099   0.301
                    H       -1.956   0.624  -0.340
                    H       -1.797  -0.799   0.206''')
        self.assertAlmostEqual(g_qm[0,2], (e2-e1)/0.002*lib.param.BOHR, 5)
        # change 1st H atom
        e1 = mfs('''O       -1.464   0.099   0.300
                    H       -1.957   0.624  -0.340
                    H       -1.797  -0.799   0.206''')
        e2 = mfs('''O       -1.464   0.099   0.300
                    H       -1.955   0.624  -0.340
                    H       -1.797  -0.799   0.206''')
        self.assertAlmostEqual(g_qm[1,0], (e2-e1)/0.002*lib.param.BOHR, 5)
        # mm gradient, change 1st MM atom
        mm_coords1 = [(1.369, 0.147,-0.395),
                      (1.894, 0.486, 0.335),
                      (0.451, 0.165,-0.083)]
        mm_mol1 = create_mm_mol(mm_coords1, mm_charges, mm_radii)
        mol1 = mol.copy()
        mol1.mm_mol = mm_mol1
        mf1 = neo.CDFT(mol1, xc='PBE0')
        e1 = mf1.kernel()
        mm_coords2 = [(1.369, 0.145,-0.395),
                      (1.894, 0.486, 0.335),
                      (0.451, 0.165,-0.083)]
        mm_mol2 = create_mm_mol(mm_coords2, mm_charges, mm_radii)
        mol2 = mol.copy()
        mol2.mm_mol = mm_mol2
        mf2 = neo.CDFT(mol2, xc='PBE0')
        e2 = mf2.kernel()
        self.assertAlmostEqual(g_mm[0,1], (e1-e2)/0.002*lib.param.BOHR, 6)

    def test_no_neo(self):
        mf = neo.CDFT(mol_1e6, xc='PBE0')
        e0 = mf.kernel()
        # kinetic energy of 1e6 basis
        mass = mol_1e6.mass[1] * nist.ATOMIC_MASS / nist.E_MASS
        ke = mol_1e6.components['n1'].intor_symmetric('int1e_kin')[0,0] / mass

        mf_dft = dft.RKS(mol_dft, xc='PBE0')
        mf1 = itrf.mm_charge(mf_dft, mm_coords, mm_charges, mm_radii)
        e1 = mf1.kernel()
        self.assertAlmostEqual(e0 - 2*ke, e1, 5)

        g = mf.Gradients()
        g_dft = itrf.mm_charge_grad(grad.RKS(mf1), mm_coords, mm_charges, mm_radii)
        numpy.testing.assert_array_almost_equal(g.grad(), g_dft.grad())
        numpy.testing.assert_array_almost_equal(g.grad_mm(), g_dft.grad_hcore_mm(mf1.make_rdm1())
                                                +g_dft.grad_nuc_mm())

    def test_no_mm(self):
        mol0 = mol.copy()
        mol0.mm_mol = None
        mf = neo.CDFT(mol0, xc='PBE0')
        e0 = mf.kernel()
        g = mf.Gradients()
        g_qm0 = g.grad()

        mm_mol1 = create_mm_mol(numpy.array(mm_coords)+100, mm_charges, mm_radii)
        mol1 = mol.copy()
        mol1.mm_mol = mm_mol1
        mf = neo.CDFT(mol1, xc='PBE0')
        e1 = mf.kernel()
        self.assertAlmostEqual(e0, e1, 7)
        g = mf.Gradients()
        g_qm1 = g.grad()
        g_mm1 = g.grad_mm()
        numpy.testing.assert_array_almost_equal(g_qm0, g_qm1)
        self.assertAlmostEqual(numpy.linalg.norm(g_mm1), 0.0, 6)

        mm_mol2 = create_mm_mol(mm_coords, numpy.zeros_like(mm_charges), mm_radii)
        mol2 = mol.copy()
        mol2.mm_mol = mm_mol2
        mf = neo.CDFT(mol2, xc='PBE0')
        e2 = mf.kernel()
        self.assertAlmostEqual(e0, e2, 8)
        g = mf.Gradients()
        g_qm2 = g.grad()
        g_mm2 = g.grad_mm()
        numpy.testing.assert_array_almost_equal(g_qm0, g_qm2)
        self.assertAlmostEqual(numpy.linalg.norm(g_mm2), 0.0, 6)

    def test_point_and_small_radius_gaussian(self):
        mf_point = neo.CDFT(mol_point, xc='PBE0')
        mf_small_radius = neo.CDFT(mol_small_radius, xc='PBE0')
        self.assertAlmostEqual(mf_point.kernel(), mf_small_radius.kernel())

        g_point = mf_point.Gradients()
        g_small_radius = mf_small_radius.Gradients()
        numpy.testing.assert_array_almost_equal(g_point.grad(), g_small_radius.grad())
        numpy.testing.assert_array_almost_equal(g_point.grad_mm(), g_small_radius.grad_mm())


if __name__ == "__main__":
    print("Full Tests for CNEO-QMMM.")
    unittest.main()
