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
from pyscf import gto, scf
from pyscf.data import nist
from pyscf.qmmm import itrf
from pyscf.qmmm.mm_mole import create_mm_mol


def setUpModule():
    global mol, mm_coords, mm_charges, mm_radii, mm_mol
    mm_coords = [(1.369, 0.146,-0.395),
                 (1.894, 0.486, 0.335),
                 (0.451, 0.165,-0.083)]
    mm_charges = [-1.040, 0.520, 0.520]
    mm_radii = [0.63, 0.32, 0.32]
    mm_mol = create_mm_mol(mm_coords, mm_charges, mm_radii)
    mol = neo.M(
        atom='''O       -1.464   0.099   0.300
                H       -1.956   0.624  -0.340
                H       -1.797  -0.799   0.206''',
        basis='631G',
        nuc_basis='pb4d',
        quantum_nuc=['H'],
        mm_mol=mm_mol)

def tearDownModule():
    global mol, mm_coords, mm_charges, mm_radii, mm_mol

class KnowValues(unittest.TestCase):
    def test_energy(self):
        mf = neo.CDFT(mol)
        mf.mf_elec.xc = 'PBE0'
        e0 = mf.kernel()
        self.assertAlmostEqual(e0, -76.23177544660282, 8)

    def test_energy_no_neo(self):
        mol_1e8 = neo.M(
            atom='''O       -1.464   0.099   0.300
                    H       -1.956   0.624  -0.340
                    H       -1.797  -0.799   0.206''',
            basis='631G',
            nuc_basis='1e8',
            quantum_nuc=['H'],
            mm_mol=mm_mol)
        mf = neo.HF(mol_1e8)
        e0 = mf.kernel()
        # kinetic energy of 1e8 basis
        mass = mol_1e8.mass[1] * nist.ATOMIC_MASS / nist.E_MASS
        ke = mol_1e8.nuc[0].intor_symmetric('int1e_kin')[0,0] / mass

        mol_hf = gto.M(
            atom='''O       -1.464   0.099   0.300
                    H       -1.956   0.624  -0.340
                    H       -1.797  -0.799   0.206''',
            basis='631G')
        mf1 = itrf.mm_charge(scf.RHF(mol_hf), mm_coords, mm_charges, mm_radii)
        e1 = mf1.kernel()
        self.assertAlmostEqual(e0 - 2*ke, e1, 7)

    def test_energy_no_mm(self):
        mol0 = mol.copy()
        mol0.mm_mol = None
        mf = neo.CDFT(mol0)
        mf.mf_elec.xc = 'PBE0'
        e0 = mf.kernel()

        mm_mol1 = create_mm_mol(numpy.array(mm_coords)+100, mm_charges, mm_radii)
        mol1 = mol.copy()
        mol1.mm_mol = mm_mol1
        mf = neo.CDFT(mol1)
        mf.mf_elec.xc = 'PBE0'
        e1 = mf.kernel()
        self.assertAlmostEqual(e0, e1, 7)

        mm_mol2 = create_mm_mol(mm_coords, numpy.zeros_like(mm_charges), mm_radii)
        mol2 = mol.copy()
        mol2.mm_mol = mm_mol2
        mf = neo.CDFT(mol2)
        mf.mf_elec.xc = 'PBE0'
        e2 = mf.kernel()
        self.assertAlmostEqual(e0, e2, 8)

if __name__ == "__main__":
    print("Full Tests for CNEO-QMMM.")
    unittest.main()
