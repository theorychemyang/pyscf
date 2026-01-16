import unittest
import numpy
from pyscf import gto
from pyscf.lo import orth

def setUpModule():
    global mol, mol1
    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None
    mol.atom = '''
         O    0.   0.       0
         1    0.   -0.757   0.587
         1    0.   0.757    0.587'''

    mol.basis = 'cc-pvdz'
    mol.build()
    mol1 = mol.copy(deep=True)
    coords = mol.atom_coords(unit='bohr').copy()
    coords[0, 1] += 1e-6
    mol1.set_geom_(coords, unit='bohr')

def tearDownModule():
    global mol, mol1
    del mol, mol1

class KnownValues(unittest.TestCase):
    def test_orth_ao_grad(self):
        c0 = orth.pre_orth_ao(mol, method='scf')

        c, c_grad = orth.orth_ao_grad(mol, 'lowdin', c0)
        c1 = orth.orth_ao(mol1, 'lowdin', c0)
        self.assertAlmostEqual(numpy.max(numpy.abs(c_grad(0)[1] - (c1-c)*1e6)), 0, 5)
        c, c_grad = orth.orth_ao_grad(mol, 'lowdin', None)
        c1 = orth.orth_ao(mol1, 'lowdin', None)
        self.assertAlmostEqual(numpy.max(numpy.abs(c_grad(0)[1] - (c1-c)*1e6)), 0, 5)
        c, c_grad = orth.orth_ao_grad(mol, 'lowdin', 'sto-3g')
        c1 = orth.orth_ao(mol1, 'lowdin', 'sto-3g')
        self.assertAlmostEqual(numpy.max(numpy.abs(c_grad(0)[1] - (c1-c)*1e6)), 0, 5)

        c, c_grad = orth.orth_ao_grad(mol, 'meta-lowdin', c0)
        c1 = orth.orth_ao(mol1, 'meta-lowdin', c0)
        self.assertAlmostEqual(numpy.max(numpy.abs(c_grad(0)[1] - (c1-c)*1e6)), 0, 5)
        c, c_grad = orth.orth_ao_grad(mol, 'meta-lowdin', None)
        c1 = orth.orth_ao(mol1, 'meta-lowdin', None)
        self.assertAlmostEqual(numpy.max(numpy.abs(c_grad(0)[1] - (c1-c)*1e6)), 0, 5)
        c, c_grad = orth.orth_ao_grad(mol, 'meta-lowdin', 'sto-3g')
        c1 = orth.orth_ao(mol1, 'meta-lowdin', 'sto-3g')
        self.assertAlmostEqual(numpy.max(numpy.abs(c_grad(0)[1] - (c1-c)*1e6)), 0, 5)

if __name__ == "__main__":
    print("Test orth ao grad")
    unittest.main()
