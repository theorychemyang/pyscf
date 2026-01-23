import unittest
from pyscf import neo

def setUpModule():
    global mol
    mol = neo.M(atom='''H 0 0 0; C 0 0 1.064; N 0 0 2.220''', basis='ccpvdz',
                quantum_nuc=[0])

def tearDownModule():
    global mol
    del mol

class KnownValues(unittest.TestCase):
    def test_scf_epc17_2(self):
        mf = neo.CDFT(mol, xc='b3lyp5', epc='17-2')
        e = mf.scf()
        mf1 = mf.copy()
        mf1.components['e'].xc = 'PBE'
        mf1.epc = None
        mf1.scf()
        self.assertAlmostEqual(mf.scf(), e, 8)

if __name__ == "__main__":
    print("Full Tests for neo copy function")
    unittest.main()
