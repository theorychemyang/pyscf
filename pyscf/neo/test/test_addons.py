import unittest

from pyscf import neo

class KnownValues(unittest.TestCase):
    def test_remove_lindep(self):
        mol = neo.M(verbose = 0,
                    atom = [('H', 0, 0, i*.5) for i in range(4)],
                    basis = ('sto-3g',[[0, [.002,1]]]))
        mf = neo.remove_linear_dep(neo.HF(mol), threshold=1e-8,
                                   lindep=1e-9).run()
        self.assertAlmostEqual(mf.e_tot, -1.6894257911772539, 7)

if __name__ == "__main__":
    print("Full Tests for neo.addons")
    unittest.main()
