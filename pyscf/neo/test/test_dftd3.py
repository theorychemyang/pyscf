#!/usr/bin/env python

import numpy
import unittest
from pyscf import neo, gto, dft, lib
try:
    from pyscf.dispersion import dftd3
except ImportError:
    dftd3 = None

class KnownValues(unittest.TestCase):
    @unittest.skipIf(dftd3 is None, "requires the dftd3 library")
    def test_H2O(self):
        mol_ref = gto.M(atom='''O     0.0000   0.0000   0.0000;
                                H     0.7574   0.5868   0.0000;
                                H    -0.7574   0.5868   0.0000;''',
                        basis='ccpvdz')
        mf_ref = dft.RKS(mol_ref, xc='b3lypg')
        mf_ref.disp = 'd3bj'
        mf_ref.scf()
        mol= neo.M(atom='''O     0.0000   0.0000   0.0000;
                           H     0.7574   0.5868   0.0000;
                           H    -0.7574   0.5868   0.0000;''',
                   basis='ccpvdz', quantum_nuc=['H'])
        mf = neo.CDFT(mol, xc='b3lypg')
        mf.disp = 'd3bj'
        mf.scf()
        self.assertAlmostEqual(mf.scf_summary['dispersion'], mf_ref.scf_summary['dispersion'], 6)

        de = mf.nuc_grad_method().kernel()
        mfs = mf.as_scanner()
        e1 = mfs('''O     0.0000   0.0000   0.0000;
                    H     0.7574   0.5868   0.0000;
                    H    -0.7584   0.5868   0.0000;''')
        e2 = mfs('''O     0.0000   0.0000   0.0000;
                    H     0.7574   0.5868   0.0000;
                    H    -0.7564   0.5868   0.0000;''')
        self.assertAlmostEqual(de[2,0], (e2-e1)/0.002*lib.param.BOHR, 5)

if __name__ == "__main__":
    print("Full Tests for dftd")
    unittest.main()
