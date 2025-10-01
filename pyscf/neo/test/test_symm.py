#!/usr/bin/env python

import unittest
from pyscf import neo

class KnownValues(unittest.TestCase):
    def test_H2(self):
        mol = neo.M(atom='H 0 0 0; H 1 0 0', basis='ccpvdz', symmetry=True)
        self.assertEqual(mol.groupname, 'Dooh')
        self.assertEqual(mol.components['e'].groupname, 'Dooh')
        self.assertEqual(mol.components['n0'].groupname, 'Coov')
        self.assertEqual(mol.components['n1'].groupname, 'Coov')

        mol = neo.M(atom='H 0 0 0; H 1 0 0', basis='ccpvdz', symmetry=True, quantum_nuc=[0])
        self.assertEqual(mol.groupname, 'Coov')
        self.assertEqual(mol.components['e'].groupname, 'Coov')
        self.assertEqual(mol.components['n0'].groupname, 'Coov')

        mol = neo.M(atom='H0 0 0 0; H1 1 0 0', basis={'H0' : 'ccpvdz', 'H1' : 'STO-3G'}, symmetry=True)
        self.assertEqual(mol.groupname, 'Coov')
        self.assertEqual(mol.components['e'].groupname, 'Coov')
        self.assertEqual(mol.components['n0'].groupname, 'Coov')
        self.assertEqual(mol.components['n1'].groupname, 'Coov')

    def test_HD(self):
        mol = neo.M(atom='H 0 0 0; H+ 1 0 0', basis={'H' : 'ccpvdz'}, symmetry=True)
        self.assertEqual(mol.groupname, 'Coov')
        self.assertEqual(mol.components['e'].groupname, 'Coov')
        self.assertEqual(mol.components['n0'].groupname, 'Coov')
        self.assertEqual(mol.components['n1'].groupname, 'Coov')

    def test_NH3(self):
        mol = neo.M(atom='''N     0.000000000000000     0.000000000000000     0.107534471726190;
                            H     0.811242503345203     0.000000000000000    -0.498021693607143;
                            H    -0.405621251672602     0.702556616526628    -0.498021693607143;
                            H    -0.405621251672602    -0.702556616526628    -0.498021693607143''',
                    basis='ccpvdz', symmetry=True)
        self.assertEqual(mol.topgroup, 'C3v')
        self.assertEqual(mol.components['e'].topgroup, 'C3v')
        self.assertEqual(mol.components['n1'].topgroup, 'Cs')
        self.assertEqual(mol.components['n2'].topgroup, 'Cs')
        self.assertEqual(mol.components['n3'].topgroup, 'Cs')

        mol = neo.M(atom='''N     0.000000000000000     0.000000000000000     0.107534471726190;
                            H     0.811242503345203     0.000000000000000    -0.498021693607143;
                            H    -0.405621251672602     0.702556616526628    -0.498021693607143;
                            H    -0.405621251672602    -0.702556616526628    -0.498021693607143''',
                    basis='ccpvdz', symmetry=True, quantum_nuc=[1,2])
        self.assertEqual(mol.topgroup, 'Cs')
        self.assertEqual(mol.components['e'].topgroup, 'Cs')
        self.assertEqual(mol.components['n1'].topgroup, 'C1')
        self.assertEqual(mol.components['n2'].topgroup, 'C1')

        mol = neo.M(atom='''N     0.000000000000000     0.000000000000000     0.107534471726190;
                            H     0.811242503345203     0.000000000000000    -0.498021693607143;
                            H    -0.405621251672602     0.702556616526628    -0.498021693607143;
                            H3   -0.405621251672602    -0.702556616526628    -0.498021693607143''',
                    basis={'N' : 'ccpvdz', 'H' : 'ccpvdz', 'H3' : '6-31G'}, symmetry=True)
        self.assertEqual(mol.topgroup, 'Cs')
        self.assertEqual(mol.components['e'].topgroup, 'Cs')
        self.assertEqual(mol.components['n1'].topgroup, 'C1')
        self.assertEqual(mol.components['n2'].topgroup, 'C1')
        self.assertEqual(mol.components['n3'].topgroup, 'Cs')

    def test_NH2D(self):
        mol = neo.M(atom='''N     0.000000000000000     0.000000000000000     0.107534471726190;
                            H     0.811242503345203     0.000000000000000    -0.498021693607143;
                            H    -0.405621251672602     0.702556616526628    -0.498021693607143;
                            H+   -0.405621251672602    -0.702556616526628    -0.498021693607143''',
                    basis='ccpvdz', symmetry=True)
        self.assertEqual(mol.topgroup, 'Cs')
        self.assertEqual(mol.components['e'].topgroup, 'Cs')
        self.assertEqual(mol.components['n1'].topgroup, 'C1')
        self.assertEqual(mol.components['n2'].topgroup, 'C1')
        self.assertEqual(mol.components['n3'].topgroup, 'Cs')

    def test_NHDHeMu(self):
        mol = neo.M(atom='''N     0.000000000000000     0.000000000000000     0.107534471726190;
                            H     0.811242503345203     0.000000000000000    -0.498021693607143;
                            H#   -0.405621251672602     0.702556616526628    -0.498021693607143;
                            H+   -0.405621251672602    -0.702556616526628    -0.498021693607143''',
                    basis='ccpvdz', symmetry=True)
        self.assertEqual(mol.topgroup, 'C1')
        self.assertEqual(mol.components['e'].topgroup, 'C1')
        self.assertEqual(mol.components['n1'].topgroup, 'C1')
        self.assertEqual(mol.components['n2'].topgroup, 'C1')
        self.assertEqual(mol.components['n3'].topgroup, 'C1')

    def test_H2O(self):
        mol = neo.M(atom='''O     0.0000   0.0000   0.0000;
                            H     0.7574   0.5868   0.0000;
                            H    -0.7574   0.586803 0.0000''',
                    basis='ccpvdz', symmetry=True)
        self.assertEqual(mol.topgroup, 'C2v')
        self.assertEqual(mol.components['e'].topgroup, 'C2v')
        self.assertEqual(mol.components['n1'].topgroup, 'Cs')
        self.assertEqual(mol.components['n2'].topgroup, 'Cs')
        mf = neo.CDFT(mol, xc='HF').run()
        grad = mf.nuc_grad_method().kernel()
        self.assertAlmostEqual(grad[1,0], -grad[2,0], 7)
        mol.set_geom_('''O     0.0000   0.0000   0.0000;
                         H     0.7574   0.5868   0.0000;
                         H    -0.7574   0.5868   0.0000''')
        self.assertEqual(mol.topgroup, 'C2v')
        mol.set_geom_('''O     0.0000   0.0000   0.0000;
                         H     0.7574   0.5868   0.0000;
                         H    -0.7574   0.58681  0.0000''')
        self.assertEqual(mol.topgroup, 'Cs')

if __name__ == "__main__":
    print("Full Tests for symmetry in CNEO")
    unittest.main()
