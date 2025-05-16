#!/usr/bin/env python

import unittest
from pyscf import neo, lib, scf


class KnownValues(unittest.TestCase):
    def test_H2O(self):
        mol = neo.M(atom='''O     0.0000   0.0000   0.0000;
                            H     0.7574   0.5868   0.0000;
                            X-H1  0.6962   0.6658   0.0000;
                            H    -0.7574   0.5868   0.0000;
                            X-H3 -0.6962   0.6658   0.0000''',
                    basis='ccpvdz', quantum_nuc=[1,3])
        '''
        The neo module will automatically label the symbols according to the order:

            O0 H1 (X-H1)(2) H3 (X-H3)(4)

        The ghost atoms will occupy spaces, therefore quantum_nuc=[1,3], not [1,2],
        because of the X-H1. X-H1 will be automatically assigned to H1, and X-H3 to H3.

        Alternatively just use quantum_nuc=['H']. Only need to care about this number
        if you only want a specific part of hydrogen to be quantized.

        Yes you can use more than one ghost for each quantum nucleus, like

            O
            H
            X-H1
            X-H1
            H
            X-H4
            X-H4
            X-H4

        with quantum_nuc=['H'] or [1,4].
        Don't forget you need to label the ghost atoms according to the global index of
        the non-ghost quantum nucleus that you want to put more basis at.
        '''
        mf = neo.KS(mol, xc='b3lypg')
        mf.conv_tol_grad = 1e-7
        mf.scf()
        '''
        The gradient is basis center gradient, including ghost atom center contributions.
        '''
        mf_grad = mf.nuc_grad_method()
        mf_grad.grid_response = True
        de = mf_grad.kernel()

        mfs = mf.as_scanner()
        e1 = mfs('''O     0.0000   0.0000   0.0000;
                    H     0.7574   0.5868   0.0000;
                    X-H1  0.6962   0.6618   0.0000;
                    H    -0.7574   0.5868   0.0000;
                    X-H3 -0.6962   0.6658   0.0000''')
        e2 = mfs('''O     0.0000   0.0000   0.0000;
                    H     0.7574   0.5868   0.0000;
                    X-H1  0.6962   0.6628   0.0000;
                    H    -0.7574   0.5868   0.0000;
                    X-H3 -0.6962   0.6658   0.0000''')
        e3 = mfs('''O     0.0000   0.0000   0.0000;
                    H     0.7574   0.5868   0.0000;
                    X-H1  0.6962   0.6638   0.0000;
                    H    -0.7574   0.5868   0.0000;
                    X-H3 -0.6962   0.6658   0.0000''')
        e4 = mfs('''O     0.0000   0.0000   0.0000;
                    H     0.7574   0.5868   0.0000;
                    X-H1  0.6962   0.6648   0.0000;
                    H    -0.7574   0.5868   0.0000;
                    X-H3 -0.6962   0.6658   0.0000''')
        e5 = mfs('''O     0.0000   0.0000   0.0000;
                    H     0.7574   0.5868   0.0000;
                    X-H1  0.6962   0.6668   0.0000;
                    H    -0.7574   0.5868   0.0000;
                    X-H3 -0.6962   0.6658   0.0000''')
        e6 = mfs('''O     0.0000   0.0000   0.0000;
                    H     0.7574   0.5868   0.0000;
                    X-H1  0.6962   0.6678   0.0000;
                    H    -0.7574   0.5868   0.0000;
                    X-H3 -0.6962   0.6658   0.0000''')
        e7 = mfs('''O     0.0000   0.0000   0.0000;
                    H     0.7574   0.5868   0.0000;
                    X-H1  0.6962   0.6688   0.0000;
                    H    -0.7574   0.5868   0.0000;
                    X-H3 -0.6962   0.6658   0.0000''')
        e8 = mfs('''O     0.0000   0.0000   0.0000;
                    H     0.7574   0.5868   0.0000;
                    X-H1  0.6962   0.6698   0.0000;
                    H    -0.7574   0.5868   0.0000;
                    X-H3 -0.6962   0.6658   0.0000''')
        fd = 1/280 * e1 + -4/105 * e2 + 1/5 * e3 + -4/5 * e4 \
             + 4/5 * e5 + -1/5 * e6 + 4/105 * e7 - 1/280 * e8
        self.assertAlmostEqual(de[2,1], fd/0.001*lib.param.BOHR, 5)

    def test_HF(self):
        mol = neo.M(atom='''H     0 0 0;
                            X-H0  0 0 0.2;
                            F     0 0 1.0''',
                    basis='ccpvdz', quantum_nuc=['H'])
        mf = neo.KS(mol, xc='M062X')
        mf.conv_tol_grad = 1e-7
        mf.scf()
        mf_grad = mf.nuc_grad_method()
        mf_grad.grid_response = True
        de = mf_grad.kernel()

        mfs = mf.as_scanner()
        e1 = mfs('''H     0 0 -0.001;
                    X-H0  0 0  0.2;
                    F     0 0  1.0''')
        e2 = mfs('''H     0 0  0.001;
                    X-H0  0 0  0.2;
                    F     0 0  1.0''')
        self.assertAlmostEqual(de[0,2], (e2-e1)/0.002*lib.param.BOHR, 5)
        e1 = mfs('''H     0 0  0;
                    X-H0  0 0  0.2;
                    F     0 0  0.999''')
        e2 = mfs('''H     0 0  0;
                    X-H0  0 0  0.2;
                    F     0 0  1.001''')
        self.assertAlmostEqual(de[2,2], (e2-e1)/0.002*lib.param.BOHR, 5)

    def test_ghost_swap(self):
        mol = neo.M(atom='''H     0 0 0;
                            X-H0  0 0 0.2;
                            F     0 0 1.0''',
                    basis='ccpvdz', quantum_nuc=[0,2])
        mf = neo.KS(mol, xc='PBE')
        e1 = mf.scf()
        mfs = mf.as_scanner()
        e2 = mfs('''H     0 0 0.2;
                    X-H0  0 0 0;
                    F     0 0 1.0''')
        self.assertAlmostEqual(e1, e2, 6)

if __name__ == "__main__":
    print("Full Tests for ghost atoms")
    unittest.main()
