#!/usr/bin/env python

import unittest
import numpy
from pyscf import neo, lib, scf
from pyscf.neo.efield import Polarizability, dipole_grad, SCFwithEfield, GradwithEfield, NEOwithEfield


class KnownValues(unittest.TestCase):
    def test_dipole_grad(self):
        mol = neo.M(atom='''C    0.5803070   0.4714570   0.4115280;
                            H+  -1.2184760  -0.1875100  -0.0282200;
                            Cl   1.8510530  -0.6893390  -0.0673470;
                            F    0.7865210   1.6508530  -0.2040030;
                            H    0.6182130   0.5951910   1.4994600;
                    ''', basis='ccpvdz')
        mf = neo.CDFT(mol, xc='b3lyp')
        mf.run()

        hess = mf.Hessian()
        hess.kernel()
        de1 = neo.hessian.dipole_grad(hess)
        print(de1)

        de2 = dipole_grad(mf)
        print(de2)

        mol1 = neo.M(atom='''C    0.5803070   0.4714570   0.4115280;
                             H+  -1.2184760  -0.1875100  -0.0282200;
                             Cl   1.8510530  -0.6893390  -0.0673470;
                             F    0.7865210   1.6508530  -0.2040030;
                             H    0.6172130   0.5951910   1.4994600;
                     ''', basis='ccpvdz')
        mf1 = neo.CDFT(mol1, xc='b3lyp')
        mf1.scf()

        mol2 = neo.M(atom='''C    0.5803070   0.4714570   0.4115280;
                             H+  -1.2184760  -0.1875100  -0.0282200;
                             Cl   1.8510530  -0.6893390  -0.0673470;
                             F    0.7865210   1.6508530  -0.2040030;
                             H    0.6192130   0.5951910   1.4994600;
                     ''', basis='ccpvdz')
        mf2 = neo.CDFT(mol2, xc='b3lyp')
        mf2.scf()

        de_finite_diff = (mf2.dip_moment(unit='au') - mf1.dip_moment(unit='au')) / 0.002 * lib.param.BOHR
        #print(de_finite_diff)

        self.assertAlmostEqual(abs(de1[-1, 0] - de_finite_diff).max(), 0, 5)
        self.assertAlmostEqual(abs(de2[-1, 0] - de_finite_diff).max(), 0, 5)


    def test_polarizability(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 0.9', basis='ccpvdz', quantum_nuc=['H'])
        mfCNEO = neo.CDFT(mol, xc='b3lyp')
        mfCNEO.run()
        polobj = Polarizability(mfCNEO)
        p = polobj.polarizability()

        mf1 = SCFwithEfield(mol, xc='b3lyp')
        mf1.efield = numpy.array([0, 0, 0.001])
        mf1.scf()
        dipole1 = mf1.dip_moment(unit='au')

        mf2 = SCFwithEfield(mol, xc='b3lyp')
        mf2.efield = numpy.array([0, 0, -0.001])
        mf2.scf()
        dipole2 = mf2.dip_moment(unit='au')

        mfNEO = neo.KS(mol,xc='b3lyp')
        mfNEO.run()
        polobj1 = Polarizability(mfNEO)
        p1 = polobj1.polarizability()

        mf3 = NEOwithEfield(mol, xc='b3lyp')
        mf3.efield = numpy.array([0, 0, 0.001])
        mf3.scf()
        dipole3 = mf3.dip_moment(unit='au')

        mf4 = NEOwithEfield(mol, xc='b3lyp')
        mf4.efield = numpy.array([0, 0, -0.001])
        mf4.scf()
        dipole4 = mf4.dip_moment(unit='au')

        self.assertAlmostEqual(p[-1,-1], (dipole1[-1] - dipole2[-1]) / 0.002, 4)
        self.assertAlmostEqual(p1[-1,-1], (dipole3[-1] - dipole4[-1]) / 0.002, 4)

    def test_grad_with_efield(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 0.8', basis='ccpvdz')
        mf = SCFwithEfield(mol, xc='b3lyp')
        mf.efield = numpy.array([0, 0, 0.01])
        mf.scf()
        grad = GradwithEfield(mf)
        grad.grid_response = True
        de = grad.kernel()

        mol1 = neo.M(atom='H 0 0 -0.001; F 0 0 0.8', basis='ccpvdz')
        mf1 = SCFwithEfield(mol1, xc='b3lyp')
        mf1.efield = numpy.array([0, 0, 0.01])
        e1 = mf1.scf()

        mol2 = neo.M(atom='H 0 0 0.001; F 0 0 0.8', basis='ccpvdz')
        mf2 = SCFwithEfield(mol2, xc='b3lyp')
        mf2.efield = numpy.array([0, 0, 0.01])
        e2 = mf2.scf()

        de_fd = (e2-e1) / 0.002 * lib.param.BOHR
        self.assertAlmostEqual(de[0,-1], de_fd, 5)


if __name__ == "__main__":
    print("Full Tests for efield")
    unittest.main()
