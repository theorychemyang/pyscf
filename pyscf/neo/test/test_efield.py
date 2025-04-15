#!/usr/bin/env python

import unittest
import numpy
from pyscf import neo, lib, scf
from pyscf.neo.efield import polarizability, dipole_grad, SCFwithEfield, GradwithEfield


class KnownValues(unittest.TestCase):
    def test_dipole_grad(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 0.9', basis='ccpvdz')
        mf = neo.CDFT(mol, xc='b3lyp')
        mf.run()

        hess = mf.Hessian()
        hess.kernel()
        de1 = neo.hessian.dipole_grad(hess)
        print(de1)

        de2 = dipole_grad(mf)
        print(de2)

        mol1 = neo.M(atom='H 0 0 -0.001; F 0 0 0.9', basis='ccpvdz')
        mf1 = neo.CDFT(mol1, xc='b3lyp')
        mf1.scf()

        mol2 = neo.M(atom='H 0 0 0.001; F 0 0 0.9', basis='ccpvdz')
        mf2 = neo.CDFT(mol2, xc='b3lyp')
        mf2.scf()

        de_finite_diff = (mf2.dip_moment(unit='au')[-1] - mf1.dip_moment(unit='au')[-1]) / 0.002 * lib.param.BOHR

        self.assertAlmostEqual(de1[0,-1,-1], de_finite_diff, 5)
        self.assertAlmostEqual(de2[0,-1,-1], de_finite_diff, 5)
        

    def test_polarizability(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 0.9', basis='ccpvdz')
        mf = neo.CDFT(mol, xc='b3lyp')
        mf.run()
        p = polarizability(mf)

        mf1 = SCFwithEfield(mol, xc='b3lyp')
        mf1.efield = numpy.array([0, 0, 0.001])
        mf1.scf()
        dipole1 = mf1.dip_moment(unit='au')

        mf2 = SCFwithEfield(mol, xc='b3lyp')
        mf2.efield = numpy.array([0, 0, -0.001])
        mf2.scf()
        dipole2 = mf2.dip_moment(unit='au')

        self.assertAlmostEqual(p[-1,-1], (dipole1[-1] - dipole2[-1]) / 0.002, 4)

    def test_grad_with_efield(self):
        mol = neo.M(atom='H 0 0 0; F 0 0 0.8', basis='ccpvdz')
        mf = SCFwithEfield(mol, xc='b3lyp')
        mf.efield = numpy.array([0, 0, 0.0001])
        mf.scf()
        grad = GradwithEfield(mf)
        de = grad.kernel()

        mol1 = neo.M(atom='H 0 0 -0.001; F 0 0 0.8', basis='ccpvdz')
        mf1 = SCFwithEfield(mol1, xc='b3lyp')
        mf1.efield = numpy.array([0, 0, 0.0001])
        e1 = mf1.scf()

        mol2 = neo.M(atom='H 0 0 0.001; F 0 0 0.8', basis='ccpvdz')
        mf2 = SCFwithEfield(mol2, xc='b3lyp')
        mf2.efield = numpy.array([0, 0, 0.0001])
        e2 = mf2.scf()
        
        de_fd = (e2-e1) / 0.002 * lib.param.BOHR
        self.assertAlmostEqual(de[0,-1], de_fd, 4)


if __name__ == "__main__":
    print("Full Tests for efield")
    unittest.main()
