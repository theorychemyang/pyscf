import numpy
import unittest
from pyscf import neo
from pyscf.neo.polarizability import Polarizability
from pyscf.neo.efield import polarizability,SCFwithEfield
class KnownValues(unittest.TestCase):
    def test_polarizability(self):
        mol = neo.M(atom='''H 0. 0. 0.
                     F  0.5   0.5   .6''', basis='ccpvdz', 
                     quantum_nuc = ['H'])
        mf = neo.CDFT(mol,xc='b3lyp')
        mf.scf()
        p = Polarizability(mf)
        ref = polarizability(mf)
        self.assertAlmostEqual(p.polarizability()[0,0], ref[0,0],4)

    def test_polarizability_numerical(self):
        mol = neo.M(atom='''H 0. 0. 0.
                     F  0.5   0.5   .6''', basis='ccpvdz', 
                     quantum_nuc = ['H'])
        mf = neo.CDFT(mol,xc='b3lyp')
        mf.scf()
        p = Polarizability(mf)
        mf1 = SCFwithEfield(mol, xc='b3lyp')
        mf1.efield = numpy.array([0.0001,0,0])
        mf1.scf()
        u1 = mf1.dip_moment(unit='au')
        mf2 = SCFwithEfield(mol, xc='b3lyp')
        mf2.efield = numpy.array([-0.0001,0,0])
        mf2.scf()
        u2 = mf2.dip_moment(unit='au')
        a = (-u2 + u1) / 0.0002
        self.assertAlmostEqual(a[0], p.polarizability()[0,0],4)

if __name__ == "__main__":
    print("Full Tests for polarizability")
    unittest.main()