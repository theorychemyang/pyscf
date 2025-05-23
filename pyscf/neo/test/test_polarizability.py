import numpy
from numpy.testing import assert_array_almost_equal
import unittest
import time
from pyscf import scf
from pyscf import neo
from pyscf.neo.polarizability import Polarizability
from pyscf.neo.efield import polarizability

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

    def test_polarizability_timing(self):
        """Test and compare execution time of both polarizability implementations"""
        n_runs = 20  # Number of runs for averaging
        
        mol = neo.M(atom='''H 0. 0. 0.
                     F  0.5   0.5   .6''', basis='ccpvdz', 
                     quantum_nuc = ['H'])
        mf = neo.CDFT(mol,xc='b3lyp')
        mf.scf()

        # Time Polarizability class implementation
        times_me = []
        for _ in range(n_runs):
            t0 = time.time()
            p = Polarizability(mf)
            pol1 = p.polarizability()
            t1 = time.time()
            times_me.append(t1 - t0)
        
        # Time efield.polarizability implementation
        times_efield = []
        for _ in range(n_runs):
            t0 = time.time()
            pol2 = polarizability(mf)
            t1 = time.time()
            times_efield.append(t1 - t0)

        # Calculate statistics
        avg_me = numpy.mean(times_me)
        std_me = numpy.std(times_me)
        avg_efield = numpy.mean(times_efield)
        std_efield = numpy.std(times_efield)

        print('\nTiming Results (averaged over {} runs):'.format(n_runs))
        print(f'Polarizability class time: {avg_me:.6f} ± {std_me:.6f} seconds')
        print(f'efield.polarizability time: {avg_efield:.6f} ± {std_efield:.6f} seconds')
        
        # Verify results match
        assert_array_almost_equal(pol1, pol2, decimal=4)

if __name__ == "__main__":
    print("Full Tests for polarizability")
    unittest.main()