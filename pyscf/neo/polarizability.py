'''
Analytical polarizability for CNEO
'''
from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.neo import cphf
from pyscf.neo import _response_functions
from pyscf.neo.hessian import gen_vind

def polarizability(polobj,with_cphf=True):
    log = logger.new_logger(polobj)
    mf = polobj._scf
    mol = mf.mol
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    #orbv = mo_coeff[:,~occidx]

    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    charge_center = numpy.einsum('i,ix->x', charges, coords) / charges.sum()
    h1 = {}
    s1 = {}
    for t, comp in mf.components.items():
        with comp.mol.with_common_orig(charge_center):
            int1e_r = comp.mol.intor_symmetric('int1e_r', comp=3)
        occidx = mo_occ[t] > 0
        mocc = mo_coeff[t][:, occidx]
        h1[t] = lib.einsum('xuv, ui, vj -> xij', int1e_r, mo_coeff[t], mocc) * comp.charge
        s1[t] = numpy.zeros_like(h1[t])
    vind = polobj.gen_vind(mf, mo_coeff, mo_occ)
    if with_cphf:
        mo1 = cphf.solve(vind, mo_energy, mo_occ, h1, s1,with_f1=True,
                         max_cycle=polobj.max_cycle_cphf,tol=polobj.conv_tol,
                         verbose=log)[0]
    else:
        raise NotImplementedError('without cphf is not implemented yet')

    e2 = lib.einsum('xpi,ypi->xy', h1['e'], mo1['e'])
    # *-1 from the definition of dipole moment. *2 for double occupancy
    e2 = (e2 + e2.T) * -2

    if mf.verbose >= logger.INFO:
        xx, yy, zz = e2.diagonal()
        log.note('Isotropic polarizability %.12g', (xx+yy+zz)/3)
        log.note('Polarizability anisotropy %.12g',
                 (.5 * ((xx-yy)**2 + (yy-zz)**2 + (zz-xx)**2))**.5)
        log.debug('Static polarizability tensor\n%s', e2)
    return e2


def hyper_polarizability(polobj, with_cphf=True):
    return NotImplementedError('hyperpolarizability is not implemented yet')


def polarizability_with_freq(polobj, freq, with_cphf=True):
    return NotImplementedError('frequency-dependent polarizability is not implemented yet')


class Polarizability(lib.StreamObject):
    def __init__(self, mf):
        mol = mf.mol
        self.mol = mol
        self.verbose = mol.verbose
        self.stdout = mol.stdout
        self._scf = mf

        self.cphf = True
        self.max_cycle_cphf = 100
        ### default: 1e-4 is enough for polarizability(the same order of magnitude as the numerical error)
        self.conv_tol = 1e-4

        self._keys = set(self.__dict__.keys())  

    def gen_vind(self, mf, mo_coeff, mo_occ):
        return gen_vind(mf, mo_coeff, mo_occ)

    polarizability = polarizability
    polarizability_with_freq = polarizability_with_freq
    hyper_polarizability = hyper_polarizability

if __name__ == '__main__':
    from pyscf import neo
    import time
    from pyscf.neo.efield import polarizability as polarizability_efield
    #compar time of both polarizability implementations
    n_runs = 20 
    
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
        pol2 = polarizability_efield(mf)
        t1 = time.time()
        times_efield.append(t1 - t0)
        
    avg_me = numpy.mean(times_me)
    std_me = numpy.std(times_me)
    avg_efield = numpy.mean(times_efield)
    std_efield = numpy.std(times_efield)

    print('\nTiming Results (averaged over {} runs):'.format(n_runs))
    print(f'Polarizability class time: {avg_me:.6f} ± {std_me:.6f} seconds')
    print(f'efield.polarizability time: {avg_efield:.6f} ± {std_efield:.6f} seconds')