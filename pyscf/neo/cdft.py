#!/usr/bin/env python

'''
Constrained nuclear-electronic orbital density functional theory
'''
import numpy
from pyscf import lib
from pyscf import scf
from pyscf.neo.ks import KS

def get_fock_add_cdft(f, int1e_r):
    return numpy.einsum('xij,x->ij', int1e_r, f)

def position_analysis(mf, fock, s1e, int1e_r):
    mo_energy, mo_coeff = mf.eig(fock, s1e)
    mo_occ = mf.get_occ(mo_energy, mo_coeff)
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    return numpy.einsum('xij,ji->x', int1e_r, dm)

class CDFT(KS):
    '''
    Example:

    >>> mol = neo.Mole()
    >>> mol.build(atom='H 0.0 0.0 0.0; C 0.0 0.0 1.064; N 0.0 0.0 2.220', basis='ccpvdz')
    >>> mf = neo.CDFT(mol)
    >>> mf.scf()
    '''

    def __init__(self, mol, **kwargs):
        self.f = numpy.zeros((mol.natm, 3))
        KS.__init__(self, mol, **kwargs)

        # set up the Hamiltonian for each quantum nuclei in cNEO
        for i in range(len(self.mol.nuc)):
            mf = self.mf_nuc[i]
            mf.nuclei_expect_position = mf.mol.atom_coord(mf.mol.atom_index)
            # the position matrix with its origin shifted to nuclear expectation position
            s1e = mf.get_ovlp(mf.mol)
            mf.int1e_r = mf.mol.intor_symmetric('int1e_r', comp=3) \
                         - numpy.array([mf.nuclei_expect_position[i] * s1e for i in range(3)])

    def get_fock_add_cdft(self):
        f_add = []
        for i in range(len(self.mol.nuc)):
            mf = self.mf_nuc[i]
            ia = mf.mol.atom_index
            f_add.append(get_fock_add_cdft(self.f[ia], mf.int1e_r))
        return f_add

    def position_analysis(self, f, mf, fock0, s1e=None):
        ia = mf.mol.atom_index
        self.f[ia] = f
        if s1e is None:
            s1e = mf.get_ovlp(mf.mol)
        return position_analysis(mf, fock0 + get_fock_add_cdft(self.f[ia], mf.int1e_r),
                                 s1e, mf.int1e_r)

    def nuc_grad_method(self):
        from pyscf.neo.grad import Gradients
        return Gradients(self)
