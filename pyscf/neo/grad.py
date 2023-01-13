#!/usr/bin/env python

'''
Analytical nuclear gradient for constrained nuclear-electronic orbital
'''
import numpy
from pyscf import lib, neo, scf
from pyscf.lib import logger
from pyscf.data import nist
from pyscf.grad.rhf import _write

# TODO: create Gradients class from GradientsMixin
#from pyscf.grad import rhf as rhf_grad
#class Gradients(rhf_grad.GradientsMixin):

class Gradients(lib.StreamObject):
    '''
    Examples::

    >>> from pyscf import neo
    >>> mol = neo.Mole()
    >>> mol.build(atom='H 0 0 0.00; C 0 0 1.064; N 0 0 2.220', basis='ccpvtz')
    >>> mf = neo.CDFT(mol)
    >>> mf.mf_elec.xc = 'b3lyp'
    >>> mf.scf()
    >>> g = neo.Gradients(mf)
    >>> g.kernel()
    '''

    def __init__(self, scf_method):
        self.verbose = scf_method.verbose
        self.mol = scf_method.mol
        self.base = scf_method
        self.max_memory = self.mol.max_memory
        self.grad = self.kernel
        self.grid_response = None
        if self.base.epc is not None:
            raise NotImplementedError('Gradient with epc is not implemented')

    def grad_elec(self, atmlst=None):
        'gradients of electrons and classic nuclei'
        g = self.base.mf_elec.nuc_grad_method()
        if self.grid_response is not None:
            g.grid_response = self.grid_response
        g.verbose = self.verbose - 1
        return g.grad(atmlst=atmlst)

    def get_hcore(self, mol):
        'part of the gradients of core Hamiltonian of quantum nucleus'
        i = mol.atom_index
        mass = self.mol.mass[i] * nist.ATOMIC_MASS / nist.E_MASS
        # minus sign for the derivative is taken w.r.t 'r' instead of 'R'
        h = -mol.intor('int1e_ipkin', comp=3) / mass
        h += mol.intor('int1e_ipnuc', comp=3) * self.mol.atom_charge(i)
        return h

    def hcore_deriv(self, atm_id, mol):
        '''The change of Coulomb interactions between quantum and classical
        nuclei due to the change of the coordinates of classical nuclei'''
        i = mol.atom_index
        with mol.with_rinv_as_nucleus(atm_id):
            vrinv = mol.intor('int1e_iprinv', comp=3) # <\nabla|1/r|>
            vrinv *= (mol.atom_charge(atm_id)*self.mol.atom_charge(i))

        return vrinv + vrinv.transpose(0,2,1)

    def grad_jcross_elec_nuc(self):
        '''get the gradient for the cross term of Coulomb interactions between
        electrons and quantum nucleus'''
        jcross = 0
        for i in range(len(self.mol.nuc)):
            index = self.mol.nuc[i].atom_index
            jcross -= scf.jk.get_jk((self.mol.elec, self.mol.elec,
                                     self.mol.nuc[i], self.mol.nuc[i]),
                                    self.base.dm_nuc[i], scripts='ijkl,lk->ij',
                                    intor='int2e_ip1', comp=3, aosym='s2kl') \
                      * self.mol.atom_charge(index)
        return jcross

    def grad_jcross_nuc_elec(self, mol):
        '''get the gradient for the cross term of Coulomb interactions between
        quantum nucleus and electrons'''
        i = mol.atom_index
        jcross = -scf.jk.get_jk((mol, mol, self.mol.elec, self.mol.elec),
                                self.base.dm_elec, scripts='ijkl,lk->ij',
                                intor='int2e_ip1', comp=3, aosym='s2kl') \
                 * self.mol.atom_charge(i)
        return jcross

    def grad_jcross_nuc_nuc(self, mol):
        '''get the gradient for the cross term of Coulomb interactions between
        quantum nuclei'''
        i = mol.atom_index
        jcross = numpy.zeros((3, mol.nao_nr(), mol.nao_nr()))
        for j in range(len(self.mol.nuc)):
            k = self.mol.nuc[j].atom_index
            if k != i:
                jcross -= scf.jk.get_jk((mol, mol, self.mol.nuc[j], self.mol.nuc[j]),
                                        self.base.dm_nuc[j], scripts='ijkl,lk->ij',
                                        intor='int2e_ip1', comp=3, aosym='s2kl') \
                          * self.mol.atom_charge(i) * self.mol.atom_charge(k)
        return jcross

    def kernel(self, atmlst=None):
        'Unit: Hartree/Bohr'
        if atmlst == None:
            atmlst = range(self.mol.natm)

        self.de = numpy.zeros((len(atmlst), 3))
        aoslices = self.mol.aoslice_by_atom()

        jcross_elec_nuc = self.grad_jcross_elec_nuc()
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia, 2:]
            # *2 for c.c.
            self.de[k] -= numpy.einsum('xij,ij->x', jcross_elec_nuc[:,p0:p1], self.base.dm_elec[p0:p1])*2

            if self.mol.quantum_nuc[ia] == True:
                for i in range(len(self.mol.nuc)):
                    if self.mol.nuc[i].atom_index == ia:
                        self.de[k] += numpy.einsum('xij,ij->x', self.get_hcore(self.mol.nuc[i]), self.base.dm_nuc[i])*2
                        jcross_nuc_elec = self.grad_jcross_nuc_elec(self.mol.nuc[i])
                        self.de[k] -= numpy.einsum('xij,ij->x', jcross_nuc_elec, self.base.dm_nuc[i])*2
                        jcross_nuc_nuc = self.grad_jcross_nuc_nuc(self.mol.nuc[i])
                        self.de[k] += numpy.einsum('xij,ij->x', jcross_nuc_nuc, self.base.dm_nuc[i])*2
            else:
                for i in range(len(self.mol.nuc)):
                    h1ao = self.hcore_deriv(ia, self.mol.nuc[i])
                    self.de[k] += numpy.einsum('xij,ij->x', h1ao, self.base.dm_nuc[i])

        grad_elec = self.grad_elec()
        self.de = grad_elec + self.de
        self._finalize()
        return self.de

    def _finalize(self):
        if self.verbose >= logger.NOTE:
            logger.note(self, '--------------- %s gradients ---------------',
                        self.base.__class__.__name__)
            _write(self, self.mol, self.de, None)
            logger.note(self, '----------------------------------------------')

    def as_scanner(self):
        if isinstance(self, lib.GradScanner):
            return self

        logger.info(self, 'Create scanner for %s', self.__class__)

        class SCF_GradScanner(self.__class__, lib.GradScanner):
            def __init__(self, g):
                lib.GradScanner.__init__(self, g)

            def __call__(self, mol_or_geom, **kwargs):
                if isinstance(mol_or_geom, neo.Mole):
                    mol = mol_or_geom
                else:
                    mol = self.mol.set_geom_(mol_or_geom, inplace=True)

                self.mol = self.base.mol = mol
                mf_scanner = self.base
                e_tot = mf_scanner(mol)
                de = self.kernel(**kwargs)
                return e_tot, de

        return SCF_GradScanner(self)

Grad = Gradients

# Inject to CDFT class
neo.CDFT.Gradients = lib.class_as_method(Gradients)
