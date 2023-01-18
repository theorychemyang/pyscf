#!/usr/bin/env python

'''
Analytical nuclear gradient for constrained nuclear-electronic orbital
'''
import numpy
from pyscf import lib, neo
from pyscf.data import nist
from pyscf.grad.rhf import _write
from pyscf.lib import logger
from pyscf.scf.jk import get_jk

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
        self.base = scf_method
        if self.base.epc is not None:
            raise NotImplementedError('Gradient with epc is not implemented')
        self.verbose = scf_method.verbose
        self.mol = scf_method.mol
        self.max_memory = self.mol.max_memory
        self.unit = 'au'

        self.grid_response = None

        self.atmlst = None
        self.de = None
        self._keys = set(self.__dict__.keys())

    def grad_elec(self, atmlst=None):
        '''gradients of electrons and classic nuclei'''
        g = self.base.mf_elec.nuc_grad_method()
        if self.grid_response is not None:
            g.grid_response = self.grid_response
        g.verbose = self.verbose - 1
        return g.grad(atmlst=atmlst)

    def get_hcore(self, mol_n):
        '''part of the gradients of core Hamiltonian of quantum nucleus'''
        mol = self.mol
        ia = mol_n.atom_index
        mass = mol.mass[ia] * nist.ATOMIC_MASS / nist.E_MASS
        # minus sign for the derivative is taken w.r.t 'r' instead of 'R'
        h = -mol_n.intor('int1e_ipkin', comp=3) / mass
        # note that kinetic energy partial derivative is actually always
        # zero, but we just keep it here because it is cheap to evaluate
        h += mol_n.intor('int1e_ipnuc', comp=3) * mol.atom_charge(ia)
        return h

    def hcore_deriv(self, atm_id, mol_n):
        '''The change of Coulomb interactions between quantum and classical
        nuclei due to the change of the coordinates of classical nuclei'''
        mol = self.mol
        ia = mol_n.atom_index
        with mol_n.with_rinv_at_nucleus(atm_id):
            vrinv = mol_n.intor('int1e_iprinv', comp=3) # <\nabla|1/r|>
            vrinv *= mol_n.atom_charge(atm_id) * mol.atom_charge(ia)
            return vrinv + vrinv.transpose(0,2,1)

    def grad_jcross_elec_nuc(self):
        '''get the gradient for the cross term of Coulomb interactions between
        electrons and quantum nucleus'''
        mol = self.mol
        jcross = 0
        for i in range(mol.nuc_num):
            index = mol.nuc[i].atom_index
            jcross -= get_jk((mol.elec, mol.elec, mol.nuc[i], mol.nuc[i]),
                             self.base.dm_nuc[i], scripts='ijkl,lk->ij',
                             intor='int2e_ip1', aosym='s2kl', comp=3) \
                      * mol.atom_charge(index)
        return jcross

    def grad_jcross_nuc_elec(self, mol_n):
        '''get the gradient for the cross term of Coulomb interactions between
        quantum nucleus and electrons'''
        mol = self.mol
        ia = mol_n.atom_index
        jcross = -get_jk((mol_n, mol_n, mol.elec, mol.elec),
                         self.base.dm_elec, scripts='ijkl,lk->ij',
                         intor='int2e_ip1', aosym='s2kl', comp=3) \
                 * mol.atom_charge(ia)
        return jcross

    def grad_jcross_nuc_nuc(self, mol_n):
        '''get the gradient for the cross term of Coulomb interactions between
        quantum nuclei'''
        mol = self.mol
        ia = mol_n.atom_index
        jcross = numpy.zeros((3, mol_n.nao_nr(), mol_n.nao_nr()))
        for j in range(mol.nuc_num):
            ja = mol.nuc[j].atom_index
            if ja != ia:
                jcross -= get_jk((mol_n, mol_n, mol.nuc[j], mol.nuc[j]),
                                 self.base.dm_nuc[j], scripts='ijkl,lk->ij',
                                 intor='int2e_ip1', aosym='s2kl', comp=3) \
                          * mol.atom_charge(ia) * mol.atom_charge(ja)
        return jcross

    # TODO:
    # This will not be necessary if Gradients class here inherits
    # from GradientsMixin, but now, copy from pyscf/grad/rhf.py
    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('\n')
        if hasattr(self.base, 'converged') and not self.base.converged:
            log.warn('Ground state %s not converged',
                     self.base.__class__.__name__)
        log.info('******** %s for %s ********',
                 self.__class__, self.base.__class__)
        if 'ANG' in self.unit.upper():
            raise NotImplementedError('unit Eh/Ang is not supported')
        else:
            log.info('unit = Eh/Bohr')
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        return self

    def kernel(self, atmlst=None):
        '''Unit: Hartree/Bohr'''
        cput0 = (logger.process_clock(), logger.perf_counter())
        mol = self.mol
        if atmlst is None:
            if self.atmlst is not None:
                atmlst = self.atmlst
            else:
                self.atmlst = atmlst = range(mol.natm)

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        de = self.grad_elec()
        aoslices = mol.aoslice_by_atom()

        jcross_elec_nuc = self.grad_jcross_elec_nuc()
        for k, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia,2:]
            # *2 for c.c.
            de[k] -= numpy.einsum('xij,ij->x', jcross_elec_nuc[:,p0:p1],
                                  self.base.dm_elec[p0:p1]) * 2.0

            if mol.quantum_nuc[ia]:
                for i in range(mol.nuc_num):
                    if mol.nuc[i].atom_index == ia:
                        de[k] += numpy.einsum('xij,ij->x',
                                              self.get_hcore(mol.nuc[i]),
                                              self.base.dm_nuc[i]) * 2.0
                        jcross_nuc_elec = self.grad_jcross_nuc_elec(mol.nuc[i])
                        de[k] -= numpy.einsum('xij,ij->x', jcross_nuc_elec,
                                              self.base.dm_nuc[i]) * 2.0
                        jcross_nuc_nuc = self.grad_jcross_nuc_nuc(mol.nuc[i])
                        de[k] += numpy.einsum('xij,ij->x', jcross_nuc_nuc,
                                              self.base.dm_nuc[i]) * 2.0
            else:
                for i in range(mol.nuc_num):
                    h1ao = self.hcore_deriv(ia, mol.nuc[i])
                    de[k] += numpy.einsum('xij,ij->x', h1ao, self.base.dm_nuc[i])

        self.de = de
        logger.timer(self, 'CNEO gradients', *cput0)
        self._finalize()
        return self.de

    grad = lib.alias(kernel, alias_name='grad')

    # TODO:
    # This will not be necessary if Gradients class here inherits
    # from GradientsMixin, but now, copy from pyscf/grad/rhf.py
    def _finalize(self):
        if self.verbose >= logger.NOTE:
            logger.note(self, '--------------- %s gradients ---------------',
                        self.base.__class__.__name__)
            _write(self, self.mol, self.de, self.atmlst)
            logger.note(self, '----------------------------------------------')

    # FIXME: scanner does not work, because CNEO needs two layers of mole's.
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
