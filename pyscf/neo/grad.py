#!/usr/bin/env python

'''
Analytical nuclear gradient for constrained nuclear-electronic orbital
'''
import numpy
from pyscf import gto, lib, neo
from pyscf.data import nist
from pyscf.grad.rhf import _write
from pyscf.lib import logger
from pyscf.scf.jk import get_jk

# TODO: create Gradients class from GradientsMixin
#from pyscf.grad import rhf as rhf_grad
#class Gradients(rhf_grad.GradientsMixin):


def grad_cneo(mf_grad, atmlst=None):
    mf = mf_grad.base
    mol = mf_grad.mol
    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)

    if atmlst is None:
        atmlst = range(mol.natm)

    hcore_deriv = []
    for x in mol.nuc:
        hcore_deriv.append(mf_grad.hcore_generator(x))

    aoslices = mol.aoslice_by_atom()
    de = numpy.zeros((len(atmlst),3))
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices[ia]
        h1ao_e = 0.0
        for j in range(mol.nuc_num):
            ja = mol.nuc[j].atom_index
            charge = mol.atom_charge(ja)
            # derivative w.r.t. electronic basis center
            shls_slice = (shl0, shl1) + (0, mol.elec.nbas) + (0, mol.nuc[j].nbas)*2
            v1en = get_jk((mol.elec, mol.elec, mol.nuc[j], mol.nuc[j]),
                          mf.dm_nuc[j], scripts='ijkl,lk->ij',
                          intor='int2e_ip1', aosym='s2kl', comp=3,
                          shls_slice=shls_slice)
            v1en *= charge
            h1ao_e += v1en * 2.0 # 2.0 for c.c.
            # nuclear hcore derivative
            h1ao_n = hcore_deriv[j](ia)
            if ja == ia:
                # derivative w.r.t. nuclear basis center
                v1ne = get_jk((mol.nuc[j], mol.nuc[j], mol.elec, mol.elec),
                              mf.dm_elec, scripts='ijkl,lk->ij',
                              intor='int2e_ip1', aosym='s2kl', comp=3)
                v1ne *= charge
                h1ao_n += v1ne + v1ne.transpose(0,2,1)
                for k in range(mol.nuc_num):
                    if k != j:
                        ka = mol.nuc[k].atom_index
                        v1nn = get_jk((mol.nuc[j], mol.nuc[j], mol.nuc[k], mol.nuc[k]),
                                      mf.dm_nuc[k], scripts='ijkl,lk->ij',
                                      intor='int2e_ip1', aosym='s2kl', comp=3)
                        v1nn *= -charge * mol.atom_charge(ka)
                        h1ao_n += v1nn + v1nn.transpose(0,2,1)
            if isinstance(h1ao_n, numpy.ndarray):
                de[i0] += numpy.einsum('xij,ij->x', h1ao_n, mf.dm_nuc[j])
        if isinstance(h1ao_e, numpy.ndarray):
            de[i0] += numpy.einsum('xij,ij->x', h1ao_e, mf.dm_elec[p0:p1])

    if log.verbose >= logger.DEBUG:
        log.debug('gradients of CNEO part')
        _write(log, mol, de, atmlst)
    return de

def get_hcore(mol_n):
    '''part of the gradients of core Hamiltonian of quantum nucleus'''
    ia = mol_n.atom_index
    mol = mol_n.super_mol
    mass = mol.mass[ia] * nist.ATOMIC_MASS / nist.E_MASS
    charge = mol.atom_charge(ia)
    # minus sign for the derivative is taken w.r.t 'r' instead of 'R'
    h = -mol_n.intor('int1e_ipkin', comp=3) / mass
    # note that kinetic energy partial derivative is actually always
    # zero, but we just keep it here because it is cheap to evaluate
    if mol._pseudo or mol_n._pseudo:
        raise NotImplementedError('Nuclear gradients for GTH PP')
    else:
        h += mol_n.intor('int1e_ipnuc', comp=3) * charge
    if mol.has_ecp():
        assert mol_n.has_ecp()
        h += mol_n.intor('ECPscalar_ipnuc', comp=3) * charge
    return h

def hcore_generator(mf_grad, mol_n):
    mol = mol_n.super_mol
    with_x2c = getattr(mf_grad.base, 'with_x2c', None)
    if with_x2c:
        raise NotImplementedError('X2C not supported')
    else:
        with_ecp = mol.has_ecp()
        if with_ecp:
            assert mol_n.has_ecp()
            ecp_atoms = set(mol_n._ecpbas[:,gto.ATOM_OF])
        else:
            ecp_atoms = ()
        ia = mol_n.atom_index
        charge = mol.atom_charge(ia)
        def hcore_deriv(atm_id):
            if atm_id == ia:
                h1 = get_hcore(mol_n)
                return h1 + h1.transpose(0,2,1)
            elif not mol.quantum_nuc[atm_id]:
                with mol_n.with_rinv_at_nucleus(atm_id):
                    vrinv = mol_n.intor('int1e_iprinv', comp=3) # <\nabla|1/r|>
                    vrinv *= mol.atom_charge(atm_id)
                    # note that ECP rinv works like ECP nuc, while regular
                    # rinv = -nuc, therefore we need a -1 factor for ECP
                    if with_ecp and atm_id in ecp_atoms:
                        vrinv -= mol_n.intor('ECPscalar_iprinv', comp=3)
                    vrinv *= charge
                return vrinv + vrinv.transpose(0,2,1)
            return 0.0
    return hcore_deriv

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

    hcore_generator = hcore_generator
    grad_cneo = grad_cneo

    def grad_elec(self, atmlst=None):
        '''gradients of electrons and classic nuclei'''
        g = self.base.mf_elec.nuc_grad_method()
        if self.grid_response is not None:
            g.grid_response = self.grid_response
        g.verbose = self.verbose - 1
        return g.grad(atmlst=atmlst)

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

        de = self.grad_cneo(atmlst=atmlst)
        self.de = de + self.grad_elec(atmlst=atmlst)
        if self.mol.symmetry:
            raise NotImplementedError('Symmetry is not supported')
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
neo.cdft.CDFT.Gradients = lib.class_as_method(Gradients)
