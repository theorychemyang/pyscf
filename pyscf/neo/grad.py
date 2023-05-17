#!/usr/bin/env python

'''
Analytical nuclear gradient for constrained nuclear-electronic orbital
'''
import numpy
from pyscf import gto, lib, neo
from pyscf.data import nist
from pyscf.grad import rhf as rhf_grad
from pyscf.lib import logger
from pyscf.scf.jk import get_jk
from pyscf.dft.numint import eval_ao, eval_rho, _scale_ao
from pyscf.neo.ks import eval_xc_nuc, eval_xc_elec
from pyscf.grad.rks import _d1_dot_


def get_vepc_elec(mf_grad):
    mf = mf_grad.base
    mol = mf_grad.mol
    ni = mf.mf_elec._numint
    grids = mf.mf_elec.grids
    nao = mol.elec.nao_nr()
    ao_loc = mol.elec.ao_loc_nr()
    ao_deriv = 1
    vmat_elec = numpy.zeros((3,nao,nao))
    for i in range(mol.nuc_num):
        ia = mol.nuc[i].atom_index
        if mol.atom_pure_symbol(ia) == 'H' and \
            (isinstance(mf.epc, str) or ia in mf.epc['epc_nuc']):
            for ao, mask, weight, coords \
                in ni.block_loop(mol.elec, grids, nao, ao_deriv):
                ao_nuc = eval_ao(mol.nuc[i], coords)
                rho_nuc = eval_rho(mol.nuc[i], ao_nuc, mf.dm_nuc[i])
                rho_nuc[rho_nuc<0.] = 0.
                rho_elec = eval_rho(mol.elec, ao[0], mf.dm_elec)
                vxc_elec_i = eval_xc_elec(mf.epc, rho_elec, rho_nuc)
                aow_elec = _scale_ao(ao[0], weight * vxc_elec_i)
                _d1_dot_(vmat_elec, mol.elec, ao[1:4], aow_elec, mask, ao_loc, True)
    return -vmat_elec

def get_vepc_nuc(mf_grad, mol, dm):
    mf = mf_grad.base
    ni = mf.mf_elec._numint
    grids = mf.mf_elec.grids
    nao = mol.nao_nr()
    ao_loc = mol.ao_loc_nr()
    # add electron epc grad
    ao_deriv = 1
    vmat_nuc = numpy.zeros((3,nao,nao))
    for ao, mask, weight, coords \
        in ni.block_loop(mol, grids, nao, ao_deriv):
        rho_nuc = eval_rho(mol, ao[0], dm)
        rho_nuc[rho_nuc<0.] = 0.
        ao_elec = eval_ao(mf.mol.elec, coords)
        rho_elec = eval_rho(mf.mol.elec, ao_elec, mf.dm_elec)
        _, vxc_nuc = eval_xc_nuc(mf.epc, rho_elec, rho_nuc)
        aow_nuc = _scale_ao(ao[0], weight * vxc_nuc)
        _d1_dot_(vmat_nuc, mol, ao[1:4], aow_nuc, mask, ao_loc, True)

    return -vmat_nuc

def grad_epc(mf_grad):
    mol = mf_grad.mol
    mf = mf_grad.base
    atmlst = range(mol.natm)
    aoslices = mol.aoslice_by_atom()
    de = numpy.zeros((len(atmlst),3))
    for j in atmlst:
        p0, p1 = aoslices[j,2:]
        vepc_elec = get_vepc_elec(mf_grad)
        de[j] += numpy.einsum('xij,ij->x', vepc_elec[:,p0:p1], mf.dm_elec[p0:p1]) * 2

    for i in range(mol.nuc_num):
        ia = mol.nuc[i].atom_index
        if mol.atom_pure_symbol(ia) == 'H' and \
            (isinstance(mf.epc, str) or ia in mf.epc['epc_nuc']):
            vepc_nuc = get_vepc_nuc(mf_grad, mol.nuc[i], mf.dm_nuc[i])
            de[ia] += numpy.einsum('xij,ij->x', vepc_nuc, mf.dm_nuc[i]) * 2

    return de

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
        rhf_grad._write(log, mol, de, atmlst)
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

def as_scanner(mf_grad):
    '''Generating a nuclear gradients scanner/solver (for geometry optimizer).

    This is different from GradientsMixin.as_scanner because (C)NEO uses two
    layers of mole objects.

    Copied from grad.rhf.as_scanner
    '''
    if isinstance(mf_grad, lib.GradScanner):
        return mf_grad

    logger.info(mf_grad, 'Create scanner for %s', mf_grad.__class__)

    class CNEO_GradScanner(mf_grad.__class__, lib.GradScanner):
        def __init__(self, g):
            lib.GradScanner.__init__(self, g)
        def __call__(self, mol_or_geom, **kwargs):
            if isinstance(mol_or_geom, neo.Mole):
                mol = mol_or_geom
            else:
                mol = self.mol.set_geom_(mol_or_geom, inplace=False)

            mf_scanner = self.base
            e_tot = mf_scanner(mol)
            self.mol = mol
            self.g_elec.mol = mol.elec

            # If second integration grids are created for RKS and UKS
            # electronic part gradients
            if getattr(self.g_elec, 'grids', None):
                self.g_elec.grids.reset(mol.elec)

            de = self.kernel(**kwargs)
            return e_tot, de
    return CNEO_GradScanner(mf_grad)


class Gradients(rhf_grad.GradientsMixin):
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
        rhf_grad.GradientsMixin.__init__(self, scf_method)
        self.grid_response = None
        self.g_elec = self.base.mf_elec.nuc_grad_method() # elec part obj
        self.g_elec.verbose = self.verbose - 1
        self._keys = self._keys.union(['grid_response', 'g_elec'])

    hcore_generator = hcore_generator
    grad_cneo = grad_cneo
    grad_epc = grad_epc

    def grad_elec(self, atmlst=None):
        '''gradients of electrons and classic nuclei'''
        if self.grid_response is not None:
            self.g_elec.grid_response = self.grid_response
        return self.g_elec.grad(atmlst=atmlst)

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
        if self.base.epc is not None:
            de += self.grad_epc()
        self.de = de + self.grad_elec(atmlst=atmlst)
        if mol.symmetry:
            raise NotImplementedError('Symmetry is not supported')
        logger.timer(self, 'CNEO gradients', *cput0)
        self._finalize()
        return self.de

    grad = lib.alias(kernel, alias_name='grad')

    as_scanner = as_scanner

Grad = Gradients

# Inject to CDFT class
neo.cdft.CDFT.Gradients = lib.class_as_method(Gradients)
