#!/usr/bin/env python

'''
Analytic nuclear gradient for constrained nuclear-electronic orbital
'''

import numpy
import warnings
from pyscf import df, gto, lib, neo, scf
from pyscf.grad import rhf as rhf_grad
from pyscf.lib import logger
from pyscf.scf import hf
from pyscf.scf.jk import get_jk
from pyscf.dft.numint import eval_ao, eval_rho, _scale_ao
from pyscf.grad.rks import _d1_dot_
from pyscf.neo.ks import precompute_epc_electron, eval_epc


def general_grad(grad_method):
    '''Modify gradient method to support general charge and mass.
    Similar to general_scf decorator in neo/hf.py
    '''
    if isinstance(grad_method, ComponentGrad):
        return grad_method

    assert (isinstance(grad_method.base, scf.hf.SCF) and
            isinstance(grad_method.base, neo.hf.Component))

    return grad_method.view(lib.make_class((ComponentGrad, grad_method.__class__)))

class ComponentGrad:
    __name_mixin__ = 'Component'

    def __init__(self, grad_method):
        self.__dict__.update(grad_method.__dict__)

    def get_hcore(self, mol=None):
        '''Core Hamiltonian first derivatives for general charged particle'''
        if mol is None: mol = self.mol

        # Kinetic and nuclear potential derivatives
        h = -mol.intor('int1e_ipkin', comp=3) / self.base.mass
        if mol._pseudo:
            raise NotImplementedError('Nuclear gradients for GTH PP')
        else:
            h -= mol.intor('int1e_ipnuc', comp=3) * self.base.charge
        if mol.has_ecp():
            h -= mol.intor('ECPscalar_ipnuc', comp=3) * self.base.charge

        # Add MM contribution if present
        if mol.super_mol.mm_mol is not None:
            mm_mol = mol.super_mol.mm_mol
            coords = mm_mol.atom_coords()
            charges = mm_mol.atom_charges()
            nao = mol.nao
            max_memory = mol.super_mol.max_memory - lib.current_memory()[0]
            blksize = int(min(max_memory*1e6/8/nao**2/3, 200))
            blksize = max(blksize, 1)

            v = 0
            if mm_mol.charge_model == 'gaussian':
                expnts = mm_mol.get_zetas()
                if mol.cart:
                    intor = 'int3c2e_ip1_cart'
                else:
                    intor = 'int3c2e_ip1_sph'
                cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas,
                                                     mol._env, intor)
                for i0, i1 in lib.prange(0, charges.size, blksize):
                    fakemol = gto.fakemol_for_charges(coords[i0:i1], expnts[i0:i1])
                    j3c = df.incore.aux_e2(mol, fakemol, intor, aosym='s1',
                                           comp=3, cintopt=cintopt)
                    v += numpy.einsum('ipqk,k->ipq', j3c, charges[i0:i1])
            else:
                for i0, i1 in lib.prange(0, charges.size, blksize):
                    j3c = mol.intor('int1e_grids_ip', grids=coords[i0:i1])
                    v += numpy.einsum('ikpq,k->ipq', j3c, charges[i0:i1])
            h += self.base.charge * v
        return h

    def hcore_generator(self, mol=None):
        if mol is None: mol = self.mol
        with_x2c = getattr(self.base, 'with_x2c', None)
        if with_x2c:
            raise NotImplementedError('X2C not supported')
        else:
            with_ecp = mol.has_ecp()
            if with_ecp:
                ecp_atoms = set(mol._ecpbas[:,gto.ATOM_OF])
            else:
                ecp_atoms = ()
            aoslices = mol.aoslice_by_atom()
            h1 = self.get_hcore(mol)
            def hcore_deriv(atm_id):
                p0, p1 = aoslices[atm_id][2:]
                # External potential gradient
                if not mol.super_mol._quantum_nuc[atm_id]:
                    with mol.with_rinv_at_nucleus(atm_id):
                        vrinv = mol.intor('int1e_iprinv', comp=3) # <\nabla|1/r|>
                        vrinv *= -mol.atom_charge(atm_id)
                        if with_ecp and atm_id in ecp_atoms:
                            vrinv += mol.intor('ECPscalar_iprinv', comp=3)
                        vrinv *= self.base.charge
                else:
                    vrinv = numpy.zeros((3, mol.nao, mol.nao))
                # Hcore gradient
                vrinv[:,p0:p1] += h1[:,p0:p1]
                return vrinv + vrinv.transpose(0,2,1)
        return hcore_deriv

    def get_veff(self, mol=None, dm=None):
        if self.base.is_nucleus: # Nucleus does not have self-type interaction
            if mol is None:
                mol = self.mol
            return numpy.zeros((3,mol.nao,mol.nao))
        else:
            assert abs(self.base.charge) == 1
            return super().get_veff(mol, dm)

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        raise AttributeError

def grad_pair_int(mol1, mol2, dm1, dm2, charge1, charge2, atmlst):
    de = numpy.zeros((len(atmlst),3))
    aoslices1 = mol1.aoslice_by_atom()
    aoslices2 = mol2.aoslice_by_atom()
    for i0, ia in enumerate(atmlst):
        shl0, shl1, p0, p1 = aoslices1[ia]
        # Derivative w.r.t. mol1
        if shl1 > shl0:
            shls_slice = (shl0, shl1) + (0, mol1.nbas) + (0, mol2.nbas)*2
            v1 = get_jk((mol1, mol1, mol2, mol2),
                        dm2, scripts='ijkl,lk->ij',
                        intor='int2e_ip1', aosym='s2kl', comp=3,
                        shls_slice=shls_slice)
            de[i0] -= 2. * charge1 * charge2 * \
                      numpy.einsum('xij,ij->x', v1, dm1[p0:p1])
        shl0, shl1, p0, p1 = aoslices2[ia]
        # Derivative w.r.t. mol2
        if shl1 > shl0:
            shls_slice = (shl0, shl1) + (0, mol2.nbas) + (0, mol1.nbas)*2
            v1 = get_jk((mol2, mol2, mol1, mol1),
                        dm1, scripts='ijkl,lk->ij',
                        intor='int2e_ip1', aosym='s2kl', comp=3,
                        shls_slice=shls_slice)
            de[i0] -= 2. * charge1 * charge2 * \
                      numpy.einsum('xij,ij->x', v1, dm2[p0:p1])
    return de

def grad_int(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    '''Calculate gradient for inter-component Coulomb interactions'''
    mf = mf_grad.base
    mol = mf_grad.mol
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff
    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)

    dm0 = mf.make_rdm1(mo_coeff, mo_occ)

    if atmlst is None:
        atmlst = range(mol.natm)

    de = numpy.zeros((len(atmlst),3))

    for (t1, t2), interaction in mf.interactions.items():
        comp1 = mf.components[t1]
        comp2 = mf.components[t2]
        dm1 = dm0[t1]
        if interaction.mf1_unrestricted:
            assert dm1.ndim > 2 and dm1.shape[0] == 2
            dm1 = dm1[0] + dm1[1]
        dm2 = dm0[t2]
        if interaction.mf2_unrestricted:
            assert dm2.ndim > 2 and dm2.shape[0] == 2
            dm2 = dm2[0] + dm2[1]
        mol1 = comp1.mol
        mol2 = comp2.mol
        de += grad_pair_int(mol1, mol2, dm1, dm2,
                            comp1.charge, comp2.charge, atmlst)

    if log.verbose >= logger.DEBUG:
        log.debug('gradients of Coulomb interaction')
        rhf_grad._write(log, mol, de, atmlst)
    return de

def grad_epc(mf_grad, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
    '''Calculate EPC gradient contributions using pre-screened grids'''
    mf = mf_grad.base
    mol = mf_grad.mol
    if mo_energy is None: mo_energy = mf.mo_energy
    if mo_occ is None:    mo_occ = mf.mo_occ
    if mo_coeff is None:  mo_coeff = mf.mo_coeff

    if atmlst is None:
        atmlst = range(mol.natm)

    de = numpy.zeros((len(atmlst),3))

    # Early return if no EPC
    if mf.epc is None:
        return de

    log = logger.Logger(mf_grad.stdout, mf_grad.verbose)

    ni = mf._numint
    grids = mf.grids

    # Get all nuclear components
    n_types = []
    mol_n = {}
    non0tab_n = {}
    vxc_n = {}
    ao_loc_n = {}

    for t, comp in mf.components.items():
        if not t.startswith('n'):
            continue
        mol_n_t = comp.mol
        ia = mol_n_t.atom_index
        if mol_n_t.super_mol.atom_pure_symbol(ia) == 'H' and \
                (isinstance(mf.epc, str) or ia in mf.epc['epc_nuc']):
            n_types.append(t)
            mol_n[t] = mol_n_t
            non0tab_n[t] = ni.make_mask(mol_n_t, grids.coords)
            nao_n = mol_n_t.nao
            vxc_n[t] = numpy.zeros((3,nao_n,nao_n))
            ao_loc_n[t] = mol_n_t.ao_loc_nr()

    if len(n_types) == 0:
        return de

    mf_e = mf.components['e']
    assert(mf._elec_grids_hash == neo.ks._hash_grids(mf_e.grids))

    dm0 = mf.make_rdm1(mo_coeff, mo_occ)

    # Get electron component
    mol_e = mf_e.mol
    dm_e = dm0['e']
    if isinstance(mf_e, scf.uhf.UHF):
        assert dm_e.ndim > 2 and dm_e.shape[0] == 2
        dm_e = dm_e[0] + dm_e[1]
    nao_e = mol_e.nao
    ao_loc_e = mol_e.ao_loc_nr()
    vxc_e = numpy.zeros((3,nao_e,nao_e))

    # Single grid loop over pre-screened points
    for ao_e, mask_e, weight, coords in ni.block_loop(mol_e, grids, nao_e, 1):
        # Get electron density and precompute EPC terms once
        rho_e = eval_rho(mol_e, ao_e[0], dm_e)
        rho_e[rho_e < 0] = 0
        common = precompute_epc_electron(mf.epc, rho_e)

        vxc_e_grid = 0

        for n_type in n_types:
            mol_n_t = mol_n[n_type]
            mask_n = non0tab_n[n_type]

            # Get nuclear density
            ao_n = eval_ao(mol_n_t, coords, deriv=1, non0tab=mask_n)
            rho_n = eval_rho(mol_n_t, ao_n[0], dm0[n_type])
            rho_n[rho_n < 0] = 0

            # Get EPC quantities
            _, vxc_n_grid, vxc_e_grid_t = eval_epc(common, rho_n)
            vxc_e_grid += vxc_e_grid_t

            # Nuclear gradient contribution
            aow = _scale_ao(ao_n[0], weight * vxc_n_grid)
            _d1_dot_(vxc_n[n_type], mol_n_t, ao_n[1:4],
                     aow, mask_n, ao_loc_n[n_type], True)

        # Electronic gradient contribution
        aow = _scale_ao(ao_e[0], weight * vxc_e_grid)
        _d1_dot_(vxc_e, mol_e, ao_e[1:4], aow, mask_e, ao_loc_e, True)

    aoslices = mol_e.aoslice_by_atom()
    for i0, ia in enumerate(atmlst):
        p0, p1 = aoslices[ia,2:]
        de[i0] -= numpy.einsum('xij,ij->x', vxc_e[:,p0:p1], dm_e[p0:p1]) * 2

    for n_type in n_types:
        aoslices = mol_n[n_type].aoslice_by_atom()
        for i0, ia in enumerate(atmlst):
            p0, p1 = aoslices[ia,2:]
            if p1 > p0:
                de[i0] -= numpy.einsum('xij,ij->x', vxc_n[n_type][:,p0:p1],
                                       dm0[n_type][p0:p1]) * 2

    if log.verbose >= logger.DEBUG:
        log.debug('gradients of EPC contribution')
        rhf_grad._write(log, mol, de, atmlst)
    return de

def grad_hcore_mm(mf_grad, dm=None, mol=None):
    '''Calculate QMMM gradient for MM atoms'''
    if mol is None:
        mol = mf_grad.mol
    mm_mol = mol.mm_mol
    if mm_mol is None:
        warnings.warn('Not a QM/MM calculation, grad_mm should not be called!')
        return None

    coords = mm_mol.atom_coords()
    charges = mm_mol.atom_charges()
    g = numpy.zeros_like(coords)
    mf = mf_grad.base
    if dm is None:
        dm = mf.make_rdm1()

    # Handle each charged component's interaction with MM
    for t, comp in mf.components.items():
        mol_comp = comp.mol
        dm_comp = dm[t]
        if mm_mol.charge_model == 'gaussian':
            expnts = mm_mol.get_zetas()

            intor = 'int3c2e_ip2'
            nao = mol_comp.nao
            max_memory = mol.max_memory - lib.current_memory()[0]
            blksize = int(min(max_memory*1e6/8/nao**2/3, 200))
            blksize = max(blksize, 1)
            cintopt = gto.moleintor.make_cintopt(mol_comp._atm, mol_comp._bas,
                                                 mol_comp._env, intor)

            for i0, i1 in lib.prange(0, charges.size, blksize):
                fakemol = gto.fakemol_for_charges(coords[i0:i1], expnts[i0:i1])
                j3c = df.incore.aux_e2(mol_comp, fakemol, intor, aosym='s1',
                                       comp=3, cintopt=cintopt)
                g[i0:i1] += numpy.einsum('ipqk,qp->ik', j3c * charges[i0:i1],
                                         dm_comp).T * comp.charge
        else:
            # From examples/qmmm/30-force_on_mm_particles.py
            # The interaction between electron density and MM particles
            # d/dR <i| (1/|r-R|) |j> = <i| d/dR (1/|r-R|) |j> = <i| -d/dr (1/|r-R|) |j>
            #   = <d/dr i| (1/|r-R|) |j> + <i| (1/|r-R|) |d/dr j>
            for i, q in enumerate(charges):
                with mol_comp.with_rinv_origin(coords[i]):
                    v = mol_comp.intor('int1e_iprinv')
                g[i] += (numpy.einsum('ij,xji->x', dm_comp, v) +
                         numpy.einsum('ij,xij->x', dm_comp, v.conj())) \
                        * -q * comp.charge
    return g

def grad_nuc_mm(mf_grad, mol=None):
    '''Nuclear gradients of the QM-MM nuclear energy
    (in the form of point charge Coulomb interactions)
    with respect to MM atoms.
    '''
    if mol is None:
        mol = mf_grad.mol
    mm_mol = mol.mm_mol
    if mm_mol is None:
        warnings.warn('Not a QM/MM calculation, grad_mm should not be called!')
        return None
    coords = mm_mol.atom_coords()
    charges = mm_mol.atom_charges()
    g_mm = numpy.zeros_like(coords)
    mol_e = mol.components['e']
    for i in range(mol_e.natm):
        q1 = mol_e.atom_charge(i)
        r1 = mol_e.atom_coord(i)
        r = lib.norm(r1-coords, axis=1)
        g_mm += q1 * numpy.einsum('i,ix,i->ix', charges, r1-coords, 1/r**3)
    return g_mm

def as_scanner(mf_grad):
    '''Generating a nuclear gradients scanner/solver (for geometry optimizer).

    This is different from GradientsBase.as_scanner because CNEO uses two
    layers of mole objects.

    Copied from grad.rhf.as_scanner
    '''
    if isinstance(mf_grad, lib.GradScanner):
        return mf_grad

    logger.info(mf_grad, 'Create scanner for %s', mf_grad.__class__)
    name = mf_grad.__class__.__name__ + CNEO_GradScanner.__name_mixin__
    return lib.set_class(CNEO_GradScanner(mf_grad),
                         (CNEO_GradScanner, mf_grad.__class__), name)

class CNEO_GradScanner(lib.GradScanner):
    def __init__(self, g):
        lib.GradScanner.__init__(self, g)

    def __call__(self, mol_or_geom, **kwargs):
        if isinstance(mol_or_geom, neo.Mole):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)

        self.reset(mol)
        mf_scanner = self.base
        if 'dm0' in kwargs:
            dm0 = kwargs.pop('dm0')
            e_tot = mf_scanner(mol, dm0=dm0)
        else:
            e_tot = mf_scanner(mol)

        for t in mf_scanner.components.keys():
            if isinstance(mf_scanner.components[t], hf.KohnShamDFT):
                if getattr(self.components[t], 'grids', None):
                    self.components[t].grids.reset(mol.components[t])
                if getattr(self.components[t], 'nlcgrids', None):
                    self.components[t].nlcgrids.reset(mol.components[t])

        de = self.kernel(**kwargs)
        return e_tot, de

class Gradients(rhf_grad.GradientsBase):
    '''Analytic gradients for CDFT

    Examples::

    >>> from pyscf import neo
    >>> mol = neo.M(atom='H 0 0 0; H 0 0 0.917', basis='ccpvdz', nuc_basis='pb4d')
    >>> mf = neo.CDFT(mol, xc='b3lyp5', epc='17-2')
    >>> mf.kernel()
    >>> g = neo.Gradients(mf)
    >>> g.kernel()
    '''
    def __init__(self, mf):
        super().__init__(mf)
        self.grid_response = None

        # Get base gradient for each component
        self.components = {}
        for t, comp in self.base.components.items():
            self.components[t] = general_grad(comp.nuc_grad_method())
        self._keys = self._keys.union(['grid_response', 'components'])

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        if self.mol.mm_mol is not None:
            logger.info(self, '** Add background charges for %s **',
                        self.__class__.__name__)
            if self.verbose >= logger.DEBUG1:
                logger.debug1(self, 'Charge      Location')
                coords = self.mol.mm_mol.atom_coords()
                charges = self.mol.mm_mol.atom_charges()
                for i, z in enumerate(charges):
                    logger.debug1(self, '%.9g    %s', z, coords[i])
            return self

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self.base.reset(self.mol)
        if sorted(self.components.keys()) == sorted(self.mol.components.keys()):
            # quantum nuc is the same, reset each component
            for t, comp in self.components.items():
                comp.reset(self.mol.components[t])
        else:
            # quantum nuc is different, need to rebuild
            self.components.clear()
            for t, comp in self.base.components.items():
                self.components[t] = general_grad(comp.nuc_grad_method())
        return self

    def grad_nuc(self, mol=None, atmlst=None):
        if mol is None: mol = self.mol
        g_qm = self.components['e'].grad_nuc(mol.components['e'], atmlst)
        if mol.mm_mol is not None:
            coords = mol.mm_mol.atom_coords()
            charges = mol.mm_mol.atom_charges()
            # nuclei lattice interaction
            mol_e = mol.components['e']
            g_mm = numpy.empty((mol_e.natm,3))
            for i in range(mol_e.natm):
                q1 = mol_e.atom_charge(i)
                r1 = mol_e.atom_coord(i)
                r = lib.norm(r1-coords, axis=1)
                g_mm[i] = -q1 * numpy.einsum('i,ix,i->x', charges, r1-coords, 1/r**3)
            if atmlst is not None:
                g_mm = g_mm[atmlst]
        else:
            g_mm = 0
        return g_qm + g_mm

    def symmetrize(self, de, atmlst=None):
        return rhf_grad.symmetrize(self.mol.components['e'], de, atmlst)

    def kernel(self, mo_energy=None, mo_coeff=None, mo_occ=None, atmlst=None):
        cput0 = (logger.process_clock(), logger.perf_counter())
        if mo_energy is None: mo_energy = self.base.mo_energy
        if mo_coeff is None: mo_coeff = self.base.mo_coeff
        if mo_occ is None: mo_occ = self.base.mo_occ
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        if self.verbose >= logger.WARN:
            self.check_sanity()
        if self.verbose >= logger.INFO:
            self.dump_flags()

        # Get gradient from each component
        de = 0
        for t, comp in self.components.items():
            if self.grid_response is not None and isinstance(comp.base, hf.KohnShamDFT):
                comp.grid_response = self.grid_response
            de += comp.grad_elec(mo_energy=mo_energy[t], mo_coeff=mo_coeff[t],
                                 mo_occ=mo_occ[t], atmlst=atmlst)

        # Add inter-component interaction gradient
        de += self.grad_int(mo_energy, mo_coeff, mo_occ, atmlst)

        # Add EPC contribution if needed
        if hasattr(self.base, 'epc') and self.base.epc is not None:
            de += self.grad_epc(mo_energy, mo_coeff, mo_occ, atmlst)

        self.de = de + self.grad_nuc(atmlst=atmlst)
        if self.mol.symmetry:
            self.de = self.symmetrize(self.de, atmlst)
        if self.base.do_disp():
            self.de += self.components['e'].get_dispersion()
        logger.timer(self, 'CNEO gradients', *cput0)
        self._finalize()
        return self.de

    grad = lib.alias(kernel, alias_name='grad')

    grad_int = grad_int
    grad_epc = grad_epc
    grad_hcore_mm = grad_hcore_mm
    grad_nuc_mm = grad_nuc_mm

    def grad_mm(self, dm=None, mol=None):
        return self.grad_hcore_mm(dm, mol) + self.grad_nuc_mm(mol)

    as_scanner = as_scanner

    def get_jk(self, mol=None, dm=None, hermi=0, omega=None):
        raise AttributeError

    def get_j(self, mol=None, dm=None, hermi=0, omega=None):
        raise AttributeError

    def get_k(self, mol=None, dm=None, hermi=0, omega=None):
        raise AttributeError

    def to_gpu(self):
        raise NotImplementedError

Grad = Gradients

# Inject to CDFT class
neo.cdft.CDFT.Gradients = lib.class_as_method(Gradients)

if __name__ == '__main__':
    mol = neo.M(atom='H 0 0 0; H 0 0 0.74', basis='ccpvdz', nuc_basis='pb4d', verbose=5)
    mf = neo.CDFT(mol, xc='PBE', epc='17-2')
    mf.scf()
    mf.nuc_grad_method().grad()
