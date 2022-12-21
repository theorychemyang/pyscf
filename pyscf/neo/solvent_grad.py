#!/usr/bin/env python

'''
Analytical nuclear gradients for NEO-ddCOSMO
'''

import numpy
from pyscf import lib
from pyscf import gto 
from pyscf import df
from pyscf.solvent import ddcosmo
from pyscf.symm import sph
from pyscf.lib import logger
from pyscf.solvent.ddcosmo_grad import make_L1, make_e_psi1, make_fi1
from pyscf.neo.solvent import make_psi
from pyscf.grad.rhf import _write
from pyscf.solvent._attach_solvent import _Solvation


def make_phi1(pcmobj, dm, r_vdw, ui, ylm_1sph, with_nuc=True):
    mol = pcmobj.mol
    natm = mol.natm
    lmax = pcmobj.lmax
    nlm = (lmax+1)**2

    if not (isinstance(dm, numpy.ndarray) and dm.ndim == 2):
        dm = dm[0] + dm[1]
    tril_dm = lib.pack_tril(dm+dm.T)
    nao = dm.shape[0]
    diagidx = numpy.arange(nao)
    diagidx = diagidx*(diagidx+1)//2 + diagidx
    tril_dm[diagidx] *= .5

    atom_coords = mol.atom_coords()
    atom_charges = mol.atom_charges()

    coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcmobj.lebedev_order)
    #extern_point_idx = ui > 0

    fi1 = make_fi1(pcmobj, pcmobj.get_atomic_radii())
    fi1[:,:,ui==0] = 0
    ui1 = -fi1

    phi1 = numpy.zeros((natm, 3, natm, nlm)) # test

    if with_nuc: # the response of classical nuclei
        ngrid_1sph = weights_1sph.size
        v_phi0 = numpy.empty((natm,ngrid_1sph))
        for ia in range(natm):
            cav_coords = atom_coords[ia] + r_vdw[ia] * coords_1sph
            d_rs = atom_coords.reshape(-1,1,3) - cav_coords
            v_phi0[ia] = numpy.einsum('z,zp->p', atom_charges, 1./lib.norm(d_rs,axis=2))
        phi1 = -numpy.einsum('n,ln,azjn,jn->azjl', weights_1sph, ylm_1sph, ui1, v_phi0)

        for ia in range(natm):
            cav_coords = atom_coords[ia] + r_vdw[ia] * coords_1sph
            for ja in range(natm):
                rs = atom_coords[ja] - cav_coords
                d_rs = lib.norm(rs, axis=1)
                v_phi = atom_charges[ja] * numpy.einsum('px,p->px', rs, 1./d_rs**3)
                tmp = numpy.einsum('n,ln,n,nx->xl', weights_1sph, ylm_1sph, ui[ia], v_phi)
                phi1[ja,:,ia] += tmp  # response of the other atoms
                phi1[ia,:,ia] -= tmp  # response of cavity grids

    int3c2e = mol._add_suffix('int3c2e')
    int3c2e_ip1 = mol._add_suffix('int3c2e_ip1')
    aoslices = mol.aoslice_by_atom()
    for ia in range(natm):
        cav_coords = atom_coords[ia] + r_vdw[ia] * coords_1sph
        #fakemol = gto.fakemol_for_charges(cav_coords[ui[ia]>0])
        fakemol = gto.fakemol_for_charges(cav_coords)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e, aosym='s1')
        v_phi = numpy.einsum('ij,ijk->k', dm, v_nj)
        phi1[:,:,ia] += numpy.einsum('n,ln,azn,n->azl', weights_1sph, ylm_1sph, ui1[:,:,ia], v_phi)

        v_e1_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip1, comp=3, aosym='s1')
        phi1_e2_nj  = numpy.einsum('ij,xijr->xr', dm, v_e1_nj)
        phi1_e2_nj += numpy.einsum('ji,xijr->xr', dm, v_e1_nj)
        phi1[ia,:,ia] += numpy.einsum('n,ln,n,xn->xl', weights_1sph, ylm_1sph, ui[ia], phi1_e2_nj)

        for ja in range(natm):
            shl0, shl1, p0, p1 = aoslices[ja]
            phi1_nj  = numpy.einsum('ij,xijr->xr', dm[p0:p1  ], v_e1_nj[:,p0:p1])
            phi1_nj += numpy.einsum('ji,xijr->xr', dm[:,p0:p1], v_e1_nj[:,p0:p1])
            phi1[ja,:,ia] -= numpy.einsum('n,ln,n,xn->xl', weights_1sph, ylm_1sph, ui[ia], phi1_nj)
    return phi1


def kernel(pcmobj, dm, verbose=None):
    mol = pcmobj.mol
    natm = mol.natm
    lmax = pcmobj.lmax
    if pcmobj.grids.coords is None:
        pcmobj.grids.build(with_non0tab=True)

    dm_elec = dm[0]
    dm_nuc = dm[1:]

    if not (isinstance(dm_elec, numpy.ndarray) and dm_elec.ndim == 2):
        # UHF density matrix
        dm_elec = dm_elec[0] + dm_elec[1]

    r_vdw = ddcosmo.get_atomic_radii(pcmobj)
    coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcmobj.lebedev_order)
    ylm_1sph = numpy.vstack(sph.real_sph_vec(coords_1sph, lmax, True))

    fi = ddcosmo.make_fi(pcmobj, r_vdw)
    ui = 1 - fi
    ui[ui<0] = 0

    cached_pol = ddcosmo.cache_fake_multipoles(pcmobj.grids, r_vdw, lmax)

    nlm = (lmax+1)**2
    L0 = ddcosmo.make_L(pcmobj, r_vdw, ylm_1sph, fi)
    L0 = L0.reshape(natm*nlm,-1)
    L1 = make_L1(pcmobj, r_vdw, ylm_1sph, fi)

    phi0 = ddcosmo.make_phi(pcmobj.pcm_elec, dm_elec, r_vdw, ui, ylm_1sph, with_nuc=True)
    phi1 = make_phi1(pcmobj.pcm_elec, dm_elec, r_vdw, ui, ylm_1sph, with_nuc=True)
    psi0 = make_psi(pcmobj.pcm_elec, dm_elec, r_vdw, cached_pol, with_nuc=True)
    for i in range(mol.nuc_num):
        ia = mol.nuc[i].atom_index
        charge = mol.atom_charge(ia)
        phi0 -= charge * ddcosmo.make_phi(pcmobj.pcm_nuc[i], dm_nuc[i], r_vdw, ui, ylm_1sph, with_nuc=False)
        phi1 -= charge * make_phi1(pcmobj.pcm_nuc[i], dm_nuc[i], r_vdw, ui, ylm_1sph, with_nuc=False)
        psi0 -= charge * make_psi(pcmobj.pcm_nuc[i], dm_nuc[i], r_vdw, cached_pol, with_nuc=False)

    L0_X = numpy.linalg.solve(L0, phi0.ravel()).reshape(natm, nlm)
    L0_S = numpy.linalg.solve(L0.T, psi0.ravel()).reshape(natm, nlm) 
    
    e_psi1 = make_e_psi1(pcmobj.pcm_elec, dm_elec, r_vdw, ui, ylm_1sph,
                         cached_pol, L0_X, L0)
    for i in range(mol.nuc_num):
        ia = mol.nuc[i].atom_index
        charge = mol.atom_charge(ia)
        e_psi1 -= charge * make_e_psi1(pcmobj.pcm_nuc[i], dm_nuc[i], r_vdw, ui, ylm_1sph,
                         cached_pol, L0_X, L0)

    dielectric = pcmobj.eps
    if dielectric > 0:
        f_epsilon = (dielectric-1.)/dielectric
    else:
        f_epsilon = 1

    de = .5 * f_epsilon * e_psi1
    de+= .5 * f_epsilon * numpy.einsum('jx,azjx->az', L0_S, phi1)
    de-= .5 * f_epsilon * numpy.einsum('aziljm,il,jm->az', L1, L0_S, L0_X)

    return de


def make_grad_object(grad_method):
    '''For grad_method in vacuum, add nuclear gradients of solvent pcmobj'''

    # Zeroth order method object must be a solvation-enabled method
    assert isinstance(grad_method.base, _Solvation)
    if grad_method.base.with_solvent.frozen:
        raise RuntimeError('Frozen solvent model is not avialbe for energy gradients')

    grad_method_class = grad_method.__class__
    class WithSolventGrad(grad_method_class):
        def __init__(self, grad_method):
            self.__dict__.update(grad_method.__dict__)
            self.de_solvent = None
            self.de_solute = None
            self._keys = self._keys.union(['de_solvent', 'de_solute'])

        # TODO: if moving to python3, change signature to
        # def kernel(self, *args, dm=None, atmlst=None, **kwargs):
        def kernel(self, *args, **kwargs):
            dm = kwargs.pop('dm', None)
            if dm is None:
                dm = []
                dm.append(self.base.mf_elec.make_rdm1(ao_repr=True))
                for i in range(self.mol.nuc_num):
                    dm.append(self.base.mf_nuc[i].make_rdm1())

            self.de_solvent = kernel(self.base.with_solvent, dm)
            self.de_solute = grad_method_class.kernel(self, *args, **kwargs)
            self.de = self.de_solute + self.de_solvent

            if self.verbose >= logger.NOTE:
                logger.note(self, '--------------- %s (+%s) gradients ---------------',
                            self.base.__class__.__name__,
                            self.base.with_solvent.__class__.__name__)
                _write(self, self.mol, self.de, None)
                logger.note(self, '----------------------------------------------')
            return self.de

        def _finalize(self):
            # disable _finalize. It is called in grad_method.kernel method
            # where self.de was not yet initialized.
            pass

    return WithSolventGrad(grad_method)