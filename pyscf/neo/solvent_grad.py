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
from pyscf.dft import gen_grid, numint
from pyscf.solvent.grad.ddcosmo_grad import make_L1, make_e_psi1, make_fi1, make_phi1
from pyscf.neo.solvent import make_psi
from pyscf.grad.rhf import _write
from pyscf.solvent._attach_solvent import _Solvation
from pyscf.grad import rks as rks_grad


def make_phi1_nuc(pcmobj, dm, r_vdw, ui, ylm_1sph):
    mol = pcmobj.mol
    natm = mol.natm
    lmax = pcmobj.lmax
    nlm = (lmax+1)**2

    tril_dm = lib.pack_tril(dm+dm.T)
    nao = dm.shape[0]
    diagidx = numpy.arange(nao)
    diagidx = diagidx*(diagidx+1)//2 + diagidx
    tril_dm[diagidx] *= .5

    atom_coords = mol.atom_coords()
    coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcmobj.lebedev_order)
    #extern_point_idx = ui > 0

    fi1 = make_fi1(pcmobj, pcmobj.get_atomic_radii())
    fi1[:,:,ui==0] = 0
    ui1 = -fi1

    phi1 = numpy.zeros((natm, 3, natm, nlm))

    int3c2e = mol._add_suffix('int3c2e')
    int3c2e_ip1 = mol._add_suffix('int3c2e_ip1')

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

        ja = mol.atom_index
        phi1[ja,:,ia] -= numpy.einsum('n,ln,n,xn->xl', weights_1sph, ylm_1sph, ui[ia], phi1_e2_nj)

    return phi1


def make_e_psi1_nuc(pcmobj, dm, cached_pol, Xvec):
    mol = pcmobj.mol
    natm = mol.natm
    lmax = pcmobj.lmax
    grids = pcmobj.grids

    ni = numint.NumInt()
    make_rho, nset, nao = ni._gen_rho_evaluator(mol, dm)
    den = numpy.empty((4,grids.weights.size))

    ao_loc = mol.ao_loc_nr()
    vmat = numpy.zeros((3,nao,nao))
    psi1 = numpy.zeros((natm,3))
    i1 = 0
    for ia, (coords, weight, weight1) in enumerate(rks_grad.grids_response_cc(grids)):
        i0, i1 = i1, i1 + weight.size
        ao = ni.eval_ao(mol, coords, deriv=1)
        mask = gen_grid.make_mask(mol, coords)
        den[:,i0:i1] = make_rho(0, ao, mask, 'GGA')

        fak_pol, leak_idx = cached_pol[mol.atom_pure_symbol(ia)]
        eta_nj = 0
        p1 = 0
        for l in range(lmax+1):
            fac = 4*numpy.pi/(l*2+1)
            p0, p1 = p1, p1 + (l*2+1)
            eta_nj += fac * numpy.einsum('mn,m->n', fak_pol[l], Xvec[ia,p0:p1])
        psi1 -= numpy.einsum('n,n,zxn->zx', den[0,i0:i1], eta_nj, weight1)
        psi1[ia] -= numpy.einsum('xn,n,n->x', den[1:4,i0:i1], eta_nj, weight)

        vtmp = numpy.zeros((3,nao,nao))
        aow = numpy.einsum('pi,p->pi', ao[0], weight*eta_nj)
        rks_grad._d1_dot_(vtmp, mol, ao[1:4], aow, mask, ao_loc, True)
        vmat += vtmp

    ja = mol.atom_index
    psi1[ja] += numpy.einsum('xij,ij->x', vmat, dm) * 2

    return psi1


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
    phi1 = make_phi1(pcmobj.pcm_elec, dm_elec, r_vdw, ui, ylm_1sph)
    psi0 = make_psi(pcmobj.pcm_elec, dm_elec, r_vdw, cached_pol, with_nuc=True)
    for i in range(mol.nuc_num):
        ia = mol.nuc[i].atom_index
        charge = mol.atom_charge(ia)
        phi0 -= charge * ddcosmo.make_phi(pcmobj.pcm_nuc[i], dm_nuc[i], r_vdw, ui, ylm_1sph, with_nuc=False)
        phi1 -= charge * make_phi1_nuc(pcmobj.pcm_nuc[i], dm_nuc[i], r_vdw, ui, ylm_1sph)
        psi0 -= charge * make_psi(pcmobj.pcm_nuc[i], dm_nuc[i], r_vdw, cached_pol, with_nuc=False)

    L0_X = numpy.linalg.solve(L0, phi0.ravel()).reshape(natm, nlm)
    L0_S = numpy.linalg.solve(L0.T, psi0.ravel()).reshape(natm, nlm)

    e_psi1 = make_e_psi1(pcmobj.pcm_elec, dm_elec, r_vdw, ui, ylm_1sph,
                         cached_pol, L0_X, L0)
    for i in range(mol.nuc_num):
        ia = mol.nuc[i].atom_index
        charge = mol.atom_charge(ia)
        e_psi1 -= charge * make_e_psi1_nuc(pcmobj.pcm_nuc[i], dm_nuc[i], cached_pol, L0_X)

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
