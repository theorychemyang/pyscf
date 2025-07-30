#!/usr/bin/env python

'''
attach ddCOSMO for NEO
'''

import copy
import numpy
from pyscf import lib
from pyscf import gto
from pyscf import df
from pyscf.lib import logger
from pyscf.dft import numint
from pyscf.solvent import ddcosmo
from pyscf.neo.cdft import CDFT
from pyscf.solvent._attach_solvent import _Solvation


def make_psi(pcmobj, dm, r_vdw, cached_pol, with_nuc=True):
    r'''
    get the \Psi vector through numerical integration

    Kwargs:
        with_nuc (bool): Mute the contribution of nuclear charges when
            computing the second order derivatives of energy.
    '''
    mol = pcmobj.mol
    natm = mol.natm
    lmax = pcmobj.lmax
    nlm = (lmax+1)**2

    dms = numpy.asarray(dm)
    is_single_dm = dms.ndim == 2
    grids = pcmobj.grids

    ni = numint.NumInt()
    max_memory = pcmobj.max_memory - lib.current_memory()[0]
    make_rho, n_dm, nao = ni._gen_rho_evaluator(mol, dms)
    dms = dms.reshape(n_dm,nao,nao)

    den = numpy.empty((n_dm, grids.weights.size))
    p1 = 0

    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, 0, max_memory):
        p0, p1 = p1, p1 + weight.size
        for i in range(n_dm):
            den[i,p0:p1] = make_rho(i, ao, mask, 'LDA')

    den *= grids.weights
    ao = None

    nelec_leak = 0
    psi = numpy.zeros((n_dm, natm, nlm))
    i1 = 0
    for ia in range(natm):
        fak_pol, leak_idx = cached_pol[mol.atom_pure_symbol(ia)]
        fac_pol = ddcosmo._vstack_factor_fak_pol(fak_pol, lmax)
        i0, i1 = i1, i1 + fac_pol.shape[1]
        nelec_leak += den[:,i0:i1][:,leak_idx].sum(axis=1)
        psi[:,ia] = -numpy.einsum('in,mn->im', den[:,i0:i1], fac_pol)
    logger.debug(pcmobj, 'electron leaks %s', nelec_leak)

    # Contribution of nuclear charges to the total density
    # The factor numpy.sqrt(4*numpy.pi) is due to the product of 4*pi * Y_0^0
    if with_nuc:
        for ia in range(natm):
            psi[:,ia,0] += numpy.sqrt(4*numpy.pi)/r_vdw[ia] * mol.atom_charge(ia)

    if is_single_dm:
        psi = psi[0]

    return psi

def make_vmat(pcmobj, dm, r_vdw, ui, ylm_1sph, cached_pol, Xvec, L, psi):
    '''
    The first order derivative of E_ddCOSMO wrt density matrix

    psi: the total electrostatic potential
    '''
    mol = pcmobj.mol
    natm = mol.natm
    lmax = pcmobj.lmax
    nlm = (lmax+1)**2

    dms = numpy.asarray(dm)
    is_single_dm = dms.ndim == 2
    grids = pcmobj.grids

    ni = numint.NumInt()
    max_memory = pcmobj.max_memory - lib.current_memory()[0]
    make_rho, n_dm, nao = ni._gen_rho_evaluator(mol, dms)
    dms = dms.reshape(n_dm,nao,nao)
    Xvec = Xvec.reshape(n_dm, natm, nlm)

    i1 = 0
    scaled_weights = numpy.empty((n_dm, grids.weights.size))
    for ia in range(natm):
        fak_pol, _ = cached_pol[mol.atom_pure_symbol(ia)]
        fac_pol = ddcosmo._vstack_factor_fak_pol(fak_pol, lmax)
        i0, i1 = i1, i1 + fac_pol.shape[1]
        scaled_weights[:,i0:i1] = numpy.einsum('mn,im->in', fac_pol, Xvec[:,ia])
    scaled_weights *= grids.weights

    shls_slice = (0, mol.nbas)
    ao_loc = mol.ao_loc_nr()
    vmat = numpy.zeros((n_dm, nao, nao))
    p1 = 0
    aow = None
    for ao, mask, weight, coords \
            in ni.block_loop(mol, grids, nao, 0, max_memory):
        p0, p1 = p1, p1 + weight.size
        for i in range(n_dm):
            aow = numint._scale_ao(ao, scaled_weights[i,p0:p1], out=aow)
            vmat[i] -= numint._dot_ao_ao(mol, ao, aow, mask, shls_slice, ao_loc)
    ao = aow = scaled_weights = None

    # <Psi, L^{-1}g> -> Psi = SL the adjoint equation to LX = g
    L_S = numpy.linalg.solve(L.reshape(natm*nlm,-1).T, psi.reshape(n_dm,-1).T)
    L_S = L_S.reshape(natm,nlm,n_dm).transpose(2,0,1)
    coords_1sph, weights_1sph = ddcosmo.make_grids_one_sphere(pcmobj.lebedev_order)
    # JCP, 141, 184108, Eq (39)
    xi_jn = numpy.einsum('n,jn,xn,ijx->ijn', weights_1sph, ui, ylm_1sph, L_S)
    extern_point_idx = ui > 0
    cav_coords = (mol.atom_coords().reshape(natm,1,3)
                  + numpy.einsum('r,gx->rgx', r_vdw, coords_1sph))
    cav_coords = cav_coords[extern_point_idx]
    xi_jn = xi_jn[:,extern_point_idx]

    max_memory = pcmobj.max_memory - lib.current_memory()[0]
    blksize = int(max(max_memory*.9e6/8/nao**2, 400))

    cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, 'int3c2e')
    vmat_tril = 0
    for i0, i1 in lib.prange(0, cav_coords.shape[0], blksize):
        fakemol = gto.fakemol_for_charges(cav_coords[i0:i1])
        v_nj = df.incore.aux_e2(mol, fakemol, intor='int3c2e', aosym='s2ij',
                                cintopt=cintopt)
        vmat_tril += numpy.einsum('xn,in->ix', v_nj, xi_jn[:,i0:i1])
    vmat += lib.unpack_tril(vmat_tril)

    if is_single_dm:
        vmat = vmat[0]

    return vmat


def _for_scf_neo(mf, solvent_obj, dm=None):
    '''Attach solvent model to NEO method.

    Kwargs:
        dm : if given, solvent does not respond to the change of density
            matrix. A frozen ddCOSMO potential is added to the results.
    '''

    oldMF = mf.__class__

    if dm is not None:
        solvent_obj.e, solvent_obj.v = solvent_obj.kernel(dm)
        solvent_obj.frozen = True

    class NEOWithSolvent(oldMF, _Solvation):
        def __init__(self, mf, solvent):
            self.__dict__.update(mf.__dict__)
            self.with_solvent = solvent
            self.e_solvent = 0
            self._keys.update(['with_solvent', 'e_solvent'])

        def dump_flags(self, verbose=None):
            oldMF.dump_flags(self, verbose)
            self.with_solvent.check_sanity()
            self.with_solvent.dump_flags(verbose)
            return self

        def reset(self, mol=None):
            self.with_solvent.reset(mol)
            return oldMF.reset(self, mol)

        # NOTE v_solvent should not be added to get_hcore for scf methods.
        # get_hcore is overloaded by many post-HF methods. Modifying
        # SCF.get_hcore may lead error.

        # Same signature as in neo.hf
        def get_fock(self, h1e=None, s1e=None, vhf=None, vint=None, dm=None, cycle=-1,
                     diis=None, diis_start_cycle=None, level_shift_factor=None,
                     damp_factor=None, fock_last=None, diis_pos='both', diis_type=3):
            # DIIS was called inside oldMF.get_fock. v_solvent, as a function of
            # dm, should be extrapolated as well. To enable it, v_solvent has to be
            # added to the fock matrix before DIIS was called.

            epcm, vpcm = self.with_solvent.kernel(dm)
            self.e_solvent = epcm
            logger.debug(self, 'Solvent Energy = %.15g', self.e_solvent)

            vhf_copy = copy.deepcopy(vhf)
            vhf_copy['e'] += vpcm[0]
            for i in range(self.mol.nuc_num):
                ia = self.mol.nuc[i].atom_index
                vhf_copy[f'n{ia}'] -= vpcm[i+1]

            return oldMF.get_fock(self, h1e, s1e, vhf_copy, vint, dm, cycle, diis,
                                  diis_start_cycle, level_shift_factor, damp_factor,
                                  fock_last, diis_pos, diis_type)


        def energy_tot(self, dm=None, h1e=None, vhf=None, vint=None):
            'add solvation energy to total energy'
            return self.e_solvent + oldMF.energy_tot(self, dm, h1e, vhf, vint)

        def nuc_grad_method(self):
            grad_method = oldMF.nuc_grad_method(self)
            return self.with_solvent.nuc_grad_method(grad_method)

        Gradients = nuc_grad_method
        '''
        def gen_response(self, *args, **kwargs):
            vind = oldMF.gen_response(self, *args, **kwargs)
            is_uhf = isinstance(self, scf.uhf.UHF)
            # singlet=None is orbital hessian or CPHF type response function
            singlet = kwargs.get('singlet', True)
            singlet = singlet or singlet is None
            def vind_with_solvent(dm1):
                v = vind(dm1)
                if self.with_solvent.equilibrium_solvation:
                    if is_uhf:
                        v_solvent = self.with_solvent._B_dot_x(dm1)
                        v += v_solvent[0] + v_solvent[1]
                    elif singlet:
                        v += self.with_solvent._B_dot_x(dm1)
                return v
            return vind_with_solvent
        '''
    mf1 = NEOWithSolvent(mf, solvent_obj)
    return mf1

def ddcosmo_for_neo(mf, solvent_obj=None, dm=None):
    if solvent_obj is None:
        solvent_obj = DDCOSMO(mf.mol)
    return _for_scf_neo(mf, solvent_obj, dm)

class DDCOSMO(ddcosmo.DDCOSMO):
    'Attach ddCOSMO model for NEO'
    def __init__(self, mol):
        super().__init__(mol)

    def build(self):
        'build solvent model for electrons and quantum nuclei'
        super().build()
        self.pcm_elec = ddcosmo.DDCOSMO(self.mol.elec)
        self.pcm_elec.eta = self.eta
        self.pcm_elec.lmax = self.lmax
        self.pcm_elec.lebedev_order = self.lebedev_order
        self.pcm_elec.build()
        self.pcm_nuc = []
        for i in range(self.mol.nuc_num):
            pcmobj = ddcosmo.DDCOSMO(self.mol.nuc[i])
            pcmobj.eta = self.eta
            pcmobj.lmax = self.lmax
            pcmobj.lebedev_order = self.lebedev_order
            pcmobj.build()
            self.pcm_nuc.append(pcmobj)

    def kernel(self, dm):

        if not self._intermediates or self.grids.coords is None:
            self.build()

        r_vdw      = self._intermediates['r_vdw'     ]
        ylm_1sph   = self._intermediates['ylm_1sph'  ]
        ui         = self._intermediates['ui'        ]
        Lmat       = self._intermediates['Lmat'      ]
        cached_pol = self._intermediates['cached_pol']

        dm_elec = dm['e']
        dm_nuc = [None] * self.mol.nuc_num

        if not (isinstance(dm_elec, numpy.ndarray) and dm_elec.ndim == 2):
            # spin-traced DM for UHF or ROHF
            dm_elec = dm_elec[0] + dm_elec[1]

        phi = ddcosmo.make_phi(self.pcm_elec, dm_elec, r_vdw, ui, ylm_1sph)
        for i in range(self.mol.nuc_num):
            ia = self.mol.nuc[i].atom_index
            charge = self.mol.atom_charge(ia)
            dm_nuc[i] = dm[f'n{ia}']
            phi -= charge * ddcosmo.make_phi(self.pcm_nuc[i], dm_nuc[i], r_vdw, ui, ylm_1sph, with_nuc=False)
        # minus sign for induced potential by quantum nuclei and the
        # contributions from class nuclei are muted to avoid double counting

        Xvec = numpy.linalg.solve(Lmat, phi.ravel()).reshape(self.mol.natm,-1)

        vmat = []
        psi = 0

        psi_e = make_psi(self.pcm_elec, dm_elec, r_vdw, cached_pol, with_nuc=True)
        psi += psi_e

        for i in range(self.mol.nuc_num):
            ia = self.mol.nuc[i].atom_index
            charge = self.mol.atom_charge(ia)
            psi_n = charge * make_psi(self.pcm_nuc[i], dm_nuc[i], r_vdw, cached_pol, with_nuc=False)
            psi -= psi_n

        vmat_e = make_vmat(self.pcm_elec, dm_elec, r_vdw, ui, ylm_1sph, cached_pol, Xvec, Lmat, psi)
        vmat.append(vmat_e)

        for i in range(self.mol.nuc_num):
            ia = self.mol.nuc[i].atom_index
            charge = self.mol.atom_charge(ia)
            vmat_n = make_vmat(self.pcm_nuc[i], dm_nuc[i], r_vdw, ui, ylm_1sph, cached_pol, Xvec, Lmat, psi)
            vmat.append(vmat_n*charge)

        dielectric = self.eps
        if dielectric > 0:
            f_epsilon = (dielectric-1.)/dielectric
        else:
            f_epsilon = 1
        epcm = .5 * f_epsilon * numpy.einsum('jx,jx', psi, Xvec)
        vpcm = .5 * f_epsilon * numpy.array(vmat, dtype=object)

        return epcm, vpcm

    def nuc_grad_method(self, grad_method):
        'For grad_method in vacuum, add nuclear gradients of solvent'
        from pyscf.neo.solvent_grad import make_grad_object
        if self.frozen:
            raise RuntimeError('Frozen solvent model is not supported for energy gradients')
        else:
            return make_grad_object(grad_method)

# inject ddcosmo
CDFT.ddCOSMO = CDFT.DDCOSMO = ddcosmo_for_neo
