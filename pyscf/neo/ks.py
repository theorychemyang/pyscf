#!/usr/bin/env python

'''
Non-relativistic Kohn-Sham for NEO-DFT
'''

import copy
import numpy
import warnings
from pyscf import dft, lib, scf
from pyscf.data import nist
from pyscf.lib import logger
from pyscf.dft.numint import (BLKSIZE, NBINS, eval_ao, eval_rho, _scale_ao,
                              _dot_ao_ao, _dot_ao_ao_sparse)
from pyscf.neo import hf

def precompute_epc_electron(epc, rho_e):
    '''Pre-compute electron-dependent terms in EPC functional.

    Args:
        epc : str or dict
            EPC functional specification. Can be '17-1', '17-2', '18-1', '18-2'
            or a dict with 'epc_type' and parameters.
        rho_e : ndarray
            Electron density on grid points

    Returns:
        dict: Pre-computed quantities for EPC
    '''
    params = {
        '17-1': (2.35, 2.4, 3.2),
        '17-2': (2.35, 2.4, 6.6),
        '18-1': (1.8, 0.1, 0.03),
        '18-2': (3.9, 0.5, 0.06)
    }
    # Parse EPC type and parameters
    if isinstance(epc, dict):
        epc_type = epc.get('epc_type', '17-2')
        if epc_type in ('17', '18'):
            a = epc['a']
            b = epc['b']
            c = epc['c']
        else:
            if epc_type not in params:
                raise ValueError(f'Unknown EPC type: {epc_type}')
            a, b, c = params[epc_type]
    else:
        epc_type = epc
        if epc_type not in params:
            raise ValueError(f'Unknown EPC type: {epc_type}')
        a, b, c = params[epc_type]

    common = {'a': a, 'b': b, 'c': c, 'type': epc_type}
    if epc_type.startswith('17'):
        common['rho_e'] = rho_e
    else:  # EPC18
        common['rho_e'] = rho_e
        common['rho_e_cbrt'] = numpy.cbrt(rho_e)
        common['rho_e_cbrt4'] = common['rho_e_cbrt']**4

    return common

def eval_epc(common, rho_n):
    '''Evaluate EPC energy and potentials using pre-computed electron quantities.

    Args:
        common : dict
            Pre-computed electron-dependent quantities from precompute_epc_electron
        rho_n : ndarray
            Nuclear density on grid points

    Returns:
        exc, vxc_n, vxc_e : Energy density and potentials
    '''
    epc_type = common['type']
    a = common['a']
    b = common['b']
    c = common['c']
    rho_e = common['rho_e']

    if epc_type.startswith('17'):
        # EPC17 form
        rho_prod = numpy.multiply(rho_e, rho_n)
        rho_sqrt = numpy.sqrt(rho_prod)
        denom = a - b * rho_sqrt + c * rho_prod
        denom2 = numpy.square(denom)

        # Energy density
        exc = -rho_e / denom

        # Nuclear potential
        numer_n = -a * rho_e + 0.5 * b * rho_e * rho_sqrt
        vxc_n = numer_n / denom2

        # Electronic potential
        numer_e = -a * rho_n + 0.5 * b * rho_n * rho_sqrt
        vxc_e = numer_e / denom2

    else:
        # EPC18 form
        rho_e_cbrt = common['rho_e_cbrt']
        rho_e_cbrt4 = common['rho_e_cbrt4']
        rho_n_cbrt = numpy.cbrt(rho_n)
        beta = rho_e_cbrt + rho_n_cbrt
        beta2 = numpy.square(beta)
        beta3 = beta * beta2
        beta5 = beta2 * beta3
        beta6 = beta3 * beta3
        denom = a - b * beta3 + c * beta6
        denom2 = numpy.square(denom)

        # Energy density
        exc = -rho_e / denom
        # Nuclear potential
        numer_n = a * rho_e - b * rho_e_cbrt4 * beta2 \
                + c * numpy.multiply(rho_e * beta5, rho_e_cbrt - rho_n_cbrt)
        vxc_n = -numer_n / denom2

        # Electronic potential
        numer_e = a * rho_n - b * rho_n_cbrt**4 * beta2 \
                + c * numpy.multiply(rho_n * beta5, rho_n_cbrt - rho_e_cbrt)
        vxc_e = -numer_e / denom2

    return exc, vxc_n, vxc_e

def _hash_grids(grids):
    return hash((
            grids.level,
            grids.atom_grid if not isinstance(grids.atom_grid, dict) else tuple(grids.atom_grid.items()),
            grids.coords.shape if grids.coords is not None else None,
            grids.coords[0,0].item() if grids.coords is not None else None,
            grids.coords[-1,-1].item() if grids.coords is not None else None,
            grids.weights.size if grids.weights is not None else None,
            grids.weights[0].item() if grids.weights is not None else None,
            grids.weights[-1].item() if grids.weights is not None else None))

class InteractionCorrelation(hf.InteractionCoulomb):
    '''Inter-component Coulomb and correlation'''
    def __init__(self, *args, epc=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.epc = epc
        self.grids = None
        self._elec_grids_hash = None
        self._skip_epc = False

    def _need_epc(self):
        if self.epc is None:
            return False
        if self.mf1_type == 'e':
            if self.mf2_type.startswith('n'):
                if self.mf2.mol.super_mol.atom_pure_symbol(self.mf2.mol.atom_index) == 'H':
                    if isinstance(self.epc, str) or \
                            self.mf2.mol.atom_index in self.epc['epc_nuc']:
                        symbol = self.mf2.mol.super_mol.atom_symbol(self.mf2.mol.atom_index)
                        if 'H+' in symbol or 'H*' in symbol or 'H#' in symbol:
                            warnings.warn('Hydrogen isotopes detected. Are you sure you want epc?')
                        return True
        if self.mf2_type == 'e':
            if self.mf1_type.startswith('n'):
                if self.mf1.mol.super_mol.atom_pure_symbol(self.mf1.mol.atom_index) == 'H':
                    if isinstance(self.epc, str) or \
                            self.mf1.mol.atom_index in self.epc['epc_nuc']:
                        symbol = self.mf1.mol.super_mol.atom_symbol(self.mf1.mol.atom_index)
                        if 'H+' in symbol or 'H*' in symbol or 'H#' in symbol:
                            warnings.warn('Hydrogen isotopes detected. Are you sure you want epc?')
                        return True
        return False

    def get_vint(self, dm, *args, no_epc=False, **kwargs):
        '''Unoptimized implementation that has duplicated electronic part
        calculations if multiple protons are present. The grids are screened
        only for this particular nucleus.'''
        vj = super().get_vint(dm, *args, **kwargs)
        # For nuclear initial guess, use Coulomb only
        if no_epc or \
                not (self.mf1_type in dm and self.mf2_type in dm and self._need_epc()):
            return vj

        if self.mf1_type == 'e':
            mf_e, dm_e = self.mf1, dm[self.mf1_type]
            if self.mf1_unrestricted:
                assert dm_e.ndim > 2 and dm_e.shape[0] == 2
                dm_e = dm_e[0] + dm_e[1]
            mf_n, dm_n = self.mf2, dm[self.mf2_type]
            n_type = self.mf2_type
        else:
            mf_e, dm_e = self.mf2, dm[self.mf2_type]
            if self.mf2_unrestricted:
                assert dm_e.ndim > 2 and dm_e.shape[0] == 2
                dm_e = dm_e[0] + dm_e[1]
            mf_n, dm_n = self.mf1, dm[self.mf1_type]
            n_type = self.mf1_type

        ni = mf_e._numint
        mol_e = mf_e.mol
        mol_n = mf_n.mol
        nao_e = mol_e.nao
        nao_n = mol_n.nao
        ao_loc_e = mol_e.ao_loc_nr()
        ao_loc_n = mol_n.ao_loc_nr()

        grids_e = mf_e.grids
        grids_changed = (self._elec_grids_hash != _hash_grids(grids_e))
        if grids_changed:
            self._skip_epc = False
        if self._skip_epc:
            return vj

        if self.grids is None or grids_changed:
            if grids_e.coords is None:
                grids_e.build(with_non0tab=True)
            self._elec_grids_hash = _hash_grids(grids_e)
            # Screen grids based on nuclear basis functions
            non0tab_n = ni.make_mask(mol_n, grids_e.coords)
            blk_index = numpy.where(numpy.any(non0tab_n > 0, axis=1))[0]

            # Skip if no nuclear basis functions
            if len(blk_index) == 0:
                self._skip_epc = True
                return vj

            # Update grid coordinates and weights
            starts = blk_index[:, None] * BLKSIZE + numpy.arange(BLKSIZE)
            mask = starts < len(grids_e.coords)
            valid_indices = starts[mask]
            self.grids = copy.copy(grids_e)
            self.grids.coords = grids_e.coords[valid_indices]
            self.grids.weights = grids_e.weights[valid_indices]
            self.grids.non0tab = ni.make_mask(mol_e, self.grids.coords)
            self.grids.screen_index = self.grids.non0tab

        grids = self.grids

        exc_sum = 0
        vxc_e = numpy.zeros((nao_e, nao_e))
        vxc_n = numpy.zeros((nao_n, nao_n))

        cutoff = grids.cutoff * 1e2
        nbins = NBINS * 2 - int(NBINS * numpy.log(cutoff) / numpy.log(grids.cutoff))
        pair_mask_e = mol_e.get_overlap_cond() < -numpy.log(ni.cutoff)

        non0tab_n = ni.make_mask(mol_n, grids.coords)

        p1 = 0
        for ao_e, mask_e, weight, coords in ni.block_loop(mol_e, grids, nao_e):
            p0, p1 = p1, p1 + weight.size
            mask_n = non0tab_n[p0//BLKSIZE:p1//BLKSIZE+1]

            rho_e = eval_rho(mol_e, ao_e, dm_e, mask_e)
            rho_e[rho_e < 0] = 0  # Ensure non-negative density
            common = precompute_epc_electron(self.epc, rho_e)

            ao_n = eval_ao(mol_n, coords, non0tab=mask_n)
            rho_n = eval_rho(mol_n, ao_n, dm_n)
            rho_n[rho_n < 0] = 0  # Ensure non-negative density

            exc, vxc_n_grid, vxc_e_grid = eval_epc(common, rho_n)

            den = rho_n * weight
            exc_sum += numpy.dot(den, exc)

            # x0.5 for vmat + vmat.T
            aow = _scale_ao(ao_n, 0.5 * weight * vxc_n_grid)
            vxc_n += _dot_ao_ao(mol_n, ao_n, aow, mask_n,
                                (0, mol_n.nbas), ao_loc_n)
            _dot_ao_ao_sparse(ao_e, ao_e, 0.5 * weight * vxc_e_grid,
                              nbins, mask_e, pair_mask_e, ao_loc_e, 1, vxc_e)

        vxc_n = vxc_n + vxc_n.conj().T
        vxc_e = vxc_e + vxc_e.conj().T

        vxc = {}
        # Assign epc correlation energy to electrons
        vxc['e'] = lib.tag_array(vj['e'] + vxc_e, exc=exc_sum, vj=vj['e'])
        vxc[n_type] = lib.tag_array(vj[n_type] + vxc_n, exc=0, vj=vj[n_type])
        return vxc

class KS(hf.HF):
    '''
    Examples::

    >>> from pyscf import neo
    >>> mol = neo.M(atom='H 0 0 0; F 0 0 0.917', quantum_nuc=[0], basis='ccpvdz', nuc_basis='pb4d')
    >>> mf = neo.KS(mol, xc='b3lyp5', epc='17-2')
    >>> mf.max_cycle = 100
    >>> mf.scf()
    -100.38833734158459
    '''

    def __init__(self, mol, *args, xc=None, epc=None, **kwargs):
        super().__init__(mol, *args, **kwargs)
        # NOTE: To prevent user error, require xc to be explicitly provided
        if xc is None:
            raise RuntimeError('Please provide electronic xc via "xc" kwarg!')
        self.xc_e = xc # Electron xc functional
        self.epc = epc # Electron-proton correlation

        for t, comp in self.mol.components.items():
            if t.startswith('n'):
                if self.epc is None:
                    mf = scf.RHF(comp)
                else:
                    mf = dft.RKS(comp, xc='HF')
                self.components[t] = hf.general_scf(mf,
                                                    charge=-1. * self.mol.atom_charge(comp.atom_index),
                                                    mass=self.mol.mass[comp.atom_index] * nist.ATOMIC_MASS
                                                         / nist.E_MASS,
                                                    is_nucleus=True,
                                                    nuc_occ_state=0)
            else:
                if self.unrestricted:
                    if self.epc is None and self.xc_e.upper() == 'HF':
                        mf = scf.UHF(comp)
                    else:
                        mf = dft.UKS(comp, xc=self.xc_e)
                else:
                    if getattr(comp, 'nhomo', None) is not None or comp.spin != 0:
                        if self.epc is None and self.xc_e.upper() == 'HF':
                            mf = scf.UHF(comp)
                        else:
                            mf = dft.UKS(comp, xc=self.xc_e)
                    else:
                        if self.epc is None and self.xc_e.upper() == 'HF':
                            mf = scf.RHF(comp)
                        else:
                            mf = dft.RKS(comp, xc=self.xc_e)
                charge = 1.
                if t.startswith('p'):
                    charge = -1.
                self.components[t] = hf.general_scf(mf, charge=charge)
        self.interactions = hf.generate_interactions(self.components, InteractionCorrelation,
                                                     self.max_memory, epc=self.epc)
        #####
        self._epc_n_types = None
        self._skip_epc = False
        if isinstance(self.components['e'], scf.hf.KohnShamDFT):
            self._numint = self.components['e']._numint
        else:
            self._numint = None
        self.grids = None
        self._elec_grids_hash = None

    def energy_elec(self, dm=None, h1e=None, vhf=None, vint=None):
        if dm is None: dm = self.make_rdm1()
        if h1e is None: h1e = self.get_hcore()
        if vint is None:
            vint = self.get_vint(self.mol, dm)
            vhf = self.get_veff(self.mol, dm)
        if vhf is None: vhf = self.get_veff(self.mol, dm)
        self.scf_summary['e1'] = 0
        self.scf_summary['coul'] = 0
        self.scf_summary['exc'] = 0
        e_elec = 0
        e2 = 0
        for t, comp in self.components.items():
            logger.debug(self, f'Component: {t}')
            e_elec_t, e2_t = comp.energy_elec(dm[t], h1e[t], vhf[t])
            e_elec += e_elec_t
            e2 += e2_t
            self.scf_summary['e1'] += comp.scf_summary['e1']
            if hasattr(vhf[t], 'exc'):
                self.scf_summary['coul'] += comp.scf_summary['coul']
                self.scf_summary['exc'] += comp.scf_summary['exc']
            elif 'e2' in comp.scf_summary:
                self.scf_summary['coul'] += comp.scf_summary['e2']
        return e_elec, e2

    def get_vint_slow(self, mol=None, dm=None):
        '''Inter-type Coulomb and possible epc, slow version'''
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        # Old code that works with InteractionCorrelation.get_vint
        vint = {}
        for t in self.components.keys():
            vint[t] = 0
        for t_pair, interaction in self.interactions.items():
            v = interaction.get_vint(dm)
            for t in t_pair:
                # Take care of tag_array, accumulate exc and vj
                # NOTE: tag_array is scheduled to be removed in the future
                v_has_tag = hasattr(v[t], 'exc')
                vint_has_tag = hasattr(vint[t], 'exc')
                if v_has_tag:
                    if vint_has_tag:
                        exc = vint[t].exc + v[t].exc
                        vj = vint[t].vj + v[t].vj
                    else:
                        exc = v[t].exc
                        vj = v[t].vj
                    vint[t] = lib.tag_array(vint[t] + v[t], exc=exc, vj=vj)
                else:
                    if vint_has_tag:
                        vint[t] = lib.tag_array(vint[t] + v[t], exc=vint[t].exc, vj=vint[t].vj + v[t])
                    else:
                        vint[t] += v[t]
        # Transfer vint to cache
        for t in self.components.keys():
            self.components[t]._vint = vint[t]
            vint[t] = 0
        return vint

    def get_vint_fast(self, mol=None, dm=None):
        '''Inter-type Coulomb and possible epc'''
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()

        # For better performance, avoid duplicated elec calculations for multiple protons

        # First get Vj
        # NOTE: super().get_vint just uses that function form. This does not mean
        # pure Coulomb will be used. The interaction class is still the Correlation one.
        # Therefore, we need to be able to get Coulomb within the Correlation class.
        vj = super().get_vint(mol, dm, no_epc=True)
        if self.epc is None:
            return vj

        mf_e = self.components['e']
        grids_e = mf_e.grids
        grids_changed = (self._elec_grids_hash != _hash_grids(grids_e))
        if grids_changed and self._epc_n_types is not None:
            if len(self._epc_n_types) > 0:
                self._skip_epc = False
        if self._skip_epc:
            return vj

        if self._epc_n_types is None:
            n_types = []

            for t_pair, interaction in self.interactions.items():
                if interaction._need_epc():
                    if t_pair[0].startswith('n'):
                        n_type  = t_pair[0]
                    else:
                        n_type  = t_pair[1]
                    n_types.append(n_type)
            self._epc_n_types = n_types
        else:
            n_types = self._epc_n_types

        if len(n_types) == 0:
            # No EPC needed
            self._skip_epc = True
            return vj

        mol_e = mf_e.mol
        ni = self._numint

        # Build epc grids. Based on elec grids, and blocks with no nuclear basis are pruned.
        if self.grids is None or grids_changed:
            if grids_e.coords is None:
                grids_e.build(with_non0tab=True)
            self._elec_grids_hash = _hash_grids(grids_e)
            total_mask = numpy.zeros(((len(grids_e.coords)+BLKSIZE-1)//BLKSIZE, 1),
                                     dtype=numpy.uint8)
            # Find all blocks where any nucleus has non-zero basis
            for n_type in n_types:
                mol_n = self.components[n_type].mol
                non0tab_n = ni.make_mask(mol_n, grids_e.coords)
                total_mask |= numpy.any(non0tab_n > 0, axis=1).reshape(-1,1)

            # Get blocks where nuclear basis functions exist
            blk_index = numpy.where(numpy.any(total_mask > 0, axis=1))[0]
            if len(blk_index) == 0:
                self._skip_epc = True # No basis overlap, skip epc
                return vj

            starts = blk_index[:, None] * BLKSIZE + numpy.arange(BLKSIZE)
            mask = starts < len(grids_e.coords)
            valid_indices = starts[mask]

            # Copy grids object but reset screened data-related attributes
            self.grids = copy.copy(grids_e)
            self.grids.coords = grids_e.coords[valid_indices]
            self.grids.weights = grids_e.weights[valid_indices]
            self.grids.non0tab = ni.make_mask(mol_e, self.grids.coords)
            self.grids.screen_index = self.grids.non0tab

        grids = self.grids

        dm_e = dm['e']
        if isinstance(mf_e, scf.uhf.UHF):
            assert dm_e.ndim > 2 and dm_e.shape[0] == 2
            dm_e = dm_e[0] + dm_e[1]

        cutoff = grids.cutoff * 1e2
        nbins = NBINS * 2 - int(NBINS * numpy.log(cutoff) / numpy.log(grids.cutoff))
        pair_mask_e = mol_e.get_overlap_cond() < -numpy.log(ni.cutoff)

        exc_sum = 0
        nao_e = mol_e.nao
        vxc_e = numpy.zeros((nao_e, nao_e))
        ao_loc_e = mol_e.ao_loc_nr()

        mol_n = {}
        non0tab_n = {}
        vxc_n = {}
        ao_loc_n = {}
        for n_type in n_types:
            mol_n_t = self.components[n_type].mol
            mol_n[n_type] = mol_n_t
            non0tab_n[n_type] = ni.make_mask(mol_n_t, grids.coords)
            nao_n = mol_n_t.nao
            vxc_n[n_type] = numpy.zeros((nao_n, nao_n))
            ao_loc_n[n_type] = mol_n_t.ao_loc_nr()

        # Loop over the screened grid only once to obtain electronic part quantities
        # even when there are multiple protons
        p1 = 0
        for ao_e, mask_e, weight, coords in ni.block_loop(mol_e, grids, nao_e):
            p0, p1 = p1, p1 + weight.size
            rho_e = eval_rho(mol_e, ao_e, dm_e, mask_e)
            rho_e[rho_e < 0] = 0  # Ensure non-negative density
            common = precompute_epc_electron(self.epc, rho_e)

            vxc_e_grid = 0
            for n_type in n_types:
                mask_n = non0tab_n[n_type][p0//BLKSIZE:p1//BLKSIZE+1]
                if numpy.all(mask_n == 0):
                    continue
                mol_n_t = mol_n[n_type]
                ao_n = eval_ao(mol_n_t, coords, non0tab=mask_n)
                rho_n = eval_rho(mol_n_t, ao_n, dm[n_type])
                rho_n[rho_n < 0] = 0  # Ensure non-negative density

                exc, vxc_n_grid, vxc_e_grid_t = eval_epc(common, rho_n)
                vxc_e_grid += vxc_e_grid_t

                den = rho_n * weight
                exc_sum += numpy.dot(den, exc)

                # x0.5 for vmat + vmat.T
                aow = _scale_ao(ao_n, 0.5 * weight * vxc_n_grid)
                vxc_n[n_type] += _dot_ao_ao(mol_n_t, ao_n, aow, mask_n,
                                            (0, mol_n_t.nbas), ao_loc_n[n_type])

            _dot_ao_ao_sparse(ao_e, ao_e, 0.5 * weight * vxc_e_grid,
                              nbins, mask_e, pair_mask_e, ao_loc_e, 1, vxc_e)

        vxc_e = vxc_e + vxc_e.conj().T
        for n_type in n_types:
            vxc_n[n_type] = vxc_n[n_type] + vxc_n[n_type].conj().T

        vxc = vj
        # Assign epc correlation energy to electrons
        # vj is dummy (all zero). Actual vj is already transferred to _vint
        vxc['e'] = lib.tag_array(self.components['e']._vint + vxc_e,
                                 exc=exc_sum, vj=self.components['e']._vint)
        for n_type in n_types:
            vxc[n_type] = lib.tag_array(self.components[n_type]._vint + vxc_n[n_type],
                                        exc=0, vj=self.components[n_type]._vint)

        # Transfer vxc to vint cache
        for t in self.components.keys():
            self.components[t]._vint = vxc[t]
            vxc[t] = 0
        return vxc

    get_vint = get_vint_fast

    def reset(self, mol=None):
        '''Reset mol and relevant attributes associated to the old mol object'''
        old_keys = sorted(self.components.keys())
        super().reset(mol=mol)
        if old_keys == sorted(self.components.keys()):
            # reset grids in interactions
            for t, comp in self.interactions.items():
                comp.grids = None
                comp._elec_grids_hash = None
                comp._skip_epc = False
        else:
            # quantum nuc is different, need to rebuild
            for t, comp in self.mol.components.items():
                if t.startswith('n'):
                    if self.epc is None:
                        mf = scf.RHF(comp)
                    else:
                        mf = dft.RKS(comp, xc='HF')
                    self.components[t] = hf.general_scf(mf,
                                                        charge=-1. * self.mol.atom_charge(comp.atom_index),
                                                        mass=self.mol.mass[comp.atom_index] * nist.ATOMIC_MASS
                                                             / nist.E_MASS,
                                                        is_nucleus=True,
                                                        nuc_occ_state=0)
                else:
                    if self.unrestricted:
                        if self.epc is None and self.xc_e.upper() == 'HF':
                            mf = scf.UHF(comp)
                        else:
                            mf = dft.UKS(comp, xc=self.xc_e)
                    else:
                        if getattr(comp, 'nhomo', None) is not None or comp.spin != 0:
                            if self.epc is None and self.xc_e.upper() == 'HF':
                                mf = scf.UHF(comp)
                            else:
                                mf = dft.UKS(comp, xc=self.xc_e)
                        else:
                            if self.epc is None and self.xc_e.upper() == 'HF':
                                mf = scf.RHF(comp)
                            else:
                                mf = dft.RKS(comp, xc=self.xc_e)
                    charge = 1.
                    if t.startswith('p'):
                        charge = -1.
                    self.components[t] = hf.general_scf(mf, charge=charge)
            self.interactions.clear()
            self.interactions.update(hf.generate_interactions(self.components,
                                                              InteractionCorrelation,
                                                              self.max_memory, epc=self.epc))
        # EPC grids
        self._epc_n_types = None
        self._skip_epc = False
        if isinstance(self.components['e'], scf.hf.KohnShamDFT):
            self._numint = self.components['e']._numint
        else:
            self._numint = None
        self.grids = None
        self._elec_grids_hash = None
        return self

if __name__ == '__main__':
    from pyscf import neo
    mol = neo.M(atom='H 0 0 0', basis='ccpvdz', nuc_basis='pb4d', verbose=5, spin=1)
    mf = neo.KS(mol, xc='PBE', epc='17-2')
    mf.scf()
