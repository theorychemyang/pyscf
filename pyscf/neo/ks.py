#!/usr/bin/env python

'''
Non-relativistic Kohn-Sham for NEO-DFT
'''
import numpy
from pyscf import scf, dft, lib
from pyscf.lib import logger
from pyscf.dft.numint import eval_ao, eval_rho, _scale_ao, _dot_ao_ao, _dot_ao_ao_sparse
from pyscf.neo.hf import HF
from pyscf.dft.gen_grid import NBINS
from pyscf.qmmm.itrf import qmmm_for_scf

def eval_xc_nuc(epc, rho_e, rho_n):
    '''evaluate e_xc and v_xc of proton on a grid (epc17)'''

    epc_type = None
    if isinstance(epc, str):
        epc_type = epc
    elif isinstance(epc, dict):
        if "epc_type" not in epc:
            epc_type = '17-2'
        else:
            epc_type = epc["epc_type"]
    else:
        raise TypeError('Only string or dictionary is allowed for epc')

    if epc_type == '17-1':
        a = 2.35
        b = 2.4
        c = 3.2
    elif epc_type == '17-2':
        a = 2.35
        b = 2.4
        c = 6.6
    elif epc_type == '18-1':
        a = 1.8
        b = 0.1
        c = 0.03
    elif epc_type == '18-2':
        a = 3.9
        b = 0.5
        c = 0.06
    elif epc_type == '17' or epc_type == '18':
        a = epc["a"]
        b = epc["b"]
        c = epc["c"]
    else:
        raise ValueError('Unsupported type of epc %s', epc_type)

    if epc_type.startswith('17'):
        rho_product = numpy.multiply(rho_e, rho_n)
        denominator = a - b * numpy.sqrt(rho_product) + c * rho_product
        exc = - numpy.multiply(rho_e, 1 / denominator)

        denominator = numpy.square(denominator)
        numerator = -a * rho_e + numpy.multiply(numpy.sqrt(rho_product), rho_e) * b * 0.5
        vxc = numpy.multiply(numerator, 1 / denominator)
    elif epc_type.startswith('18'):
        rho_e_cr = numpy.cbrt(rho_e)
        rho_n_cr = numpy.cbrt(rho_n)
        beta = rho_e_cr + rho_n_cr
        denominator = a - b * beta**3 + c * beta**6
        exc = - numpy.multiply(rho_e, 1 / denominator)

        denominator = numpy.square(denominator)
        numerator = a * rho_e - b * numpy.multiply(rho_e_cr**4, numpy.square(beta))\
                    + c * numpy.multiply(numpy.multiply(rho_e, beta**5), rho_e_cr - rho_n_cr)
        vxc = - numpy.multiply(numerator, 1 / denominator)
    else:
        raise ValueError('Unsupported type of epc %s', epc_type)

    return exc, vxc

def eval_xc_elec(epc, rho_e, rho_n):
    '''evaluate v_xc of electrons on a grid (only the epc part)'''

    epc_type = None
    if isinstance(epc, str):
        epc_type = epc
    elif isinstance(epc, dict):
        if "epc_type" not in epc:
            epc_type = '17-2'
        else:
            epc_type = epc["epc_type"]
    else:
        raise TypeError('Only string or dictionary is allowed for epc')

    if epc_type == '17-1':
        a = 2.35
        b = 2.4
        c = 3.2
    elif epc_type == '17-2':
        a = 2.35
        b = 2.4
        c = 6.6
    elif epc_type == '18-1':
        a = 1.8
        b = 0.1
        c = 0.03
    elif epc_type == '18-2':
        a = 3.9
        b = 0.5
        c = 0.06
    elif epc_type == '17' or epc_type == '18':
        a = epc["a"]
        b = epc["b"]
        c = epc["c"]
    else:
        raise ValueError('Unsupported type of epc %s', epc_type)

    if epc_type.startswith('17'):
        rho_product = numpy.multiply(rho_e, rho_n)
        denominator = a - b * numpy.sqrt(rho_product) + c * rho_product
        denominator = numpy.square(denominator)
        numerator = -a * rho_n + numpy.multiply(numpy.sqrt(rho_product), rho_n) * b * 0.5
        vxc = numpy.multiply(numerator, 1 / denominator)
    elif epc_type.startswith('18'):
        rho_e_cr = numpy.cbrt(rho_e)
        rho_n_cr = numpy.cbrt(rho_n)
        beta = rho_e_cr + rho_n_cr
        denominator = a - b * beta**3 + c * beta**6
        denominator = numpy.square(denominator)
        numerator = a * rho_n - b * numpy.multiply(rho_n_cr**4, numpy.square(beta))\
                    + c * numpy.multiply(numpy.multiply(rho_n, beta**5), rho_n_cr - rho_e_cr)
        vxc = - numpy.multiply(numerator, 1 / denominator)
    else:
        raise ValueError('Unsupported type of epc %s', epc_type)

    return vxc


class KS(HF):
    '''
    Examples::

    >>> from pyscf import neo
    >>> mol = neo.Mole()
    >>> mol.build(atom='H 0 0 0; F 0 0 0.917', quantum_nuc=[0], basis='ccpvdz', nuc_basis='pb4d')
    >>> mf = neo.KS(mol, epc='17-2')
    >>> mf.max_cycle = 100
    >>> mf.scf()
    -100.38833734158459
    '''

    def __init__(self, mol, epc=None, **kwargs):
        HF.__init__(self, mol, **kwargs)
        self.epc = epc # electron-proton correlation: '17-1' or '17-2' can be used

        # set up Hamiltonian for electrons
        if self.unrestricted:
            self.mf_elec = dft.UKS(mol.elec)
        else:
            self.mf_elec = dft.RKS(mol.elec)
        if 'df_ee' in kwargs:
            df_ee = kwargs['df_ee']
        else:
            df_ee = False
        if 'auxbasis_e' in kwargs:
            auxbasis_e = kwargs['auxbasis_e']
        else:
            auxbasis_e = None
        if 'only_dfj_e' in kwargs:
            only_dfj_e = kwargs['only_dfj_e']
        else:
            only_dfj_e = False
        if df_ee:
            self.mf_elec = self.mf_elec.density_fit(auxbasis=auxbasis_e,
                                                    only_dfj=only_dfj_e)
        if self.mol.mm_mol is not None:
            self.mf_elec = qmmm_for_scf(self.mf_elec, self.mol.mm_mol)
        # need to repeat these lines because self.mf_elec got overwritten
        self.mf_elec.xc = 'b3lyp' # use b3lyp as the default xc functional for electrons
        self.mf_elec.get_hcore = self.get_hcore_elec
        self.mf_elec.hcore_static = None
        if self.epc is not None:
            self.mf_elec.get_veff = self.get_veff_elec_epc
        if mol.elec.nhomo is not None:
            self.mf_elec.get_occ = self.get_occ_elec(self.mf_elec)

        # set up Hamiltonian for each quantum nuclei
        for i in range(mol.nuc_num):
            ia = self.mol.nuc[i].atom_index
            # only support electron-proton correlation
            if self.epc is not None and self.mol.atom_pure_symbol(ia) == 'H' \
                and (isinstance(self.epc, str) or ia in self.epc['epc_nuc']):
                self.mf_nuc[i] = dft.RKS(self.mol.nuc[i])
                mf_nuc = self.mf_nuc[i]
                # need to repeat these lines because self.mf_nuc got overwritten
                mf_nuc.occ_state = 0 # for Delta-SCF
                mf_nuc.get_occ = self.get_occ_nuc(mf_nuc)
                mf_nuc.get_hcore = self.get_hcore_nuc(mf_nuc)
                mf_nuc.hcore_static = None
                mf_nuc.get_veff = self.get_veff_nuc_epc
                mf_nuc.energy_qmnuc = self.energy_qmnuc

    def get_veff_nuc_epc(self, mol, dm, dm_last=None, vhf_last=None, hermi=1):
        '''Add EPC contribution to nuclear veff'''
        nao = mol.nao_nr()
        ao_loc = mol.ao_loc_nr()
        nnuc = 0
        excsum = 0
        vmat = numpy.zeros((nao, nao))

        grids = self.mf_elec.grids
        ni = self.mf_elec._numint

        aow = None
        for ao, mask, weight, coords in ni.block_loop(mol, grids, nao):
            aow = numpy.ndarray(ao.shape, order='F', buffer=aow)
            ao_elec = eval_ao(self.mol.elec, coords)
            if self.dm_elec.ndim > 2:
                rho_elec = eval_rho(self.mol.elec, ao_elec, self.dm_elec[0] + self.dm_elec[1])
            else:
                rho_elec = eval_rho(self.mol.elec, ao_elec, self.dm_elec)
            ao_nuc = eval_ao(mol, coords)
            rho_nuc = eval_rho(mol, ao_nuc, dm)
            rho_nuc[rho_nuc<0.] = 0.
            exc, vxc = eval_xc_nuc(self.epc, rho_elec, rho_nuc)
            den = rho_nuc * weight
            nnuc += den.sum()
            excsum += numpy.dot(den, exc)
            # times 0.5 because vmat + vmat.T
            aow = _scale_ao(ao_nuc, 0.5 * weight * vxc, out=aow)
            vmat += _dot_ao_ao(mol, ao_nuc, aow, mask, (0, mol.nbas), ao_loc)
        logger.debug(self, 'The number of nuclei: %.5f', nnuc)
        vmat += vmat.conj().T
        # attach E_ep to vmat to retrieve later
        vmat = lib.tag_array(vmat, exc=excsum, ecoul=0, vj=0, vk=0)
        return vmat

    def get_veff_elec_epc(self, mol, dm, dm_last=None, vhf_last=None, hermi=1):
        '''Add EPC contribution to electronic veff'''
        nao = mol.nao_nr()
        ao_loc = mol.ao_loc_nr()
        vmat = numpy.zeros((nao, nao))
        grids = self.mf_elec.grids
        ni = self.mf_elec._numint
        cutoff = grids.cutoff * 1e2
        nbins = NBINS * 2 - int(NBINS * numpy.log(cutoff) / numpy.log(grids.cutoff))
        pair_mask = mol.get_overlap_cond() < -numpy.log(ni.cutoff)
        for i in range(self.mol.nuc_num):
            ia = self.mol.nuc[i].atom_index
            if self.mol.atom_pure_symbol(ia) == 'H' \
                and (isinstance(self.epc, str) or ia in self.epc['epc_nuc']):
                for ao, mask, weight, coords in ni.block_loop(mol, grids, nao):
                    if dm.ndim > 2:
                        rho_elec = eval_rho(mol, ao, dm[0] + dm[1])
                    else:
                        rho_elec = eval_rho(mol, ao, dm)
                    ao_nuc = eval_ao(self.mol.nuc[i], coords)
                    rho_nuc = eval_rho(self.mol.nuc[i], ao_nuc, self.dm_nuc[i])
                    rho_nuc[rho_nuc<0.] = 0.
                    vxc_i = eval_xc_elec(self.epc, rho_elec, rho_nuc)
                    wv = weight * vxc_i
                    # times 0.5 because vmat + vmat.T
                    _dot_ao_ao_sparse(ao, ao, 0.5*wv, nbins, mask, pair_mask, ao_loc, 1, vmat)
        vmat += vmat.conj().T
        if dm.ndim > 2:
            veff = dft.uks.get_veff(self.mf_elec, mol, dm, dm_last, vhf_last, hermi)
        else:
            veff = dft.rks.get_veff(self.mf_elec, mol, dm, dm_last, vhf_last, hermi)
        vxc = lib.tag_array(veff + vmat, ecoul=veff.ecoul, exc=veff.exc, vj=veff.vj, vk=veff.vk)
        return vxc

    def energy_qmnuc(self, mf_nuc, h1n, dm_nuc, veff_n=None):
        '''energy of quantum nuclei by NEO-DFT'''
        n1 = numpy.einsum('ij,ji', h1n, dm_nuc)
        ia = mf_nuc.mol.atom_index
        if self.mol.atom_pure_symbol(ia) == 'H' and self.epc is not None \
            and (isinstance(self.epc, str) or ia in self.epc['epc_nuc']):
            if veff_n is None:
                veff_n = mf_nuc.get_veff(mf_nuc.mol, dm_nuc)
            n1 += veff_n.exc
        logger.debug(self, 'Energy of %s (%3d): %s', self.mol.atom_symbol(ia), ia, n1)
        return n1

    def reset(self, mol=None):
        '''Reset mol and relevant attributes associated to the old mol object'''
        super().reset(mol=mol)
        # point to correct ``self'' for overriden functions
        if self.epc is not None:
            self.mf_elec.get_veff = self.get_veff_elec_epc
            for i in range(mol.nuc_num):
                ia = self.mol.nuc[i].atom_index
                # only support electron-proton correlation
                if self.mol.atom_pure_symbol(ia) == 'H' \
                    and (isinstance(self.epc, str) or ia in self.epc['epc_nuc']):
                    mf_nuc = self.mf_nuc[i]
                    mf_nuc.get_veff = self.get_veff_nuc_epc
        return self
