#!/usr/bin/env python

'''
Non-relativistic Kohn-Sham for NEO-DFT
'''
import numpy
from pyscf import scf, dft, lib
from pyscf.lib import logger
from pyscf.dft.numint import eval_ao, eval_rho, _scale_ao, _dot_ao_ao
from pyscf.neo.hf import HF

def eval_xc_nuc(epc, rho_e, rho_n):
    '''evaluate e_xc and v_xc of proton on a grid (epc17)'''
    a = 2.35
    b = 2.4

    if epc == '17-1':
        c = 3.2
    elif epc == '17-2':
        c = 6.6
    else:
        raise ValueError('Unsupported type of epc %s', epc)

    rho_product = numpy.multiply(rho_e, rho_n)
    denominator = a - b * numpy.sqrt(rho_product) + c * rho_product
    exc = - numpy.multiply(rho_e, 1 / denominator)

    denominator = numpy.square(denominator)
    numerator = -a * rho_e + numpy.multiply(numpy.sqrt(rho_product), rho_e) * b * 0.5
    vxc = numpy.multiply(numerator, 1 / denominator)

    return exc, vxc

def eval_xc_elec(epc, rho_e, rho_n):
    '''evaluate e_xc and v_xc of electrons on a grid (only the epc part)'''
    a = 2.35
    b = 2.4

    if epc == '17-1':
        c = 3.2
    elif epc == '17-2':
        c = 6.6
    else:
        raise ValueError('Unsupported type of epc %s', epc)

    rho_product = numpy.multiply(rho_e, rho_n)
    denominator = a - b * numpy.sqrt(rho_product) + c * rho_product
    denominator = numpy.square(denominator)
    numerator = -a * rho_n + numpy.multiply(numpy.sqrt(rho_product), rho_n) * b * 0.5
    vxc = numpy.multiply(numerator, 1 / denominator)

    return vxc


class KS(HF):
    '''
    Example:
    >>> from pyscf import neo
    >>> mol = neo.Mole()
    >>> mol.build(atom='H 0 0 0; F 0 0 0.917', basis='ccpvdz', quantum_nuc=[0])
    >>> mf = neo.KS(mol, epc='17-2')
    >>> mf.scf()
    '''

    def __init__(self, mol, unrestricted=False, epc=None):
        HF.__init__(self, mol)

        if mol.elec.nhomo is not None:
            unrestricted = True
        self.unrestricted = unrestricted
        self.epc = epc # electron-proton correlation: '17-1' or '17-2' can be used

        # set up Hamiltonian for electrons
        if self.unrestricted == True:
            self.mf_elec = dft.UKS(mol.elec)
        else:
            self.mf_elec = dft.RKS(mol.elec)
        # need to repeat these lines because self.mf_elec got overwritten
        self.mf_elec.xc = 'b3lyp' # use b3lyp as the default xc functional for electrons
        self.mf_elec.get_hcore = self.get_hcore_elec
        if self.epc is not None:
            self.mf_elec.get_veff = self.get_veff_elec_epc
        self.mf_elec.super_mf = self
        if mol.elec.nhomo is not None:
            self.mf_elec.get_occ = self.get_occ_elec(self.mf_elec)

        # build grids (Note: high-density grids are needed since nuclei is more localized than electrons)
        self.mf_elec.grids.build(with_non0tab=False)

        # set up Hamiltonian for each quantum nuclei
        for i in range(mol.nuc_num):
            ia = self.mol.nuc[i].atom_index
            # only support electron-proton correlation
            if self.epc is not None and self.mol.atom_pure_symbol(ia) == 'H':
                self.mf_nuc[i] = dft.RKS(self.mol.nuc[i])
                mf_nuc = self.mf_nuc[i]
                # need to repeat these lines because self.mf_nuc got overwritten
                mf_nuc.occ_state = 0 # for Delta-SCF
                mf_nuc.get_occ = self.get_occ_nuc(mf_nuc)
                mf_nuc.get_hcore = self.get_hcore_nuc
                mf_nuc.get_veff = self.get_veff_nuc_epc
                mf_nuc.energy_qmnuc = self.energy_qmnuc
                mf_nuc.super_mf = self

    def get_veff_nuc_epc(self, mol, dm, dm_last=None, vhf_last=None, hermi=1):
        '''Add EPC contribution to nuclear veff'''
        nao = mol.nao_nr()
        shls_slice = (0, mol.nbas)
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
            exc, vxc = eval_xc_nuc(self.epc, rho_elec, rho_nuc)
            den = rho_nuc * weight
            nnuc += den.sum()
            excsum += numpy.dot(den, exc)
            # times 0.5 because vmat + vmat.T
            aow = _scale_ao(ao_nuc, 0.5 * weight * vxc, out=aow)
            vmat += _dot_ao_ao(mol, ao_nuc, aow, mask, shls_slice, ao_loc)
        logger.debug(self, 'The number of nuclei: %.5f', nnuc)
        vmat += vmat.conj().T
        # attach E_ep to vmat to retrieve later
        vmat = lib.tag_array(vmat, exc=excsum, ecoul=0, vj=0, vk=0)
        return vmat

    def get_veff_elec_epc(self, mol, dm, dm_last=None, vhf_last=None, hermi=1):
        '''Add EPC contribution to electronic veff'''
        nao = mol.nao_nr()
        shls_slice = (0, mol.nbas)
        ao_loc = mol.ao_loc_nr()
        vmat = numpy.zeros((nao, nao))

        grids = self.mf_elec.grids
        ni = self.mf_elec._numint

        aow = None
        for i in range(self.mol.nuc_num):
            ia = self.mol.nuc[i].atom_index
            if self.mol.atom_pure_symbol(ia) == 'H':
                for ao, mask, weight, coords in ni.block_loop(mol, grids, nao):
                    aow = numpy.ndarray(ao.shape, order='F', buffer=aow)
                    ao_elec = eval_ao(mol, coords)
                    if dm.ndim > 2:
                        rho_elec = eval_rho(mol, ao_elec, dm[0] + dm[1])
                    else:
                        rho_elec = eval_rho(mol, ao_elec, dm)
                    ao_nuc = eval_ao(self.mol.nuc[i], coords)
                    rho_nuc = eval_rho(self.mol.nuc[i], ao_nuc, self.dm_nuc[i])
                    vxc_i = eval_xc_elec(self.epc, rho_elec, rho_nuc)
                    # times 0.5 because vmat + vmat.T
                    aow = _scale_ao(ao_elec, 0.5 * weight * vxc_i, out=aow)
                    vmat += _dot_ao_ao(mol, ao_elec, aow, mask, shls_slice, ao_loc)
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
        if self.mol.atom_pure_symbol(ia) == 'H' and self.epc is not None:
            if veff_n is None:
                veff_n = mf_nuc.get_veff(mf_nuc.mol, dm_nuc)
            n1 += veff_n.exc
        logger.debug(self, 'Energy of %s (%3d): %s', self.mol.atom_symbol(ia), ia, n1)
        return n1
