#!/usr/bin/env python

'''
Non-relativistic Kohn-Sham for NEO-DFT
'''
import numpy
from pyscf import scf, dft, lib
from pyscf.lib import logger
from pyscf.dft.numint import eval_ao, eval_rho, _scale_ao, _dot_ao_ao
from pyscf.neo.hf import HF

class KS(HF):
    '''
    Example:
    >>> from pyscf import neo
    >>> mol = neo.Mole()
    >>> mol.build(atom = 'H 0 0 0; F 0 0 0.917', basis = 'ccpvdz', quantum_nuc=[0])
    >>> mf = neo.KS(mol, epc='17-2')
    >>> mf.scf()
    '''

    def __init__(self, mol, unrestricted=False, epc=None, verbose=4):
        HF.__init__(self, mol, verbose=verbose)

        self.unrestricted = unrestricted
        self.epc = epc # electron-proton correlation: '17-1' or '17-2' can be used

        # set up Hamiltonian for electrons
        if self.unrestricted == True:
            self.mf_elec = dft.UKS(mol.elec)
        else:
            self.mf_elec = dft.RKS(mol.elec)

        self.mf_elec.xc = 'b3lyp' # use b3lyp as the default xc functional for electrons
        self.mf_elec.get_hcore = self.get_hcore_elec
        if self.epc is not None:
            self.mf_elec.get_veff = self.get_veff_elec_epc

        # build grids (Note: high-density grids are needed since nuclei is more localized than electrons)
        self.mf_elec.grids.build(with_non0tab = False)
        self.mf_elec.verbose = self.verbose - 1

        # set up Hamiltonian for each quantum nuclei
        for i in range(len(self.mol.nuc)):
            ia = self.mol.nuc[i].atom_index
            if self.epc is not None and self.mol.atom_symbol(ia) == 'H':  # only support electron-proton correlation
                self.mf_nuc[i] = dft.RKS(self.mol.nuc[i])
                self.mf_nuc[i].get_veff = self.get_veff_nuc_epc
                #self.mf_nuc[i].conv_tol = 1e-8
                #self.mf_nuc[i].verbose = self.verbose
            else:
                self.mf_nuc[i] = scf.RHF(self.mol.nuc[i])
                self.mf_nuc[i].get_veff = self.get_veff_nuc_bare

            self.mf_nuc[i].occ_state = 0 # for delta SCF
            self.mf_nuc[i].get_occ = self.get_occ_nuc(self.mf_nuc[i])
            self.mf_nuc[i].get_init_guess = self.get_init_guess_nuc
            self.mf_nuc[i].get_hcore = self.get_hcore_nuc
            self.mf_nuc[i].verbose = self.verbose - 1
            self.dm_nuc[i] = self.get_init_guess_nuc(self.mf_nuc[i])


    def eval_xc_nuc(self, rho_e, rho_n):
        'evaluate e_xc and v_xc of proton on a grid (epc17)'
        a = 2.35
        b = 2.4

        if self.epc == '17-1':
            c = 3.2 #TODO solve the convergence issue for epc17-1
        elif self.epc == '17-2':
            c = 6.6
        else:
            raise ValueError('Unsupported type of epc %s', self.epc)

        rho_product = numpy.multiply(rho_e, rho_n)
        denominator = a - b*numpy.sqrt(rho_product) + c*rho_product
        exc = - numpy.multiply(rho_e, 1/denominator)

        denominator = numpy.square(denominator)
        numerator = -a * rho_e + numpy.multiply(numpy.sqrt(rho_product), rho_e)*b/2
        vxc = numpy.multiply(numerator, 1/denominator)

        return exc, vxc

    def eval_xc_elec(self, rho_e, rho_n):
        'evaluate e_xc and v_xc of electrons on a grid (only the epc part)'
        a = 2.35
        b = 2.4

        if self.epc == '17-1':
            c = 3.2 #TODO solve the convergence issue for epc17-1
        elif self.epc == '17-2':
            c = 6.6
        else:
            raise ValueError('Unsupported type of epc %s', self.epc)

        rho_product = numpy.multiply(rho_e, rho_n)
        denominator = a - b*numpy.sqrt(rho_product) + c*rho_product
        exc = - numpy.multiply(rho_n, 1/denominator)

        denominator = numpy.square(denominator)
        numerator = -a * rho_n + numpy.multiply(numpy.sqrt(rho_product), rho_n)*b/2
        vxc = numpy.multiply(numerator, 1/denominator)

        return exc, vxc

    def get_veff_nuc_epc(self, mol, dm, dm_last=None, vhf_last=None, hermi=1):
        # def nr_rks_nuc(self, mol, dm):
        'get the effective potential for proton of NEO-DFT'

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
            if self.unrestricted == True:
                rho_elec = eval_rho(self.mol.elec, ao_elec, self.dm_elec[0]+self.dm_elec[1])
            else:
                rho_elec = eval_rho(self.mol.elec, ao_elec, self.dm_elec)
            ao_nuc = eval_ao(mol, coords)
            rho_nuc = eval_rho(mol, ao_nuc, dm)

            exc, vxc = self.eval_xc_nuc(rho_elec, rho_nuc)
            den = rho_nuc * weight
            nnuc += den.sum()
            excsum += numpy.dot(den, exc)
            aow = _scale_ao(ao_nuc, .5*weight*vxc, out=aow) # *0.5 because vmat + vmat.T
            vmat += _dot_ao_ao(mol, ao_nuc, aow, mask, shls_slice, ao_loc)

        logger.debug(self, 'the number of nuclei: %.5f', nnuc)
        vmat += vmat.conj().T
        vmat = lib.tag_array(vmat, exc=excsum, ecoul=0, vj=0, vk=0)
        return vmat

    def get_veff_nuc2(self, mol, dm, dm_last=None, vhf_last=None, hermi=1):
        # TODO DIIS for scf of quantum nucleus
        if dm_last is None:
            veff = self.nr_rks_nuc(mol, dm)
        else:
            ddm = numpy.asarray(dm) - numpy.asarray(dm_last)
            veff = self.nr_rks_nuc(mol, ddm) + numpy.asarray(vhf_last)
        return veff

    def get_veff_elec_epc(self, mol, dm, dm_last=None, vhf_last=None, hermi=1):
        'get the effective potential for electrons of NEO-DFT'

        nao = mol.nao_nr()
        shls_slice = (0, mol.nbas)
        ao_loc = mol.ao_loc_nr()
        excsum = 0
        vmat = numpy.zeros((nao, nao))

        grids = self.mf_elec.grids
        ni = self.mf_elec._numint

        aow = None
        for i in range(len(self.mol.nuc)):
            ia = self.mol.nuc[i].atom_index
            if self.mol.atom_symbol(ia) == 'H':
                for ao, mask, weight, coords in ni.block_loop(mol, grids, nao):
                    aow = numpy.ndarray(ao.shape, order='F', buffer=aow)
                    ao_elec = eval_ao(mol, coords)
                    if self.unrestricted == True:
                        rho_elec = eval_rho(mol, ao_elec, dm[0]+dm[1])
                    else:
                        rho_elec = eval_rho(mol, ao_elec, dm)

                    ao_nuc = eval_ao(self.mol.nuc[i], coords)
                    rho_nuc = eval_rho(self.mol.nuc[i], ao_nuc, self.dm_nuc[i])

                    exc_i, vxc_i = self.eval_xc_elec(rho_elec, rho_nuc)
                    den = rho_elec * weight
                    excsum += numpy.dot(den, exc_i)
                    aow = _scale_ao(ao_elec, .5*weight*vxc_i, out=aow) # *0.5 because vmat + vmat.T
                    vmat += _dot_ao_ao(mol, ao_elec, aow, mask, shls_slice, ao_loc)

        vmat += vmat.conj().T
        if self.unrestricted == True:
            veff = dft.uks.get_veff(self.mf_elec, mol, dm, dm_last, vhf_last, hermi)
        else:
            veff = dft.rks.get_veff(self.mf_elec, mol, dm, dm_last, vhf_last, hermi)
        vxc = lib.tag_array(veff + vmat, ecoul = veff.ecoul, exc = veff.exc, vj = veff.vj, vk = veff.vk)
        return vxc

    def energy_qmnuc(self, mf_nuc, h1n, dm_nuc):
        'energy of quantum nuclei by NEO-DFT'
        n1 = numpy.einsum('ij,ji', h1n, dm_nuc)
        ia = mf_nuc.mol.atom_index
        if self.mol.atom_symbol(ia) == 'H' and self.epc is not None:
            veff = mf_nuc.get_veff(mf_nuc.mol, dm_nuc)
            n1 += veff.exc
        #logger.debug(self, 'Energy of %s: %s', self.mol.atom_symbol(ia), n1)
        return n1
