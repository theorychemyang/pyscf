#!/usr/bin/env python
# Author: Kurt Brorsen (brorsenk@missouri.edu)

import numpy
from pyscf import lib, ao2mo, neo
from pyscf import mp
from timeit import default_timer as timer
from pyscf.neo.ao2mo import *

class MP2(lib.StreamObject):
    def __init__(self, mf):

        self.mf  = mf
        self.mp_ee = mp.MP2(self.mf.mf_elec)

        if(self.mf.unrestricted==True):
            raise NotImplementedError('NEO-MP2 is for RHF wave functions only')

        if(len(self.mf.mol.nuc)>1):
            raise NotImplementedError('NEO-MP2 is for single quantum nuclei')

    def kernel(self):

        emp2_ee = self.mp_ee.kernel()[0]

        e_nocc = self.mf.mf_elec.mo_coeff[:,self.mf.mf_elec.mo_occ>0].shape[1]
        e_tot  = self.mf.mf_elec.mo_coeff[0,:].shape[0]
        e_nvir = e_tot - e_nocc

        p_nocc = self.mf.mf_nuc[0].mo_coeff[:,self.mf.mf_nuc[0].mo_occ>0].shape[1]
        p_tot  = self.mf.mf_nuc[0].mo_coeff[0,:].shape[0]
        p_nvir = p_tot - p_nocc

        eia = self.mf.mf_elec.mo_energy[:e_nocc,None] - self.mf.mf_elec.mo_energy[None,e_nocc:]
        ejb = self.mf.mf_nuc[0].mo_energy[:p_nocc,None] - self.mf.mf_nuc[0].mo_energy[None,p_nocc:]

        start = timer()
        eri_ep = neo.ao2mo.ep_ovov(self.mf)
        finish = timer()

        print('time for ep ao2mo transform = ',finish-start)

        start = timer()
        emp2_ep = 0.0
        for i in range(e_nocc):
            gi = numpy.asarray(eri_ep[i*e_nvir:(i+1)*e_nvir])
            gi = gi.reshape(e_nvir,p_nocc,p_nvir)
            t2i = gi/lib.direct_sum('a+jb->ajb', eia[i], ejb)
            emp2_ep += numpy.einsum('ajb,ajb', t2i, gi)

        emp2_ep = 2.0 * emp2_ep
        end = timer()
        print('time for python mp2',end-start)

        return emp2_ee, emp2_ep


if __name__ == '__main__':

    mol = neo.Mole()
    mol.build(atom='''H 0 0 0; F 0 0 1.15; F 0 0 -1.15''', basis='ccpvdz', quantum_nuc=[0], charge=-1)
    mf = neo.HF(mol)
    energy = mf.scf()

    emp2_ee, emp2_ep = MP2(mf).kernel()

    print('emp2_ee = ',emp2_ee)
    print('emp2_ep = ',emp2_ep)
    print('total neo-mp2 = ',energy+emp2_ee+emp2_ep)
