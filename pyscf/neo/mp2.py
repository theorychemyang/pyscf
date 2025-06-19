#!/usr/bin/env python
# Author: Kurt Brorsen (brorsenk@missouri.edu)

import numpy
from pyscf import lib, neo, scf
from pyscf import mp
from timeit import default_timer as timer
from pyscf.neo.ao2mo import ep_ovov

class MP2(lib.StreamObject):
    def __init__(self, mf, nuclei_only=False):

        self.mf  = mf
        self.mol = mf.mol
        self.nuclei_only = nuclei_only
        self.mp_e = mp.MP2(self.mf.mf_elec)
        self.mp_n = []
        for i in range(self.mol.nuc_num):
            self.mp_n.append(mp.MP2(mf.mf_nuc[i]))


        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
        self.max_memory = mf.max_memory

        if isinstance(mf.mf_elec, scf.uhf.UHF):
            raise NotImplementedError('NEO-MP2 is for RHF wave functions only')


    def kernel(self):

        emp2_ee = 0
        if not self.nuclei_only:
            emp2_ee = self.mp_e.kernel()[0]

        e_nocc = self.mp_e.nocc
        e_nvir = self.mp_e.nmo - e_nocc

        emp2_ep = 0.0
        for i in range(self.mol.nuc_num):
            n_nocc = self.mp_n[i].nocc
            n_nvir = self.mp_n[i].nmo - n_nocc

            eia = self.mf.mf_elec.mo_energy[:e_nocc,None] - self.mf.mf_elec.mo_energy[None,e_nocc:]
            ejb = self.mf.mf_nuc[i].mo_energy[:n_nocc,None] - self.mf.mf_nuc[i].mo_energy[None,n_nocc:]

            start = timer()
            eri_ep = ep_ovov(self.mf, i)
            finish = timer()

            print('time for ep ao2mo transform = ', finish-start)

            start = timer()

            for i in range(e_nocc):
                gi = numpy.asarray(eri_ep[i*e_nvir:(i+1)*e_nvir])
                gi = gi.reshape(e_nvir, n_nocc, n_nvir)
                t2i = gi/lib.direct_sum('a+jb->ajb', eia[i], ejb)
                emp2_ep += numpy.einsum('ajb,ajb', t2i, gi)

            emp2_ep += 2.0 * emp2_ep
            end = timer()
            print('time for python mp2', end-start)

        return emp2_ee, emp2_ep


if __name__ == '__main__':

    mol = neo.Mole()
    mol.build(atom='''O 0 0 0; H 0 0.7 0.7; H 0 0.7 -0.7''', basis='ccpvdz')
    mf = neo.HF(mol)
    energy = mf.scf()

    emp2_ee, emp2_ep = MP2(mf).kernel()

    print('emp2_ee = ', emp2_ee)
    print('emp2_ep = ', emp2_ep)
    print('total neo-mp2 = ', energy + emp2_ee + emp2_ep)
