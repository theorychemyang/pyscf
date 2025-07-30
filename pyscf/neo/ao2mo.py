#!/usr/bin/env python
# Author: Kurt Brorsen (brorsenk@missouri.edu)

import numpy
import pyscf.ao2mo as ao2mo
from timeit import default_timer as timer

def ep_setup(mf, i=0, j=0, ep=False):

    if not ep:
        mol_tot = mf.mol.elec + mf.mol.nuc[i]
        tot1  = mf.mf_elec.mo_coeff[0,:].shape[0]
        tot2  = mf.mf_nuc[i].mo_coeff[0,:].shape[0]
    else:
        mol_tot = mf.mol.nuc[i] + mf.mol.nuc[j]
        tot1  = mf.mf_nuc[i].mo_coeff[0,:].shape[0]
        tot2  = mf.mf_nuc[j].mo_coeff[0,:].shape[0]

    eri = mol_tot.intor('int2e',aosym='s8')

    mo_coeff_tot = numpy.zeros((tot1+tot2,tot1+tot2))

    if not ep:
        mo_coeff_tot[:tot1,:tot1] = mf.mf_elec.mo_coeff
        mo_coeff_tot[tot1:,tot1:] = mf.mf_nuc[i].mo_coeff
    else:
        mo_coeff_tot[:tot1,:tot1] = mf.mf_nuc[i].mo_coeff
        mo_coeff_tot[tot1:,tot1:] = mf.mf_nuc[j].mo_coeff

    return eri, mo_coeff_tot

def ep_full(mf, i=0):

    eri, mo_coeff_tot = ep_setup(mf,i)

    e_tot  = mf.mf_elec.mo_coeff[0,:].shape[0]
    #p_tot  = mf.mf_nuc[0].mo_coeff[0,:].shape[0]

    c_e= mo_coeff_tot[:,:e_tot]
    c_n= mo_coeff_tot[:,e_tot:]

    eri_ep = ao2mo.incore.general(eri,(c_e,c_e,c_n,c_n),compact=False)

    return eri_ep

def ep_ovov(mf, i=0):

    eri, mo_coeff_tot = ep_setup(mf,i)

    e_nocc = mf.mf_elec.mo_coeff[:,mf.mf_elec.mo_occ>0].shape[1]
    e_tot  = mf.mf_elec.mo_coeff[0,:].shape[0]
    #e_nvir = e_tot - e_nocc

    p_nocc = mf.mf_nuc[0].mo_coeff[:,mf.mf_nuc[0].mo_occ>0].shape[1]
    #p_tot  = mf.mf_nuc[0].mo_coeff[0,:].shape[0]
    #p_nvir = p_tot - p_nocc

    co_e= mo_coeff_tot[:,:e_nocc]
    cv_e= mo_coeff_tot[:,e_nocc:e_tot]

    co_n=mo_coeff_tot[:,e_tot:e_tot+p_nocc]
    cv_n=mo_coeff_tot[:,e_tot+p_nocc:]

    start = timer()
    eri_ep = ao2mo.incore.general(eri,(co_e,cv_e,co_n,cv_n),compact=False)
    finish = timer()

    print('real time for ovov integral transformation = ',finish-start)

    return eri_ep
