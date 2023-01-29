#!/usr/bin/env python

'''
(C)NEO response functions
'''

from pyscf import lib, neo

# import _response_functions to load gen_response methods in SCF class
from pyscf.scf import _response_functions  # noqa


def _gen_neo_response(mf, hermi=0, max_memory=None):
    '''Generate a function to compute the product of (C)NEO response function
    and electronic/nuclear density matrices.
    '''
    assert isinstance(mf, neo.HF)
    mol = mf.mol
    if max_memory is None:
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory*.8-mem_now)

    def vind(dm1e_symm, dm1e_partial, dm1n_partial):
        '''
        Input:
        dm1e_symm can be of shape (...,nao,nao) for RHF/RKS or
        (2,...,nao,nao) for UHF/UKS.
        dm1e_partial is of shape (...,nao,nao) regardless of RKS/UKS, because
        it is total density instead of spin densities.
        dm1n_partial must be a list, even if there is only one quantum nucleus.
        Each element is of shape (...,nao,nao)

        The reason we have two dm1e's (symm and partial) is that, for e-e
        response, the product is symmetric, but for e-n (and n-n), it is not.

        Output:
        v1e is of the same shape as dm1e_*, for e-e and e-n parts.
        v1n is a list and of the same shape as dm1n_partial, for n-e and n-n parts.
        Note that quantum nuclei do not have self-interaction so there should not
        be a symm part for n-n, nor does it require dm1n_symm.
        '''

        # electron-electron response
        vresp_e = mf.mf_elec.gen_response(hermi=hermi, max_memory=max_memory)
        v1e = vresp_e(dm1e_symm)

        # effect of nuclear density matrix change on electronic Fock
        # this effect is insensitive to electronic spin
        if dm1e_symm.size > dm1e_partial.size:
            # this means we have unrestricted dm1e_symm,
            # and v1e has two components
            for i in range(mol.nuc_num):
                v1en = mf.get_j_e_dm_n(i, dm1n_partial[i]) * 2.0
                v1e[0] += v1en
                v1e[1] += v1en
        else:
            for i in range(mol.nuc_num):
                v1e += mf.get_j_e_dm_n(i, dm1n_partial[i]) * 2.0

        v1n = [None] * mol.nuc_num
        for i in range(mol.nuc_num):
            # effect of electronic density matrix change on nuclear Fock
            # this effect is insensitive to electronic spin, because
            # dm1e_partial is already summed over spin (total density)
            # note that here 2.0 is still used instead of 4.0 even if it is RKS,
            # because if RHF/RKS is used, dm1e_partial will be doubled before
            # it is passed to this function
            v1n[i] = mf.get_j_n_dm_e(i, dm1e_partial) * 2.0
            # effect of other nuclear density matrix change on nuclear Fock
            for j in range(mol.nuc_num):
                if j != i:
                    v1n[i] += mf.get_j_nn(i, j, dm1n_partial[j]) * 2.0
        return v1e, v1n

    return vind

neo.hf.HF.gen_response = _gen_neo_response
