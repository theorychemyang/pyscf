#!/usr/bin/env python

'''
(C)NEO response functions
'''

from pyscf import lib, neo

# import _response_functions to load gen_response methods in SCF class
from pyscf.scf import _response_functions  # noqa


def _gen_neo_response(mf, mo_coeff=None, mo_occ=None, hermi=0, max_memory=None, no_epc=False):
    '''Generate a function to compute the product of (C)NEO response function
    and electronic/nuclear density matrices.
    '''
    assert isinstance(mf, neo.HF)
    if isinstance(mf, neo.KS):
        # TODO: Integrate EPC response calculation into _gen_neo_response
        if mf.epc is not None and not no_epc:
            raise NotImplementedError('Response with EPC kernel is not available.')
    if mo_coeff is None: mo_coeff = mf.mo_coeff
    if mo_occ is None: mo_occ = mf.mo_occ

    if max_memory is None:
        mem_now = lib.current_memory()[0]
        max_memory = max(2000, mf.max_memory*.8-mem_now)

    def vind(dm1):
        '''
        Args:
            dm1: dictionary of density matrices, same format as make_rdm1
                For 'e', can be RHF (...,nao,nao) or UHF (2,...,nao,nao)
                For 'nX', RHF (...,nao,nao)

        Returns:
            dictionary of potentials, same format as input dm1
        '''
        v1 = {}
        # intra-component response (except quantum nuclei)
        for t in mf.components.keys():
            if not t.startswith('n'):
                vresp = mf.components[t].gen_response(hermi=hermi,
                                                      max_memory=max_memory)
                v1[t] = vresp(dm1[t])

        # Process each interaction
        for (t1, t2), interaction in mf.interactions.items():
            if isinstance(mf, neo.KS):
                vint = interaction.get_vint(dm1, no_epc=no_epc)
            else:
                vint = interaction.get_vint(dm1)
            v1[t1] = v1.get(t1, 0) + vint[t1]
            v1[t2] = v1.get(t2, 0) + vint[t2]

        return v1

    return vind

neo.hf.HF.gen_response = _gen_neo_response
