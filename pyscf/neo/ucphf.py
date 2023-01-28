#!/usr/bin/env python

'''
Unrestricted coupled perturbed Hartree-Fock solver for
constrained nuclear-electronic orbital method
'''

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.neo.cphf import mo1s_disassembly


def solve(fvind, mf_e, mf_n, h1e, h1n, s1e=None, with_f1n=False,
          max_cycle=30, tol=1e-9, hermi=False, verbose=logger.WARN):
    '''
    Args:
        fvind : function
            Given density matrix, compute response function density matrix
            products. If requested, also the position constraint part.
        mf_e : SCF class
            Electronic wave function.
        mf_n : list
            A list of nuclear wave functions.
        h1e : ndarray
            MO space electronic H1.
        h1n : list
            A list of MO space nuclear H1.
        s1e : ndaray
            MO space electronic S1.
        with_f1n : boolean
            If True, triggers CNEO.

    Kwargs:
        hermi : boolean
            Whether the matrix defined by fvind is Hermitian or not.
    '''
    if s1e is None:
        return solve_nos1(fvind, mf_e, mf_n, h1e, h1n, with_f1n=with_f1n,
                          max_cycle=max_cycle, tol=tol, hermi=hermi,
                          verbose=verbose)
    else:
        return solve_withs1(fvind, mf_e, mf_n, h1e, h1n, s1e, with_f1n=with_f1n,
                            max_cycle=max_cycle, tol=tol, hermi=hermi,
                            verbose=verbose)
kernel = solve

# h1 shape is (:,nvir,nocc)
def solve_nos1(fvind, mf_e, mf_n, h1e, h1n, with_f1n=False,
               max_cycle=30, tol=1e-9, hermi=False, verbose=logger.WARN):
    '''For field independent basis. First order overlap matrix is zero'''
    raise NotImplementedError('CP equation without s1 is not implemented yet.')

# h1 shape is (:,nocc+nvir,nocc)
def solve_withs1(fvind, mf_e, mf_n, h1e, h1n, s1e, with_f1n=False,
                 max_cycle=30, tol=1e-9, hermi=False, verbose=logger.WARN):
    '''For field dependent basis. First order overlap matrix is non-zero.

    Kwargs:
        hermi : boolean
            Whether the matrix defined by fvind is Hermitian or not.

    Returns:
        First order electronic orbital coefficients (in MO basis), first order
        elecronic orbital energy matrix and first order nuclear orbital
        coefficients. If requested, also first order nuclear position Lagrange
        multipliers for CNEO.
    '''
    log = logger.new_logger(verbose=verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    mol = mf_e.mol.super_mol
    total_size = 0 # size of mo1

    mo_energy_e = mf_e.mo_energy
    mo_occ_e = mf_e.mo_occ
    occidx_e_a = mo_occ_e[0] > 0
    occidx_e_b = mo_occ_e[1] > 0
    viridx_e_a = ~occidx_e_a
    viridx_e_b = ~occidx_e_b
    e_a_e_a = mo_energy_e[0][viridx_e_a]
    e_a_e_b = mo_energy_e[1][viridx_e_b]
    e_i_e_a = mo_energy_e[0][occidx_e_a]
    e_i_e_b = mo_energy_e[1][occidx_e_b]
    e_ai_e_a = 1. / lib.direct_sum('a-i->ai', e_a_e_a, e_i_e_a)
    e_ai_e_b = 1. / lib.direct_sum('a-i->ai', e_a_e_b, e_i_e_b)
    nvir_e_a, nocc_e_a = e_ai_e_a.shape
    nvir_e_b, nocc_e_b = e_ai_e_b.shape
    nmo_e_a = nocc_e_a + nvir_e_a
    nmo_e_b = nocc_e_b + nvir_e_b

    e_size = nmo_e_a * nocc_e_a + nmo_e_b * nocc_e_b # size of mo1e
    total_size += e_size

    s1e_a = s1e[0].reshape(-1,nmo_e_a,nocc_e_a)
    s1e_b = s1e[1].reshape(-1,nmo_e_b,nocc_e_b)
    hs_a = mo1base_e_a = h1e[0].reshape(-1,nmo_e_a,nocc_e_a) - s1e_a*e_i_e_a
    hs_b = mo1base_e_b = h1e[1].reshape(-1,nmo_e_b,nocc_e_b) - s1e_b*e_i_e_b
    mo_e1_e_a = hs_a[:,occidx_e_a].copy()
    mo_e1_e_b = hs_b[:,occidx_e_b].copy()

    mo1base_e_a[:,viridx_e_a] *= -e_ai_e_a
    mo1base_e_b[:,viridx_e_b] *= -e_ai_e_b
    mo1base_e_a[:,occidx_e_a] = -s1e_a[:,occidx_e_a] * .5
    mo1base_e_b[:,occidx_e_b] = -s1e_b[:,occidx_e_b] * .5
    nset = s1e_a.shape[0]
    mo1base_e = numpy.hstack((mo1base_e_a.reshape(nset,-1), mo1base_e_b.reshape(nset,-1)))

    occidx_n = []
    viridx_n = []
    e_a_n = []
    e_i_n = []
    e_ai_n = []
    nvir_n = []
    nocc_n = []
    nmo_n = []
    mo1base_n = []
    n_size = []
    for i in range(len(mf_n)):
        occidx_n.append(mf_n[i].mo_occ > 0)
        viridx_n.append(mf_n[i].mo_occ == 0)
        e_a_n.append(mf_n[i].mo_energy[viridx_n[-1]])
        e_i_n.append(mf_n[i].mo_energy[occidx_n[-1]])
        e_ai_n.append(1. / lib.direct_sum('a-i->ai', e_a_n[-1], e_i_n[-1]))
        tmp1, tmp2 = e_ai_n[-1].shape
        nvir_n.append(tmp1)
        nocc_n.append(tmp2)
        nmo_n.append(tmp1 + tmp2)
        n_size.append(nmo_n[-1] * nocc_n[-1])
        total_size += n_size[-1]
        mo1base_n.append(h1n[i].reshape(-1,nmo_n[-1],nocc_n[-1]))
        mo1base_n[-1][:,viridx_n[-1]] *= -e_ai_n[-1]
        mo1base_n[-1][:,occidx_n[-1]] = 0

    def vind_vo(mo1s):
        mo1e, mo1n, f1n = mo1s_disassembly(mo1s, total_size, e_size, n_size,
                                           with_f1n=with_f1n)
        if f1n is not None:
            for i in range(len(f1n)):
                f1n[i] = f1n[i].reshape(-1,3)
        ve, vn, rfn = fvind(mo1e, mo1n, f1n=f1n)
        ve = ve.reshape(mo1base_e.shape)
        v1a = ve[:,:nmo_e_a*nocc_e_a].reshape(nset,nmo_e_a,nocc_e_a)
        v1b = ve[:,nmo_e_a*nocc_e_a:].reshape(nset,nmo_e_b,nocc_e_b)
        v1a[:,viridx_e_a] *= e_ai_e_a
        v1b[:,viridx_e_b] *= e_ai_e_b
        v1a[:,occidx_e_a] = 0
        v1b[:,occidx_e_b] = 0
        for i in range(len(vn)):
            vn[i] = vn[i].reshape(-1,nmo_n[i],nocc_n[i])
            vn[i][:,viridx_n[i]] *= e_ai_n[i]
            vn[i][:,occidx_n[i]] = 0
        if rfn is not None:
            for i in range(len(f1n)):
                ia = mf_n[i].mol.atom_index
                charge = mol.atom_charge(ia)
                # NOTE: this 2*charge factor is purely empirical, because equation
                # r * mo1 = 0 is insensitive to the factor mathematically, but
                # the factor will change the numerical solution
                # TODO: find the best factor
                rfn[i] = rfn[i] * 2.0 * charge - f1n[i]
                # note that f got subtracted because krylov solver solves (1+a)x=b
        if with_f1n:
            return numpy.concatenate((ve,
                                      numpy.concatenate(vn, axis=None),
                                      numpy.concatenate(rfn, axis=None)),
                                     axis=None)
        else:
            return numpy.concatenate((ve,
                                      numpy.concatenate(vn, axis=None)),
                                     axis=None)
    if with_f1n:
        mo1base = numpy.concatenate((mo1base_e,
                                     numpy.concatenate(mo1base_n, axis=None),
                                     numpy.zeros(3*len(mf_n)*len(mo1base_e))),
                                    axis=None)
    else:
        mo1base = numpy.concatenate((mo1base_e,
                                     numpy.concatenate(mo1base_n, axis=None)),
                                    axis=None)
    mo1s = lib.krylov(vind_vo, mo1base,
                      tol=tol, max_cycle=max_cycle, hermi=hermi, verbose=log)
    mo1e, mo1n, f1n = mo1s_disassembly(mo1s, total_size, e_size, n_size,
                                       with_f1n=with_f1n)
    mo1e = mo1e.reshape(mo1base_e.shape)
    for i in range(len(mo1n)):
        mo1n[i] = mo1n[i].reshape(mo1base_n[i].shape)
    if f1n is not None:
        for i in range(len(f1n)):
            f1n[i] = f1n[i].reshape(-1,3)
    log.timer('krylov solver in CNEO CPHF', *t0)

    v1mo_e, _, _ = fvind(mo1e, mo1n, f1n=f1n)
    v1a = v1mo_e[:,:nmo_e_a*nocc_e_a].reshape(nset,nmo_e_a,nocc_e_a)
    v1b = v1mo_e[:,nmo_e_a*nocc_e_a:].reshape(nset,nmo_e_b,nocc_e_b)
    mo1e_a = mo1e[:,:nmo_e_a*nocc_e_a].reshape(nset,nmo_e_a,nocc_e_a)
    mo1e_b = mo1e[:,nmo_e_a*nocc_e_a:].reshape(nset,nmo_e_b,nocc_e_b)
    # TODO: for electronic only CPHF, they have (1+A)x=b -> x=b-Ax to
    # increase accuracy. This is a bit more complicated for CNEO CPHF.

    mo_e1_e_a += mo1e_a[:,occidx_e_a] * lib.direct_sum('i-j->ij', e_i_e_a, e_i_e_a)
    mo_e1_e_b += mo1e_b[:,occidx_e_b] * lib.direct_sum('i-j->ij', e_i_e_b, e_i_e_b)
    mo_e1_e_a += v1a[:,occidx_e_a]
    mo_e1_e_b += v1b[:,occidx_e_b]

    if isinstance(h1e[0], numpy.ndarray) and h1e[0].ndim == 2:
        # get rid of the redundant shape[0]=1 if h1e[0].ndim == 2:
        mo1e_a, mo1e_b = mo1e_a[0], mo1e_b[0]
        mo_e1_e_a, mo_e1_e_b = mo_e1_e_a[0], mo_e1_e_b[0]
        for i in range(len(mo1n)):
            mo1n[i] = mo1n[i].reshape(h1n[i].shape)
        if f1n is not None:
            for i in range(len(f1n)):
                f1n[i] = f1n[i].reshape(3)
    return (mo1e_a, mo1e_b), (mo_e1_e_a, mo_e1_e_b), mo1n, f1n
