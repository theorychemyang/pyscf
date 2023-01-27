#!/usr/bin/env python

'''
Restricted coupled perturbed Hartree-Fock solver for
constrained nuclear-electronic orbital method
'''

import numpy
from pyscf import lib
from pyscf.lib import logger


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
    occidx_e = mo_occ_e > 0
    viridx_e = mo_occ_e == 0
    e_a_e = mo_energy_e[viridx_e]
    e_i_e = mo_energy_e[occidx_e]
    e_ai_e = 1. / lib.direct_sum('a-i->ai', e_a_e, e_i_e)
    nvir_e, nocc_e = e_ai_e.shape
    nmo_e = nocc_e + nvir_e

    e_size = nmo_e * nocc_e # size of mo1e
    total_size += e_size

    s1e = s1e.reshape(-1,nmo_e,nocc_e)
    hs = mo1base_e = h1e.reshape(-1,nmo_e,nocc_e) - s1e*e_i_e
    mo_e1_e = hs[:,occidx_e,:].copy()

    mo1base_e[:,viridx_e] *= -e_ai_e
    mo1base_e[:,occidx_e] = -s1e[:,occidx_e] * .5

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
        for i in range(len(mo1n)):
            mo1n[i] = mo1n[i].reshape(h1n[i].shape)
        if f1n is not None:
            for i in range(len(f1n)):
                f1n[i] = f1n[i].reshape(-1,3)
        ve, vn, rfn = fvind(mo1e.reshape(h1e.shape), mo1n, f1n=f1n)
        ve = ve.reshape(-1,nmo_e,nocc_e)
        ve[:,viridx_e,:] *= e_ai_e
        ve[:,occidx_e,:] = 0
        for i in range(len(vn)):
            vn[i] = vn[i].reshape(-1,nmo_n[i],nocc_n[i])
            vn[i][:,viridx_n[i],:] *= e_ai_n[i]
            vn[i][:,occidx_n[i],:] = 0
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
    if f1n is not None:
        for i in range(len(f1n)):
            f1n[i] = f1n[i].reshape(-1,3)
    log.timer('krylov solver in CNEO CPHF', *t0)
    if log.verbose >= logger.DEBUG1:
        # analyze the error
        ax = vind_vo(mo1s) + mo1s
        log.debug1(f'{numpy.linalg.norm(ax-mo1base)=}')
        ax_e, ax_n, ax_f = mo1s_disassembly(ax, total_size, e_size, n_size, with_f1n=with_f1n)
        ax_e = ax_e.reshape(-1,nmo_e,nocc_e)
        ax_e[:,viridx_e] /= -e_ai_e
        b_e = mo1base_e.copy()
        b_e = b_e.reshape(-1,nmo_e,nocc_e)
        b_e[:,viridx_e] /= -e_ai_e
        log.debug1(f'{numpy.linalg.norm(ax_e-b_e)=}')
        for i in range(len(mf_n)):
            ax_n[i] = ax_n[i].reshape(-1,nmo_n[i],nocc_n[i])
            ax_n[i][:,viridx_n[i]] /= -e_ai_n[i]
            b_n = mo1base_n[i].copy()
            b_n = b_n.reshape(-1,nmo_n[i],nocc_n[i])
            b_n[:,viridx_n[i]] /= -e_ai_n[i]
            log.debug1(f'{i=} {numpy.linalg.norm(ax_n[i]-b_n)=}')
            if ax_f is not None:
                log.debug1(f'{i=} {numpy.linalg.norm(ax_f[i])=}')

    # to feed to fvind, mo1n should be of this shape
    for i in range(len(mo1n)):
        mo1n[i] = mo1n[i].reshape(h1n[i].shape)
    #v1mo_e, v1mo_n, rf1mo_n = fvind(mo1e.reshape(h1e.shape), mo1n, f1n=f1n)
    v1mo_e, _, _ = fvind(mo1e.reshape(h1e.shape), mo1n, f1n=f1n)
    v1mo_e = v1mo_e.reshape(-1,nmo_e,nocc_e)
    # TODO: for electronic only CPHF, they have (1+A)x=b -> x=b-Ax to
    # increase accuracy. This is a bit more complicated for CNEO CPHF.
    # mo1e[:,viridx_e] = ?
    # mo1n[i][:,viridx_n[i]] = ?
    # f1n[i] = ?
    # get back to the correct shape for following operations
    for i in range(len(mo1n)):
        mo1n[i] = mo1n[i].reshape(mo1base_n[i].shape)

    # mo_e1 has the same symmetry as the first order Fock matrix (hermitian or
    # anti-hermitian). mo_e1 = v1mo - s1*lib.direct_sum('i+j->ij',e_i,e_i)
    mo_e1_e += mo1e[:,occidx_e] * lib.direct_sum('i-j->ij', e_i_e, e_i_e)
    mo_e1_e += v1mo_e[:,occidx_e,:]

    if h1e.ndim == 2:
        # get rid of the redundant shape[0]=1 if h1e.ndim == 2:
        for i in range(len(mo1n)):
            mo1n[i] = mo1n[i].reshape(h1n[i].shape)
        if f1n is not None:
            for i in range(len(f1n)):
                f1n[i] = f1n[i].reshape(3)
        return mo1e.reshape(h1e.shape), mo_e1_e.reshape(nocc_e,nocc_e), \
               mo1n, f1n
    else:
        return mo1e, mo_e1_e, mo1n, f1n

def mo1s_disassembly(mo1s, total_size, e_size, n_size, with_f1n=False):
    '''Transfer mo1 from a 1-d array to 1-d arrays containing data of
    mo1e and mo1n. Also f1n, if requested'''
    if with_f1n:
        total_size += 3 * len(n_size)
    assert mo1s.size % total_size == 0
    nset = mo1s.size // total_size

    mo1s = mo1s.ravel()
    n_e = nset * e_size
    index = n_e
    mo1e = mo1s[:index]

    mo1n = []
    for n in n_size:
        add = nset * n
        mo1n.append(mo1s[index:index+add])
        index += add

    f1n = None
    if with_f1n:
        f1n = []
        for n in n_size:
            add = nset * 3
            f1n.append(mo1s[index:index+add])
            index += add

    assert index == mo1s.size
    return mo1e, mo1n, f1n
