#!/usr/bin/env python

'''
Restricted coupled perturbed Hartree-Fock solver for
constrained nuclear-electronic orbital method
'''

import numpy
from pyscf import lib
from pyscf.lib import logger


def solve(fvind, mo_energy, mo_occ, h1, s1=None, with_f1=False,
          max_cycle=100, tol=1e-9, hermi=False, verbose=logger.WARN,
          level_shift=0):
    '''
    Args:
        fvind : function
            Given density matrix, compute response function density matrix
            products. If requested, also the position constraint part.
        with_f1 : boolean
            If True, triggers CNEO.
    '''
    if s1 is None:
        return solve_nos1(fvind, mo_energy, mo_occ, h1, with_f1,
                          max_cycle, tol, hermi, verbose, level_shift)
    else:
        return solve_withs1(fvind, mo_energy, mo_occ, h1, s1, with_f1,
                            max_cycle, tol, hermi, verbose, level_shift)
kernel = solve

# h1 shape is (:,nvir,nocc)
def solve_nos1(fvind, mo_energy, mo_occ, h1, with_f1=False,
               max_cycle=100, tol=1e-9, hermi=False, verbose=logger.WARN,
               level_shift=0):
    '''For field independent basis. First order overlap matrix is zero'''
    raise NotImplementedError('(C)NEO-CP equation without s1 is not implemented yet.')

# h1 shape is (:,nocc+nvir,nocc)
def solve_withs1(fvind, mo_energy, mo_occ, h1, s1, with_f1=False,
                 max_cycle=100, tol=1e-9, hermi=False, verbose=logger.WARN,
                 level_shift=0):
    '''For field dependent basis. First order overlap matrix is non-zero.
    The first order orbitals are set to
    C^1_{ij} = -1/2 S1
    e1 = h1 - s1*e0 + (e0_j-e0_i)*c1 + vhf[c1]

    Returns:
        First order orbital coefficients (in MO basis) and first order orbital
        energy matrix. If requested, also first order nuclear position Lagrange
        multipliers for CNEO.
    '''
    assert not hermi
    log = logger.new_logger(verbose=verbose)
    t0 = (logger.process_clock(), logger.perf_counter())

    occidx = {}
    viridx = {}
    e_i = {}
    e_a = {}
    e_ai = {}
    nocc = {}
    nmo = {}
    hs = {}
    scale = {}
    mo1base = []
    is_component_unrestricted = {}
    nov = {}
    total_mo1 = 0
    total_f1 = 0
    sorted_keys = sorted(mo_occ.keys())
    for t in sorted_keys:
        mo_occ[t] = numpy.asarray(mo_occ[t])
        if mo_occ[t].ndim > 1: # unrestricted
            assert not t.startswith('n')
            assert mo_occ[t].shape[0] == 2
            is_component_unrestricted[t] = True
            occidxa = mo_occ[t][0] > 0
            occidxb = mo_occ[t][1] > 0
            occidx[t] = (occidxa, occidxb)
            viridxa = ~occidxa
            viridxb = ~occidxb
            viridx[t] = (viridxa, viridxb)
            nocca = numpy.count_nonzero(occidxa)
            noccb = numpy.count_nonzero(occidxb)
            nocc[t] = (nocca, noccb)
            nmoa = mo_occ[t][0].size
            nmob = mo_occ[t][1].size
            nmo[t] = (nmoa, nmob)
            mo_energy[t] = numpy.asarray(mo_energy[t])
            assert mo_energy[t].ndim > 1 and mo_energy[t].shape[0] == 2
            ei_a = mo_energy[t][0][occidxa]
            ei_b = mo_energy[t][1][occidxb]
            e_i[t] = (ei_a, ei_b)
            ea_a = mo_energy[t][0][viridxa]
            ea_b = mo_energy[t][1][viridxb]
            e_a[t] = (ea_a, ea_b)
            eai_a = 1 / (ea_a[:,None] + level_shift - ei_a)
            eai_b = 1 / (ea_b[:,None] + level_shift - ei_b)
            e_ai[t] = (eai_a, eai_b)
            s1_a = s1[t][0].reshape(-1,nmoa,nocca)
            nset = s1_a.shape[0]
            s1_b = s1[t][1].reshape(nset,nmob,noccb)
            hs_a = h1[t][0].reshape(nset,nmoa,nocca) - s1_a * ei_a
            hs_b = h1[t][1].reshape(nset,nmob,noccb) - s1_b * ei_b
            hs[t] = [hs_a, hs_b]
            mo1base_a = hs_a.copy()
            mo1base_b = hs_b.copy()
            mo1base_a[:,viridxa] *= -eai_a
            mo1base_b[:,viridxb] *= -eai_b
            mo1base_a[:,occidxa] = -s1_a[:,occidxa] * .5
            mo1base_b[:,occidxb] = -s1_b[:,occidxb] * .5
            mo1base.append(mo1base_a.reshape(nset,-1))
            mo1base.append(mo1base_b.reshape(nset,-1))
            nov[t] = nocca * nmoa + noccb * nmob
        else: # restricted
            is_component_unrestricted[t] = False
            occidx[t] = mo_occ[t] > 0
            viridx[t] = mo_occ[t] == 0
            e_a[t] = mo_energy[t][viridx[t]]
            e_i[t] = mo_energy[t][occidx[t]]
            e_ai[t] = 1 / (e_a[t][:,None] + level_shift - e_i[t])
            nvir, nocc[t] = e_ai[t].shape
            nmo[t] = nocc[t] + nvir
            if with_f1 and t.startswith('n'):
                # In theory, this scale factor should be 1/(LUMO-HOMO) to be
                # consistent with NEO-CPHF blocks, but in practical tests it
                # can lead to nonconvergence. In the test for a molecule with
                # 24 atoms (9 of which are hydrogen), I get the following results:
                #  | scale | #cycles | NEO-CPHF error | position error |
                #  |  1.0  |    44   |     2.9e-10    |    1.4e-10     |
                #  |  1.6  |    49   |     1.7e-12    |    5.8e-13     |
                #  |  1.8  |    51   |     1.9e-13    |    1.2e-13     |
                #  |  2.0  |    54   |     6.9e-14    |    6.2e-14     |
                #  |  2.2  |    55   |     4.5e-14    |    8.9e-15     |
                #  |  2.4  |    57   |     5.5e-14    |    8.9e-16     |
                #  |  2.6  |    60   |     5.0e-14    |    4.4e-16     |
                #  |  2.8  |    63   |     5.0e-14    |    5.8e-16     |
                #  |  5.0  |    85   |     7.7e-14    |    1.2e-15     |
                #  | 10.0  |   138   |     4.2e-14    |    1.2e-15     |
                # Roughly 2.0 can be chosen to balance the accuracy and efficiency
                # However, this does not work well for heavier quantum nuclei, e.g.,
                # full quantum H-F molecule in the test.
                # Previously the empirical scale factor was chosen as 2 * charge,
                # but now this function does not have access to the charge, so using
                # a slightly weird empirical formula:
                # Instead of
                #scale[t] = 1. / (e_a[t][0] + level_shift - e_i[t][-1])
                # use
                scale[t] = (e_a[t][0] + level_shift - e_i[t][-1]) * 100.
                # This will give roughly 2.0 for hydrogen. For heavier nuclei, this
                # will give a large number. I guess it is likely to fail the krylov
                # solver again if someone calculates with multiple heavy quantum nuclei,
                # but this is the best compromise I can find now (works for multiple
                # hydrogen and doesn't fail the one heavy nucleus full quantum test).
                # Here I collect the data for full quantum H-F, H2O and CH3:
                # | molecule | scale for H | scale for heavy | NEO-CPHF error | position error |
                # |    HF    |     45.9    |      1.53       |     3.7e-7     |    3.5e-8      |
                # |    HF    |     2.18    |      65.2       |     4.5e-6     |    1.7e-6      |
                # |    H2O   |     44.3    |      1.80       |     8.8e-14    |    1.7e-15     |
                # |    H2O   |     2.26    |      55.6       |     1.3e-5     |    4.0e-6      |
                # |    CH3   |     41.5    |      2.84       |     6.3e-14    |    9.6e-15     |
                # |    CH3   |     2.41    |      35.2       |     2.0e-5     |    1.4e-5      |
                # The error is much larger, but at least the frequency test does not fail.
                total_f1 += 3

            s1[t] = s1[t].reshape(-1,nmo[t],nocc[t])
            hs[t] = h1[t].reshape(-1,nmo[t],nocc[t]) - s1[t]*e_i[t]

            mo1base_a = hs[t].copy()
            mo1base_a[:,viridx[t]] *= -e_ai[t]
            mo1base_a[:,occidx[t]] = -s1[t][:,occidx[t]] * .5
            mo1base.append(mo1base_a.reshape(-1,nmo[t]*nocc[t]))
            nov[t] = nmo[t] * nocc[t]
        total_mo1 += nov[t]
    if with_f1:
        nset = mo1base[0].shape[0]
        for t in sorted_keys:
            if t.startswith('n'):
                mo1base.append(numpy.zeros((nset,3)))
    mo1base = numpy.hstack(mo1base)

    def vind_vo(mo1_and_f1):
        mo1_and_f1 = mo1_and_f1.reshape(-1,total_mo1+total_f1)
        mo1_array = mo1_and_f1[:,:total_mo1]
        mo1 = {}
        offset = 0
        for t in sorted_keys:
            mo1[t] = mo1_array[:,offset:offset+nov[t]]
            if not is_component_unrestricted[t]:
                mo1[t] = mo1[t].reshape(-1, nmo[t], nocc[t])
            offset += nov[t]
        f1 = None
        if with_f1:
            f1_array = mo1_and_f1[:,total_mo1:]
            f1 = {}
            offset = 0
            for t in sorted_keys:
                if t.startswith('n'):
                    f1[t] = f1_array[:,offset:offset+3]
                    offset += 3
        v, r = fvind(mo1, f1=f1)
        for t in v:
            if level_shift != 0:
                v[t] -= mo1[t] * level_shift
            if is_component_unrestricted[t]:
                nmoa, nmob = nmo[t]
                nocca, noccb = nocc[t]
                eai_a, eai_b = e_ai[t]
                occidxa, occidxb = occidx[t]
                viridxa, viridxb = viridx[t]
                v1a = v[t][:,:nmoa*nocca].reshape(-1,nmoa,nocca)
                v1b = v[t][:,nmoa*nocca:].reshape(-1,nmob,noccb)
                v1a[:,viridxa] *= eai_a
                v1b[:,viridxb] *= eai_b
                v1a[:,occidxa] = 0
                v1b[:,occidxb] = 0
            else:
                v[t] = v[t].reshape(-1, nmo[t], nocc[t])
                v[t][:,viridx[t],:] *= e_ai[t]
                v[t][:,occidx[t],:] = 0
            v[t] = v[t].reshape(-1, nov[t])
        if with_f1 and r is not None:
            for t in r:
                # NOTE: this scale factor is somewhat empirical. The goal is
                # to try to bring the position constraint equation r * mo1 = 0
                # to be of a similar magnitude as compared to the conventional
                # CPHF equations.
                r[t] = r[t] * scale[t] - f1[t]
                # NOTE: f1 got subtracted because krylov solver solves (1+a)x=b
            return numpy.hstack([v[k] for k in sorted_keys]
                                + [r[k] for k in sorted_keys if k in r]).ravel()
        return numpy.hstack([v[k] for k in sorted_keys]).ravel()
    # TODO: remove ravel. See https://github.com/pyscf/pyscf/issues/2702

    mo1_and_f1 = lib.krylov(vind_vo, mo1base.ravel(),
                            tol=tol, max_cycle=max_cycle, hermi=hermi, verbose=log)
    mo1_and_f1 = mo1_and_f1.reshape(-1,total_mo1+total_f1)
    mo1_array = mo1_and_f1[:,:total_mo1]
    mo1 = {}
    offset = 0
    for t in sorted_keys:
        mo1[t] = mo1_array[:,offset:offset+nov[t]]
        offset += nov[t]
        if is_component_unrestricted[t]:
            mo1[t] = mo1[t].reshape(-1, nov[t])
            nset = mo1[t].shape[0]
            nmoa, nmob = nmo[t]
            nocca, noccb = nocc[t]
            occidxa, occidxb = occidx[t]
            mo1_a = mo1[t][:,:nmoa*nocca].reshape(nset,nmoa,nocca)
            mo1_b = mo1[t][:,nmoa*nocca:].reshape(nset,nmob,noccb)
            mo1_a[:,occidxa] = -s1[t][0][:,occidxa] * .5
            mo1_b[:,occidxb] = -s1[t][1][:,occidxb] * .5
        else:
            mo1[t] = mo1[t].reshape(-1, nmo[t], nocc[t])
            mo1[t][:,occidx[t]] = -s1[t][:,occidx[t]] * .5
    f1 = None
    if with_f1:
        f1_array = mo1_and_f1[:,total_mo1:]
        f1 = {}
        offset = 0
        for t in sorted_keys:
            if t.startswith('n'):
                f1[t] = f1_array[:,offset:offset+3]
                offset += 3
    log.timer('krylov solver in CNEO CPHF', *t0)

    mo_e1 = {}
    v1mo, r1mo = fvind(mo1, f1=f1)
    for t in sorted_keys:
        if is_component_unrestricted[t]:
            nmoa, nmob = nmo[t]
            nocca, noccb = nocc[t]
            occidxa, occidxb = occidx[t]
            hs_a, hs_b = hs[t]
            hs_a += v1mo[t][:,:nmoa*nocca].reshape(-1,nmoa,nocca)
            hs_b += v1mo[t][:,nmoa*nocca:].reshape(-1,nmob,noccb)
            mo1_a = mo1[t][:,:nmoa*nocca].reshape(-1,nmoa,nocca)
            mo1_b = mo1[t][:,nmoa*nocca:].reshape(-1,nmob,noccb)
            ei_a, ei_b = e_i[t]
            # NOTE: the refinement step x=b-Ax does not seem to work well for (C)NEO-CPHF
            #ea_a, ea_b = e_a[t]
            #viridxa, viridxb = viridx[t]
            #mo1_a[:,viridxa] = hs_a[:,viridxa] / (ei_a - ea_a[:,None])
            #mo1_b[:,viridxb] = hs_b[:,viridxb] / (ei_b - ea_b[:,None])
            mo_e1_a = hs_a[:,occidxa]
            mo_e1_b = hs_b[:,occidxb]
            mo_e1_a += mo1_a[:,occidxa] * (ei_a[:,None] - ei_a)
            mo_e1_b += mo1_b[:,occidxb] * (ei_b[:,None] - ei_b)
            if isinstance(h1[t][0], numpy.ndarray) and h1[t][0].ndim == 2:
                mo1_a, mo1_b = mo1_a[0], mo1_b[0]
                mo_e1_a, mo_e1_b = mo_e1_a[0], mo_e1_b[0]
            else:
                assert h1[t][0].ndim == 3
            mo1[t] = (mo1_a, mo1_b)
            mo_e1[t] = (mo_e1_a, mo_e1_b)
        else:
            hs[t] += v1mo[t].reshape(-1, nmo[t], nocc[t])
            # NOTE: the refinement step x=b-Ax does not seem to work well for (C)NEO-CPHF
            #mo1[t][:,viridx[t]] = hs[t][:,viridx[t]] / (e_i[t] - e_a[t][:,None])
            #if f1 is not None and t in f1:
            #    f1[t] -= r1mo[t] * scale[t]
            #     DEBUG: Verify nuclear r * mo1
            #    print(f'[DEBUG] norm(r * mo1) for {t}: {numpy.linalg.norm(r1mo[t])}')

            mo_e1[t] = hs[t][:,occidx[t],:]
            mo_e1[t] += mo1[t][:,occidx[t]] * (e_i[t][:,None] - e_i[t])

            if isinstance(h1[t], numpy.ndarray) and h1[t].ndim == 2:
                mo1[t] = mo1[t][0]
                mo_e1[t] = mo_e1[t][0]
                if f1 is not None and t in f1:
                    f1[t] = f1[t][0]
            else:
                assert h1[t].ndim == 3

    # DEBUG: verify the solution
    if log.verbose >= logger.DEBUG1:
        x = []
        for t in sorted_keys:
            if is_component_unrestricted[t]:
                x.append(mo1[t][0].reshape(-1,nmo[t][0]*nocc[t][0]))
                x.append(mo1[t][1].reshape(-1,nmo[t][1]*nocc[t][1]))
            else:
                x.append(mo1[t].reshape(-1,nov[t]))
        if with_f1:
            for t in sorted_keys:
                if t.startswith('n'):
                    x.append(f1[t])
        x = numpy.hstack(x)
        ax = vind_vo(x).reshape(x.shape) + x
        log.debug1(f'[DEBUG] CPHF error: {numpy.abs(ax - mo1base).max()}')
        if mo1base.shape[1] > total_mo1:
            log.debug1(f'[DEBUG] error for NEO part:   {numpy.abs(ax - mo1base)[:,:total_mo1].max()}')
            log.debug1(f'[DEBUG] error for constraint: {numpy.abs(ax - mo1base)[:,total_mo1:].max()}')

    return mo1, mo_e1, f1
