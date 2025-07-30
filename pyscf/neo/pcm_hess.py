import numpy
from pyscf import lib, gto, df, scf
from pyscf.lib import logger
from pyscf.solvent.hessian import pcm as pcm_hess
from pyscf.solvent.pcm import PI
from pyscf.neo import pcm_grad
from pyscf.neo.pcm import _get_charge_from_mol_comp

def get_dvgrids(pcmobj, dm, atmlst):
    assert pcmobj._intermediates is not None

    mol = pcmobj.mol
    gridslice    = pcmobj.surface['gslice_by_atom']
    charge_exp   = pcmobj.surface['charge_exp']
    grid_coords  = pcmobj.surface['grid_coords']
    ngrids = grid_coords.shape[0]

    atom_coords = mol.atom_coords(unit='B')
    atom_charges = numpy.asarray(mol.atom_charges(), dtype=numpy.float64)
    atom_coords = atom_coords[atmlst]
    atom_charges = atom_charges[atmlst]
    fakemol_nuc = gto.fakemol_for_charges(atom_coords)
    fakemol = gto.fakemol_for_charges(grid_coords, expnt=charge_exp**2)
    int2c2e_ip1 = mol._add_suffix('int2c2e_ip1')
    v_ng_ip1 = gto.mole.intor_cross(int2c2e_ip1, fakemol_nuc, fakemol)
    v_ng_ip1 = numpy.array(v_ng_ip1)
    dV_on_charge_dx = numpy.einsum('dAq,A->Adq', v_ng_ip1, atom_charges)

    v_ng_ip2 = gto.mole.intor_cross(int2c2e_ip1, fakemol, fakemol_nuc)
    v_ng_ip2 = numpy.array(v_ng_ip2)
    for i_atom in atmlst:
        g0,g1 = gridslice[i_atom]
        dV_on_charge_dx[i_atom,:,g0:g1] += numpy.einsum('dqA,A->dq', v_ng_ip2[:,g0:g1,:], atom_charges)

    super_mol = mol.super_mol
    for t, mol in super_mol.components.items():
        charge = _get_charge_from_mol_comp(super_mol, t)
        nao = mol.nao
        max_memory = pcmobj.max_memory - lib.current_memory()[0]
        blksize = int(max(max_memory*.9e6/8/nao**2/3, 400))
        aoslice = mol.aoslice_by_atom()
        int3c2e_ip1 = mol._add_suffix('int3c2e_ip1')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip1)
        dIdA = numpy.empty([len(atmlst), 3, ngrids])
        for g0, g1 in lib.prange(0, ngrids, blksize):
            fakemol = gto.fakemol_for_charges(grid_coords[g0:g1], expnt=charge_exp[g0:g1]**2)
            v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip1, aosym='s1', cintopt=cintopt)
            v_nj = numpy.einsum('dijq,ij->diq', v_nj, dm[t] + dm[t].T)
            dvj = numpy.asarray([numpy.sum(v_nj[:,p0:p1,:], axis=1) for p0,p1 in aoslice[:,2:]])
            dIdA[atmlst,:,g0:g1] = dvj[atmlst,:,:]

        dV_on_charge_dx[atmlst,:,:] -= charge * dIdA[atmlst,:,:]

        int3c2e_ip2 = mol._add_suffix('int3c2e_ip2')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip2)
        dIdC = numpy.empty([3,ngrids])
        for g0, g1 in lib.prange(0, ngrids, blksize):
            fakemol = gto.fakemol_for_charges(grid_coords[g0:g1], expnt=charge_exp[g0:g1]**2)
            q_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip2, aosym='s1', cintopt=cintopt)
            dIdC[:,g0:g1] = numpy.einsum('dijq,ij->dq', q_nj, dm[t])
        for i_atom in atmlst:
            g0,g1 = gridslice[i_atom]
            dV_on_charge_dx[i_atom,:,g0:g1] -= charge * dIdC[:,g0:g1]

    return dV_on_charge_dx

def get_dqsym_dx_fix_K_R(pcmobj, dm, atmlst, dV_on_charge_dx=None):
    if dV_on_charge_dx is None:
        dV_on_charge_dx = get_dvgrids(pcmobj, dm, atmlst)
    K = pcmobj._intermediates['K']
    R = pcmobj._intermediates['R']
    R_dVdx = numpy.einsum('ij,Adj->Adi', R, dV_on_charge_dx)
    K_1_R_dVdx = pcm_hess.einsum_ij_Adj_Adi_inverseK(K, R_dVdx)
    K_1T_dVdx = pcm_hess.einsum_ij_Adj_Adi_inverseK(K.T, dV_on_charge_dx)
    RT_K_1T_dVdx = numpy.einsum('ij,Adj->Adi', R.T, K_1T_dVdx)
    dqdx_fix_K_R = 0.5 * (K_1_R_dVdx + RT_K_1T_dVdx)

    return dqdx_fix_K_R

def get_dqsym_dx(pcmobj, dm, atmlst, dV_on_charge_dx=None):
    return pcm_hess.get_dqsym_dx_fix_vgrids(pcmobj, atmlst) \
           + get_dqsym_dx_fix_K_R(pcmobj, dm, atmlst, dV_on_charge_dx)

def analytical_hess_nuc(pcmobj, dqdx, verbose=None):
    if not pcmobj._intermediates:
        pcmobj.build()
    mol = pcmobj.mol
    log = logger.new_logger(mol, verbose)
    t1 = (logger.process_clock(), logger.perf_counter())

    q_sym        = pcmobj._intermediates['q_sym']
    gridslice    = pcmobj.surface['gslice_by_atom']
    grid_coords  = pcmobj.surface['grid_coords']
    exponents    = pcmobj.surface['charge_exp']

    ngrids = q_sym.shape[0]

    atom_coords = mol.atom_coords(unit='B')
    atom_charges = numpy.asarray(mol.atom_charges(), dtype=numpy.float64)
    fakemol_nuc = gto.fakemol_for_charges(atom_coords)
    fakemol = gto.fakemol_for_charges(grid_coords, expnt=exponents**2)

    d2e_from_d2I = numpy.zeros([mol.natm, mol.natm, 3, 3])

    int2c2e_ip1ip2 = mol._add_suffix('int2c2e_ip1ip2')
    d2I_dAdC = gto.mole.intor_cross(int2c2e_ip1ip2, fakemol_nuc, fakemol)
    d2I_dAdC = d2I_dAdC.reshape(3, 3, mol.natm, ngrids)
    for i_atom in range(mol.natm):
        g0,g1 = gridslice[i_atom]
        d2e_from_d2I[:, i_atom, :, :] += \
            numpy.einsum('A,dDAq,q->AdD', atom_charges, d2I_dAdC[:, :, :, g0:g1], q_sym[g0:g1])
        d2e_from_d2I[i_atom, :, :, :] += \
            numpy.einsum('A,dDAq,q->AdD', atom_charges, d2I_dAdC[:, :, :, g0:g1], q_sym[g0:g1])

    int2c2e_ipip1 = mol._add_suffix('int2c2e_ipip1')
    # # Some explanations here:
    # # Why can we use the ip1ip2 here? Because of the translational invariance
    # # $\frac{\partial^2 I_{AC}}{\partial A^2} + \frac{\partial^2 I_{AC}}{\partial A \partial C} = 0$
    # # Why not using the ipip1 here? Because the nuclei, a point charge,
    # # is handled as a Gaussian charge with exponent = 1e16.
    # # This causes severe numerical problem in function int2c2e_ip1ip2,
    # # and make the main diagonal of hessian garbage.
    # d2I_dA2 = gto.mole.intor_cross(int2c2e_ipip1, fakemol_nuc, fakemol)
    d2I_dA2 = -gto.mole.intor_cross(int2c2e_ip1ip2, fakemol_nuc, fakemol)
    d2I_dA2 = d2I_dA2 @ q_sym
    d2I_dA2 = d2I_dA2.reshape(3, 3, mol.natm)
    for i_atom in range(mol.natm):
        d2e_from_d2I[i_atom, i_atom, :, :] += atom_charges[i_atom] * d2I_dA2[:, :, i_atom]

    d2I_dC2 = gto.mole.intor_cross(int2c2e_ipip1, fakemol, fakemol_nuc)
    d2I_dC2 = d2I_dC2 @ atom_charges
    d2I_dC2 = d2I_dC2.reshape(3, 3, ngrids)
    for i_atom in range(mol.natm):
        g0,g1 = gridslice[i_atom]
        d2e_from_d2I[i_atom, i_atom, :, :] += d2I_dC2[:, :, g0:g1] @ q_sym[g0:g1]

    d2e_from_dIdq = numpy.zeros([mol.natm, mol.natm, 3, 3])
    for i_atom in range(mol.natm):
        for i_xyz in range(3):
            d2e_from_dIdq[i_atom, :, i_xyz, :] = pcm_grad.grad_nuc(pcmobj, q_sym = dqdx[i_atom, i_xyz, :])

    d2e = d2e_from_d2I - d2e_from_dIdq

    t1 = log.timer_debug1('solvent hessian d(dVnuc/dx * q)/dx contribution', *t1)
    return d2e

def analytical_hess_qv(pcmobj, dm, mol, dqdx, verbose=None):
    if not pcmobj._intermediates:
        pcmobj.build()
    nao = mol.nao
    log = logger.new_logger(mol, mol.verbose)
    t1 = (logger.process_clock(), logger.perf_counter())

    gridslice   = pcmobj.surface['gslice_by_atom']
    charge_exp  = pcmobj.surface['charge_exp']
    grid_coords = pcmobj.surface['grid_coords']
    q_sym       = pcmobj._intermediates['q_sym']

    aoslice = mol.aoslice_by_atom()
    aoslice = numpy.array(aoslice)

    d2e_from_d2I = numpy.zeros([mol.natm, mol.natm, 3, 3])

    max_memory = pcmobj.max_memory - lib.current_memory()[0]
    blksize = int(max(max_memory*.9e6/8/nao**2/9, 400))
    ngrids = q_sym.shape[0]
    int3c2e_ipip1 = mol._add_suffix('int3c2e_ipip1')
    cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ipip1)
    d2I_dA2 = numpy.zeros([9, nao, nao])
    for g0, g1 in lib.prange(0, ngrids, blksize):
        fakemol = gto.fakemol_for_charges(grid_coords[g0:g1], expnt=charge_exp[g0:g1]**2)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ipip1, aosym='s1', cintopt=cintopt)
        d2I_dA2 += numpy.einsum('dijq,q->dij', v_nj, q_sym[g0:g1])
    d2I_dA2 = d2I_dA2.reshape([3, 3, nao, nao])
    for i_atom in range(mol.natm):
        p0,p1 = aoslice[i_atom, 2:]
        d2e_from_d2I[i_atom, i_atom, :, :] += \
            numpy.einsum('ij,dDij->dD', dm[p0:p1, :], d2I_dA2[:, :, p0:p1, :])
        d2e_from_d2I[i_atom, i_atom, :, :] += \
            numpy.einsum('ij,dDij->dD', dm[:, p0:p1], d2I_dA2[:, :, p0:p1, :].transpose(0,1,3,2))
    d2I_dA2 = None

    int3c2e_ipvip1 = mol._add_suffix('int3c2e_ipvip1')
    cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ipvip1)
    d2I_dAdB = numpy.zeros([9, nao, nao])
    for g0, g1 in lib.prange(0, ngrids, blksize):
        fakemol = gto.fakemol_for_charges(grid_coords[g0:g1], expnt=charge_exp[g0:g1]**2)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ipvip1, aosym='s1', cintopt=cintopt)
        d2I_dAdB += numpy.einsum('dijq,q->dij', v_nj, q_sym[g0:g1])
    d2I_dAdB = d2I_dAdB.reshape([3, 3, nao, nao])
    for i_atom in range(mol.natm):
        pi0,pi1 = aoslice[i_atom, 2:]
        for j_atom in range(mol.natm):
            pj0,pj1 = aoslice[j_atom, 2:]
            d2e_from_d2I[i_atom, j_atom, :, :] += \
                numpy.einsum('ij,dDij->dD', dm[pi0:pi1, pj0:pj1], d2I_dAdB[:, :, pi0:pi1, pj0:pj1])
            d2e_from_d2I[i_atom, j_atom, :, :] += \
                numpy.einsum('ij,dDij->dD', dm[pj0:pj1, pi0:pi1], d2I_dAdB[:, :, pi0:pi1, pj0:pj1].transpose(0,1,3,2))
    d2I_dAdB = None

    for j_atom in range(mol.natm):
        g0,g1 = gridslice[j_atom]
        int3c2e_ip1ip2 = mol._add_suffix('int3c2e_ip1ip2')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip1ip2)
        fakemol = gto.fakemol_for_charges(grid_coords[g0:g1], expnt=charge_exp[g0:g1]**2)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip1ip2, aosym='s1', cintopt=cintopt)
        d2I_dAdC = numpy.einsum('dijq,q->dij', v_nj, q_sym[g0:g1])
        d2I_dAdC = d2I_dAdC.reshape([3, 3, nao, nao])

        for i_atom in range(mol.natm):
            p0,p1 = aoslice[i_atom, 2:]
            d2e_from_d2I[i_atom, j_atom, :, :] += \
                numpy.einsum('ij,dDij->dD', dm[p0:p1, :], d2I_dAdC[:, :, p0:p1, :])
            d2e_from_d2I[i_atom, j_atom, :, :] += \
                numpy.einsum('ij,dDij->dD', dm[:, p0:p1], d2I_dAdC[:, :, p0:p1, :].transpose(0,1,3,2))

            d2e_from_d2I[j_atom, i_atom, :, :] += \
                numpy.einsum('ij,dDij->dD', dm[p0:p1, :], d2I_dAdC[:, :, p0:p1, :].transpose(1,0,2,3))
            d2e_from_d2I[j_atom, i_atom, :, :] += \
                numpy.einsum('ij,dDij->dD', dm[:, p0:p1], d2I_dAdC[:, :, p0:p1, :].transpose(1,0,3,2))
    d2I_dAdC = None

    int3c2e_ipip2 = mol._add_suffix('int3c2e_ipip2')
    cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ipip2)
    d2I_dC2 = numpy.empty([9, ngrids])
    for g0, g1 in lib.prange(0, ngrids, blksize):
        fakemol = gto.fakemol_for_charges(grid_coords[g0:g1], expnt=charge_exp[g0:g1]**2)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ipip2, aosym='s1', cintopt=cintopt)
        d2I_dC2[:,g0:g1] = numpy.einsum('dijq,ij->dq', v_nj, dm)
    d2I_dC2 = d2I_dC2.reshape([3, 3, ngrids])
    for i_atom in range(mol.natm):
        g0,g1 = gridslice[i_atom]
        d2e_from_d2I[i_atom, i_atom, :, :] += d2I_dC2[:, :, g0:g1] @ q_sym[g0:g1]
    d2I_dC2 = None

    d2e_from_dIdq = numpy.zeros([mol.natm, mol.natm, 3, 3])
    for i_atom in range(mol.natm):
        for i_xyz in range(3):
            d2e_from_dIdq[i_atom, :, i_xyz, :] = pcm_grad.grad_qv(pcmobj, dm, mol,
                                                                  q_sym = dqdx[i_atom, i_xyz, :])

    d2e = d2e_from_d2I + d2e_from_dIdq
    d2e *= -1

    t1 = log.timer_debug1('solvent hessian d(dI/dx * q)/dx contribution', *t1)
    return d2e

def analytical_hess_solver(pcmobj, dVdx, verbose=None):
    if not pcmobj._intermediates:
        pcmobj.build()
    mol = pcmobj.mol
    log = logger.new_logger(mol, mol.verbose)
    t1 = (logger.process_clock(), logger.perf_counter())

    natom = mol.natm
    atmlst = range(natom) # Attention: This cannot be split

    gridslice    = pcmobj.surface['gslice_by_atom']
    v_grids      = pcmobj._intermediates['v_grids']
    A            = pcmobj._intermediates['A']
    D            = pcmobj._intermediates['D']
    S            = pcmobj._intermediates['S']
    K            = pcmobj._intermediates['K']
    R            = pcmobj._intermediates['R']
    q            = pcmobj._intermediates['q']
    f_epsilon    = pcmobj._intermediates['f_epsilon']

    ngrids = q.shape[0]

    vK_1 = numpy.linalg.solve(K.T, v_grids)

    if pcmobj.method.upper() in ['C-PCM', 'CPCM', 'COSMO']:
        dF, _ = pcm_hess.get_dF_dA(pcmobj.surface)
        _, dS, dSii = pcm_hess.get_dD_dS(pcmobj.surface, dF, with_D=False, with_S=True)

        # dR = 0, dK = dS
        # d(S-1 R) = - S-1 dS S-1 R
        # d2(S-1 R) = (S-1 dS S-1 dS S-1 R) + (S-1 dS S-1 dS S-1 R) - (S-1 d2S S-1 R)
        dSdx_dot_q = pcm_hess.get_dS_dot_q(dS, dSii, q, atmlst, gridslice)
        S_1_dSdx_dot_q = pcm_hess.einsum_ij_Adj_Adi_inverseK(K, dSdx_dot_q)
        dSdx_dot_q = None
        VS_1_dot_dSdx = pcm_hess.get_dST_dot_q(dS, dSii, vK_1, atmlst, gridslice)
        dS = None
        dSii = None
        d2e_from_d2KR = numpy.einsum('Adi,BDi->ABdD', VS_1_dot_dSdx, S_1_dSdx_dot_q) * 2

        _, d2S = pcm_hess.get_d2D_d2S(pcmobj.surface, with_D=False, with_S=True)
        d2F, _ = pcm_hess.get_d2F_d2A(pcmobj.surface)
        d2Sii = pcm_hess.get_d2Sii(pcmobj.surface, dF, d2F)
        dF = None
        d2F = None
        d2e_from_d2KR -= pcm_hess.get_v_dot_d2S_dot_q(d2S, d2Sii, vK_1, q, natom, gridslice)
        d2S = None
        d2Sii = None

        dK_1Rv = -S_1_dSdx_dot_q
        dvK_1R = -pcm_hess.einsum_Adi_ij_Adj_inverseK(VS_1_dot_dSdx, K) @ R

    elif pcmobj.method.upper() in ['IEF-PCM', 'IEFPCM', 'SMD']:
        dF, dA = pcm_hess.get_dF_dA(pcmobj.surface)
        dD, dS, dSii = pcm_hess.get_dD_dS(pcmobj.surface, dF, with_D=True, with_S=True)

        # dR = f_eps/(2*pi) * (dD*A + D*dA)
        # dK = dS - f_eps/(2*pi) * (dD*A*S + D*dA*S + D*A*dS)

        # d2R = f_eps/(2*pi) * (d2D*A + dD*dA + dD*dA + D*d2A)
        # d2K = d2S - f_eps/(2*pi) * (d2D*A*S + D*d2A*S + D*A*d2S
        #                             + dD*dA*S + dD*dA*S + dD*A*dS + dD*A*dS + D*dA*dS + D*dA*dS)
        # The terms showing up twice on equation above (dD*dA + dD*dA for example)
        # refer to dD/dx * dA/dy + dD/dy * dA/dx,
        # since D is not symmetric, they are not the same.

        # d(K-1 R) = - K-1 dK K-1 R + K-1 dR
        # d2(K-1 R) = (K-1 dK K-1 dK K-1 R) + (K-1 dK K-1 dK K-1 R) - (K-1 d2K K-1 R) - (K-1 dK K-1 dR)
        #             - (K-1 dK K-1 dR) + (K-1 d2R)
        f_eps_over_2pi = f_epsilon/(2.0*PI)

        dSdx_dot_q = pcm_hess.get_dS_dot_q(dS, dSii, q, atmlst, gridslice)
        DA = D*A
        dKdx_dot_q = dSdx_dot_q - f_eps_over_2pi * numpy.einsum('ij,Adj->Adi', DA, dSdx_dot_q)
        dAdx_dot_Sq = pcm_hess.get_dA_dot_q(dA, S @ q, atmlst)
        dKdx_dot_q -= f_eps_over_2pi * numpy.einsum('ij,Adj->Adi', D, dAdx_dot_Sq)
        AS = (A * S.T).T # It's just diag(A) @ S
        ASq = AS @ q
        dDdx_dot_ASq = pcm_hess.get_dD_dot_q(dD, ASq, atmlst, gridslice, ngrids)
        dKdx_dot_q -= f_eps_over_2pi * dDdx_dot_ASq
        dDdx_dot_ASq = None

        K_1_dot_dKdx_dot_q = pcm_hess.einsum_ij_Adj_Adi_inverseK(K, dKdx_dot_q)
        dKdx_dot_q = None

        vK_1_dot_dSdx = pcm_hess.get_dST_dot_q(dS, dSii, vK_1, atmlst, gridslice)
        vK_1_dot_dKdx = vK_1_dot_dSdx
        vK_1_dot_dSdx = None
        vK_1_dot_dDdx = pcm_hess.get_dDT_dot_q(dD, vK_1, atmlst, gridslice, ngrids)
        vK_1_dot_dKdx -= f_eps_over_2pi * numpy.einsum('ij,Adj->Adi', AS.T, vK_1_dot_dDdx)
        AS = None
        vK_1D = D.T @ vK_1
        vK_1D_dot_dAdx = pcm_hess.get_dA_dot_q(dA, vK_1D, atmlst)
        vK_1_dot_dKdx -= f_eps_over_2pi * numpy.einsum('ij,Adj->Adi', S.T, vK_1D_dot_dAdx)
        vK_1DA = DA.T @ vK_1
        DA = None
        vK_1DA_dot_dSdx = pcm_hess.get_dST_dot_q(dS, dSii, vK_1DA, atmlst, gridslice)
        dS = None
        dSii = None
        vK_1_dot_dKdx -= f_eps_over_2pi * vK_1DA_dot_dSdx
        vK_1DA_dot_dSdx = None

        d2e_from_d2KR  = numpy.einsum('Adi,BDi->ABdD', vK_1_dot_dKdx, K_1_dot_dKdx_dot_q)
        d2e_from_d2KR += numpy.einsum('Adi,BDi->BADd', vK_1_dot_dKdx, K_1_dot_dKdx_dot_q)

        d2F, d2A = pcm_hess.get_d2F_d2A(pcmobj.surface)
        vK_1_d2K_q  = pcm_hess.get_v_dot_d2A_dot_q(d2A, vK_1D, S @ q)
        vK_1_d2R_V  = pcm_hess.get_v_dot_d2A_dot_q(d2A, vK_1D, v_grids)
        d2A = None
        d2Sii = pcm_hess.get_d2Sii(pcmobj.surface, dF, d2F)
        dF = None
        d2F = None
        d2D, d2S = pcm_hess.get_d2D_d2S(pcmobj.surface, with_D=True, with_S=True)
        vK_1_d2K_q += pcm_hess.get_v_dot_d2D_dot_q(d2D, vK_1, ASq, natom, gridslice)
        vK_1_d2R_V += pcm_hess.get_v_dot_d2D_dot_q(d2D, vK_1, A * v_grids, natom, gridslice)
        d2D = None
        vK_1_d2K_q += pcm_hess.get_v_dot_d2S_dot_q(d2S, d2Sii, vK_1DA, q, natom, gridslice)
        vK_1_d2K_q += numpy.einsum('Adi,BDi->ABdD', vK_1_dot_dDdx, dAdx_dot_Sq)
        vK_1_d2K_q += numpy.einsum('Adi,BDi->ABdD', vK_1_dot_dDdx * A, dSdx_dot_q)
        vK_1_d2K_q += numpy.einsum('Adi,BDi->ABdD', vK_1D_dot_dAdx, dSdx_dot_q)
        vK_1_d2K_q += numpy.einsum('Adi,BDi->BADd', vK_1_dot_dDdx, dAdx_dot_Sq)
        vK_1_d2K_q += numpy.einsum('Adi,BDi->BADd', vK_1_dot_dDdx * A, dSdx_dot_q)
        vK_1_d2K_q += numpy.einsum('Adi,BDi->BADd', vK_1D_dot_dAdx, dSdx_dot_q)
        vK_1_d2K_q *= -f_eps_over_2pi
        vK_1_d2K_q += pcm_hess.get_v_dot_d2S_dot_q(d2S, d2Sii, vK_1, q, natom, gridslice)
        d2S = None
        d2Sii = None

        d2e_from_d2KR -= vK_1_d2K_q

        dAdx_dot_V = pcm_hess.get_dA_dot_q(dA, v_grids, atmlst)
        dDdx_dot_AV = pcm_hess.get_dD_dot_q(dD, A * v_grids, atmlst, gridslice, ngrids)
        dRdx_dot_V = f_eps_over_2pi * (dDdx_dot_AV + numpy.einsum('ij,Adj->Adi', D, dAdx_dot_V))
        dDdx_dot_AV = None

        K_1_dot_dRdx_dot_V = pcm_hess.einsum_ij_Adj_Adi_inverseK(K, dRdx_dot_V)
        dRdx_dot_V = None

        d2e_from_d2KR -= numpy.einsum('Adi,BDi->ABdD', vK_1_dot_dKdx, K_1_dot_dRdx_dot_V)
        d2e_from_d2KR -= numpy.einsum('Adi,BDi->BADd', vK_1_dot_dKdx, K_1_dot_dRdx_dot_V)

        vK_1_d2R_V += numpy.einsum('Adi,BDi->ABdD', vK_1_dot_dDdx, dAdx_dot_V)
        vK_1_d2R_V += numpy.einsum('Adi,BDi->BADd', vK_1_dot_dDdx, dAdx_dot_V)
        vK_1_d2R_V *= f_eps_over_2pi

        d2e_from_d2KR += vK_1_d2R_V

        dK_1Rv = -K_1_dot_dKdx_dot_q + K_1_dot_dRdx_dot_V

        VK_1D_dot_dAdx = pcm_hess.get_dA_dot_q(dA, (D.T @ vK_1).T, atmlst)
        VK_1_dot_dDdx = pcm_hess.get_dDT_dot_q(dD, vK_1, atmlst, gridslice, ngrids)
        VK_1_dot_dRdx = f_eps_over_2pi * (VK_1D_dot_dAdx + VK_1_dot_dDdx * A)

        dvK_1R = -pcm_hess.einsum_Adi_ij_Adj_inverseK(vK_1_dot_dKdx, K) @ R + VK_1_dot_dRdx

    elif pcmobj.method.upper() in ['SS(V)PE']:
        dF, dA = pcm_hess.get_dF_dA(pcmobj.surface)
        dD, dS, dSii = pcm_hess.get_dD_dS(pcmobj.surface, dF, with_D=True, with_S=True)

        # dR = f_eps/(2*pi) * (dD*A + D*dA)
        # dK = dS - f_eps/(4*pi) * (dD*A*S + D*dA*S + D*A*dS + dST*AT*DT + ST*dAT*DT + ST*AT*dDT)

        # d2R = f_eps/(2*pi) * (d2D*A + dD*dA + dD*dA + D*d2A)
        # d2K = d2S - f_eps/(4*pi)
        #             * (d2D*A*S + D*d2A*S + D*A*d2S
        #                + dD*dA*S + dD*dA*S + dD*A*dS + dD*A*dS + D*dA*dS + D*dA*dS
        #                + d2ST*AT*DT + ST*d2AT*DT + ST*AT*d2DT
        #                + dST*dAT*DT + dST*dAT*DT + dST*AT*dDT + dST*AT*dDT + ST*dAT*dDT + ST*dAT*dDT)
        f_eps_over_2pi = f_epsilon/(2.0*PI)
        f_eps_over_4pi = f_epsilon/(4.0*PI)

        dSdx_dot_q = pcm_hess.get_dS_dot_q(dS, dSii, q, atmlst, gridslice)
        DA = D*A
        dKdx_dot_q = dSdx_dot_q - f_eps_over_4pi * numpy.einsum('ij,Adj->Adi', DA, dSdx_dot_q)
        dAdx_dot_Sq = pcm_hess.get_dA_dot_q(dA, S @ q, atmlst)
        dKdx_dot_q -= f_eps_over_4pi * numpy.einsum('ij,Adj->Adi', D, dAdx_dot_Sq)
        AS = (A * S.T).T # It's just diag(A) @ S
        ASq = AS @ q
        dDdx_dot_ASq = pcm_hess.get_dD_dot_q(dD, ASq, atmlst, gridslice, ngrids)
        dKdx_dot_q -= f_eps_over_4pi * dDdx_dot_ASq
        dDdx_dot_ASq = None
        dDdxT_dot_q = pcm_hess.get_dDT_dot_q(dD, q, atmlst, gridslice, ngrids)
        dKdx_dot_q -= f_eps_over_4pi * numpy.einsum('ij,Adj->Adi', AS.T, dDdxT_dot_q)
        dAdxT_dot_DT_q = pcm_hess.get_dA_dot_q(dA, D.T @ q, atmlst)
        dKdx_dot_q -= f_eps_over_4pi * numpy.einsum('ij,Adj->Adi', S.T, dAdxT_dot_DT_q)
        AT_DT_q = DA.T @ q
        dSdxT_dot_AT_DT_q = pcm_hess.get_dS_dot_q(dS, dSii, AT_DT_q, atmlst, gridslice)
        dKdx_dot_q -= f_eps_over_4pi * dSdxT_dot_AT_DT_q
        dSdxT_dot_AT_DT_q = None

        K_1_dot_dKdx_dot_q = pcm_hess.einsum_ij_Adj_Adi_inverseK(K, dKdx_dot_q)
        dKdx_dot_q = None

        vK_1_dot_dSdx = pcm_hess.get_dST_dot_q(dS, dSii, vK_1, atmlst, gridslice)
        vK_1_dot_dKdx = vK_1_dot_dSdx
        vK_1_dot_dSdx = None
        vK_1_dot_dDdx = pcm_hess.get_dDT_dot_q(dD, vK_1, atmlst, gridslice, ngrids)
        vK_1_dot_dKdx -= f_eps_over_4pi * numpy.einsum('ij,Adj->Adi', AS.T, vK_1_dot_dDdx)
        vK_1D_dot_dAdx = pcm_hess.get_dA_dot_q(dA, D.T @ vK_1, atmlst)
        vK_1_dot_dKdx -= f_eps_over_4pi * numpy.einsum('ij,Adj->Adi', S.T, vK_1D_dot_dAdx)
        vK_1DA = DA.T @ vK_1
        vK_1DA_dot_dSdx = pcm_hess.get_dST_dot_q(dS, dSii, vK_1DA, atmlst, gridslice)
        vK_1_dot_dKdx -= f_eps_over_4pi * vK_1DA_dot_dSdx
        vK_1DA_dot_dSdx = None
        vK_1_dot_dSdxT = pcm_hess.get_dS_dot_q(dS, dSii, vK_1, atmlst, gridslice)
        dS = None
        dSii = None
        vK_1_dot_dKdx -= f_eps_over_4pi * numpy.einsum('ij,Adj->Adi', DA, vK_1_dot_dSdxT)
        DA = None
        vK_1_ST_dot_dAdxT = pcm_hess.get_dA_dot_q(dA, (S @ vK_1).T, atmlst)
        vK_1_dot_dKdx -= f_eps_over_4pi * numpy.einsum('ij,Adj->Adi', D, vK_1_ST_dot_dAdxT)
        vK_1_ST_AT = AS @ vK_1
        AS = None
        vK_1_ST_AT_dot_dDdxT = pcm_hess.get_dD_dot_q(dD, vK_1_ST_AT, atmlst, gridslice, ngrids)
        vK_1_dot_dKdx -= f_eps_over_4pi * vK_1_ST_AT_dot_dDdxT
        vK_1_ST_AT_dot_dDdxT = None

        d2e_from_d2KR  = numpy.einsum('Adi,BDi->ABdD', vK_1_dot_dKdx, K_1_dot_dKdx_dot_q)
        d2e_from_d2KR += numpy.einsum('Adi,BDi->BADd', vK_1_dot_dKdx, K_1_dot_dKdx_dot_q)

        d2F, d2A = pcm_hess.get_d2F_d2A(pcmobj.surface)
        vK_1_d2K_q  = pcm_hess.get_v_dot_d2A_dot_q(d2A, (D.T @ vK_1).T, S @ q)
        vK_1_d2K_q += pcm_hess.get_v_dot_d2A_dot_q(d2A, (S @ vK_1).T, D.T @ q)
        vK_1_d2R_V  = pcm_hess.get_v_dot_d2A_dot_q(d2A, (D.T @ vK_1).T, v_grids)
        d2A = None
        d2Sii = pcm_hess.get_d2Sii(pcmobj.surface, dF, d2F)
        dF = None
        d2F = None
        d2D, d2S = pcm_hess.get_d2D_d2S(pcmobj.surface, with_D=True, with_S=True)
        vK_1_d2K_q += pcm_hess.get_v_dot_d2D_dot_q(d2D, vK_1, ASq, natom, gridslice)
        vK_1_d2K_q += pcm_hess.get_v_dot_d2DT_dot_q(d2D, vK_1_ST_AT, q, natom, gridslice)
        vK_1_d2R_V += pcm_hess.get_v_dot_d2D_dot_q(d2D, vK_1, A * v_grids, natom, gridslice)
        d2D = None
        vK_1_d2K_q += pcm_hess.get_v_dot_d2S_dot_q(d2S, d2Sii, vK_1DA, q, natom, gridslice)
        vK_1_d2K_q += pcm_hess.get_v_dot_d2ST_dot_q(d2S, d2Sii, vK_1, AT_DT_q, natom, gridslice)
        vK_1_d2K_q += numpy.einsum('Adi,BDi->ABdD', vK_1_dot_dDdx, dAdx_dot_Sq)
        vK_1_d2K_q += numpy.einsum('Adi,BDi->ABdD', vK_1_dot_dDdx * A, dSdx_dot_q)
        vK_1_d2K_q += numpy.einsum('Adi,BDi->ABdD', vK_1D_dot_dAdx, dSdx_dot_q)
        vK_1_d2K_q += numpy.einsum('Adi,BDi->ABdD', vK_1_dot_dSdxT, dAdxT_dot_DT_q)
        vK_1_d2K_q += numpy.einsum('Adi,BDi->ABdD', vK_1_dot_dSdxT * A, dDdxT_dot_q)
        vK_1_d2K_q += numpy.einsum('Adi,BDi->ABdD', vK_1_ST_dot_dAdxT, dDdxT_dot_q)
        vK_1_d2K_q += numpy.einsum('Adi,BDi->BADd', vK_1_dot_dDdx, dAdx_dot_Sq)
        vK_1_d2K_q += numpy.einsum('Adi,BDi->BADd', vK_1_dot_dDdx * A, dSdx_dot_q)
        vK_1_d2K_q += numpy.einsum('Adi,BDi->BADd', vK_1D_dot_dAdx, dSdx_dot_q)
        vK_1_d2K_q += numpy.einsum('Adi,BDi->BADd', vK_1_dot_dSdxT, dAdxT_dot_DT_q)
        vK_1_d2K_q += numpy.einsum('Adi,BDi->BADd', vK_1_dot_dSdxT * A, dDdxT_dot_q)
        vK_1_d2K_q += numpy.einsum('Adi,BDi->BADd', vK_1_ST_dot_dAdxT, dDdxT_dot_q)
        vK_1_d2K_q *= -f_eps_over_4pi
        vK_1_d2K_q += pcm_hess.get_v_dot_d2S_dot_q(d2S, d2Sii, vK_1, q, natom, gridslice)
        d2S = None
        d2Sii = None

        d2e_from_d2KR -= vK_1_d2K_q

        dAdx_dot_V = pcm_hess.get_dA_dot_q(dA, v_grids, atmlst)
        dDdx_dot_AV = pcm_hess.get_dD_dot_q(dD, A * v_grids, atmlst, gridslice, ngrids)
        dRdx_dot_V = f_eps_over_2pi * (dDdx_dot_AV + numpy.einsum('ij,Adj->Adi', D, dAdx_dot_V))
        dDdx_dot_AV = None

        K_1_dot_dRdx_dot_V = pcm_hess.einsum_ij_Adj_Adi_inverseK(K, dRdx_dot_V)

        d2e_from_d2KR -= numpy.einsum('Adi,BDi->ABdD', vK_1_dot_dKdx, K_1_dot_dRdx_dot_V)
        d2e_from_d2KR -= numpy.einsum('Adi,BDi->BADd', vK_1_dot_dKdx, K_1_dot_dRdx_dot_V)

        vK_1_d2R_V += numpy.einsum('Adi,BDi->ABdD', vK_1_dot_dDdx, dAdx_dot_V)
        vK_1_d2R_V += numpy.einsum('Adi,BDi->BADd', vK_1_dot_dDdx, dAdx_dot_V)
        vK_1_d2R_V *= f_eps_over_2pi

        d2e_from_d2KR += vK_1_d2R_V

        dK_1Rv = -K_1_dot_dKdx_dot_q + K_1_dot_dRdx_dot_V

        VK_1D_dot_dAdx = pcm_hess.get_dA_dot_q(dA, (D.T @ vK_1).T, atmlst)
        VK_1_dot_dDdx = pcm_hess.get_dDT_dot_q(dD, vK_1, atmlst, gridslice, ngrids)
        VK_1_dot_dRdx = f_eps_over_2pi * (VK_1D_dot_dAdx + VK_1_dot_dDdx * A)

        dvK_1R = -pcm_hess.einsum_Adi_ij_Adj_inverseK(vK_1_dot_dKdx, K) @ R + VK_1_dot_dRdx

    else:
        raise RuntimeError(f"Unknown implicit solvent model: {pcmobj.method}")

    d2e = d2e_from_d2KR

    d2e -= numpy.einsum('Adi,BDi->BADd', dvK_1R, dVdx)
    d2e -= numpy.einsum('Adi,BDi->ABdD', dVdx, dK_1Rv)

    d2e *= 0.5
    t1 = log.timer_debug1('solvent hessian d(V * dK-1R/dx * V)/dx contribution', *t1)
    return d2e

def analytical_grad_vmat(pcmobj, dqdx, mol, atmlst=None, verbose=None):
    '''
    dv_solv / da
    '''
    if not pcmobj._intermediates:
        pcmobj.build()
    nao = mol.nao
    log = logger.new_logger(mol, mol.verbose)
    t1 = (logger.process_clock(), logger.perf_counter())

    if atmlst is None:
        atmlst = range(mol.natm)

    gridslice    = pcmobj.surface['gslice_by_atom']
    charge_exp   = pcmobj.surface['charge_exp']
    grid_coords  = pcmobj.surface['grid_coords']
    q_sym        = pcmobj._intermediates['q_sym']
    ngrids = q_sym.shape[0]

    aoslice = mol.aoslice_by_atom()
    aoslice = numpy.array(aoslice)

    dIdx = numpy.empty([len(atmlst), 3, nao, nao])

    max_memory = pcmobj.max_memory - lib.current_memory()[0]
    blksize = int(max(max_memory*.9e6/8/nao**2/3, 400))
    int3c2e_ip1 = mol._add_suffix('int3c2e_ip1')
    cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip1)
    dIdA = numpy.zeros([3, nao, nao])
    for g0, g1 in lib.prange(0, ngrids, blksize):
        fakemol = gto.fakemol_for_charges(grid_coords[g0:g1], expnt=charge_exp[g0:g1]**2)
        v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip1, aosym='s1', cintopt=cintopt)
        dIdA += numpy.einsum('dijq,q->dij', v_nj, q_sym[g0:g1])

    for i_atom in atmlst:
        p0,p1 = aoslice[i_atom, 2:]
        dIdx[i_atom, :, :, :] = 0
        dIdx[i_atom, :, p0:p1, :] += dIdA[:, p0:p1, :]
        dIdx[i_atom, :, :, p0:p1] += dIdA[:, p0:p1, :].transpose(0,2,1)

    int3c2e_ip2 = mol._add_suffix('int3c2e_ip2')
    cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e_ip2)
    for i_atom in atmlst:
        g0,g1 = gridslice[i_atom]
        fakemol = gto.fakemol_for_charges(grid_coords[g0:g1], expnt=charge_exp[g0:g1]**2)
        q_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e_ip2, aosym='s1', cintopt=cintopt)
        dIdC = numpy.einsum('dijq,q->dij', q_nj, q_sym[g0:g1])
        dIdx[i_atom, :, :, :] += dIdC

    dV_on_molecule_dx = dIdx

    int3c2e = mol._add_suffix('int3c2e')
    cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e)
    for i_atom in atmlst:
        for i_xyz in range(3):
            dIdx_from_dqdx = numpy.zeros([nao, nao])
            for g0, g1 in lib.prange(0, ngrids, blksize):
                fakemol = gto.fakemol_for_charges(grid_coords[g0:g1], expnt=charge_exp[g0:g1]**2)
                v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e, aosym='s1', cintopt=cintopt)
                dIdx_from_dqdx += numpy.einsum('ijq,q->ij', v_nj, dqdx[i_atom, i_xyz, g0:g1])
            dV_on_molecule_dx[i_atom, i_xyz, :, :] += dIdx_from_dqdx

    t1 = log.timer_debug1('computing solvent grad veff', *t1)
    return dV_on_molecule_dx

def make_hess_object(base_method):
    from pyscf.solvent._attach_solvent import _Solvation
    from pyscf.hessian.rhf import HessianBase
    if isinstance(base_method, HessianBase):
        # For backward compatibility. The input argument is a gradient object in
        # previous implementations.
        base_method = base_method.base

    # Must be a solvent-attached method
    assert isinstance(base_method, _Solvation)
    with_solvent = base_method.with_solvent
    if with_solvent.frozen:
        raise RuntimeError('Frozen solvent model is not avialbe for energy hessian')

    vac_hess = base_method.undo_solvent().Hessian()
    vac_hess.base = base_method
    name = with_solvent.__class__.__name__ + vac_hess.__class__.__name__
    return lib.set_class(WithSolventHess(vac_hess),
                         (WithSolventHess, vac_hess.__class__), name)

class WithSolventHess:
    _keys = {'de_solvent', 'de_solute'}

    def __init__(self, hess_method):
        self.__dict__.update(hess_method.__dict__)
        self.de_solvent = None
        self.de_solute = None

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.base.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, WithSolventHess, name_mixin))
        del obj.de_solvent
        del obj.de_solute
        return obj

    def to_gpu(self):
        raise NotImplementedError

    def kernel(self, *args, dm=None, atmlst=None, **kwargs):
        dm = kwargs.pop('dm', None)
        if dm is None:
            dm = self.base.make_rdm1(ao_repr=True)
        solvent = self.base.with_solvent
        if not solvent._intermediates:
            solvent.build()
        dm_cache = solvent._intermediates.get('dm', None)
        if dm_cache is not None and isinstance(dm_cache, dict):
            is_dm_similar = True
            for t, comp in dm.items():
                if t in dm_cache:
                    if numpy.linalg.norm(dm_cache[t] - comp) >= 1e-10:
                        is_dm_similar = False
                        break
                else:
                    is_dm_similar = False
                    break
            if not is_dm_similar:
                solvent._get_vind(dm)
        else:
            solvent._get_vind(dm)
        is_equilibrium = solvent.equilibrium_solvation
        solvent.equilibrium_solvation = True
        mol = self.mol
        for t, comp in self.base.components.items():
            if dm[t].ndim == 3:
                dm[t] = dm[t][0] + dm[t][1]
        dVdx = get_dvgrids(solvent, dm, range(mol.natm))
        dqdx = get_dqsym_dx(solvent, dm, range(mol.natm), dV_on_charge_dx=dVdx)
        self.de_solvent  =    analytical_hess_nuc(solvent, dqdx, verbose=self.verbose)
        self.de_solvent += analytical_hess_solver(solvent, dVdx, verbose=self.verbose)
        for t, comp in self.base.components.items():
            charge = comp.charge
            self.de_solvent += charge * analytical_hess_qv(solvent, dm[t], mol.components[t],
                                                           dqdx, verbose=self.verbose)
        self.de_solute = super().kernel(*args, **kwargs)
        self.de = self.de_solute + self.de_solvent
        solvent.equilibrium_solvation = is_equilibrium
        return self.de

    def make_h1(self, mo_coeff, mo_occ, chkfile=None, atmlst=None, verbose=None):
        if atmlst is None:
            atmlst = range(self.mol.natm)
        h1ao = super().make_h1(mo_coeff, mo_occ, atmlst=atmlst, verbose=verbose)
        mol = self.mol
        solvent = self.base.with_solvent
        if not solvent._intermediates:
            solvent.build()
        dm = self.base.make_rdm1(ao_repr=True)
        dm_cache = solvent._intermediates.get('dm', None)
        if dm_cache is not None and isinstance(dm_cache, dict):
            is_dm_similar = True
            for t, comp in dm.items():
                if t in dm_cache:
                    if numpy.linalg.norm(dm_cache[t] - comp) >= 1e-10:
                        is_dm_similar = False
                        break
                else:
                    is_dm_similar = False
                    break
            if not is_dm_similar:
                solvent._get_vind(dm)
        else:
            solvent._get_vind(dm)
        for t, comp in self.base.components.items():
            if isinstance(comp, scf.uhf.UHF):
                dm[t] = dm[t][0] + dm[t][1]
        dqdx = get_dqsym_dx(solvent, dm, atmlst)
        for t, comp in self.base.components.items():
            dv = analytical_grad_vmat(solvent, dqdx, mol.components[t],
                                      atmlst=atmlst, verbose=verbose)
            if isinstance(comp, scf.uhf.UHF):
                for i0, ia in enumerate(atmlst):
                    h1ao[t][:,i0] += comp.charge * dv[i0]
            elif isinstance(comp, scf.hf.RHF):
                for i0, ia in enumerate(atmlst):
                    h1ao[t][i0] += comp.charge * dv[i0]
            else:
                raise NotImplementedError('Base object is not supported')
        return h1ao

    def _finalize(self):
        # disable _finalize. It is called in grad_method.kernel method
        # where self.de was not yet initialized.
        pass
