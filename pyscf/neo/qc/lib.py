#!/usr/bin/env python
#
import numpy
import itertools
from scipy import sparse

def dump_qc_header(log):
    log.note("\n------- Quantum Computing Protocol -------")
    return

def dump_analysis_header(log):
    log.note("\n--------------------------------------------")
    log.note("-------------- QC Analysis -----------------")
    log.note("--------------------------------------------\n")
    log.note("Ordering of orbitals in qubit basis states")
    log.note("is according to orbital energy for each")
    log.note("particle type. Electrons always go first")
    log.note("if multiple types of particles are involved:\n")
    log.note("|tot> = |e> \\otimes |p2> \\otimes |p3> ...")
    return

def dump_spin_order(log, a_id, b_id):
    log.note("\nSpin ordering in electronic qubit basis states: ")
    log.note("alpha: %s", a_id)
    log.note("beta : %s", b_id)
    return

def dump_ucc_level(log, ucc_level, bool_con=False):
    if bool_con:
        str_con = 'Constrained '
    else:
        str_con = ''
    if ucc_level == 2:
        log.note("\n--- %sUCCSD Calculation --- \n", str_con)
    elif ucc_level == 3:
        log.note("\n----%sUCCSD[T^ep] Calculation ----", str_con)
        log.note("--- T^ep operator 1: t2_e*t1_p ----")
        log.note("--- T^ep operator 2: t1_e*t1_p1*t1_p2 ---- \n")
    elif ucc_level == 4:
        log.note("\n---- %sUCCSD[T^ep][Q^ep] Calculation -----", str_con)
        log.note("------ T^ep operator 1: t2_e*t1_p --------")
        log.note("------ T^ep operator 2: t1_e*t1_p1*t1_p2 -------")
        log.note("------ Q^ep operator: t2_e*t1_p1*t1_p2 ------- \n")
    else:
        raise NotImplementedError('UCC level not available')
    return

def ca1_op(idx, create, destroy, pid):
    op1 = create[pid][idx[0]] @ destroy[pid][idx[1]]
    return op1

def ca2_op(idx, create, destroy, pid):
    op2 = create[pid][idx[0]] @ create[pid][idx[1]] @\
          destroy[pid][idx[2]] @ destroy[pid][idx[3]]
    return op2

def kron_I(n_qubit):
    I = sparse.csr_matrix(numpy.array([[1.0, 0.0],[0.0, 1.0]]))
    a = sparse.kron(I,I,'csr')
    for i in range(2,n_qubit):
        a = sparse.kron(a,I,'csr')
    return a

def S_squared_operator(n_qubit_e, s_ab, s_ba, e_sort, create, destroy):
    ''''Valid for both RHF and UHF references.
    Phase and overlap information is contained in
    mixed alpha/beta overlap matrices s_ab and s_ba (relevant for uhf).
    '''
    dim = create[0][0].shape[0]
    # Spin alpha/beta locations for electron in a_id/b_id
    a_id = []
    b_id =[]

    for i in range(n_qubit_e):
        if (e_sort[i]<(n_qubit_e//2)):
            a_id.append(i)
        else:
            b_id.append(i)
    Sz  = sparse.csr_matrix((dim,dim),dtype=complex)
    Sz2 = sparse.csr_matrix((dim,dim),dtype=complex)
    Sp  = sparse.csr_matrix((dim,dim),dtype=complex)
    Sm  = sparse.csr_matrix((dim,dim),dtype=complex)
    for p in range(n_qubit_e // 2):
        pa = a_id[p]
        pb = b_id[p]
        Sz += 0.50*(create[0][pa] @ destroy[0][pa] - create[0][pb] @ destroy[0][pb])
        for q in range(n_qubit_e // 2):
            qa = a_id[q]
            qb = b_id[q]
            Sp += s_ab[p,q]*(create[0][pa] @ destroy[0][qb])
            Sm += s_ba[p,q]*(create[0][pb] @ destroy[0][qa])
            Sz2 += 0.25*( create[0][pa] @ destroy[0][pa] @ create[0][qa] @ destroy[0][qa]
                       -  create[0][pa] @ destroy[0][pa] @ create[0][qb] @ destroy[0][qb]
                       -  create[0][pb] @ destroy[0][pb] @ create[0][qa] @ destroy[0][qa]
                       +  create[0][pb] @ destroy[0][pb] @ create[0][qb] @ destroy[0][qb] )
    Spm = Sp @ Sm
    S_squared = Spm - Sz + Sz2
    return S_squared, Sz, a_id, b_id

def mo_to_spinor(moa, mob, ea, eb):
    ''' Block rhf or uhf spin-orbitals into spinor representation.
    Order according to orbital energy.
    '''
    nao = moa.shape[0]
    etot  = numpy.hstack((ea,eb))
    motot = numpy.block([[moa, numpy.zeros_like(mob)],[numpy.zeros_like(moa),mob]])
    motot = motot[:,etot.argsort()]
    motota = motot[:nao,:] # alpha orb block dim = nao x nso
    mototb = motot[nao:,:] # beta  orb block dim = nao x nso
    return motota, mototb, motot

def UCC_energy(t, Hamiltonian, tau, tau_dag, nt_amp, psi_HF):
    ham_dim = Hamiltonian.shape[0]
    UCC_ansatz = sparse.csr_matrix((ham_dim, ham_dim), dtype=complex)
    for i in range(nt_amp):
        UCC_ansatz += t[i]*(tau[i]-tau_dag[i])

    UCC_psi = sparse.linalg.expm_multiply(UCC_ansatz, psi_HF)
    E = numpy.real(UCC_psi.conj().T @ Hamiltonian @ UCC_psi).item()
    return E

def UCC_energy_shift(t, Hamiltonian, tau, tau_dag, nt_amp, psi_HF,\
        num_nuc, axis, r_op, r_coeff, bool_S2, S2_op, S2_coeff, S2_target):

    ham_dim = Hamiltonian.shape[0]
    UCC_ansatz = sparse.csr_matrix((ham_dim, ham_dim), dtype=complex)
    for i in range(nt_amp):
        UCC_ansatz += t[i]*(tau[i]-tau_dag[i])
    UCC_psi = sparse.linalg.expm_multiply(UCC_ansatz, psi_HF)
    E = numpy.real(UCC_psi.conj().T @ Hamiltonian @ UCC_psi).item()

    for k in range(num_nuc):
        if axis is not None:
            r_axis = numpy.real(UCC_psi.conj().T @ r_op[k,axis] @ UCC_psi).item()
            r_sq = r_axis*r_axis
        else:
            r_x = numpy.real(UCC_psi.conj().T @ r_op[k,0] @ UCC_psi).item()
            r_y = numpy.real(UCC_psi.conj().T @ r_op[k,1] @ UCC_psi).item()
            r_z = numpy.real(UCC_psi.conj().T @ r_op[k,2] @ UCC_psi).item()
            r_sq = r_x*r_x + r_y*r_y + r_z*r_z
        E += r_coeff*r_sq
    # here we add constraint contribution from electronic <S^2> if desired
    if bool_S2:
        S2 = numpy.real(UCC_psi.conj().T @ S2_op @ UCC_psi).item()
        E += S2_coeff*(S2 - S2_target)**2
    return E


def fci_wf_analysis(log, state, n_qubit_tot, str_lab, tol=1e-10):
    wf_dim = state.shape[0]
    basis_list = list(itertools.product([0,1], repeat=n_qubit_tot))
    hilbert_dim = len(basis_list)
    log.note("\n       ---" + str_lab + " Wave Function Data ---")
    log.note("  FCI coefficient            Basis State")
    for i in range(wf_dim):
        if abs(state[i]) > tol:
            real_coeff = numpy.real(state[i].item())
            log.note("%18.15f     %s", real_coeff, str(basis_list[i]))
    return

def construct_UCC_wf(nt_amp, t_amp, tau, tau_dag, psi_HF):
    ham_dim = psi_HF.shape[0]
    UCC_ansatz = sparse.csr_matrix((ham_dim, ham_dim), dtype=complex)
    for i in range(nt_amp):
        UCC_ansatz += t_amp[i]*(tau[i]-tau_dag[i])
    UCC_psi = sparse.linalg.expm_multiply(UCC_ansatz, psi_HF)
    return UCC_psi

def column_vec(vector):
    if vector.ndim == 1:
        vector = vector[:, numpy.newaxis]
    elif vector.shape[0] == 1:
        vector = vector.T
    return vector

def pc_projection(n_qubit_e, n_qubit_p, nocc_e):
    '''particle conserving projection'''
    def qubit_state(val):
        return numpy.array([1, 0]) if val == 0 else numpy.array([0, 1])

    npqb = 0
    for i in range(len(n_qubit_p)):
        npqb += n_qubit_p[i]
    n_qubit_tot = n_qubit_e + npqb
    dim = 2**n_qubit_tot
    tmp_state = []
    final_state = []

    # electrons
    e_basis_states = []
    for config in itertools.combinations(range(n_qubit_e), nocc_e):
        state = numpy.zeros(n_qubit_e)
        for idx in config:
            state[idx] = 1
        e_state = numpy.array([qubit_state(s) for s in state])
        e_tensor_prod = e_state[0]
        for qubit in e_state[1:]:
            e_tensor_prod = numpy.kron(e_tensor_prod, qubit)         
        e_basis_states.append(e_tensor_prod)

    # protons
    p_basis_list = []
    for i in range(len(n_qubit_p)):
        p_basis_states = []
        for config in itertools.combinations(range(n_qubit_p[i]), 1):
            state = numpy.zeros(n_qubit_p[i])
            for idx in config:
                state[idx] = 1
            p_state = numpy.array([qubit_state(s) for s in state])
            p_tensor_prod = p_state[0]
            for qubit in p_state[1:]:
                p_tensor_prod = numpy.kron(p_tensor_prod, qubit)         
            p_basis_states.append(p_tensor_prod)
        p_basis_list.append(p_basis_states)

    # take first tensor product |e> \times |p1>
    for i in range(len(e_basis_states)):
        for j in range(len(p_basis_list[0])):
            comp_state = numpy.kron(e_basis_states[i], p_basis_list[0][j])
            final_state.append(comp_state)

    # remaining tensor products
    for k in range(1,len(n_qubit_p)):
        new_tmp_state = []
        for i in range(len(final_state)):
            for j in range(len(p_basis_list[k])):
                comp_state = numpy.kron(final_state[i],p_basis_list[k][j])
                new_tmp_state.append(comp_state)
        final_state = new_tmp_state

    proj_op = sparse.csr_matrix((dim, dim), dtype=complex)
    for i in range(len(final_state)):
        vector = sparse.csr_matrix(final_state[i])
        vector = column_vec(vector)
        proj_op += vector @ vector.conj().T

    return proj_op
