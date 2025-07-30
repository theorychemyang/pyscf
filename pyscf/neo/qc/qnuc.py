#!/usr/bin/env python
#
import copy
import numpy
import time
from pyscf import lib
from pyscf import scf, ao2mo
import scipy
from scipy.optimize import minimize, NonlinearConstraint
from pyscf import neo
import itertools
import math
from scipy import sparse
from scipy.sparse.linalg import eigsh
from pyscf.lib import logger
from pyscf.neo.qc.elec import number_operator_e, t1_op_e, t2_op_e,\
                              Ham_elec, parse_mf_elec, HF_state,\
                              make_rdm1_e, vN_entropy_1rdm_e
from pyscf.neo.qc import lib as qc_lib

def dump_neo_qc_info(self, log):
    nvirt_so = self.n_qubit_e - self.nocc_so_e
    nvirt_qn = 0
    n_qnuc_so = 0
    nocc_p = len(self.mf_nuc)
    for i in range(len(self.mf_nuc)):
        n_qnuc_so += self.n_qubit_p[i]
        nvirt_qn += self.n_qubit_p[i] - 1 # always only 1 for distinguishable particle
    log.note("\noccupied spin-orbitals electrons: %g", self.nocc_so_e)
    log.note("virtual  spin-orbitals electrons: %g", nvirt_so)
    log.note("total    spin-orbitals electrons: %g", self.n_qubit_e)
    log.note("\noccupied spin-orbitals quantum nuclei: %g", len(self.mf_nuc))
    log.note("virtual  spin-orbitals quantum nuclei: %g", nvirt_qn)
    log.note("total    spin-orbitals quantum nuclei: %g", n_qnuc_so)
    log.note("\nnuclear    qubits: %g", self.n_qubit_tot-self.n_qubit_e)
    log.note("electronic qubits: %g", self.n_qubit_e)
    log.note("total      qubits: %g", self.n_qubit_tot)
    log.note("Fock space dimension: %g", 2**self.n_qubit_tot)

    Hamiltonian = self.hamiltonian
    psi_HF = self.psi_hf
    S2_op = self.s2_op
    pos_op = self.r_op

    E_HF = numpy.real(psi_HF.conj().T @ Hamiltonian @ psi_HF)
    HF_S2 = numpy.real(psi_HF.conj().T @ S2_op @ psi_HF)
    log.note("\nCalculating <psi_HF| H |psi_HF> as sanity check")
    log.note("Hartree-Fock Energy: %-18.15f",E_HF.item())
    log.note("Hartree-Fock S_e^2: %-18.15f",HF_S2.item())
    for k in range(len(self.mf_nuc)):
        rx_hf = numpy.real(psi_HF.conj().T @ pos_op[k,0] @ psi_HF).item()
        ry_hf = numpy.real(psi_HF.conj().T @ pos_op[k,1] @ psi_HF).item()
        rz_hf = numpy.real(psi_HF.conj().T @ pos_op[k,2] @ psi_HF).item()
        nuc_lab = self.mf_nuc[k].mol.atom_symbol(k)
        log.note("Hartree-Fock <%s>: %-15.7e %-15.7e %-15.7e", nuc_lab, rx_hf, ry_hf, rz_hf)

    fci_exc_e = math.comb(self.n_qubit_e, self.nocc_so_e)
    log.note("\nparticle-conserving determinants electrons: %g", fci_exc_e)
    fci_exc_p = []
    fci_exc_tot = fci_exc_e
    for k in range(len(self.mf_nuc)):
        fci_tmp_p = math.comb(self.n_qubit_p[k], 1)
        nuc_lab = self.mf_nuc[k].mol.atom_symbol(k)
        log.note("particle-conserving determinants %s: %g",nuc_lab, fci_tmp_p)
        fci_exc_tot *= fci_tmp_p
        fci_exc_p.append(fci_tmp_p)
    log.note("\nNumber of quantum particles: %g",self.nocc_so_e + nocc_p)
    log.note("Hilbert-subspace-particle-conserving dimension: %g",fci_exc_tot)
    return

def cas_selection(mf_nuc, cas_orb_nuc):
    '''returns list nuc_coeff containing
    MO matrices with orbitals according 
    to cas_orb_nuc specification
    '''
    nuc_coeff = []
    for i in range(len(mf_nuc)):
        coeff_mat = mf_nuc[i].mo_coeff[:, cas_orb_nuc]
        nuc_coeff.append(coeff_mat)
    return nuc_coeff

def axis_finder(mol):
    ''' simple code to find if we have a linear molecule
        if so, report along what axis '''
    from pyscf.hessian.thermo import rotation_const, _get_rotor_type
    # find if this is a linear molecule
    mass = mol.mass
    atom_coords = mol.atom_coords()
    mass_center = numpy.einsum('z,zx->x', mass, atom_coords) / mass.sum()
    atom_coords = atom_coords - mass_center
    rot_const = rotation_const(mass, atom_coords, 'GHz')
    rotor_type = _get_rotor_type(rot_const)
    if rotor_type == 'LINEAR':
        # if a linear molecule, detect if the molecule is along x/y/z axis
        if numpy.abs(atom_coords[:,0]).max() < 1e-6:
            if numpy.abs(atom_coords[:,1]).max() < 1e-6:
                axis = 2
            elif numpy.abs(atom_coords[:,2]).max() < 1e-6:
                axis = 1
        elif numpy.abs(atom_coords[:,1]).max() < 1e-6:
            if numpy.abs(atom_coords[:,2]).max() < 1e-6:
                axis = 0
        else:
            # if not along an axis, warn
            print('This is molecule is linear, but was not put along x/y/z axis. Symmetry will be OFF.',
                  flush=True)
            return None
    else:
        print('Not a linear molecule. Symmetry will be OFF.', flush=True)
        return None
    return axis

def get_qubit_qnuc(nuc_coeff):
    '''returns (1) list of number of qubits associated with each quantum nucleus
    (2) total number of quantum nuclear qubits
    '''
    n_qubit_p = []
    n_qubit_p_tot = 0
    for i in range(len(nuc_coeff)):
        n_qubit_p.append(nuc_coeff[i].shape[1])
        n_qubit_p_tot += n_qubit_p[i]
    return n_qubit_p, n_qubit_p_tot

def fci_index(C_FCI, nocc_e, mf_nuc, Num_op_e, Num_op_p, S2_op, bool_ge=False):
    '''Parses state or set of states based on particle conservation
    in each Hilbert subspace.
    returns: 1) fci_idx: array of indices for states of interest
             2) fci_pnum: array containing total particle number
                          for corresponding state index in fci_idx
                          (these should all be the same)
             3) fci_s2: array containing <S_e^2> for
                        corresponding state index in fci_idx
    '''
    num_states = C_FCI.shape[1]
    pnum_tol = 1e-5
    fci_idx = []
    fci_pnum = []
    fci_s2 = []
    for i in range(num_states):
        if (num_states > 1):
            wf_tmp = C_FCI[:,i]
        else:
            wf_tmp = C_FCI

        fci_pnum_tot = 0.0
        S2 = numpy.real(wf_tmp.conj().T @ S2_op @ wf_tmp).item()
        fci_pnum_e = numpy.real(wf_tmp.conj().T @ Num_op_e @ wf_tmp).item()
        pnum_diff_e = abs(fci_pnum_e - 1.0*nocc_e)
        pnum_diff_p = 0.0
        fci_pnum_p_sum = 0.0
        for k in range(len(mf_nuc)):
            fci_pnum_p = numpy.real(wf_tmp.conj().T @ Num_op_p[k] @ wf_tmp).item()
            pnum_diff_p += abs(fci_pnum_p - 1.0)
            fci_pnum_p_sum += fci_pnum_p
        fci_pnum_tot = fci_pnum_e + fci_pnum_p_sum

        if ((pnum_diff_e + pnum_diff_p) < pnum_tol):
            fci_idx.append(i)
            fci_pnum.append(fci_pnum_tot)
            fci_s2.append(S2)
            if bool_ge:
                return fci_idx, fci_pnum, fci_s2
    return fci_idx, fci_pnum, fci_s2

def basis_vec_list(particle_id, n_qubit_e, n_qubit_p, tot_id):
    '''returns ordered list of basis vectors associated
    with a given quantum particle.
    Basis vectors are constructed with identity matrices
    according to:
    I^{\otimes(n_qubit_left)} |ket_of_interest> \otimes I^{\otimes(n_qubit_right)}
    '''
    n_qubit_left = 0
    n_qubit_right = 0
    if particle_id == str(0):
        n_qubit_poi = n_qubit_e
    else:
        n_qubit_poi = n_qubit_p[int(particle_id)-1]
    for c in tot_id:
        ci = int(c)
        if ci < int(particle_id):
            if ci==0: n_qubit_left += n_qubit_e
            else: n_qubit_left += n_qubit_p[ci-1]
        elif ci > int(particle_id):
            n_qubit_right += n_qubit_p[ci-1]

    II = sparse.csr_matrix(numpy.array([[1.0, 0.0],[0.0, 1.0]]))
    q0 = sparse.csr_matrix(numpy.array([[1.0+0.0j],[0.0+0.0j]]))
    q1 = sparse.csr_matrix(numpy.array([[0.0+0.0j],[1.0+0.0j]]))
    qubit_basis_list = []
    basis_list = list(itertools.product([0,1], repeat=n_qubit_poi))
    hilbert_dim = len(basis_list)
    vec_bas_list = numpy.zeros((n_qubit_poi, hilbert_dim),dtype=object)
    for i in range(hilbert_dim):
        for j in range(n_qubit_poi):
            mat_id = basis_list[i][j]
            if mat_id == 0: mat = q0
            else: mat = q1
            vec_bas_list[j,i] = mat
    for i in range(hilbert_dim):
        mat0 = vec_bas_list[0,i]
        mat1 = vec_bas_list[1,i]
        a = sparse.kron(mat0,mat1,'csr')
        for j in range(2,n_qubit_poi):
            mat3 = vec_bas_list[j,i]
            a = sparse.kron(a,mat3,'csr')
        for l in range(n_qubit_left):
            a = sparse.kron(II,a,'csr')
        for l in range(n_qubit_right):
            a = sparse.kron(a,II,'csr')
        qubit_basis_list.append(a)

    return qubit_basis_list

def vN_entropy(vector, str_id, n_qubit_e, n_qubit_p, tol=1e-15):
    S_vN = 0.0
    vector = qc_lib.column_vec(vector)
    rho = vector @ vector.conj().T
    n_ptyp = len(n_qubit_p) + 1 # number of particle types
    # trace over full composite Hilbert space
    if len(str_id) == n_ptyp:
        e_rho = scipy.linalg.eigh(rho)
        for i in range(len(e_rho)):
            if e_rho[i] > tol:
                S_vN -= e_rho[i]*numpy.log(e_rho[i])
    # trace over subspace
    else:
        n_qubit_tot = n_qubit_e
        for i in range(len(n_qubit_p)):
            n_qubit_tot += n_qubit_p[i]
        n_qubit_reduced = n_qubit_tot
        tot_id = ''
        trace_id = ''
        for i in range(n_ptyp):
            tot_id += str(i)
            bool_match = False
            for j in range(len(str_id)):
                if str(i) == str_id[j]:
                    bool_match = True
            if not bool_match: trace_id += str(i)
        rho_prev = copy.deepcopy(rho)
        for k in range(len(trace_id)):
            particle_id = trace_id[k]
            if particle_id == str(0):
                n_qubit_reduced -= n_qubit_e
            else:
                n_qubit_reduced -= n_qubit_p[int(particle_id)-1]
            subsys_bas = basis_vec_list(particle_id, n_qubit_e, n_qubit_p, tot_id)
            hilbert_dim_tmp = 2**n_qubit_reduced
            rho_tmp = sparse.csr_matrix((hilbert_dim_tmp, hilbert_dim_tmp), dtype=complex)
            for i in range(len(subsys_bas)):
                rho_tmp += subsys_bas[i].conj().T @ rho_prev @ subsys_bas[i]
            rho_prev = copy.deepcopy(rho_tmp)
            tot_id = tot_id.replace(particle_id,'')
        e_rho_e = scipy.linalg.eigh(rho_tmp, eigvals_only=True)
        for i in range(len(e_rho_e)):
            if e_rho_e[i] > tol:
                S_vN -= e_rho_e[i]*numpy.log(e_rho_e[i])

    return S_vN

def make_rdm1_n(vector, create, destroy):
    '''Make 1-RDM for quantum nuclei
       rho_ij = < a_j^+ a_i >

       Returns:
           List of 1-RDMs for quantum nuclei
    '''
    n_ptyp = len(create[1:])
    rho = []
    vector = qc_lib.column_vec(vector)
    for k in range(n_ptyp):
        p_dim = len(create[k+1])
        rho_tmp = numpy.zeros((p_dim, p_dim), dtype=complex)
        for i in range(p_dim):
            for j in range(p_dim):
                rho_tmp[i,j] = (vector.conj().T @ create[k+1][j] @ destroy[k+1][i] @ vector).item()
        rho.append(rho_tmp)
    return rho

def vN_entropy_1rdm_p(log, vector, str_method, mf_nuc, create, destroy, tol=1e-15):
    preamble = "SvN 1RDM "
    rho = make_rdm1_n(vector, create, destroy)
    for k in range(len(mf_nuc)):
        S_vN = 0.0
        lab_prot = mf_nuc[k].mol.atom_symbol(k)
        label = preamble + str_method + ' ' + lab_prot + ": %15.8e"
        e_rho = scipy.linalg.eigh(rho[k], eigvals_only=True)
        for i in range(len(e_rho)):
            if e_rho[i] > tol:
                S_vN -= e_rho[i]*numpy.log(e_rho[i])
        log.note(label, S_vN)
    return

def compute_vN_entropy(log, state, n_qubit_e, n_qubit_p, mf_nuc,
                       str_con, str_method, str_vn_level):
    if not (str_vn_level=="electron" or str_vn_level=="all"):
        raise NotImplementedError("invalid von Neumann entropy input")
    pnum = len(mf_nuc) + 1

    label = str_con + "SvN " + str_method + " "
    e_lab = "e"
    tot_lab = e_lab

    if str_vn_level == 'electron': # calculate electron vN entropy only
        SvN = vN_entropy(state, '0', n_qubit_e, n_qubit_p)
        log.note(label + e_lab + ': %15.8e', SvN)
    elif pnum == 2: # 2 particle case
        for i in range(len(mf_nuc)+1):
            str_id_i = str(i)
            if i==0:
                lab_id_i = e_lab
            else:
                lab_id_i = mf_nuc[i-1].mol.atom_symbol(i-1)
            SvN = vN_entropy(state, str_id_i, n_qubit_e, n_qubit_p)
            label_i = label + lab_id_i + ": %15.8e"
            log.note(label_i, SvN)
    elif pnum == 3: # 3 particle case
        for i in range(len(mf_nuc)+1):
            str_id_i = str(i)
            if i==0:
                lab_id_i = e_lab
            else:
                lab_id_i = mf_nuc[i-1].mol.atom_symbol(i-1)
            SvN = vN_entropy(state, str_id_i, n_qubit_e, n_qubit_p)
            label_i = label + lab_id_i + ": %15.8e"
            log.note(label_i, SvN)
            for j in range(i+1,len(mf_nuc)+1):
                lab_id_j = mf_nuc[j-1].mol.atom_symbol(j-1)
                lab_id_ij = lab_id_i + lab_id_j
                str_id_j = str(j)
                str_id_ij = str_id_i + str_id_j
                label_ij = label + lab_id_ij + ": %15.8e"
                SvN = vN_entropy(state, str_id_ij, n_qubit_e, n_qubit_p)
                log.note(label_ij, SvN)
    else: # more than 3 particles, only calculate electron case and give reason
        SvN = vN_entropy(state, '0', n_qubit_e, n_qubit_p)
        log.note(label + e_lab + ': %12.7e', SvN)
        log.note("NOTE: von Neumann entropy for more than 3 particles")
        log.note("      is only programmed for electrons: -Tr[rho_e ln(rho_e)]")
    return

def number_operator(n_qubit_tot, n_qubit_e, n_qubit_p, create, destroy, str_kind):
    '''
    Can return single operator or list of operators
    if 'composite': single operator on full composite Hilbert space
    if 'electron' : single operator for electronic subspace
    if 'proton'   : list of proton operators on each proton's subspace
    Returns:
           requested number operator,
    '''
    dim = 2**n_qubit_tot
    Num_op = sparse.csr_matrix((dim, dim),dtype=complex)
    Num_op_list = []
    if str_kind == 'composite':
        for pid in range(len(n_qubit_p)+1):
            for p in range(len(create[pid])):
                Num_op += create[pid][p] @ destroy[pid][p]
        Num_op_return = Num_op
    elif str_kind == 'electron':
        Num_op_return = number_operator_e(n_qubit_tot, n_qubit_e, create, destroy)
    elif str_kind == 'proton':
        for k in range(len(n_qubit_p)):
            Num_op_p = sparse.csr_matrix((dim, dim),dtype=complex)
            for p in range(n_qubit_p[k]):
                Num_op_p += create[k+1][p] @ destroy[k+1][p]
            Num_op_list.append(Num_op_p)
        Num_op_return = Num_op_list

    return Num_op_return

def position_operator(mf_nuc, nuc_coeff, n_qubit_p, create, destroy, bool_c_shift):
    ''' 
    r_op[0,0]: second quantized r_x operator for quantum nucleus 0
    r_op[0,1]: second quantized r_y operator for quantum nucleus 0
    r_op[0,2]: second quantized r_z operator for quantum nucleus 0

    Returns: r_op
         num_nuc x 3 matrix of matrices
    '''
    r_op = numpy.zeros((len(mf_nuc), 3),dtype=object)
    rmo = numpy.zeros((len(mf_nuc), 3),dtype=object)
    for i in range(len(mf_nuc)):
        mo_coeff = nuc_coeff[i]
        if not bool_c_shift:
            int1p_r = mf_nuc[i].mol.intor_symmetric('int1e_r', comp=3)
        else:
            r_target = mf_nuc[i].mol.atom_coord(mf_nuc[i].mol.atom_index)
            s1p = mf_nuc[i].get_ovlp(mf_nuc[i].mol)
            int1p_r = mf_nuc[i].mol.intor_symmetric('int1e_r', comp=3) \
                      - numpy.asarray([r_target[i] * s1p for i in range(3)])
        rmo[i,0] = mo_coeff.conj().T @ int1p_r[0] @ mo_coeff
        rmo[i,1] = mo_coeff.conj().T @ int1p_r[1] @ mo_coeff
        rmo[i,2] = mo_coeff.conj().T @ int1p_r[2] @ mo_coeff
    for k in range(len(mf_nuc)):
        for p in range(n_qubit_p[k]):
            for q in range(n_qubit_p[k]):
                idx = [p, q]
                op1 = qc_lib.ca1_op(idx, create, destroy, k+1)
                r_op[k,0] += rmo[k,0][p,q]*op1
                r_op[k,1] += rmo[k,1][p,q]*op1
                r_op[k,2] += rmo[k,2][p,q]*op1
    return r_op

def momentum_operator(mf_nuc, nuc_coeff, n_qubit_p, create, destroy):
    '''
    mom_op[0,0]: second quantized p_x operator for quantum nucleus 0
    mom_op[0,1]: second quantized p_y operator for quantum nucleus 0
    mom_op[0,2]: second quantized p_z operator for quantum nucleus 0

    int1e_ipovlp returns ao matrix of <a|nabla|b> along some direction.
    Thus, need to multiply final result by imaginary factor below since
    p = -i*hbar*nabla

    Returns: mom_op
        num_nuc x 3 matrix of matrices
    '''
    mom_op = numpy.zeros((len(mf_nuc), 3),dtype=object)
    mom_mo = numpy.zeros((len(mf_nuc), 3),dtype=object)
    for i in range(len(mf_nuc)):
        mo_coeff = nuc_coeff[i]
        int1p_mom = -1.0j*mf_nuc[i].mol.intor('int1e_ipovlp', comp=3).transpose(0,2,1)
        mom_mo[i,0] = mo_coeff.conj().T @ int1p_mom[0] @ mo_coeff
        mom_mo[i,1] = mo_coeff.conj().T @ int1p_mom[1] @ mo_coeff
        mom_mo[i,2] = mo_coeff.conj().T @ int1p_mom[2] @ mo_coeff
    for k in range(len(mf_nuc)):
        for p in range(n_qubit_p[k]):
            for q in range(n_qubit_p[k]):
                idx = [p, q]
                op1 = qc_lib.ca1_op(idx, create, destroy, k+1)
                mom_op[k,0] += mom_mo[k,0][p,q]*op1
                mom_op[k,1] += mom_mo[k,1][p,q]*op1
                mom_op[k,2] += mom_mo[k,2][p,q]*op1
    return mom_op

def t1_op_p(nuc_coeff, create, destroy):
    nocc_so = 1 # distinguishable so always 1 occupied quantum nuclear orbital
    tau = []
    tau_dag = []
    for k in range(len(nuc_coeff)):
        nvirt_so = nuc_coeff[k].shape[1] - nocc_so
        tmp = []
        tmp_dag = []
        for i in range(nocc_so):
            for a in range(nvirt_so):
                idx = [a + nocc_so, i]
                t1 = qc_lib.ca1_op(idx, create, destroy, k + 1)
                t1_dag = t1.conj().T
                tmp.append(t1)
                tmp_dag.append(t1_dag)
        tau.append(tmp)
        tau_dag.append(tmp_dag)
    return tau, tau_dag

def t2_op_ep(nocc_so_e, nvirt_so_e, num_p, create, destroy, t1_e, t1_p):
    tau = []
    tau_dag = []
    for k in range(num_p):
        for i in range(len(t1_e)):
            for j in range(len(t1_p[k])):
                t2 = t1_e[i] @ t1_p[k][j]
                t2_dag = t2.conj().T
                tau.append(t2)
                tau_dag.append(t2_dag)
    return tau, tau_dag

def t2_op_pp(num_p, create, destroy, t1_p):
    tau = []
    tau_dag = []
    for k in range(num_p):
        k_len = len(t1_p[k])
        tmp_k = []
        tmp_k_dag = []
        for l in range(k+1, num_p):
            l_len = len(t1_p[l])
            tmp_l = []
            tmp_l_dag = []
            for i in range(k_len):
                for j in range(l_len):
                    t2 = t1_p[k][i] @ t1_p[l][j]
                    t2_dag = t2.conj().T
                    tmp_l.append(t2)
                    tmp_l_dag.append(t2_dag)
            tmp_k.append(tmp_l)
            tmp_k_dag.append(tmp_l_dag)
        tau.append(tmp_k)
        tau_dag.append(tmp_k_dag)
    return tau, tau_dag

def t3_op_2e1p(nocc_so_e, nvirt_so_e, create, destroy, t2_e, t1_p):
    tau = []
    tau_dag = []
    for i in range(len(t2_e)):
        for k in range(len(t1_p)):
            for j in range(len(t1_p[k])):
                t3 = t2_e[i] @ t1_p[k][j]
                t3_dag = t3.conj().T
                tau.append(t3)
                tau_dag.append(t3_dag)
    return tau, tau_dag

def t3_op_1e1p1p(nocc_so_e, nvirt_so_e, create, destroy, t1_e, t2_p):
    tau = []
    tau_dag = []
    for i in range(len(t1_e)):
        for k in range(len(t2_p)):
            for l in range(len(t2_p[k])):
                for j in range(len(t2_p[k][l])):
                    t3 = t1_e[i] @ t2_p[k][l][j]
                    t3_dag = t3.conj().T
                    tau.append(t3)
                    tau_dag.append(t3_dag)
    return tau, tau_dag

def t4_op_ep(nocc_so_e, nvirt_so_e, create, destroy, t2_e, t2_p):
    tau = []
    tau_dag = []
    for i in range(len(t2_e)):
        for k in range(len(t2_p)):
            for l in range(len(t2_p[k])):
                for j in range(len(t2_p[k][l])):
                    t4 = t2_e[i] @ t2_p[k][l][j]
                    t4_dag = t4.conj().T
                    tau.append(t4)
                    tau_dag.append(t4_dag)
    return tau, tau_dag

def ucc_op_list_neo(nocc_so_e, nvirt_so_e, create, destroy, nuc_coeff, ucc_level):
    '''Pure quantum nuclear operators return nested lists that are indexed
    according to quantum nuclear identity.
    Example: t1_op_p returns list[k][z] where k is nucleus, z is operator
             t2_op_pp returns list [k][l][z] where k,l is nuclear combo, z is operator
    Pure electron operators and mixed ep operators return single list of operators
    '''
    num_p = len(nuc_coeff)
    if (ucc_level > (nocc_so_e + num_p)):
        raise NotImplementedError("Requested UCC excitation level exceeds number of "
                                  "particles")
    if ucc_level > 2 and (nocc_so_e > 2 or num_p > 2):
        raise NotImplementedError("UCC level > 2 only implemented for 2e/2p systems")

    # electrons
    t1_e, t1_e_dag = t1_op_e(nocc_so_e, nvirt_so_e, create, destroy)
    t2_e, t2_e_dag = t2_op_e(nocc_so_e, nvirt_so_e, create, destroy)
    tau = t1_e + t2_e
    tau_dag = t1_e_dag + t2_e_dag

    # nested list for 1 particle nuclear excitations
    t1_p, t1_p_dag =  t1_op_p(nuc_coeff, create, destroy)
    for k in range(len(t1_p)):
        tau += t1_p[k]
        tau_dag += t1_p_dag[k]

    # nested list for qnuc-qnuc interaction if more than one nucleus
    if (num_p > 1):
        t2_pp, t2_pp_dag =  t2_op_pp(num_p, create, destroy, t1_p)
        for k in range(len(t2_pp)):
            for l in range(len(t2_pp[k])):
                tau += t2_pp[k][l]
                tau_dag += t2_pp_dag[k][l]

    # mixed ep term
    t2_ep, t2_ep_dag = t2_op_ep(nocc_so_e, nvirt_so_e, num_p, create, destroy, t1_e, t1_p)
    tau += t2_ep
    tau_dag += t2_ep_dag

    # two types of 3 particle excitations
    if ucc_level > 2:
        # t2_e*t1_p
        t3_2e1p, t3_2e1p_dag = t3_op_2e1p(nocc_so_e, nvirt_so_e, create, destroy, t2_e, t1_p)
        tau += t3_2e1p
        tau_dag += t3_2e1p_dag
        # t1_e*t1_p1*t1_p2
        t3_1e1p1p, t3_1e1p1p_dag = t3_op_1e1p1p(nocc_so_e, nvirt_so_e, create, destroy, t1_e, t2_pp)
        tau += t3_1e1p1p
        tau_dag += t3_1e1p1p_dag

    # t2_e*t1_p1*t1_p2
    if ucc_level > 3:
        t4_ep, t4_ep_dag = t4_op_ep(nocc_so_e, nvirt_so_e, create, destroy, t2_e, t2_pp)
        tau += t4_ep
        tau_dag += t4_ep_dag
    return tau, tau_dag

def HF_state_cneo(n_occ_e, n_qubit_e, nocc_p, n_qubit_p):
    psi_HF_e = HF_state(n_occ_e, n_qubit_e)
    psi_HF_p = []
    q0 = numpy.array([[1.0+0.0j],[0.0+0.0j]])
    q1 = numpy.array([[0.0+0.0j],[1.0+0.0j]])
    for k in range(nocc_p):
        a = numpy.kron(q1,q0)
        for i in range(2,n_qubit_p[k]):
            a = numpy.kron(a,q0)
        psi_HF_p.append(a)
    psi_HF_tot = numpy.kron(psi_HF_e,psi_HF_p[0])
    for k in range(1,nocc_p):
        psi_HF_tot = numpy.kron(psi_HF_tot,psi_HF_p[k])
    return psi_HF_tot

def qnuc_nn(mf, nuc_coeff):
    mf_nuc = mf.mf_nuc
    eri_nn_mo = []
    for i in range(len(mf_nuc)):
        eri_nn_mo.append([None]*len(mf_nuc))
    for i in range(len(mf_nuc)):
        for j in range(i+1,len(mf_nuc)):
            mf1 = mf_nuc[i]
            mf2 = mf_nuc[j]
            eri_nn = mf._eri_nn[i][j]
            coeff1 = nuc_coeff[i]
            coeff2 = nuc_coeff[j]
            num_i = coeff1.shape[1]
            num_j = coeff2.shape[1]
            charge = mf.mol.atom_charge(mf1.mol.atom_index) * mf.mol.atom_charge(mf2.mol.atom_index)
            c_ij = charge*ao2mo.incore.general(eri_nn, (coeff1, coeff1, coeff2, coeff2),compact=False)\
                   .reshape(num_i, num_i, num_j, num_j)
            eri_nn_mo[i][j] = c_ij
    return eri_nn_mo

def qnuc_ne(mf, mf_nuc, nuc_coeff, moa_e, mob_e):
    eri_ne_mo = []
    num_e = moa_e.shape[1]
    for i in range(len(mf_nuc)):
        charge = -1.0*mf.mol.atom_charge(mf_nuc[i].mol.atom_index)
        coeff_n = nuc_coeff[i]
        num_n = coeff_n.shape[1]
        eri_ne = mf._eri_ne[i]
        c_ne_a = charge*ao2mo.incore.general(eri_ne, (coeff_n, coeff_n, moa_e, moa_e),compact=False)\
                 .reshape(num_n, num_n, num_e, num_e)
        c_ne_b = charge*ao2mo.incore.general(eri_ne, (coeff_n, coeff_n, mob_e, mob_e),compact=False)\
                 .reshape(num_n, num_n, num_e, num_e)
        c_ne = c_ne_a + c_ne_b
        eri_ne_mo.append(c_ne)
    return eri_ne_mo

def Ham_neo(mf, nuc_coeff, moa, mob, ea, eb, create, destroy, n_qubit_e, n_qubit_p, n_qubit_tot, tol=1e-12):

    mf_elec = mf.mf_elec
    mf_nuc = mf.mf_nuc

    if len(mf_nuc)>0:
        hao = mf_elec.hcore_static # need static (true) hcore
    else:
        hao = mf_elec.get_hcore()
    eri_ao = mf_elec._eri
    Ham_ee = Ham_elec(mf, moa, mob, ea, eb, eri_ao, hao, create, destroy, n_qubit_e)
    E_nuc = mf_elec.energy_nuc()

    # electron-qnuc interaction
    motota, mototb, motot = qc_lib.mo_to_spinor(moa, mob, ea, eb)
    eri_ne_mo = qnuc_ne(mf, mf_nuc, nuc_coeff, motota, mototb)

    # quantum nn interaction if there is more than 1 nucleus
    if (len(mf_nuc) > 1):
        eri_nn_mo = qnuc_nn(mf, nuc_coeff)

    # list of core hamiltonians for quantum nuclei
    hmo_p = []
    for i in range(len(mf_nuc)):
        hao_p = mf_nuc[i].hcore_static # need static (true) hcore
        hmo_tmp_p = nuc_coeff[i].transpose() @ hao_p @ nuc_coeff[i]
        hmo_p.append(hmo_tmp_p)

    h_dim = 2**n_qubit_tot
    Hamiltonian = sparse.csr_matrix((h_dim, h_dim),dtype=complex)

    # electronic contribution
    Hamiltonian += Ham_ee

    # quantum nuclear core
    for k in range(len(mf_nuc)):
        for p in range(n_qubit_p[k]):
            for q in range(n_qubit_p[k]):
                hval_p = abs(hmo_p[k][p,q])
                if hval_p < tol: continue
                idx = [p, q]
                op_hpq = qc_lib.ca1_op(idx, create, destroy, k+1)
                Hamiltonian += hmo_p[k][p,q]*op_hpq

    # electron-qnuc interaction
    for k in range(len(mf_nuc)):
        for p in range(n_qubit_p[k]):
            for q in range(n_qubit_p[k]):
                for r in range(n_qubit_e):
                    for s in range(n_qubit_e):
                        neval = abs(eri_ne_mo[k][p,q,r,s])
                        if neval < tol: continue
                        idxe = [r, s]
                        idxp = [p, q]
                        ope = qc_lib.ca1_op(idxe, create, destroy, 0)
                        opp = qc_lib.ca1_op(idxp, create, destroy, k+1)
                        Hamiltonian += eri_ne_mo[k][p,q,r,s]*(ope @ opp)

    # quantum nn interaction if there is more than one quantum nucleus
    if (len(mf_nuc)) > 1:
        for k in range(len(mf_nuc)):
            for l in range(k+1,len(mf_nuc)):
                for p in range(n_qubit_p[k]):
                    for q in range(n_qubit_p[k]):
                        for r in range(n_qubit_p[l]):
                            for s in range(n_qubit_p[l]):
                                nnval = abs(eri_nn_mo[k][l][p,q,r,s])
                                if nnval < tol: continue
                                idx1 = [p, q]
                                idx2 = [r, s]
                                op1 = qc_lib.ca1_op(idx1, create, destroy, k+1)
                                op2 = qc_lib.ca1_op(idx2, create, destroy, l+1)
                                Hamiltonian += eri_nn_mo[k][l][p,q,r,s]*(op1 @ op2)

    # Add nuclear repulsion
    Hamiltonian += E_nuc*qc_lib.kron_I(n_qubit_tot)
    return Hamiltonian

def fci_constrained(f, Ham0, mf_nuc, nocc_e, Num_op_e, Num_op_p, n_qubit_e, n_qubit_p, S2_op,
                    r_op, axis, full_diag, proj_op):
    '''Constrained FCI procedure is hard coded (for now) to find the ground state.
    '''
    ham_loc = copy.deepcopy(Ham0)
    if axis is not None:
        r_final = numpy.zeros(len(mf_nuc))
    else:
        r_final = numpy.zeros(len(mf_nuc)*3)

    for k in range(len(mf_nuc)):
        if axis is not None:
            ham_axis = f[k]*r_op[k,axis]
            ham_loc += ham_axis
        else:
            ham_r_x = f[3*k+0]*r_op[k,0]
            ham_r_y = f[3*k+1]*r_op[k,1]
            ham_r_z = f[3*k+2]*r_op[k,2]
            ham_loc += ham_r_x + ham_r_y + ham_r_z

    if full_diag:
        E_FCI, C_FCI = numpy.linalg.eigh(ham_loc.todense())
    else:
        # Project out non-particle-conserving blocks in Fock space
        ham_loc = proj_op @ ham_loc @ proj_op
        E_FCI, C_FCI = sparse.linalg.eigsh(ham_loc, k=1, which='SA')
    fci_idx, fci_pnum, fci_s2 = fci_index(C_FCI, nocc_e, mf_nuc, Num_op_e, Num_op_p, S2_op, True)
    ia = fci_idx[0]
    for k in range(len(mf_nuc)):
        if axis is not None:
            r_final[k] = numpy.real(C_FCI[:,ia].conj().T @ r_op[k,axis] @ C_FCI[:,ia]).item()
        else:
            r_final[3*k+0] = numpy.real(C_FCI[:,ia].conj().T @ r_op[k,0] @ C_FCI[:,ia]).item()
            r_final[3*k+1] = numpy.real(C_FCI[:,ia].conj().T @ r_op[k,1] @ C_FCI[:,ia]).item()
            r_final[3*k+2] = numpy.real(C_FCI[:,ia].conj().T @ r_op[k,2] @ C_FCI[:,ia]).item()
    return r_final

def constrained_ucc_energy(t, tau, tau_dag, nt_amp, psi_HF, num_nuc, r_op, r_coeff,
        bool_S2, S2_op, S2_coeff, S2_target):

    ham_dim = psi_HF.shape[0]
    UCC_ansatz = sparse.csr_matrix((ham_dim, ham_dim), dtype=complex)

    for i in range(nt_amp):
        UCC_ansatz += t[i]*(tau[i]-tau_dag[i])
    UCC_psi = sparse.linalg.expm_multiply(UCC_ansatz, psi_HF)
    E_add = 0.0
    for k in range(num_nuc):
        r_x = numpy.real(UCC_psi.conj().T @ r_op[k,0] @ UCC_psi).item()
        r_y = numpy.real(UCC_psi.conj().T @ r_op[k,1] @ UCC_psi).item()
        r_z = numpy.real(UCC_psi.conj().T @ r_op[k,2] @ UCC_psi).item()
        r_sq = r_x*r_x + r_y*r_y + r_z*r_z
        E_add += r_coeff*r_sq
    # here we add constraint contribution from electronic <S^2> if desired
    if bool_S2:
        S2 = numpy.real(UCC_psi.conj().T @ S2_op @ UCC_psi).item()
        E_add += S2_coeff*(S2 - S2_target)**2
    return E_add

def JW_array_cneo(n_qubit_e, n_qubit_p, n_qubit_p_tot, op_id):
    '''
    Returns: tot_list
        Nested list of operators according to tot_list[pid][k]
        pid = particle identification, electrons are first (0)
        k = element of a set of creation/annihilation operators associated with pid
        i.e., k \in {a_1, a_2, ...} or
            k \in {a_1^\dagger, a_2^\dagger, ...}
    '''
    tot_list = []

    # Pauli Matrices
    II = sparse.csr_matrix(numpy.array([[1.0, 0.0],[0.0, 1.0]]))
    X = sparse.csr_matrix(numpy.array([[0.0, 1.0],[1.0, 0.0]]))
    Y = sparse.csr_matrix(numpy.array([[0.0, -1.0j],[1.0j, 0.0]]))
    Z = sparse.csr_matrix(numpy.array([[1.0, 0.0],[0.0, -1.0]]))
    s_plus = (X + 1.0j*Y)/2.0
    s_minus = (X - 1.0j*Y)/2.0
    if (op_id=="creation"):
        ladder_op = s_minus
    elif (op_id=="annihilation"):
        ladder_op = s_plus
    else:
        raise NotImplementedError("Invalid JW operator designation")

    # electronic loop
    op_list = []
    for i in range(n_qubit_e):
        if i == 0:
            mat1 = ladder_op
            mat2 = II
        elif i == 1:
            mat1 = Z
            mat2 = ladder_op
        else:
            mat1 = Z
            mat2 = Z
        a = sparse.kron(mat1,mat2,'csr')
        for j in range(2,n_qubit_e):
            if i == j:
                mat3=ladder_op
            elif (i<j):
                mat3 = II
            else:
                mat3 = Z
            a = sparse.kron(a,mat3,'csr')
        for k in range(n_qubit_p_tot):
            a = sparse.kron(a,II,'csr')
        op_list.append(a)
    tot_list.append(op_list)
    # protonic loop
    #Z = II; #test setting Z to identity for protons
    mat0 = qc_lib.kron_I(n_qubit_e)
    n_qubit_previous = 0
    for k in range(len(n_qubit_p)): # k loop over distinguishable nuclei
        op_list = []
        n_qubit_after = n_qubit_p_tot - n_qubit_previous - n_qubit_p[k]
        mat_tmp = copy.deepcopy(mat0)
        for m in range(n_qubit_previous):
            mat_tmp = sparse.kron(mat_tmp,II,'csr')
        for i in range(n_qubit_p[k]):
            if i == 0:
                mat1 = ladder_op
                mat2 = II
            elif i == 1:
                mat1 = Z
                mat2 = ladder_op
            else:
                mat1 = Z
                mat2 = Z
            a = sparse.kron(mat1,mat2,'csr')
            for j in range(2,n_qubit_p[k]):
                if i == j:
                    mat3 = ladder_op
                elif (i<j):
                    mat3 = II
                else:
                    mat3 = Z
                a = sparse.kron(a,mat3,'csr')
            a = sparse.kron(mat_tmp,a,'csr')
            for l in range(n_qubit_after):
                a = sparse.kron(a,II,'csr')
            op_list.append(a)
        n_qubit_previous += n_qubit_p[k]
        tot_list.append(op_list)
    return tot_list

def entropy(n_qubit_e, n_qubit_p, num_e, indices, state_vec, tol=1e-15):

    if isinstance(indices, (int, numpy.integer)):
        indices = [indices]

    bl = qc_lib.basis_list(n_qubit_e, n_qubit_p, num_e, indices, complement=True)
    vec = qc_lib.column_vec(state_vec)
    rho = vec @ vec.conj().T

    rho_red = sum(mat.conj().T @ rho @ mat for mat in bl)
    e_red = scipy.linalg.eigh(rho_red, eigvals_only=True)
    svn = sum(eig*numpy.log(eig) for eig in e_red if eig>tol)

    return -svn

class QC_NEO_BASE(lib.StreamObject):

    def __init__(self, mf, cas_orb_nuc=None, c_shift=False):
        # these are fixed, not inputs
        self.verbose = mf.verbose
        self.mf = mf
        self.mf_nuc = None
        self.n_qubit_e = None
        self.n_qubit_p = None
        self.n_qubit_tot = None
        self.hamiltonian = None
        self.create = None
        self.destroy = None
        self.num_op_e = None
        self.num_op_p = None
        self.s2_op = None
        self.sz_op = None
        self.a_id = None
        self.b_id = None
        self.r_op = None
        self.psi_hf = None
        self.nocc_so_e = None
        self.nuc_coeff = None
        # input
        self.c_shift = c_shift
        self.cas_orb_nuc = cas_orb_nuc

        if not isinstance(mf, neo.HF):
            raise NotImplementedError('NEO QC Protocol must take NEO mf object')

    def qc_components(self, c_shift=None):
        log = logger.new_logger(self.mf, self.verbose)
        time_qc_components = logger.process_clock()

        if c_shift is None:
            c_shift = self.c_shift
        else:
            self.c_shift = c_shift

        if c_shift:
            log.note("\nOrigin of each quantum nuclear position operator has been shifted")
            log.note("to the corresponding nuclear basis function center")
        if self.mf.mf_positron is not None:
            raise NotImplementedError('QC CNEO code does not currently work for positrons')

        mf = self.mf
        mf_nuc = mf.mf_nuc
        mf_elec = mf.mf_elec
        cas_orb_nuc = self.cas_orb_nuc

        # parse electronic mf object
        moa, mob, ea, eb, na, nb, noa, nob = parse_mf_elec(mf_elec)

        # mixed alpha/beta overlap
        s1e = mf.get_ovlp()
        s_ab = moa.conj().T @ s1e @ mob
        s_ba = mob.conj().T @ s1e @ moa

        nvirta  = na - noa
        nvirtb  = nb - nob
        nocc_so_e = noa + nob
        nvirt_so_e = nvirta + nvirtb
        ntot_so_e = nocc_so_e + nvirt_so_e
        nocc_p = len(mf_nuc)

        # CAS procedure, select all specified MOs
        # Applies CAS selection to all nuclei uniformly
        if cas_orb_nuc is None:
            nuc_coeff = []
            for i in range(len(mf_nuc)):
                nuc_coeff.append(mf_nuc[i].mo_coeff)
        else:
            log.note("\nCAS selection of nuclear orbitals: %s", cas_orb_nuc)
            nuc_coeff = cas_selection(mf_nuc, cas_orb_nuc)

        n_qubit_e = ntot_so_e
        n_qubit_tot = n_qubit_e
        n_qubit_p_tot = 0
        n_qubit_p = []
        for i in range(len(mf_nuc)):
            n_qubit_p.append(nuc_coeff[i].shape[1])
            n_qubit_tot += n_qubit_p[i]
            n_qubit_p_tot += n_qubit_p[i]

        if n_qubit_tot > 14:
            log.warn("Number of qubits is %g, recommended MAX is 14, ideal is <= 12", n_qubit_tot)

        # form creation/annihilation lists
        create = JW_array_cneo(ntot_so_e, n_qubit_p, n_qubit_p_tot, "creation")
        destroy = JW_array_cneo(ntot_so_e, n_qubit_p, n_qubit_p_tot, "annihilation")

        # construct Hamiltonian
        Hamiltonian = Ham_neo(mf, nuc_coeff, moa, mob, ea, eb, create, destroy, n_qubit_e, n_qubit_p, n_qubit_tot)

        etot  = numpy.hstack((ea,eb))
        Num_op_e = number_operator(n_qubit_tot, n_qubit_e, n_qubit_p, create, destroy, 'electron')
        Num_op_p = number_operator(n_qubit_tot, n_qubit_e, n_qubit_p, create, destroy, 'proton')
        S2_op, Sz_op, a_id, b_id  = qc_lib.S_squared_operator(n_qubit_e, s_ab, s_ba,
                                                              etot.argsort(), create, destroy)
        pos_op = position_operator(mf_nuc, nuc_coeff, n_qubit_p, create, destroy, c_shift)
        psi_HF = HF_state_cneo(nocc_so_e, n_qubit_e, nocc_p, n_qubit_p)

        self.mf_nuc = mf_nuc
        self.n_qubit_e = n_qubit_e
        self.n_qubit_p = n_qubit_p
        self.n_qubit_tot = n_qubit_tot
        self.hamiltonian = Hamiltonian
        self.create = create
        self.destroy = destroy
        self.num_op_e = Num_op_e
        self.num_op_p = Num_op_p
        self.s2_op = S2_op
        self.sz_op = Sz_op
        self.a_id = a_id
        self.b_id = b_id
        self.r_op = pos_op
        self.psi_hf = psi_HF
        self.nocc_so_e = nocc_so_e
        self.nuc_coeff = nuc_coeff

        log.timer("Construction of QC components: ", time_qc_components)
        return

class QC_FCI_NEO(QC_NEO_BASE):
    ''' FCI calculation using matrix diagonalization in qubit basis

    Examples:

    >>> from pyscf import scf
    >>> from pyscf import neo
    >>> from pyscf.neo import qc
    >>> mol = neo.M(atom='H 0 0 0; H 0.74 0 0', basis='STO-3G',
    >>>             quantum_nuc = [0,1], nuc_basis = '1s1p', cart=True, spin=0)
    >>> mf = neo.HF(mol)
    >>> mf.scf()
    >>> qc_mf = qc.QC_FCI_NEO(mf)
    >>> qc_mf.kernel()
    ------- Quantum Computing Protocol -------
    FCI Energy: -1.057677072326754
    '''

    def __init__(self, mf, cas_orb_nuc=None, c_shift=False):
        QC_NEO_BASE.__init__(self, mf, cas_orb_nuc, c_shift)
        self.e = None
        self.c = None
        self.num = None
        self.s2 = None
        self.num_e = None

    def kernel(self, full_diag=False, nstates=1, num_e=None):
        log = logger.new_logger(self.mf, self.verbose)
        time_kernel = logger.process_clock()

        qc_lib.dump_qc_header(log)

        # get all quantum computing components needed for calculation
        self.qc_components()

        # FCI Hamiltonian
        ham_kern = self.hamiltonian
        ham_dim = ham_kern.shape[0]

        # Fock space target for number of electrons
        if num_e is None:
            num_e = self.nocc_so_e
            self.num_e = num_e
        else:
            self.num_e = num_e

        # Diagonalize
        if full_diag:
            E_FCI, C_FCI = numpy.linalg.eigh(ham_kern.todense()) #eigh should use dense matrix
        else:
            # Project out all non-particle-conserving blocks in Fock space
            proj_op = qc_lib.pc_projection(self.n_qubit_e, self.n_qubit_p, num_e)
            ham_kern = proj_op @ ham_kern @ proj_op
            E_FCI, C_FCI = sparse.linalg.eigsh(ham_kern, k=nstates, which='SA')
            # guarantee proper sorting
            sorted_indices = numpy.argsort(E_FCI)
            E_FCI = E_FCI[sorted_indices]
            C_FCI = C_FCI[:, sorted_indices]

        fci_idx, fci_pnum, fci_s2 = fci_index(C_FCI, num_e, self.mf_nuc,
                                              self.num_op_e, self.num_op_p, self.s2_op)

        E_FCI_final = []
        C_FCI_final = numpy.zeros((ham_dim, len(fci_idx)), dtype=complex)
        for i in range(len(fci_idx)):
            E_FCI_final.append(E_FCI[fci_idx[i]])
            C_FCI_final[:,i] = C_FCI[:, fci_idx[i]].reshape(-1)

        # eigsh may give complex coefficients, so convert to real and renormalize
        # --> this is always a valid procedure if the Hamiltonian is real-valued
        C_FCI_final = qc_lib.convert_coeff(C_FCI_final)

        self.e = E_FCI_final
        self.c = C_FCI_final
        self.num = fci_pnum
        self.s2 = fci_s2

        log.timer("FCI Diagonalization and Sorting: ", time_kernel)
        log.note("\nFCI Energy: %-20.15f", E_FCI_final[0])
        return E_FCI_final, C_FCI_final, fci_pnum, fci_s2

    def analyze(self, nstates=1, vn_level='electron', verbose=None):
        if verbose is None: verbose = self.verbose
        log = logger.new_logger(self.mf, verbose)

        qc_lib.dump_analysis_header(log)
        qc_lib.dump_spin_order(log, self.a_id, self.b_id)
        dump_neo_qc_info(self,log)

        # Print FCI particle-conserving states
        log.note("\n------------------- FCI RESULTS -------------------")
        log.note("  State        Energy        S_e^2   Particle Number")
        for i in range(len(self.e)):
            log.note("%5i %20.15f %7.3f %11.3f", i, self.e[i], self.s2[i], self.num[i])

        if nstates > len(self.e): nstates = len(self.e)
        for i in range(nstates):
            log.note("\n------------------- STATE %g -------------------\n", i)
            c_iter = self.c[:,i]
            log.note("E FCI %g: %-20.15f", i, self.e[i])
            log.note("<S_e^2>: %-20.15f", self.s2[i])
            log.note("\nQuantum Nuclear Expectation Values: ")
            for k in range(len(self.mf_nuc)):
                rx_fci = numpy.real(c_iter.conj().T @ self.r_op[k,0] @ c_iter).item()
                ry_fci = numpy.real(c_iter.conj().T @ self.r_op[k,1] @ c_iter).item()
                rz_fci = numpy.real(c_iter.conj().T @ self.r_op[k,2] @ c_iter).item()
                nuc_lab = self.mf_nuc[k].mol.atom_symbol(k)
                log.note("<%s>: %-15.7e %-15.7e %-15.7e", nuc_lab, rx_fci, ry_fci, rz_fci)

            qc_lib.fci_wf_analysis(log, self.c[:,i], self.n_qubit_tot, '')

            log.note("\n--- von Neumann Entropy Data ---")
            compute_vN_entropy(log, self.c[:,i], self.n_qubit_e, self.n_qubit_p,
                               self.mf_nuc, '','FCI', vn_level)
            log.note("\n--- 1-RDM von Neumann Entropy Data ---")
            vN_entropy_1rdm_e(log, self.c[:,i], 'FCI', self.create, self.destroy)
            vN_entropy_1rdm_p(log, self.c[:,i], 'FCI', self.mf_nuc, self.create, self.destroy)

        return

    def entropy(self, indices, state_vec=None):
        if state_vec is None:
            state_vec = self.c[:,0]
        return entropy(self.n_qubit_e, self.n_qubit_p, self.num_e, indices, state_vec)

    def make_rdm1_e(self, state_vec=None):
        if state_vec is None:
            state_vec = self.c[:,0]
        return make_rdm1_e(state_vec, self.create, self.destroy)

    def make_rdm1_n(self, state_vec=None):
        if state_vec is None:
            state_vec = self.c[:,0]
        return make_rdm1_n(state_vec, self.create, self.destroy)

class QC_CFCI_NEO(QC_NEO_BASE):
    ''' Constrained FCI calculation using matrix diagonalization in qubit basis

    Examples:

    >>> from pyscf import scf
    >>> from pyscf import neo
    >>> from pyscf.neo import qc
    >>> mol = neo.M(atom='H 0 0 0; H 0.74 0 0', basis='STO-3G',
    >>>             quantum_nuc = [1], nuc_basis = '2s1p', cart=True, spin=0)
    >>> mf = neo.HF(mol)
    >>> mf.scf()
    >>> qc_mf = qc.QC_CFCI_NEO(mf)
    >>> qc_mf.kernel()
    ------- Quantum Computing Protocol -------
    Origin of each quantum nuclear position operator has been shifted
    to the corresponding nuclear basis function center
    Lagrange multiplier starting guess:
    H [0. 0. 0.]
    opt.success: True
    opt.message: The solution converged.
    Constrained FCI Energy: -1.096593699077798
    '''

    def __init__(self, mf, cas_orb_nuc=None, c_shift=True):
        QC_NEO_BASE.__init__(self, mf, cas_orb_nuc, c_shift)
        self.e = None
        self.c = None
        self.num = None
        self.num_e = None
        self.s2 = None
        self.f = None

    def kernel(self, lm_guess='hf', f=None, full_diag=False, num_e=None):
        log = logger.new_logger(self.mf, self.verbose)
        time_kernel = logger.process_clock()

        qc_lib.dump_qc_header(log)

        # get all quantum computing components needed for calculation
        self.qc_components()

        mf = self.mf
        mf_nuc = self.mf_nuc
        num_op_e = self.num_op_e
        num_op_p = self.num_op_p
        s2_op = self.s2_op
        r_op = self.r_op
        n_qubit_e = self.n_qubit_e
        n_qubit_p = self.n_qubit_p

        # For CNEO-FCI, ham_kern is not to be modified
        ham_kern = self.hamiltonian
        ham_dim = ham_kern.shape[0]

        if f is None:
            if isinstance(mf, neo.CDFT) and lm_guess=='hf':
                f = copy.deepcopy(mf.f[mf_nuc[0].mol.atom_index])
                for i in range(1,len(mf_nuc)):
                    ia = mf_nuc[i].mol.atom_index
                    f = numpy.hstack((f, mf.f[ia]))
            elif lm_guess=='random':
                f = 0.1*(2.0*numpy.random.rand(len(mf_nuc)*3) - 1.0)
            elif lm_guess=='zero':
                f = numpy.zeros(len(mf_nuc)*3)
            else:
                f = numpy.zeros(len(mf_nuc)*3)

        log.note("\nLagrange multiplier starting guess: ")
        for i in range(len(mf_nuc)):
            nuc_lab = mf_nuc[i].mol.atom_symbol(i)
            log.note("%s %s ", nuc_lab, f[3*i:3*(i+1)])

        # find out if molecule is linear along some axis
        self.axis = axis_finder(mf.mol)
        axis = self.axis
        if axis is not None:
            f_ = numpy.zeros(len(mf_nuc))
            for i in range(len(mf_nuc)):
                f_[i] = f[3*i + axis]
            f.fill(0.0)
        else:
            f_ = f

        # Fock space target for number of electrons
        if num_e is None:
            num_e = self.nocc_so_e
            self.num_e = num_e
        else:
            self.num_e = num_e

        # Projector for particle-conserving blocks in Fock space
        proj_op = qc_lib.pc_projection(n_qubit_e, n_qubit_p, num_e)

        opt = scipy.optimize.root(fci_constrained, f_, args=(ham_kern, mf_nuc, num_e, num_op_e,
                                  num_op_p, n_qubit_e, n_qubit_p, s2_op, r_op, axis, full_diag, proj_op),
                                  method = 'hybr')
        log.note("\nopt.success: %s", opt.success)
        log.note("opt.message: %s \n", opt.message)

        if opt.success:
            if axis is not None:
                for i in range(len(mf_nuc)):
                    f[3*i + axis] = opt.x[i]
            else:
                f = opt.x
            ham_loc = copy.deepcopy(ham_kern)
            for k in range(len(mf_nuc)):
                ham_r_x = f[3*k+0]*r_op[k,0]
                ham_r_y = f[3*k+1]*r_op[k,1]
                ham_r_z = f[3*k+2]*r_op[k,2]
                ham_loc += ham_r_x + ham_r_y + ham_r_z

            if full_diag:
                E_CFCI, C_CFCI = numpy.linalg.eigh(ham_loc.todense())
            else:
                ham_loc = proj_op @ ham_loc @ proj_op
                E_CFCI, C_CFCI = sparse.linalg.eigsh(ham_loc, k=1, which='SA')
            cfci_idx, cfci_pnum, cfci_s2 = fci_index(C_CFCI, num_e, mf_nuc, num_op_e,
                                                     num_op_p, s2_op, True)
            cia = cfci_idx[0]
            cpnum = cfci_pnum[0]
            cs2 = cfci_s2[0]

            # eigsh may give complex coefficients, so convert to real and renormalize
            # --> this is always a valid procedure if the Hamiltonian is real-valued
            C_CFCI_final = qc_lib.convert_coeff(C_CFCI[:,cia].reshape(ham_dim, 1))
            E_CFCI_final = numpy.real(C_CFCI_final.conj().T @ ham_kern @ C_CFCI_final).item()

            self.e = E_CFCI_final
            self.c = C_CFCI_final
            self.num = cpnum
            self.s2 = cs2
            self.f = f
            log.note("\nConstrained FCI Energy: %-20.15f", E_CFCI_final)
        else:
            log.note("Constrained FCI Failed...\n")

        log.timer("Constrained FCI Procedure: ", time_kernel)
        return E_CFCI_final, C_CFCI_final, cpnum, cs2

    def analyze(self, vn_level='electron', verbose=None):
        if verbose is None: verbose = self.verbose
        log = logger.new_logger(self.mf, verbose)

        qc_lib.dump_analysis_header(log)
        qc_lib.dump_spin_order(log, self.a_id, self.b_id)
        dump_neo_qc_info(self,log)

        log.note("\n------------------- STATE 0 -------------------\n")
        log.note("E CFCI: %-20.15f", self.e)
        log.note("<S_e^2>: %-20.15f", self.s2)
        log.note("Particles: %-11.7f", self.num)
        log.note("\nQuantum Nuclear Expectation Values: ")
        for k in range(len(self.mf_nuc)):
            nuc_lab = self.mf_nuc[k].mol.atom_symbol(k)
            rx_fci = numpy.real(self.c.conj().T @ self.r_op[k,0] @ self.c).item()
            ry_fci = numpy.real(self.c.conj().T @ self.r_op[k,1] @ self.c).item()
            rz_fci = numpy.real(self.c.conj().T @ self.r_op[k,2] @ self.c).item()
            log.note("<%s>: %-15.7e %-15.7e %-15.7e", nuc_lab, rx_fci, ry_fci, rz_fci)

        log.note("\nConstrained FCI Lagrange multiplier: ")
        for i in range(len(self.mf_nuc)):
            nuc_lab = self.mf_nuc[i].mol.atom_symbol(i)
            f_tmp = self.f[3*i:3*(i+1)]
            log.note("%s: %s", nuc_lab, f_tmp)

        qc_lib.fci_wf_analysis(log, self.c, self.n_qubit_tot, '')

        log.note("\n--- von Neumann Entropy Data ---")
        compute_vN_entropy(log, self.c, self.n_qubit_e, self.n_qubit_p,
                           self.mf_nuc, '','CFCI', vn_level)
        log.note("\n--- 1-RDM von Neumann Entropy Data ---")
        vN_entropy_1rdm_e(log, self.c, 'CFCI', self.create, self.destroy)
        vN_entropy_1rdm_p(log, self.c, 'CFCI', self.mf_nuc, self.create, self.destroy)
        return

    def entropy(self, indices, state_vec=None):
        if state_vec is None:
            state_vec = self.c
        return entropy(self.n_qubit_e, self.n_qubit_p, self.num_e, indices, state_vec)

    def make_rdm1_e(self, state_vec=None):
        if state_vec is None:
            state_vec = self.c
        return make_rdm1_e(state_vec, self.create, self.destroy)

    def make_rdm1_n(self, state_vec=None):
        if state_vec is None:
            state_vec = self.c
        return make_rdm1_n(state_vec, self.create, self.destroy)

class QC_UCC_NEO(QC_NEO_BASE):
    ''' Unitary coupled-cluster calculation using minimization in qubit basis

    Examples:

    >>> from pyscf import scf
    >>> from pyscf import neo
    >>> from pyscf.neo import qc
    >>> mol = neo.M(atom='H 0 0 0; H 0.74 0 0', basis='STO-3G',
    >>>             quantum_nuc = [0,1], nuc_basis = '1s1p', cart=True, spin=0)
    >>> mf = neo.HF(mol)
    >>> mf.scf()
    >>> qc_mf = qc.QC_UCC_NEO(mf)
    >>> qc_mf.kernel()
    ------- Quantum Computing Protocol -------
    --- UCCSD Calculation ---
    number of cluster amplitudes: 44
    res.success: True
    res.message: Optimization terminated successfully.
    UCC Energy: -1.057617041568029
    '''

    def __init__(self, mf, cas_orb_nuc=None, c_shift=False):
        QC_NEO_BASE.__init__(self, mf, cas_orb_nuc, c_shift)
        self.e = None
        self.c = None
        self.num = None
        self.s2 = None
        self.t = None

    def kernel(self, ucc_level=2, adv_init=False, conv_tol=None, method='BFGS'):
        log = logger.new_logger(self.mf, self.verbose)
        time_kernel = logger.process_clock()

        qc_lib.dump_qc_header(log)

        # get all quantum computing components needed for calculation
        self.qc_components()

        ham_kern = self.hamiltonian
        psi_HF = self.psi_hf
        S2_op = self.s2_op
        nocc_so_e = self.nocc_so_e
        mf_nuc = self.mf_nuc
        nuc_coeff = self.nuc_coeff
        create = self.create
        destroy = self.destroy
        n_qubit_e = self.n_qubit_e
        nvirt_so_e = n_qubit_e - nocc_so_e

        tau, tau_dag = ucc_op_list_neo(nocc_so_e, nvirt_so_e, create, destroy,
                                       nuc_coeff, ucc_level)
        nt_amp = len(tau)
        t_amp = numpy.zeros(nt_amp)

        if adv_init:
            t1_e, t1_e_dag = t1_op_e(nocc_so_e, nvirt_so_e, create, destroy)
            t2_e, t2_e_dag = t2_op_e(nocc_so_e, nvirt_so_e, create, destroy)
            tau_e = t1_e + t2_e
            tau_dag_e = t1_e_dag + t2_e_dag
            nt_amp_e = len(tau_e)
            t_amp_e = numpy.zeros(nt_amp_e)
            res_init = minimize(lambda z: qc_lib.UCC_energy(z, ham_kern, tau_e, tau_dag_e, nt_amp_e,
                           psi_HF), t_amp_e, tol=None, method='BFGS')
            theta_e = res_init.x
            log.note("\nAdvanced cluster amplitude initialization")
            log.note("Number of cluster amplitudes used in initialization: %g", nt_amp_e)
            log.note("\nres_init.success: %s", res_init.success)
            log.note("res_init.message: %s \n", res_init.message)
            if res_init.success:
                for i in range(len(theta_e)):
                    t_amp[i] = theta_e[i]
                E_init = res_init.fun
                log.note("Initialized UCC Energy: %-20.15f", E_init)

        if method == 'SLSQP':
            maxiter = 500
            opt_dic = {'maxiter': maxiter}
        else:
            opt_dic = None
        qc_lib.dump_ucc_level(log, ucc_level)
        log.note("number of cluster amplitudes: %g", nt_amp)
        res = minimize(lambda z: qc_lib.UCC_energy(z, ham_kern, tau, tau_dag, nt_amp,
                                            psi_HF), t_amp, tol=conv_tol, options=opt_dic,
                                            method=method)
        log.note("\nres.success: %s", res.success)
        log.note("res.message: %s \n", res.message)
        if res.success:
            theta = res.x
            E_UCC = res.fun
            psi_UCC = qc_lib.construct_UCC_wf(nt_amp, theta, tau, tau_dag, psi_HF)
            ucc_idx, ucc_pnum, ucc_s2 = fci_index(psi_UCC, nocc_so_e, mf_nuc,
                                                  self.num_op_e, self.num_op_p, S2_op)

            self.e = E_UCC
            self.c = psi_UCC
            self.num = ucc_pnum[0]
            self.s2 = ucc_s2[0]
            self.t = theta
            log.note("\nUCC Energy: %-20.15f", E_UCC)
        else:
            log.note("\nUCC Failed...")
        log.timer("UCC Procedure: ", time_kernel)
        return E_UCC, psi_UCC, ucc_pnum[0], ucc_s2[0]

    def analyze(self, vn_level='electron', verbose=None):
        if verbose is None: verbose = self.verbose
        log = logger.new_logger(self.mf, verbose)

        qc_lib.dump_analysis_header(log)
        qc_lib.dump_spin_order(log, self.a_id, self.b_id)
        dump_neo_qc_info(self,log)

        log.note("\n------------------- STATE 0 -------------------\n")
        log.note("E UCC: %-20.15f", self.e)
        log.note("<S_e^2>: %-20.15f", self.s2)
        log.note("Particles: %-11.7f", self.num)
        log.note("Amplitudes: \n%s\n", self.t)
        log.note("Quantum Nuclear Expectation Values: ")
        for k in range(len(self.mf_nuc)):
            nuc_lab = self.mf_nuc[k].mol.atom_symbol(k)
            rx_ucc = numpy.real(self.c.conj().T @ self.r_op[k,0] @ self.c).item()
            ry_ucc = numpy.real(self.c.conj().T @ self.r_op[k,1] @ self.c).item()
            rz_ucc = numpy.real(self.c.conj().T @ self.r_op[k,2] @ self.c).item()
            log.note("<%s>: %-15.7e %-15.7e %-15.7e", nuc_lab, rx_ucc, ry_ucc, rz_ucc)

        log.note("\n--- von Neumann Entropy Data ---")
        compute_vN_entropy(log, self.c, self.n_qubit_e, self.n_qubit_p,
                           self.mf_nuc, '','UCC', vn_level)
        log.note("\n--- 1-RDM von Neumann Entropy Data ---")
        vN_entropy_1rdm_e(log, self.c, 'UCC', self.create, self.destroy)
        vN_entropy_1rdm_p(log, self.c, 'UCC', self.mf_nuc, self.create, self.destroy)
        return

    def entropy(self, indices, state_vec=None):
        if state_vec is None:
            state_vec = self.c
        return entropy(self.n_qubit_e, self.n_qubit_p, self.nocc_so_e, indices, state_vec)

    def make_rdm1_e(self, state_vec=None):
        if state_vec is None:
            state_vec = self.c
        return make_rdm1_e(state_vec, self.create, self.destroy)

    def make_rdm1_n(self, state_vec=None):
        if state_vec is None:
            state_vec = self.c
        return make_rdm1_n(state_vec, self.create, self.destroy)

class QC_CUCC_NEO(QC_NEO_BASE):
    ''' Constrained unitary coupled-cluster calculation using
        constrained minimization in qubit basis

    Examples:

    >>> from pyscf import scf
    >>> from pyscf import neo
    >>> from pyscf.neo import qc
    >>> mol = neo.M(atom='H 0 0 0; H 0.74 0 0', basis='STO-3G',
    >>>             quantum_nuc = [1], nuc_basis = '2s1p', cart=True, spin=0)
    >>> mf = neo.CDFT(mol)
    >>> mf.mf_elec.xc = 'HF'
    >>> mf.scf()
    >>> qc_mf = qc.QC_CUCC_NEO(mf)
    >>> qc_mf.kernel()
    ------- Quantum Computing Protocol -------
    Origin of each quantum nuclear position operator has been shifted
    to the corresponding nuclear basis function center
    --- Constrained UCCSD Calculation ---
    number of cluster amplitudes: 25
    res.success: True
    res.message: Optimization terminated successfully
    Constrained UCC Energy: -1.096541479568951
    '''

    def __init__(self, mf, cas_orb_nuc=None, c_shift=True):
        QC_NEO_BASE.__init__(self, mf, cas_orb_nuc, c_shift)
        self.e = None
        self.c = None
        self.num = None
        self.s2 = None
        self.t = None

    def kernel(self, ucc_level=2, adv_init=False, nuc_adv_init=False, neo_init=False, conv_tol=None,
               r_coeff=None, S2_constraint=False, S2_coeff=None, S2_target=0.0, method='BFGS'):
        log = logger.new_logger(self.mf, self.verbose)
        time_kernel = logger.process_clock()

        qc_lib.dump_qc_header(log)

        # get all quantum computing components needed for calculation
        self.qc_components()

        ham_kern = self.hamiltonian
        psi_HF = self.psi_hf
        S2_op = self.s2_op
        nocc_so_e = self.nocc_so_e
        mf_nuc = self.mf_nuc
        nuc_coeff = self.nuc_coeff
        create = self.create
        destroy = self.destroy
        n_qubit_e = self.n_qubit_e
        r_op = self.r_op
        nvirt_so_e = n_qubit_e - nocc_so_e

        tau, tau_dag = ucc_op_list_neo(nocc_so_e, nvirt_so_e, create, destroy,
                                       nuc_coeff, ucc_level)
        nt_amp = len(tau)
        t_amp = numpy.zeros(nt_amp)

        if adv_init:
            t1_e, t1_e_dag = t1_op_e(nocc_so_e, nvirt_so_e, create, destroy)
            t2_e, t2_e_dag = t2_op_e(nocc_so_e, nvirt_so_e, create, destroy)
            tau_e = t1_e + t2_e
            tau_dag_e = t1_e_dag + t2_e_dag
            if nuc_adv_init:
                num_p = len(nuc_coeff)
                t1_p, t1_p_dag =  t1_op_p(nuc_coeff, create, destroy)
                for k in range(len(t1_p)):
                    tau_e += t1_p[k]
                    tau_dag_e += t1_p_dag[k]
                if (num_p > 1):
                    t2_pp, t2_pp_dag =  t2_op_pp(num_p, create, destroy, t1_p)
                    for k in range(len(t2_pp)):
                        for l in range(len(t2_pp[k])):
                            tau_e += t2_pp[k][l]
                            tau_dag_e += t2_pp_dag[k][l]
            nt_amp_e = len(tau_e)
            t_amp_e = numpy.zeros(nt_amp_e)
            res_init = minimize(lambda z: qc_lib.UCC_energy(z, ham_kern, tau_e, tau_dag_e, nt_amp_e,
                           psi_HF), t_amp_e, tol=None, method='BFGS')
            theta_e = res_init.x
            log.note("\nAdvanced cluster amplitude initialization")
            log.note("Number of cluster amplitudes used in initialization: %g", nt_amp_e)
            log.note("\nres_init.success: %s", res_init.success)
            log.note("res_init.message: %s \n", res_init.message)
            if res_init.success:
                for i in range(len(theta_e)):
                    t_amp[i] = theta_e[i]
                E_init = res_init.fun
                log.note("Initialized UCC Energy: %-20.15f", E_init)

            if neo_init and res_init.success:
                nt_amp_neo = len(tau)
                res_neo = minimize(lambda z: qc_lib.UCC_energy(z, ham_kern, tau, tau_dag, nt_amp_neo,
                                psi_HF), t_amp, tol=None, method='BFGS')
                theta_neo = res_neo.x
                log.note("\nNEO cluster amplitude initialization")
                log.note("\nres_neo.success: %s", res_neo.success)
                log.note("res_neo.message: %s \n", res_neo.message)
                if res_neo.success:
                    for i in range(len(theta_neo)):
                        t_amp[i] = theta_neo[i]
                    E_init_neo = res_neo.fun
                    log.note("Initialized NEO UCC Energy: %-20.15f", E_init_neo)

        qc_lib.dump_ucc_level(log, ucc_level, True)
        log.note("number of cluster amplitudes: %g", nt_amp)

        if r_coeff is None:
            r_coeff = 1e4
        if S2_constraint and S2_coeff is None:
            S2_coeff = 1e4

        res = minimize(lambda z: qc_lib.UCC_energy_shift(z, ham_kern, tau, tau_dag, nt_amp,
                psi_HF, len(mf_nuc), r_op, r_coeff, S2_constraint, S2_op, S2_coeff, S2_target),
                t_amp, tol=conv_tol, method=method)
        log.note("\nres.success: %s", res.success)
        log.note("res.message: %s \n", res.message)

        if res.success:
            theta = res.x
            psi_UCC = qc_lib.construct_UCC_wf(nt_amp, theta, tau, tau_dag, psi_HF)
            E_UCC = numpy.real(psi_UCC.conj().T @ ham_kern @ psi_UCC).item()
            ucc_idx, ucc_pnum, ucc_s2 = fci_index(psi_UCC, nocc_so_e, mf_nuc,
                                                  self.num_op_e, self.num_op_p, S2_op)
            self.e = E_UCC
            self.c = psi_UCC
            self.num = ucc_pnum[0]
            self.s2 = ucc_s2[0]
            self.t = theta
            log.note("\nConstrained UCC Energy: %-20.15f", E_UCC)
        else:
            log.note("\nConstrained UCC Failed...")
        log.timer("Constrained UCC Procedure: ", time_kernel)
        return E_UCC, psi_UCC, ucc_pnum[0], ucc_s2[0]

    def analyze(self, vn_level='electron', verbose=None):
        if verbose is None: verbose = self.verbose
        log = logger.new_logger(self.mf, verbose)

        qc_lib.dump_analysis_header(log)
        qc_lib.dump_spin_order(log, self.a_id, self.b_id)
        dump_neo_qc_info(self,log)

        log.note("\n------------------- STATE 0 -------------------\n")
        log.note("E CUCC: %-20.15f", self.e)
        log.note("<S_e^2>: %-20.15f", self.s2)
        log.note("Particles: %-11.7f", self.num)
        log.note("Amplitudes: \n%s\n", self.t)
        log.note("Quantum Nuclear Expectation Values: ")
        for k in range(len(self.mf_nuc)):
            nuc_lab = self.mf_nuc[k].mol.atom_symbol(k)
            rx_ucc = numpy.real(self.c.conj().T @ self.r_op[k,0] @ self.c).item()
            ry_ucc = numpy.real(self.c.conj().T @ self.r_op[k,1] @ self.c).item()
            rz_ucc = numpy.real(self.c.conj().T @ self.r_op[k,2] @ self.c).item()
            log.note("<%s>: %-15.7e %-15.7e %-15.7e", nuc_lab, rx_ucc, ry_ucc, rz_ucc)

        log.note("\n--- von Neumann Entropy Data ---")
        compute_vN_entropy(log, self.c, self.n_qubit_e, self.n_qubit_p,
                           self.mf_nuc, '','CUCC', vn_level)
        log.note("\n--- 1-RDM von Neumann Entropy Data ---")
        vN_entropy_1rdm_e(log, self.c, 'CUCC', self.create, self.destroy)
        vN_entropy_1rdm_p(log, self.c, 'CUCC', self.mf_nuc, self.create, self.destroy)
        return

    def entropy(self, indices, state_vec=None):
        if state_vec is None:
            state_vec = self.c
        return entropy(self.n_qubit_e, self.n_qubit_p, self.nocc_so_e, indices, state_vec)

    def make_rdm1_e(self, state_vec=None):
        if state_vec is None:
            state_vec = self.c
        return make_rdm1_e(state_vec, self.create, self.destroy)

    def make_rdm1_n(self, state_vec=None):
        if state_vec is None:
            state_vec = self.c
        return make_rdm1_n(state_vec, self.create, self.destroy)
