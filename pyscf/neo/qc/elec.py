#!/usr/bin/env python
#
import copy
import numpy
from pyscf import lib
from pyscf import scf, ao2mo
import scipy
from scipy.optimize import minimize
from pyscf import neo
import itertools
import math
from scipy import sparse
from pyscf.lib import logger
from pyscf.neo.qc import lib as qc_lib

def dump_elec_qc_info(self, log):
    nvirt_so = self.n_qubit - self.nocc_so
    log.note("\noccupied spin-orbitals electrons: %g",self.nocc_so)
    log.note("virtual  spin-orbitals electrons: %g",nvirt_so)
    log.note("total    spin-orbitals electrons: %g",self.n_qubit)
    log.note("\ntotal qubits: %g",self.n_qubit)
    log.note("Fock space dimension: %g",2**self.n_qubit)

    Hamiltonian = self.hamiltonian
    psi_HF = self.psi_hf
    S2_op = self.s2_op

    E_HF = numpy.real(psi_HF.conj().T @ Hamiltonian @ psi_HF)
    HF_S2 = numpy.real(psi_HF.conj().T @ S2_op @ psi_HF)
    log.note("\nCalculating <psi_HF| H |psi_HF> as sanity check")
    log.note("Hartree-Fock Energy: %-18.15f",E_HF.item())
    log.note("Hartree-Fock S_e^2 : %-18.15f",HF_S2.item())

    fci_exc_e = math.comb(self.n_qubit, self.nocc_so)
    log.note("\nparticle-conserving determinants electrons: %g", fci_exc_e)
    log.note("\nNumber of quantum particles: %g", self.nocc_so)
    log.note("Hilbert-subspace-particle-conserving dimension: %g", fci_exc_e)
    return

def parse_mf_elec(mf):
    if isinstance(mf, scf.hf.RHF):
        moa = mf.mo_coeff[:,:]
        ea = mf.mo_energy[:]
        mob = moa
        eb = ea
        na = moa.shape[1]
        nb = na
        noa = int(numpy.sum(mf.mo_occ)//2)
        nob = noa
    elif isinstance(mf, scf.uhf.UHF):
        moa = mf.mo_coeff[0][:]
        ea = mf.mo_energy[0][:]
        mob = mf.mo_coeff[1][:]
        eb = mf.mo_energy[1][:]
        na = moa.shape[1]
        nb = mob.shape[1]
        noa = numpy.sum(mf.mo_occ[0]>0)
        nob = numpy.sum(mf.mo_occ[1]>0)
    else:
        raise TypeError('Reference must be RHF or UHF')
    return moa, mob, ea, eb, na, nb, noa, nob

def fci_index_e(C_FCI, nocc_e, Num_op_e, S2_op, bool_ge=False):
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

        S2 = numpy.real(wf_tmp.conj().T @ S2_op @ wf_tmp).item()
        fci_pnum_e = numpy.real(wf_tmp.conj().T @ Num_op_e @ wf_tmp).item()
        pnum_diff_e = abs(fci_pnum_e-1.0*nocc_e)
        if (pnum_diff_e  < pnum_tol):
            fci_idx.append(i)
            fci_pnum.append(fci_pnum_e)
            fci_s2.append(S2)
            if bool_ge:
                return fci_idx, fci_pnum, fci_s2
    return fci_idx, fci_pnum, fci_s2

def number_operator_e(n_qubit_tot, n_qubit_e, create, destroy):
    dim = 2**n_qubit_tot
    Num_op = sparse.csr_matrix((dim, dim),dtype=complex)
    for p in range(n_qubit_e):
        Num_op += create[0][p] @ destroy[0][p]
    return Num_op

def make_rdm1_e(vector, create, destroy):
    '''Make 1-RDM for electrons
       rho_ij = < a_i^+ a_j >
    '''
    e_dim = len(create[0])
    rho = numpy.zeros((e_dim, e_dim), dtype=complex)
    vector = qc_lib.column_vec(vector)
    for i in range(e_dim):
        for j in range(e_dim):
            rho[i,j] = (vector.conj().T @ create[0][i] @ destroy[0][j] @ vector).item()
    return rho

#def vN_entropy_1rdm_e(log, vector, str_method, create, destroy, tol=1e-15):
#    S_vN = 0.0
#    preamble = "SvN 1RDM "
#    rho = make_rdm1_e(vector, create, destroy)
#    label = preamble + str_method + " e: %15.8e"
#    e_rho = scipy.linalg.eigh(rho, eigvals_only=True)
#    for i in range(len(e_rho)):
#        if e_rho[i] > tol:
#            S_vN -= e_rho[i]*numpy.log(e_rho[i])
#    log.note(label, S_vN)
#    return

def t1_op_e(nocc_so, nvirt_so, create, destroy):
    tau = []
    tau_dag = []
    for i in range(nocc_so):
        for a in range(nvirt_so):
            idx = [a+nocc_so, i]
            t1 = qc_lib.ca1_op(idx, create, destroy, 0)
            t1_dag = t1.conj().T
            tau.append(t1)
            tau_dag.append(t1_dag)
    return tau, tau_dag

def t2_op_e(nocc_so, nvirt_so, create, destroy):
    tau = []
    tau_dag = []
    for i in range(nocc_so):
        for j in range(i+1,nocc_so):
            for a in range(nvirt_so):
                for b in range(a+1, nvirt_so):
                    idx = [a + nocc_so, b + nocc_so, j, i]
                    t2 = qc_lib.ca2_op(idx, create, destroy, 0)
                    t2_dag = t2.conj().T
                    tau.append(t2)
                    tau_dag.append(t2_dag)
    return tau, tau_dag

def HF_state(n_occ, n_qubit):
    q0 = numpy.array([[1.0+0.0j],[0.0+0.0j]])
    q1 = numpy.array([[0.0+0.0j],[1.0+0.0j]])
    if n_occ == 0:
        a = numpy.kron(q0,q0)
        for i in range(2,n_qubit):
            a = numpy.kron(a,q0)
    elif n_occ == 1:
        a = numpy.kron(q1,q0)
        for i in range(2,n_qubit):
            a = numpy.kron(a,q0)
    else:
        a = numpy.kron(q1,q1)
        for i in range(2,n_qubit):
            if i<n_occ: mat = q1
            else: mat=q0
            a = numpy.kron(a,mat)
    return a

def calc_eri(eri_ao, moa, mob, ea, eb, n_qubit_e):
    motota, mototb, motot = qc_lib.mo_to_spinor(moa, mob, ea, eb)
    eri_ee = eri_ao
    eriaa = ao2mo.incore.general(eri_ee, (motota,motota,motota,motota), compact=False)\
            .reshape(n_qubit_e, n_qubit_e, n_qubit_e, n_qubit_e)
    eriab = ao2mo.incore.general(eri_ee, (motota,motota,mototb,mototb), compact=False)\
            .reshape(n_qubit_e, n_qubit_e, n_qubit_e, n_qubit_e)
    eriba = ao2mo.incore.general(eri_ee, (mototb,mototb,motota,motota), compact=False)\
            .reshape(n_qubit_e, n_qubit_e, n_qubit_e, n_qubit_e)
    eribb = ao2mo.incore.general(eri_ee, (mototb,mototb,mototb,mototb), compact=False)\
            .reshape(n_qubit_e, n_qubit_e, n_qubit_e, n_qubit_e)
    eri_ee_mo = eriaa + eriab + eriba + eribb
    return eri_ee_mo

def calc_hmo(hao, moa, mob, ea, eb):
    motota, mototb, motot = qc_lib.mo_to_spinor(moa, mob, ea, eb)
    hao = numpy.kron(numpy.eye(2), hao)
    hmo = motot.transpose() @ hao @ motot
    return hmo

def Ham_elec(mf, moa, mob, ea, eb, eri_ao, hao, create, destroy, n_qubit_e, tol=1e-12):
    h_dim = create[0][0].shape[0]
    if eri_ao is not None:
        eri_ee_mo = calc_eri(eri_ao, moa, mob, ea, eb, n_qubit_e)
    hmo = calc_hmo(hao, moa, mob, ea, eb)
    Hamiltonian = sparse.csr_matrix((h_dim, h_dim), dtype=complex)

    # electronic core
    for p in range(n_qubit_e):
        for q in range(n_qubit_e):
            hval = abs(hmo[p,q])
            if hval < tol: continue
            idx = [p, q]
            op_hpq = qc_lib.ca1_op(idx, create, destroy, 0)
            Hamiltonian += hmo[p,q]*op_hpq

    # electronic ee interaction
    if eri_ao is not None:
        for p in range(n_qubit_e):
            for q in range(n_qubit_e):
                for r in range(n_qubit_e):
                    for s in range(n_qubit_e):
                        eeval = abs(eri_ee_mo[p,q,r,s])
                        if eeval < tol: continue
                        idx = [p, r, s, q] # Mulliken notation
                        op_pqrs = qc_lib.ca2_op(idx, create, destroy, 0)
                        Hamiltonian += 0.5*eri_ee_mo[p,q,r,s]*op_pqrs

    #if not (isinstance(mf, neo.CDFT) or isinstance(mf, neo.HF)):
    #    E_nuc = mf.energy_nuc()
    #    Hamiltonian += E_nuc*qc_lib.kron_I(n_qubit_e)
    return Hamiltonian

def JW_array(n_qubit, op_id):
    tot_list = []
    op_list = []
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
        raise ValueError("Invalid JW operator designation")
    for i in range(n_qubit):
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
        for j in range(2,n_qubit):
            if i == j:
                mat3 = ladder_op
            elif (i<j):
                mat3 = II
            else:
                mat3 = Z
            a = sparse.kron(a,mat3,'csr')
        op_list.append(a)
    tot_list.append(op_list)
    return tot_list

class QC_ELEC_BASE(lib.StreamObject):

    def __init__(self, mf):
        self.verbose = mf.verbose
        self.mf = mf
        self.n_qubit = None
        self.hamiltonian = None
        self.create = None
        self.destroy = None
        self.num_op = None
        self.s2_op = None
        self.sz_op = None
        self.a_id = None
        self.b_id = None
        self.psi_hf = None
        self.nocc_so = None

        if isinstance(mf, neo.HF):
            raise TypeError('Electronic QC Protocol cannot take NEO mf object')

    def qc_components(self):
        log = logger.new_logger(self.mf, self.verbose)
        time_qc_components = logger.process_clock()

        mf = self.mf

        # find out what kind of calculation we have
        moa, mob, ea, eb, na, nb, noa, nob = parse_mf_elec(mf)

        # mixed alpha/beta overlap
        s1e = mf.get_ovlp()
        s_ab = moa.conj().T @ s1e @ mob
        s_ba = mob.conj().T @ s1e @ moa

        nvirta  = na - noa
        nvirtb  = nb - nob
        nocc_so = noa + nob
        nvirt_so = nvirta + nvirtb
        ntot_so = nocc_so + nvirt_so

        n_qubit_e = ntot_so

        if n_qubit_e > 14:
            log.warn("Number of qubits is %g, recommended max is 14", n_qubit_e)

        # form creation/annihilation lists
        create = JW_array(n_qubit_e, "creation")
        destroy = JW_array(n_qubit_e, "annihilation")

        # construct Hamiltonian
        hao = mf.get_hcore()
        eri_ao = mf._eri
        Hamiltonian = Ham_elec(mf, moa, mob, ea, eb, eri_ao, hao, create, destroy, n_qubit_e)

        etot  = numpy.hstack((ea,eb))
        Num_op_e = number_operator_e(n_qubit_e, n_qubit_e, create, destroy)
        S2_op, Sz_op, a_id, b_id  = qc_lib.S_squared_operator(n_qubit_e, s_ab, s_ba,
                                                              etot.argsort(), create, destroy)
        psi_HF = HF_state(nocc_so, n_qubit_e)

        self.n_qubit = n_qubit_e
        self.hamiltonian = Hamiltonian
        self.create = create
        self.destroy = destroy
        self.num_op = Num_op_e
        self.s2_op = S2_op
        self.sz_op = Sz_op
        self.a_id = a_id
        self.b_id = b_id
        self.psi_hf = psi_HF
        self.nocc_so = nocc_so

        log.timer("Construction of QC components: ", time_qc_components)
        return

class QC_FCI_ELEC(QC_ELEC_BASE):
    ''' FCI calculation using matrix diagonalization in qubit basis

    Examples:

    >>> from pyscf import gto, scf
    >>> from pyscf import neo
    >>> from pyscf.neo import qc
    >>> mol = gto.M(atom='H 0 0 0; H 0.74 0 0', basis = '6-31G',
    >>>             spin = 0)
    >>> mf = scf.RHF(mol)
    >>> mf.scf()
    >>> qc_mf = qc.QC_FCI_ELEC(mf)
    >>> qc_mf.kernel()
    ------- Quantum Computing Protocol -------
    FCI Energy:   -1.151672544961238
    '''

    def __init__(self, mf):
        super().__init__(mf)
        self.e = None
        self.c = None
        self.num = None
        self.num_e = None
        self.s2 = None

    def kernel(self, full_diag=False, nstates=1, num_e=None):
        log = logger.new_logger(self.mf, self.verbose)
        time_kernel = logger.process_clock()

        qc_lib.dump_qc_header(log)

        # get all quantum computing components needed for calculation
        self.qc_components()

        # Get FCI Hamiltonian
        ham_kern = self.hamiltonian
        ecore = self.mf.energy_nuc()
        ham_dim = ham_kern.shape[0]

        # Fock space target for number of electrons
        if num_e is None:
            num_e = self.nocc_so
            self.num_e = num_e
        else:
            self.num_e = num_e

        # Verify Hamiltonian is real
        max_i = max_imag_comp(ham_kern)
        if max_i > 1e-15:
            log.warn(f"Maximum |imaginary part| in Hamiltonian: {max_i:.5e}")

        # Diagonalize
        if full_diag:
            ham_kern.real
            E_FCI, C_FCI = numpy.linalg.eigh(ham_kern.todense()) #eigh should use dense matrix
        else:
            # Project out all non-particle-conserving blocks in Fock space
            proj_op = qc_lib.pc_projection(self.n_qubit, None, num_e)
            ham_kern = proj_op @ ham_kern @ proj_op
            ham_kern.real
            E_FCI, C_FCI = sparse.linalg.eigsh(ham_kern, k=nstates, which='SA')
            # guarantee proper sorting
            sorted_indices = numpy.argsort(E_FCI)
            E_FCI = E_FCI[sorted_indices]
            C_FCI = C_FCI[:, sorted_indices]

        fci_idx, fci_pnum, fci_s2 = fci_index_e(C_FCI, num_e, self.num_op, self.s2_op)

        E_FCI_final = []
        C_FCI_final = numpy.zeros((ham_dim, len(fci_idx)), dtype=complex)
        for i in range(len(fci_idx)):
            E_FCI_final.append(E_FCI[fci_idx[i]] + ecore)
            C_FCI_final[:,i] = C_FCI[:, fci_idx[i]].reshape(-1)

        self.e = E_FCI_final
        self.c = C_FCI_final
        self.num = fci_pnum
        self.s2 = fci_s2

        log.timer("FCI Diagonalization and Sorting: ", time_kernel)
        log.note("\nFCI Energy: %20.15f", E_FCI_final[0])
        return E_FCI_final, C_FCI_final, fci_pnum, fci_s2

    def analyze(self, nstates=1, verbose = None):
        if verbose is None: verbose = self.verbose
        log = logger.new_logger(self.mf, verbose)

        qc_lib.dump_analysis_header(log)
        qc_lib.dump_spin_order(log, self.a_id, self.b_id)
        dump_elec_qc_info(self, log)

        log.note("\n------------- ELECTRONIC FCI RESULTS -------------")
        log.note("  State        Energy        S_e^2   Particle Number")
        for i in range(len(self.e)):
            log.note("%5i %20.15f %7.3f %11.3f", i, self.e[i], self.s2[i], self.num[i])

        if nstates > len(self.e): nstates = len(self.e)
        for i in range(nstates):
            log.note("\n------------------- STATE %g -------------------\n", i)
            log.note("E FCI %g: %-20.15f", i, self.e[i])
            log.note("<S^2>  : %-20.15f", self.s2[i])
            qc_lib.fci_wf_analysis(log, self.c[:,i], self.n_qubit, '')

        return

    def make_rdm1(self, state_vec=None):
        if state_vec is None:
            state_vec = self.c[:,0]
        return make_rdm1_e(state_vec, self.create, self.destroy)

class QC_UCC_ELEC(QC_ELEC_BASE):
    ''' Unitary coupled-cluster calculation using minimization in qubit basis

    Examples:

    >>> from pyscf import gto, scf
    >>> from pyscf import neo
    >>> from pyscf.neo import qc
    >>> mol = gto.M(atom='H 0 0 0; H 0.74 0 0', basis = '6-31G',
    >>>             spin = 0)
    >>> mf = scf.RHF(mol)
    >>> mf.scf()
    >>> qc_mf = qc.QC_UCC_ELEC(mf)
    >>> qc_mf.kernel()
    ------- Quantum Computing Protocol -------
    --- UCCSD Calculation ---
    number of cluster amplitudes: 27
    res.success: True
    res.message: Optimization terminated successfully.
    UCC Energy: -1.151672544961234
    '''
    def __init__(self, mf):
        super().__init__(mf)
        self.e = None
        self.c = None
        self.num = None
        self.s2 = None
        self.t = None

    def kernel(self, conv_tol=None, method='BFGS'):
        log = logger.new_logger(self.mf, self.verbose)
        time_kernel = logger.process_clock()

        qc_lib.dump_qc_header(log)

        # get all quantum computing components needed for calculation
        self.qc_components()

        ham_kern = self.hamiltonian
        ecore = self.mf.energy_nuc()
        psi_HF = self.psi_hf
        S2_op = self.s2_op
        nocc_so = self.nocc_so
        n_qubit = self.n_qubit
        create = self.create
        destroy = self.destroy
        nvirt_so = n_qubit - nocc_so

        t1_e, t1_e_dag = t1_op_e(nocc_so, nvirt_so, create, destroy)
        t2_e, t2_e_dag = t2_op_e(nocc_so, nvirt_so, create, destroy)
        tau = t1_e + t2_e
        tau_dag = t1_e_dag + t2_e_dag

        nt_amp = len(tau)
        t_amp = numpy.zeros(nt_amp)

        qc_lib.dump_ucc_level(log, 2)
        log.note("number of cluster amplitudes: %g", nt_amp)
        res = minimize(lambda z: qc_lib.UCC_energy(z, ham_kern, tau, tau_dag, nt_amp,
                                            psi_HF), t_amp, tol=conv_tol, method=method)
        log.note("\nres.success: %s", res.success)
        log.note("res.message: %s \n", res.message)

        theta = res.x
        E_UCC = res.fun
        psi_UCC = qc_lib.construct_UCC_wf(nt_amp, theta, tau, tau_dag, psi_HF)
        ucc_idx, ucc_pnum, ucc_s2 = fci_index_e(psi_UCC, nocc_so, self.num_op, S2_op)

        self.e = E_UCC + ecore
        self.c = psi_UCC
        self.num = ucc_pnum[0]
        self.s2 = ucc_s2[0]
        self.t = theta

        if res.success:
            log.note("\nUCC Energy: %-20.15f", E_UCC+ecore)
        else:
            log.note("\nUCC Failed: %-20.15f", E_UCC+ecore)

        log.timer("UCC Procedure: ", time_kernel)
        return E_UCC, psi_UCC, ucc_pnum[0], ucc_s2[0]

    def analyze(self, verbose=None):
        if verbose is None: verbose = self.verbose
        log = logger.new_logger(self.mf, verbose)

        qc_lib.dump_analysis_header(log)
        qc_lib.dump_spin_order(log, self.a_id, self.b_id)
        dump_elec_qc_info(self, log)

        log.note("\n------------------- STATE 0 -------------------\n")
        log.note("E UCC: %-20.15f", self.e)
        log.note("<S_e^2>: %-20.15f", self.s2)
        log.note("Particles: %-11.7f \n", self.num)
        log.note("Amplitudes: \n%s\n", self.t)

        return

    def make_rdm1(self, state_vec=None):
        if state_vec is None:
            state_vec = self.c
        return make_rdm1_e(state_vec, self.create, self.destroy)
