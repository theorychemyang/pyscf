from pyscf import neo
from pyscf.neo.qc.qnuc import QC_FCI_NEO, QC_UCC_NEO, QC_CFCI_NEO, QC_CUCC_NEO
from pyscf.neo.qc.elec import QC_FCI_ELEC, QC_UCC_ELEC

def FCI(mf, cas_orb_nuc=None, c_shift=False):
    if isinstance(mf, neo.HF):
        fcisolver = QC_FCI_NEO(mf, cas_orb_nuc, c_shift)
    else:
        fcisolver = QC_FCI_ELEC(mf)
    return fcisolver

def UCC(mf, cas_orb_nuc=None, c_shift=False):
    if isinstance(mf, neo.HF):
        uccsolver = QC_UCC_NEO(mf, cas_orb_nuc, c_shift)
    else:
        uccsolver = QC_UCC_ELEC(mf)
    return uccsolver

def CFCI(mf, cas_orb_nuc=None, c_shift=True):
    fcisolver = QC_CFCI_NEO(mf, cas_orb_nuc, c_shift)
    return fcisolver

def CUCC(mf, cas_orb_nuc=None, c_shift=True):
    uccsolver = QC_CUCC_NEO(mf, cas_orb_nuc)
    return uccsolver
