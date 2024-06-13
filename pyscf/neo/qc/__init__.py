from pyscf import neo
from pyscf.neo.qc.qnuc import QC_FCI_NEO, QC_UCC_NEO, QC_CFCI_NEO, QC_CUCC_NEO
from pyscf.neo.qc.elec import QC_FCI_ELEC, QC_UCC_ELEC

def FCI(mf, cas_orb=None):
    if isinstance(mf, neo.HF):
        fcisolver = QC_FCI_NEO(mf, cas_orb)
    else:
        fcisolver = QC_FCI_ELEC(mf)
    return fcisolver

def UCC(mf, cas_orb=None):
    if isinstance(mf, neo.HF):
        uccsolver = QC_UCC_NEO(mf, cas_orb)
    else:
        uccsolver = QC_UCC_ELEC(mf)
    return uccsolver

def CFCI(mf, cas_orb=None):
    fcisolver = QC_CFCI_NEO(mf, cas_orb)
    return fcisolver

def CUCC(mf):
    uccsolver = QC_CUCC_NEO(mf)
    return uccsolver
