#!/usr/bin/env python

from pyscf.neo.mole import Mole, M
from pyscf.neo.hf import HF
from pyscf.neo.ks import KS
from pyscf.neo.cdft import CDFT
from pyscf.neo.grad import Gradients
from pyscf.neo.hessian import Hessian
from pyscf.neo.solvent import DDCOSMO
try:
    from pyscf.neo.ase import Pyscf_NEO, Pyscf_DFT
except ImportError:
    pass
from pyscf.neo.mp2 import MP2
from pyscf.neo.fci_n_minus_2_resolution import FCI
from pyscf.neo.efield import SCFwithEfield, GradwithEfield, polarizability
from pyscf.neo import pcm
from pyscf.neo.addons import *
from pyscf.neo.ctddft import CTDDirect, CTDDFT
from pyscf.neo.tdgrad import Gradients as TDGradients

def PCM(method_or_mol, solvent_obj=None, dm=None):
    if isinstance(method_or_mol, Mole):
        return pcm.PCM4NEO(method_or_mol)
    elif isinstance(method_or_mol, HF):
        return pcm.pcm_for_neo_scf(method_or_mol, solvent_obj, dm)
    else:
        raise NotImplementedError
