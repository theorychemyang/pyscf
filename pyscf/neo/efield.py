#!/usr/bin/env python
'''
CNEO with electric field
'''
import numpy
from pyscf.neo import cphf, CDFT
from pyscf.neo.grad import Gradients
from pyscf.neo.hessian import gen_vind
from pyscf import lib

def solve_mo1(mf, mo_energy, mo_coeff, mo_occ, h1ao=None,
              fx=None, atmlst=None, max_memory=4000, verbose=None,
              max_cycle=100, level_shift=0):
    'Solve CNEO-CPKS with the perturbation of electric fields'

    mol = mf.mol
    
    if fx is None:
        fx = gen_vind(mf, mo_coeff, mo_occ)

    mocc = {}
    for t in mo_coeff.keys():
        mocc[t] = mo_coeff[t][:, mo_occ[t]>0]

    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    charge_center = numpy.einsum('i,ix->x', charges, coords) / charges.sum()

    h1vo = {}
    s1 = {}
    for t, comp in mf.components.items():
        with comp.mol.with_common_orig(charge_center):
            int1e_r = comp.mol.intor_symmetric('int1e_r', comp=3)
            h1vo[t] = numpy.einsum('xuv, ui, vj -> xij', int1e_r, mo_coeff[t], mocc[t]) * comp.charge

        s1[t] = numpy.zeros_like(h1vo[t])
    
    mo1, e1, _ = cphf.solve(fx, mo_energy, mo_occ, h1vo, s1,
                            with_f1=True, verbose=mf.verbose,
                            max_cycle=max_cycle, level_shift=level_shift)

    return mo1, e1

def polarizability(mf):
    'polarizability in CNEO'

    mol = mf.mol

    mo1 = solve_mo1(mf, mf.mo_energy, mf.mo_coeff, mf.mo_occ)[0]

    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    charge_center = numpy.einsum('i,ix->x', charges, coords) / charges.sum()

    with mol.components['e'].with_common_orig(charge_center):
        int_r = mol.components['e'].intor_symmetric('int1e_r', comp=3)

    mo_coeff = mf.components['e'].mo_coeff
    mo_occ = mf.components['e'].mo_occ
    mocc = mo_coeff[:, mo_occ>0]
    h1 = numpy.einsum('xpq,pi,qj->xij', int_r, mo_coeff, mocc)
    
    e2 = numpy.einsum('xpi,ypi->xy', h1, mo1['e'])

    # *-1 from the definition of dipole moment. *2 for double occupancy
    e2 = (e2 + e2.T) * -2

    return e2


def dipole_grad(mf):
    'Analytic nuclear gradients of dipole moments in CNEO'

    mol = mf.mol
    natm = mol.natm

    dm = mf.make_rdm1()

    de = numpy.zeros((natm, 3, 3))

    # contribution from nuclei
    for i in range(natm): 
        de[i] = numpy.eye(3) * mol.atom_charge(i) 

    
    mo1, e1 = solve_mo1(mf, mf.mo_energy, mf.mo_coeff, mf.mo_occ)

    mf_hess = mf.Hessian()
    h1ao = mf_hess.make_h1(mf.mo_coeff, mf.mo_occ) # 1st order skeleton derivative of Fock matrix

    mf_e = mf.components['e']
    mol_e = mol.components['e']

    mo_coeff = mf_e.mo_coeff
    mo_occ = mf_e.mo_occ
    mo_energy = mf_e.mo_energy
    nao = mo_coeff.shape[0]

    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    charge_center = numpy.einsum('i,ix->x', charges, coords) / charges.sum()

    with mol_e.with_common_orig(charge_center): 
        int1e_irp = - mol_e.intor("int1e_irp", comp=9)
        
    s1a = - mol_e.intor('int1e_ipovlp')

    # contribution from electrons
    for a in range(natm):
        p0, p1 = mol_e.aoslice_by_atom()[a, 2:]
        
        h2ao = numpy.zeros((9, nao, nao))
        h2ao[:,:,p0:p1] += int1e_irp[:,:,p0:p1] # nable is on ket in int1e_irp
        h2ao[:,p0:p1] += int1e_irp[:,:,p0:p1].transpose(0, 2, 1)
        de[a] -= numpy.einsum('xuv,uv->x', h2ao, dm['e']).reshape(3, 3).T

        h1vo = numpy.einsum('xuv, ui, vj -> xij', h1ao['e'][a], mo_coeff[:,mo_occ>0], mo_coeff)
        de[a] -= 4 * numpy.einsum('xij,tji->xt', h1vo, mo1['e']) 

        s1ao = numpy.zeros((3, nao, nao))
        s1ao[:,p0:p1] += s1a[:,p0:p1]
        s1ao[:,:,p0:p1] += s1a[:,p0:p1].transpose(0,2,1)

        s1ii = numpy.einsum('ui, vj, xuv -> xij', mo_coeff[:,mo_occ>0], mo_coeff[:,mo_occ>0], s1ao)
        de[a] += 2*numpy.einsum('xij, tij -> xt', s1ii, e1['e']) 

        s1ij = numpy.einsum('ui, vj, xuv -> xij', mo_coeff[:,mo_occ>0], mo_coeff, s1ao)
        de[a] += 4*numpy.einsum('i, tji, xij -> xt', mo_energy[mo_occ>0], mo1['e'], s1ij) 

    # contribution from quantum nuclei
    for t, comp in mf.components.items():
        if comp.is_nucleus:
            mo_coeff_n = comp.mo_coeff
            mo_occ_n = comp.mo_occ

            for a in range(natm):
                h1vo = numpy.einsum('xuv, ui, vj -> xij', h1ao[t][a], mo_coeff_n[:,mo_occ_n>0], mo_coeff_n)
                de[a] -= 2 * numpy.einsum('xij,tji->xt', h1vo, mo1[t]) # single occupancy for quantum nuclei, *2 for c.c.

    return de


class SCFwithEfield(CDFT):
    '''CNEO with electric field'''
    _keys = {'efield'}

    def __init__(self, mol, *args, **kwargs):
        CDFT.__init__(self, mol, *args, **kwargs)
        self.efield = numpy.array([0, 0, 0]) # unit: a.u. ( 1 a.u. = 5.14e11 V/m ? )
        self.mol = mol
        
    def get_hcore(self, mol=None):
        if mol is None: mol = self.mol
        hcore = {}
        for t, comp in mol.components.items():
            hcore[t] = self.components[t].get_hcore(mol=comp)
            comp.set_common_orig([0, 0, 0])  # The gauge origin for dipole integral
            hcore[t] += numpy.einsum('x,xij->ij', self.efield, comp.intor('int1e_r', comp=3)) * self.components[t].charge

        return hcore
    
    def energy_nuc(self):
        enuc = self.components['e'].energy_nuc()

        nuclear_charges = self.mol.components['e'].atom_charges()  
        nuclear_coords = self.mol.atom_coords()    

        E_nuc_field = -numpy.sum([Z * numpy.dot(self.efield, R) for Z, R in zip(nuclear_charges, nuclear_coords)])

        return enuc + E_nuc_field


class GradwithEfield(Gradients):
    '''CNEO gradients with external electric field'''
    _keys = {'mf'}

    def __init__(self, mf):
        super().__init__(mf)
        self.mf = self.base = mf
        self._efield = mf.efield

        mol_e = self.mol.components['e']
        h = self.components['e'].get_hcore(mol=mol_e)
        nao = mol_e.nao
        with mol_e.with_common_orig([0, 0, 0]):
            int1e_irp = - mol_e.intor('int1e_irp', comp=9).reshape(3, 3, nao, nao)
        h += numpy.einsum('z,zxij->xji', numpy.array(self._efield), int1e_irp) * self.components['e'].base.charge

        self.components['e'].get_hcore = lambda *args: h
    
    def grad_nuc(self, atmlst=None):
        gs = super().grad_nuc(atmlst)
        charges = self.mol.atom_charges()

        gs -= numpy.einsum('i,x->ix', charges, self._efield)
        if atmlst is not None:
            gs = gs[atmlst]
        return gs

SCFwithEfield.Gradients = lib.class_as_method(GradwithEfield)

if __name__ == '__main__':
    from pyscf import neo
    mol = neo.M(atom='H 0 0 0; F 0 0 0.8', basis='ccpvdz')
    mf = SCFwithEfield(mol, xc='b3lyp')
    mf.efield = numpy.array([0, 0, 0.001])
    #mf.conv_tol = 1e-14
    mf.scf()

    grad = mf.Gradients()
    grad.grid_response = True
    g = grad.kernel()
    


