'''
Interface for PySCF and ASE
'''

from ase.calculators.calculator import Calculator
from ase.units import Bohr, Hartree
from pyscf.data import nist
from pyscf import neo
from pyscf import gto, dft, tddft
from pyscf.lib import logger
from pyscf.tdscf.rhf import oscillator_strength
from pyscf.neo import ctddft, tdgrad


# from examples/scf/17-stability.py
def stable_opt_internal(mf):
    log = logger.new_logger(mf)
    if hasattr(mf, 'components'):
        mf_elec = mf.components['e']
    else:
        mf_elec = mf
    mo1, _, stable, _ = mf_elec.stability(return_status=True)
    cyc = 0
    while (not stable and cyc < 10):
        log.note('Try to optimize orbitals until stable, attempt %d' % cyc)
        dm1 = mf_elec.make_rdm1(mo1, mf_elec.mo_occ)
        mf = mf.run(dm1)
        mo1, _, stable, _ = mf_elec.stability(return_status=True)
        cyc += 1
    if not stable:
        log.note('Stability Opt failed after %d attempts' % cyc)
    return mf


class Pyscf_NEO(Calculator):
    """CNEO-DFT PySCF calculator"""

    implemented_properties = ['energy', 'forces', 'dipole', 'excited_energies',
                              'oscillator_strength']

    def __init__(self, basis='ccpvdz', nuc_basis='pb4d', charge=0, spin=0,
                 quantum_nuc=['H'], xc='b3lyp',
                 add_solvent=False,        # add implict solvent model ddCOSMO
                 pcm_eps=None,             # PCM solvent model, see pyscf.solvent.pcm
                 pcm_method=None,          # PCM solvent model, see pyscf.solvent.pcm
                 efield=None,              # external electric field
                 run_tda=False,            # run TDA calculations
                 disp=False,               # add dispersion correction (such as d3, d3bj, d4)
                 add_vv10=False,           # add dispersion correction VV10
                 epc=None,                 # add eletron proton correlation
                 atom_grid=None,           # (99,590) or even (99,974) for accuracy
                 grid_response=False,      # recommended for meta-GGA
                 init_guess=None,          # 'huckel' for unrestricted might be good
                 conv_tol=None,            # 1e-11~1e-12 for tight convergence
                 conv_tol_grad=None,       # 1e-7~1e-8 for tight convergence
                 den_fit=False,            # density-fitting
                 den_fit_basis=None,       # DF aux basis
                 den_fit_ne=False,         # if density fit the nuclear-electronic Coulomb
                 force_unrestricted=False, # can force mf to be unrestricted
                 stable_opt=False,         # if check stability
                 **kwargs):
        Calculator.__init__(self, **kwargs)
        self.basis = basis
        self.nuc_basis = nuc_basis
        self.charge = charge
        self.spin = spin
        self.quantum_nuc = quantum_nuc
        self.xc = xc
        self.add_vv10 = add_vv10
        self.epc = epc
        self.atom_grid = atom_grid
        self.grid_response = grid_response
        self.conv_tol = conv_tol
        self.conv_tol_grad = conv_tol_grad
        self.den_fit = den_fit
        self.den_fit_basis = den_fit_basis
        self.den_fit_ne = den_fit_ne
        self.init_guess = init_guess
        ###
        self.unrestricted = force_unrestricted
        self.add_solvent = add_solvent
        self.pcm_eps = pcm_eps
        self.pcm_method = pcm_method # NEO-PCM can work with scanners
        self.efield = efield
        self.run_tda = run_tda
        self.disp = disp
        self.stable_opt = stable_opt
        if self.add_solvent or self.run_tda or self.stable_opt or self.efield is not None:
            # TODO: see if some of them can work with scanners
            self.scanner_available = False
        else:
            # initialize a fake mol then create scanners
            mol = neo.M(atom='H 0 0 0; F 0 0 0.9')
            mf = self.create_mf(mol)
            self.mf_scanner = mf.as_scanner()
            self.mf_grad_scanner = mf.nuc_grad_method().set(grid_response=self.grid_response).as_scanner()
            self.scanner_available = True

    def get_mol_from_atoms(self, atoms):
        """Convert ASE atoms to PySCF NEO mol"""
        symbols = atoms.get_chemical_symbols()
        ase_masses = atoms.get_masses()
        positions = atoms.get_positions()
        atom_pyscf = []
        for i, symbol in enumerate(symbols):
            if symbol == 'Mu':
                atom_pyscf.append(['H*', tuple(positions[i])])
            elif symbol == 'D':
                atom_pyscf.append(['H+', tuple(positions[i])])
            elif symbol == 'H':
                # this is for person who does not want to modify ase
                # by changing the mass array, pyscf still accepts H as D
                if abs(ase_masses[i]-0.114) < 0.01:
                    atom_pyscf.append(['H*', tuple(positions[i])])
                elif abs(ase_masses[i]-2.014) < 0.02:
                    atom_pyscf.append(['H+', tuple(positions[i])])
                else:
                    atom_pyscf.append(['%s' % symbol, tuple(positions[i])])
            else:
                atom_pyscf.append(['%s' % symbol, tuple(positions[i])])
        mol = neo.M(atom=atom_pyscf, quantum_nuc=self.quantum_nuc, basis=self.basis,
                    nuc_basis=self.nuc_basis, charge=self.charge, spin=self.spin)
        return mol

    def create_mf(self, mol):
        if self.den_fit:
            mf = neo.CDFT(mol, xc=self.xc, unrestricted=self.unrestricted,
                          epc=self.epc).density_fit(auxbasis=self.den_fit_basis,
                                                    df_ne=self.den_fit_ne)
        else:
            mf = neo.CDFT(mol, xc=self.xc, unrestricted=self.unrestricted,
                          epc=self.epc)
        if self.pcm_eps is not None and self.pcm_method is not None:
            mf = mf.PCM()
            mf.with_solvent.eps = self.pcm_eps
            mf.with_solvent.method = self.pcm_method
        if self.efield is not None:
            from pyscf.neo.efield import SCFwithEfield
            mf = SCFwithEfield(mol, xc=self.xc, epc=self.epc)
            mf.efield = self.efield
        if self.atom_grid is not None:
            mf.components['e'].grids.atom_grid = self.atom_grid
        if self.add_vv10:
            mf.components['e'].nlc = 'VV10'
            mf.components['e'].grids.prune = None
            mf.components['e'].nlcgrids.atom_grid = (50,194)
            mf.components['e'].nlcgrids.prune = dft.gen_grid.sg1_prune
        if self.init_guess is not None:
            mf.init_guess = self.init_guess # string or array or dict
        if self.conv_tol is not None:
            mf.conv_tol = self.conv_tol
        if self.conv_tol_grad is not None:
            mf.conv_tol_grad = self.conv_tol_grad
        return mf

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        mol = self.get_mol_from_atoms(atoms)
        if self.scanner_available:
            if 'forces' in properties:
                e_tot, de = self.mf_grad_scanner(mol)
                mf = self.mf_grad_scanner.base
            else:
                e_tot = self.mf_scanner(mol)
                mf = self.mf_scanner
        else:
            mf = self.create_mf(mol)
            if self.disp:
                mf.disp = self.disp
            if self.add_solvent:
                mf = mf.ddCOSMO()
            # TODO: last step dm0 for non-scanner?
            mf.scf()
            if self.stable_opt:
                mf = stable_opt_internal(mf)
            e_tot = mf.e_tot
            if 'forces' in properties:
                de = mf.nuc_grad_method().set(grid_response=self.grid_response).grad()
        self.results['energy'] = e_tot * Hartree
        if 'forces' in properties:
            self.results['forces'] = -de * Hartree / Bohr

        #if 'dipole' in properties: # somehow ASE MD does not request dipole. How to enable?
        if True:
            self.results['dipole'] = mf.dip_moment()

        if self.run_tda:
            # calculate excited energies and oscillator strength by TDDFT/TDA
            td = tddft.TDA(mf.mf_elec)
            e, xy = td.kernel()
            os = oscillator_strength(td, e=e, xy=xy)

            self.results['excited_energies'] = e * nist.HARTREE2EV
            self.results['oscillator_strength'] = os

class Pyscf_TDNEO(Pyscf_NEO):
    '''CNEO-TDDFT PySCF calculator'''
    implemented_properties = ['energy', 'forces', 'ground_energy', 'excited_energies']
    def __init__(self, state=1,
                 nstates=3,
                 is_davidson = False,
                 max_space = 40,
                 **kwargs):
        super().__init__(**kwargs)
        self.run_tda = False
        if not self.scanner_available:
            raise NotImplementedError('mf_scanner not initialized')
        if self.epc is not None:
            raise NotImplementedError('epc not supported for CNEO-TDDFT gradient')

        self.scanner_available = False
        self.state = state
        self.nstates = nstates
        self.is_davidson = is_davidson
        self.max_space = max_space

        mol = neo.M(atom='H 0 0 0; F 0 0 0.9')
        td_mf, td_grad = self.create_tdmf(mol)
        self.td_scanner = td_mf.as_scanner()
        self.td_grad_scanner = td_grad.as_scanner(state=self.state)
        self.scanner_available = True

    def create_tdmf(self, mol):
        mf = self.create_mf(mol)
        if self.is_davidson:
            td_mf = ctddft.CTDDFT(mf)
            td_mf.nstates = self.nstates
            td_mf.max_space = self.max_space
        else:
            td_mf = ctddft.CTDDirect(mf)
            td_mf.nstates = self.nstates

        td_grad = tdgrad.Gradients(td_mf)
        td_grad.state = self.state

        return td_mf, td_grad

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        mol = self.get_mol_from_atoms(atoms)
        if not self.scanner_available:
            raise ValueError('td scanner not initialized')

        if 'forces' in properties:
            e_tot, de = self.td_grad_scanner(mol)
            td_mf = self.td_grad_scanner.base
        else:
            e_tot = self.td_scanner(mol)[self.state-1]
            td_mf = self.td_scanner

        self.results['energy'] = e_tot * Hartree
        if 'forces' in properties:
            self.results['forces'] = -de * Hartree / Bohr
        if 'ground_energy' in properties:
            e_gs = td_mf._scf.e_tot
            self.results['ground_energy'] = e_gs
        if 'excited_energies' in properties:
            e_ex = td_mf.e
            self.results['excited_energies'] = e_ex * nist.HARTREE2EV



class Pyscf_DFT(Calculator):
    """DFT PySCF calculator"""

    implemented_properties = ['energy', 'forces', 'dipole', 'excited_energies',
                              'oscillator_strength']

    def __init__(self, basis='ccpvdz', charge=0, spin=0, xc='b3lyp',
                 add_solvent=False,        # add implict solvent model ddCOSMO
                 pcm_eps=None,             # PCM solvent model, see pyscf.solvent.pcm
                 pcm_method=None,          # PCM solvent model, see pyscf.solvent.pcm
                 efield=None,              # external electric field
                 run_tda=False,            # run TDA calculations
                 disp=False,               # add dispersion correction (such as d3, d3bj, d4)
                 add_vv10=False,           # add dispersion correction VV10
                 atom_grid=None,           # (99,590) or even (99,974) for accuracy
                 grid_response=False,      # recommended for meta-GGA
                 init_guess=None,          # 'huckel' for unrestricted might be good
                 conv_tol=None,            # 1e-11~1e-12 for tight convergence
                 conv_tol_grad=None,       # 1e-7~1e-8 for tight convergence
                 den_fit=False,            # density-fitting
                 den_fit_basis=None,       # DF aux basis
                 force_unrestricted=False, # can force mf to be unrestricted
                 stable_opt=False,         # if check stability
                 **kwargs):
        Calculator.__init__(self, **kwargs)
        self.basis = basis
        self.charge = charge
        self.spin = spin
        self.xc = xc
        self.add_vv10 = add_vv10
        self.atom_grid = atom_grid
        self.grid_response = grid_response
        self.conv_tol = conv_tol
        self.conv_tol_grad = conv_tol_grad
        self.den_fit = den_fit
        self.den_fit_basis = den_fit_basis
        self.init_guess = init_guess
        ###
        self.unrestricted = force_unrestricted
        self.pcm_eps = pcm_eps
        self.pcm_method = pcm_method
        self.add_solvent = add_solvent
        self.efield = efield
        self.run_tda = run_tda
        self.disp = disp
        self.stable_opt = stable_opt
        if self.add_solvent or self.run_tda or self.stable_opt or self.efield is not None:
            # TODO: see if some of them can work with scanners
            self.scanner_available = False
        else:
            # initialize a fake mol then create scanners
            mol = gto.M(atom='H 0 0 0; F 0 0 0.9')
            mf = self.create_mf(mol)
            self.mf_scanner = mf.as_scanner()
            self.mf_grad_scanner = mf.nuc_grad_method().set(grid_response=self.grid_response).as_scanner()
            self.scanner_available = True

    def get_mol_from_atoms(self, atoms):
        """Convert ASE atoms to PySCF mol"""
        symbols = atoms.get_chemical_symbols()
        ase_masses = atoms.get_masses()
        positions = atoms.get_positions()
        atom_pyscf = []
        for i, symbol in enumerate(symbols):
            if symbol == 'Mu':
                atom_pyscf.append(['H*', tuple(positions[i])])
            elif symbol == 'D':
                atom_pyscf.append(['H+', tuple(positions[i])])
            elif symbol == 'H':
                # this is for person who does not want to modify ase
                # by changing the mass array, pyscf still accepts H as D
                if abs(ase_masses[i]-0.114) < 0.01:
                    atom_pyscf.append(['H*', tuple(positions[i])])
                elif abs(ase_masses[i]-2.014) < 0.02:
                    atom_pyscf.append(['H+', tuple(positions[i])])
                else:
                    atom_pyscf.append(['%s' % symbol, tuple(positions[i])])
            else:
                atom_pyscf.append(['%s' % symbol, tuple(positions[i])])
        mol = gto.M(atom=atom_pyscf, basis=self.basis,
                    charge=self.charge, spin=self.spin)
        return mol

    def create_mf(self, mol):
        if self.unrestricted:
            mf = dft.UKS(mol)
        else:
            if self.spin != 0:
                mf = dft.UKS(mol)
            else:
                mf = dft.RKS(mol)
        if self.den_fit:
            mf = mf.density_fit(auxbasis=self.den_fit_basis)
        if self.efield is not None:
            from pyscf.prop.infrared.efield import SCFwithEfield
            mf = SCFwithEfield(mol)
            mf.efield = self.efield
        mf.xc = self.xc
        if self.pcm_eps is not None and self.pcm_method is not None:
            mf = mf.PCM()
            mf.with_solvent.eps = self.pcm_eps
            mf.with_solvent.method = self.pcm_method
        if self.atom_grid is not None:
            mf.grids.atom_grid = self.atom_grid
        if self.add_vv10:
            mf.nlc = 'VV10'
            mf.grids.prune = None
            mf.nlcgrids.atom_grid = (50,194)
            mf.nlcgrids.prune = dft.gen_grid.sg1_prune
        if self.init_guess is not None:
            mf.init_guess = self.init_guess # string or array
        if self.conv_tol is not None:
            mf.conv_tol = self.conv_tol
        if self.conv_tol_grad is not None:
            mf.conv_tol_grad = self.conv_tol_grad
        return mf

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        mol = self.get_mol_from_atoms(atoms)
        if self.scanner_available:
            if 'forces' in properties:
                e_tot, de = self.mf_grad_scanner(mol)
                mf = self.mf_grad_scanner.base
            else:
                e_tot = self.mf_scanner(mol)
                mf = self.mf_scanner
        else:
            mf = self.create_mf(mol)
            if self.disp:
                mf.disp = self.disp
            if self.add_solvent:
                from pyscf import solvent
                mf = mf.ddCOSMO()
            # TODO: last step dm0 for non-scanner?
            mf.scf()
            if self.stable_opt:
                mf = stable_opt_internal(mf)
            e_tot = mf.e_tot
            if 'forces' in properties:
                de = mf.nuc_grad_method().set(grid_response=self.grid_response).grad()
        self.results['energy'] = e_tot * Hartree
        if 'forces' in properties:
            self.results['forces'] = -de * Hartree / Bohr

        #if 'dipole' in properties: # somehow ASE MD does not request dipole. How to enable?
        if True:
            self.results['dipole'] = mf.dip_moment()

        if self.run_tda:
            td = tddft.TDA(mf)
            e, xy = td.kernel()
            os = oscillator_strength(td, e=e, xy=xy)

            self.results['excited_energies'] = e * nist.HARTREE2EV
            self.results['oscillator_strength'] = os
