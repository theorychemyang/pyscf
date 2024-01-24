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


# from examples/scf/17-stability.py
def stable_opt_internal(mf):
    log = logger.new_logger(mf)
    if hasattr(mf, 'mf_elec'):
        mf_elec = mf.mf_elec
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
                 force_unrestricted=False, # can force mf to be unrestricted
                 stable_opt=False,         # if check stability
                 force_fresh_init=False,   # force to use fresh guess even if scanner
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
        self.init_guess = init_guess
        if isinstance(init_guess, dict):
            self.dm0 = [init_guess['e']] + init_guess['n']
        else:
            self.dm0 = None
        self.force_fresh_init = force_fresh_init
        if self.force_fresh_init:
            self.dm0 = None
        ###
        if self.spin == 0 and not force_unrestricted:
            self.unrestricted = False
        else:
            self.unrestricted = True
        self.add_solvent = add_solvent
        self.run_tda = run_tda
        self.disp = disp
        self.stable_opt = stable_opt
        if self.add_solvent or self.run_tda or self.stable_opt:
            # TODO: see if some of them can work with scanners
            self.scanner_available = False
        else:
            # initialize a fake mol then create scanners
            mol = neo.M(atom='H 0 0 0; F 0 0 0.9')
            mf = self.create_mf(mol)
            self.mf_scanner = mf.as_scanner()
            self.mf_grad_scanner = mf.Gradients().set(grid_response=self.grid_response).as_scanner()
            self.scanner_available = True

    def get_mol_from_atoms(self, atoms):
        """Convert ASE atoms to PySCF NEO mol"""
        symbols = atoms.get_chemical_symbols()
        ase_masses = atoms.get_masses()
        positions = atoms.get_positions()
        atom_pyscf = []
        for i in range(len(symbols)):
            if symbols[i] == 'Mu':
                atom_pyscf.append(['H*', tuple(positions[i])])
            elif symbols[i] == 'D':
                atom_pyscf.append(['H+', tuple(positions[i])])
            elif symbols[i] == 'H':
                # this is for person who does not want to modify ase
                # by changing the mass array, pyscf still accepts H as D
                if abs(ase_masses[i]-0.114) < 0.01:
                    atom_pyscf.append(['H*', tuple(positions[i])])
                elif abs(ase_masses[i]-2.014) < 0.02:
                    atom_pyscf.append(['H+', tuple(positions[i])])
                else:
                    atom_pyscf.append(['%s' % symbols[i], tuple(positions[i])])
            else:
                atom_pyscf.append(['%s' % symbols[i], tuple(positions[i])])
        mol = neo.M(atom=atom_pyscf, quantum_nuc=self.quantum_nuc, basis=self.basis,
                    nuc_basis=self.nuc_basis, charge=self.charge, spin=self.spin)
        return mol

    def create_mf(self, mol):
        if self.den_fit:
            mf = neo.CDFT(mol, unrestricted=self.unrestricted, epc=self.epc,
                          df_ee=self.den_fit, auxbasis_e=self.den_fit_basis)
        else:
            mf = neo.CDFT(mol, unrestricted=self.unrestricted, epc=self.epc)
        mf.mf_elec.xc = self.xc
        if self.atom_grid is not None:
            mf.mf_elec.grids.atom_grid = self.atom_grid
        if self.add_vv10:
            mf.mf_elec.nlc = 'VV10'
            mf.mf_elec.grids.prune = None
            mf.mf_elec.nlcgrids.atom_grid = (50,194)
            mf.mf_elec.nlcgrids.prune = dft.gen_grid.sg1_prune
        if self.init_guess is not None and not isinstance(self.init_guess, dict):
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
            # TODO: can pass external dm0 to scanners.
            # By default scanners will use last step dm, and pass None
            # to use fresh initial guess.
            # NOTE: if for the external dm0 you do really only want
            # electronic dm0, pass to scanners a list with one ndarray in it:
            #    [dm0_e]
            # so that nuclear dm will have fresh initial guesses.
            # If you pass dm0_e ndarray directly to them, they can still run,
            # but they will automatically construct last step nuclear
            # dm as dm0. If you want that behavior, then pass the array.
            if 'forces' in properties:
                if self.dm0 is not None or self.force_fresh_init:
                    e_tot, de = self.mf_grad_scanner(mol, dm0=self.dm0)
                else:
                    e_tot, de = self.mf_grad_scanner(mol)
                mf = self.mf_grad_scanner.base
            else:
                if self.dm0 is not None or self.force_fresh_init:
                    e_tot = self.mf_scanner(mol, dm0=self.dm0)
                else:
                    e_tot = self.mf_scanner(mol)
                mf = self.mf_scanner
        else:
            mf = self.create_mf(mol)
            if self.disp:
                mf.disp = self.disp
            if self.add_solvent:
                mf.scf(cycle=0) # TODO: remove this
                mf = mf.ddCOSMO()
            # TODO: last step dm0 for non-scanner?
            mf.scf(dm0=self.dm0)
            if self.stable_opt:
                mf = stable_opt_internal(mf)
            e_tot = mf.e_tot
            if 'forces' in properties:
                de = mf.Gradients().set(grid_response=self.grid_response).grad()
        self.results['energy'] = e_tot * Hartree
        if 'forces' in properties:
            self.results['forces'] = -de * Hartree / Bohr

       #if 'dipole' in properties: # somehow ASE MD does not request dipole. How to enable?
        if True:
            dm1 = mf.mf_elec.make_rdm1()
            dip_elec = mf.dip_moment(mol.elec, dm1) # dipole of electrons and classical nuclei
            dip_nuc = 0
            for i in range(len(mf.mf_nuc)):
                ia = mf.mf_nuc[i].mol.atom_index
                dip_nuc += mol.atom_charge(ia) * mf.mf_nuc[i].nuclei_expect_position * nist.AU2DEBYE

            self.results['dipole'] = dip_elec + dip_nuc

        if self.run_tda:
            # calculate excited energies and oscillator strength by TDDFT/TDA
            td = tddft.TDA(mf.mf_elec)
            e, xy = td.kernel()
            os = oscillator_strength(td, e=e, xy=xy)

            self.results['excited_energies'] = e * nist.HARTREE2EV
            self.results['oscillator_strength'] = os



class Pyscf_DFT(Calculator):
    """DFT PySCF calculator"""

    implemented_properties = ['energy', 'forces', 'dipole', 'excited_energies',
                              'oscillator_strength']

    def __init__(self, basis='ccpvdz', charge=0, spin=0, xc='b3lyp',
                 add_solvent=False,        # add implict solvent model ddCOSMO
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
                 force_fresh_init=False,   # force to use fresh guess even if scanner
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
        if isinstance(init_guess, dict):
            self.init_guess = init_guess['e']
            self.dm0 = self.init_guess
        else:
            self.init_guess = init_guess
            self.dm0 = None
        # if init_guess is an dict, every step this dm0 is used.
        # if init_guess is an array, it can work as a pyscf init_guess
        # for just the first step, then following steps use scanners' default:
        # last step guess.
        self.force_fresh_init = force_fresh_init
        if self.force_fresh_init:
            self.dm0 = None
        ###
        if self.spin == 0 and not force_unrestricted:
            self.unrestricted = False
        else:
            self.unrestricted = True
        self.add_solvent = add_solvent
        self.run_tda = run_tda
        self.disp = disp
        self.stable_opt = stable_opt
        if self.add_solvent or self.run_tda or self.stable_opt:
            # TODO: see if some of them can work with scanners
            self.scanner_available = False
        else:
            # initialize a fake mol then create scanners
            mol = gto.M(atom='H 0 0 0; F 0 0 0.9')
            mf = self.create_mf(mol)
            self.mf_scanner = mf.as_scanner()
            self.mf_grad_scanner = mf.Gradients().set(grid_response=self.grid_response).as_scanner()
            self.scanner_available = True

    def get_mol_from_atoms(self, atoms):
        """Convert ASE atoms to PySCF mol"""
        symbols = atoms.get_chemical_symbols()
        ase_masses = atoms.get_masses()
        positions = atoms.get_positions()
        atom_pyscf = []
        for i in range(len(symbols)):
            if symbols[i] == 'Mu':
                atom_pyscf.append(['H*', tuple(positions[i])])
            elif symbols[i] == 'D':
                atom_pyscf.append(['H+', tuple(positions[i])])
            elif symbols[i] == 'H':
                # this is for person who does not want to modify ase
                # by changing the mass array, pyscf still accepts H as D
                if abs(ase_masses[i]-0.114) < 0.01:
                    atom_pyscf.append(['H*', tuple(positions[i])])
                elif abs(ase_masses[i]-2.014) < 0.02:
                    atom_pyscf.append(['H+', tuple(positions[i])])
                else:
                    atom_pyscf.append(['%s' % symbols[i], tuple(positions[i])])
            else:
                atom_pyscf.append(['%s' % symbols[i], tuple(positions[i])])
        mol = gto.M(atom=atom_pyscf, basis=self.basis,
                    charge=self.charge, spin=self.spin)
        return mol

    def create_mf(self, mol):
        if self.unrestricted:
            mf = dft.UKS(mol)
        else:
            mf = dft.RKS(mol)
        if self.den_fit:
            mf = mf.density_fit(auxbasis=self.den_fit_basis)
        mf.xc = self.xc
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
            # TODO: can pass external dm0 to scanners.
            # By default scanners will use last step dm, and pass None
            # to use fresh initial guess.
            if 'forces' in properties:
                if self.dm0 is not None or self.force_fresh_init:
                    e_tot, de = self.mf_grad_scanner(mol, dm0=self.dm0)
                else:
                    e_tot, de = self.mf_grad_scanner(mol)
                mf = self.mf_grad_scanner.base
            else:
                if self.dm0 is not None or self.force_fresh_init:
                    e_tot = self.mf_scanner(mol, dm0=self.dm0)
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
            mf.scf(dm0=self.dm0)
            if self.stable_opt:
                mf = stable_opt_internal(mf)
            e_tot = mf.e_tot
            if 'forces' in properties:
                de = mf.Gradients().set(grid_response=self.grid_response).grad()
        self.results['energy'] = e_tot * Hartree
        if 'forces' in properties:
            self.results['forces'] = -de * Hartree / Bohr

       #if 'dipole' in properties: # somehow ASE MD does not request dipole. How to enable?
        if True:
            dm1 = mf.make_rdm1()
            self.results['dipole'] = mf.dip_moment(mol, dm1)

        if self.run_tda:
            td = tddft.TDA(mf)
            e, xy = td.kernel()
            os = oscillator_strength(td, e=e, xy=xy)

            self.results['excited_energies'] = e * nist.HARTREE2EV
            self.results['oscillator_strength'] = os
