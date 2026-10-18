'''
Interface for PySCF and ASE

SCF initial-guess extrapolation during MD is opt-in for Pyscf_NEO,
Pyscf_DFT and their TD calculators::

    atoms.calc = Pyscf_NEO(..., extrapolator=True)
    dyn = VelocityVerlet(atoms, timestep=0.5 * units.fs)
    dyn.run(100)

True selects projected G-Ext with four snapshots. Use extrapolator='aspc'
to select a method, or extrapolator={'scheme': 'aspc', 'history_length': 4}
to set its parameters. Explicit predictor objects are also accepted.
At each geometry, extrapolation supplies an initial density for a full SCF
calculation; energies and forces are evaluated from the converged state.
Each new calculated geometry advances the prediction history automatically;
repeated property requests at the same geometry retain one history entry.
Enable this only when each new geometry is the next fixed-timestep MD point.
After SCF converges, the calculator stores its orbitals and the current
geometry in the prediction history.

A geometry optimizer can evaluate trial geometries that it later rejects;
storing those trials as consecutive MD points gives the predictor the wrong
history. Leave extrapolator=None for geometry optimization. MD integrators
that evaluate intermediate geometries need the same care.
Extrapolation can also accelerate geometry optimization: see Askarpour
et al., doi:10.1021/acs.jctc.4c01417, and Song and Lao,
doi:10.1039/D6CP01718E. The history handling needed for those optimization
methods is currently unimplemented in this calculator.
The default extrapolator=None keeps ordinary scanner behavior for geometry
optimization and single-point calculations.
TD calculators predict only the underlying ground-state density; excitation
amplitudes are computed by the ordinary TD solver at each geometry.
'''

from contextlib import nullcontext

from ase.calculators.calculator import Calculator
from ase.units import Bohr, Hartree
from pyscf.data import nist
from pyscf import neo
from pyscf import gto, dft, tddft, scf
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


def _make_extrapolator(option):
    '''Convert ASE's boolean/string/dictionary options to a predictor.'''
    from pyscf.neo.extrapolation import Extrapolator
    if option is True:
        return Extrapolator()
    if option is False:
        return None
    if isinstance(option, str):
        return Extrapolator(scheme=option)
    if isinstance(option, dict):
        return Extrapolator(**option)
    return option


def _run_scf(calc, mol, forces, mf=None):
    '''Run ground-state SCF using a scanner or a supplied fresh SCF object.'''
    scanner = mf is None
    if calc.extrapolator is None and scanner:
        if forces:
            return calc.mf_grad_scanner(mol)
        return calc.mf_scanner(mol), None
    if scanner:
        mf = calc.mf_scanner
    predictor = calc.extrapolator
    dm0 = None if predictor is None else predictor.kernel(mf, mol)
    if forces and scanner:
        calc.mf_grad_scanner.reset(mol)
    # GPU gradient scanners forward kwargs to the gradient kernel. Supply
    # dm0 to SCF directly. Fresh objects can also refine their orbitals through
    # stability optimization; record its final Fock for the stored orbitals.
    with nullcontext() if predictor is None else predictor.record_fock(mf):
        if scanner:
            energy = mf(mol, dm0=dm0)
        else:
            mf.scf(dm0=dm0)
            if calc.stable_opt:
                stable_opt_internal(mf)
            energy = mf.e_tot
    if predictor is not None and not mf.converged:
        raise RuntimeError('SCF failed; Extrapolator history is unchanged')
    gradient = None
    if forces:
        if scanner:
            gradient = calc.mf_grad_scanner.kernel()
        else:
            gradient = mf.nuc_grad_method().set(grid_response=calc.grid_response).grad()
    if predictor is not None:
        predictor.update(mf)
    return energy, gradient


def _run_td(calc, mol, forces, reuse=False):
    '''Share ground-state SCF guess extrapolation and TD solves between calculators.'''
    if reuse:
        # Separate atoms.get_potential_energy() then atoms.get_forces() calls
        # at unchanged coordinates reuse the converged SCF and TD amplitudes;
        # only the excited-state gradient remains to be evaluated.
        # Energy-only TD scanner calls update the TD molecule. Its gradient
        # object needs that molecule too; keep the converged SCF/TD state.
        calc.td_grad_scanner.mol = calc.td_scanner.mol
        gradient = calc.td_grad_scanner.kernel(state=calc.state) if forces else None
        return calc.results['energy'] / Hartree, gradient
    initialize = calc.td_scanner is None
    if calc.extrapolator is None and not initialize:
        if forces:
            return calc.td_grad_scanner(mol)
        return calc.td_scanner(mol)[calc.state-1], None
    mf = calc.create_mf(mol) if initialize else calc.td_scanner._scf
    predictor = calc.extrapolator
    dm0 = None if predictor is None else predictor.kernel(mf, mol)
    if not initialize:
        calc.td_grad_scanner.reset(mol)
    with nullcontext() if predictor is None else predictor.record_fock(mf):
        if initialize:
            mf.kernel(dm0=dm0)
        else:
            mf(mol, dm0=dm0)
    if predictor is not None and not mf.converged:
        raise RuntimeError('SCF failed; Extrapolator history is unchanged')
    if initialize:
        # TD constructors run SCF when orbitals are absent. Supplying the
        # converged reference makes first-point SCF and Fock capture explicit.
        td, grad = calc.create_tdmf(mol, mf)
        calc.td_grad_scanner = grad.as_scanner(state=calc.state)
        calc.td_scanner = calc.td_grad_scanner.base
        calc.mf_scanner = calc.td_scanner._scf
    # Calling the TD kernel directly avoids the scanner's second SCF call.
    calc.td_scanner.kernel()
    energy = mf.e_tot + calc.td_scanner.e[calc.state-1]
    gradient = calc.td_grad_scanner.kernel(state=calc.state) if forces else None
    if predictor is not None:
        predictor.update(mf)
    return energy, gradient


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
                 den_fit_nn=False,         # if density fit the nuclear-nuclear Coulomb
                 to_gpu=False,             # use GPU4PySCF
                 force_unrestricted=False, # can force mf to be unrestricted
                 stable_opt=False,         # if check stability
                 extrapolator=None,        # SCF guess extrapolation during MD
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
        self.den_fit_nn = den_fit_nn
        self.to_gpu = to_gpu
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
        self.extrapolator = _make_extrapolator(extrapolator)
        if self.add_solvent or self.run_tda or self.stable_opt or self.efield is not None:
            # TODO: see if some of them can work with scanners
            self.scanner_available = False
        elif hasattr(self, 'create_tdmf'):
            # TD setup waits for the first real molecule to avoid an SCF on
            # the placeholder used by ground-state scanner construction.
            self.scanner_available = True
        else:
            # initialize a fake mol then create scanners
            mol = neo.M(atom='H 0 0 0; F 0 0 0.9')
            mf = self.create_mf(mol)
            self.mf_grad_scanner = mf.nuc_grad_method().set(grid_response=self.grid_response).as_scanner()
            self.mf_scanner = self.mf_grad_scanner.base
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
                                                    df_ne=self.den_fit_ne,
                                                    df_nn=self.den_fit_nn)
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
        if self.to_gpu:
            mf = mf.to_gpu()
        return mf

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        mol = self.get_mol_from_atoms(atoms)
        if not system_changes and 'energy' in self.results:
            # Separate atoms.get_potential_energy() then atoms.get_forces()
            # calls at unchanged coordinates reuse this converged SCF state;
            # only the gradient remains to be evaluated.
            mf = self._last_mf
            e_tot = mf.e_tot
            if 'forces' in properties:
                de = mf.nuc_grad_method().set(grid_response=self.grid_response).grad()
        elif self.scanner_available:
            mf = self.mf_scanner
            e_tot, de = _run_scf(self, mol, 'forces' in properties)
        else:
            mf = self.create_mf(mol)
            if self.disp:
                mf.disp = self.disp
            if self.add_solvent:
                mf = mf.ddCOSMO()
            e_tot, de = _run_scf(self, mol, 'forces' in properties, mf)
        self._last_mf = mf
        self.results['energy'] = e_tot * Hartree
        if 'forces' in properties:
            self.results['forces'] = -de * Hartree / Bohr

        #if 'dipole' in properties: # somehow ASE MD does not request dipole. How to enable?
        if True:
            self.results['dipole'] = mf.dip_moment()

        if self.run_tda and (system_changes or 'excited_energies' not in self.results):
            # calculate excited energies and oscillator strength by TDDFT/TDA
            td = tddft.TDA(mf.mf_elec)
            e, xy = td.kernel()
            os = oscillator_strength(td, e=e, xy=xy)

            self.results['excited_energies'] = e * nist.HARTREE2EV
            self.results['oscillator_strength'] = os

class Pyscf_TDNEO(Pyscf_NEO):
    '''CNEO-TDDFT PySCF calculator'''
    implemented_properties = ['energy', 'forces', 'excitation-energy']
    def __init__(self, state=1,
                 nstates=3,
                 is_davidson=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.run_tda = False
        if not self.scanner_available:
            raise RuntimeError('mf_scanner not initialized')
        if self.epc is not None:
            raise NotImplementedError('EPC not supported for CNEO-TDDFT gradient')

        self.scanner_available = False
        self.state = state
        self.nstates = nstates
        self.is_davidson = is_davidson

        self.td_scanner = None
        self.td_grad_scanner = None
        self.scanner_available = True

    def create_tdmf(self, mol, mf=None):
        if mf is None:
            mf = self.create_mf(mol)
        if self.is_davidson:
            td_mf = mf.TDDFT()
            td_mf.nstates = self.nstates
        else:
            td_mf = mf.TDDirect()
            td_mf.nstates = self.nstates

        td_grad = td_mf.Gradients()
        td_grad.state = self.state

        return td_mf, td_grad

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        mol = self.get_mol_from_atoms(atoms)
        if not self.scanner_available:
            raise RuntimeError('td scanner not initialized')

        e_tot, de = _run_td(self, mol, 'forces' in properties,
                            reuse=not system_changes and 'energy' in self.results)
        td_mf = self.td_scanner

        self.results['energy'] = e_tot * Hartree
        if 'forces' in properties:
            self.results['forces'] = -de * Hartree / Bohr
        if 'excitation-energy' in properties:
            e_ex = td_mf.e
            self.results['excitation-energy'] = e_ex * nist.HARTREE2EV



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
                 to_gpu=False,             # use GPU4PySCF
                 force_unrestricted=False, # can force mf to be unrestricted
                 stable_opt=False,         # if check stability
                 extrapolator=None,        # SCF guess extrapolation during MD
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
        self.to_gpu = to_gpu
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
        self.extrapolator = _make_extrapolator(extrapolator)
        if self.add_solvent or self.run_tda or self.stable_opt or self.efield is not None:
            # TODO: see if some of them can work with scanners
            self.scanner_available = False
        elif hasattr(self, 'create_tdmf'):
            self.scanner_available = True
        else:
            # initialize a fake mol then create scanners
            mol = gto.M(atom='H 0 0 0; F 0 0 0.9')
            mf = self.create_mf(mol)
            self.mf_grad_scanner = mf.nuc_grad_method().set(grid_response=self.grid_response).as_scanner()
            self.mf_scanner = self.mf_grad_scanner.base
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
        if self.to_gpu:
            mf = mf.to_gpu()
        return mf

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        mol = self.get_mol_from_atoms(atoms)
        if not system_changes and 'energy' in self.results:
            # Separate atoms.get_potential_energy() then atoms.get_forces()
            # calls at unchanged coordinates reuse this converged SCF state;
            # only the gradient remains to be evaluated.
            mf = self._last_mf
            e_tot = mf.e_tot
            if 'forces' in properties:
                de = mf.nuc_grad_method().set(grid_response=self.grid_response).grad()
        elif self.scanner_available:
            mf = self.mf_scanner
            e_tot, de = _run_scf(self, mol, 'forces' in properties)
        else:
            mf = self.create_mf(mol)
            if self.disp:
                mf.disp = self.disp
            if self.add_solvent:
                from pyscf import solvent
                mf = mf.ddCOSMO()
            e_tot, de = _run_scf(self, mol, 'forces' in properties, mf)
        self._last_mf = mf
        self.results['energy'] = e_tot * Hartree
        if 'forces' in properties:
            self.results['forces'] = -de * Hartree / Bohr

        #if 'dipole' in properties: # somehow ASE MD does not request dipole. How to enable?
        if True:
            self.results['dipole'] = mf.dip_moment()

        if self.run_tda and (system_changes or 'excited_energies' not in self.results):
            td = tddft.TDA(mf)
            e, xy = td.kernel()
            os = oscillator_strength(td, e=e, xy=xy)

            self.results['excited_energies'] = e * nist.HARTREE2EV
            self.results['oscillator_strength'] = os


class Pyscf_TDDFT(Pyscf_DFT):
    '''TDDFT PySCF calculator'''
    implemented_properties = ['energy', 'forces', 'excitation-energy']
    def __init__(self, state=1,
                 nstates=3,
                 **kwargs):
        super().__init__(**kwargs)
        self.run_tda = False
        if not self.scanner_available:
            raise RuntimeError('mf_scanner not initialized')

        self.scanner_available = False
        self.state = state
        self.nstates = nstates

        self.td_scanner = None
        self.td_grad_scanner = None
        self.scanner_available = True

    def create_tdmf(self, mol, mf=None):
        if mf is None:
            mf = self.create_mf(mol)
        td_mf = mf.TDDFT()
        td_mf.nstates = self.nstates

        td_grad = td_mf.Gradients()
        td_grad.state = self.state

        return td_mf, td_grad

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        mol = self.get_mol_from_atoms(atoms)
        if not self.scanner_available:
            raise RuntimeError('td scanner not initialized')

        e_tot, de = _run_td(self, mol, 'forces' in properties,
                            reuse=not system_changes and 'energy' in self.results)
        td_mf = self.td_scanner

        self.results['energy'] = e_tot * Hartree
        if 'forces' in properties:
            self.results['forces'] = -de * Hartree / Bohr
        if 'excitation-energy' in properties:
            e_ex = td_mf.e
            self.results['excitation-energy'] = e_ex * nist.HARTREE2EV
