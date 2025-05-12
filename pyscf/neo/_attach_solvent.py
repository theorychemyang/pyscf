import copy
import numpy
from pyscf import lib, scf
from pyscf.lib import logger
from pyscf.solvent._attach_solvent import _Solvation

def _for_neo_scf(mf, solvent_obj, dm=None):
    '''Add solvent model to NEO-SCF (HF and DFT) method.

    Kwargs:
        dm : if given, solvent does not respond to the change of density
            matrix. A frozen potential is added to the results.
    '''
    if isinstance(mf, _Solvation):
        mf.with_solvent = solvent_obj
        return mf

    if dm is not None:
        solvent_obj.e, solvent_obj.v = solvent_obj.kernel(dm)
        solvent_obj.frozen = True

    sol_mf = NEOSCFWithSolvent(mf, solvent_obj)
    name = solvent_obj.__class__.__name__ + mf.__class__.__name__
    return lib.set_class(sol_mf, (NEOSCFWithSolvent, mf.__class__), name)

class NEOSCFWithSolvent(_Solvation):
    _keys = {'with_solvent'}

    def __init__(self, mf, solvent):
        self.__dict__.update(mf.__dict__)
        self.with_solvent = solvent

    def undo_solvent(self):
        cls = self.__class__
        name_mixin = self.with_solvent.__class__.__name__
        obj = lib.view(self, lib.drop_class(cls, NEOSCFWithSolvent, name_mixin))
        del obj.with_solvent
        return obj

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        self.with_solvent.check_sanity()
        self.with_solvent.dump_flags(verbose)
        return self

    def reset(self, mol=None):
        self.with_solvent.reset(mol)
        return super().reset(mol)

    # Note v_solvent should not be added to get_hcore for scf methods.
    # get_hcore is overloaded by many post-HF methods. Modifying
    # SCF.get_hcore may lead error.

    def get_veff(self, mol=None, dm=None, *args, **kwargs):
        vhf = super().get_veff(mol, dm, *args, **kwargs)
        with_solvent = self.with_solvent
        if not with_solvent.frozen:
            with_solvent.e, with_solvent.v = with_solvent.kernel(dm)
        e_solvent, v_solvent = with_solvent.e, with_solvent.v

        # NOTE: v_solvent should not be added to vhf in this place. This is
        # because vhf is used as the reference for direct_scf in the next
        # iteration. If v_solvent is added here, it may break direct SCF.
        for t, comp in vhf.items():
            vhf[t] = lib.tag_array(comp, e_solvent=e_solvent, v_solvent=v_solvent[t])
        return vhf

    def get_fock(mf, h1e=None, s1e=None, vhf=None, vint=None, dm=None, cycle=-1,
                 diis=None, diis_start_cycle=None, level_shift_factor=None,
                 damp_factor=None, fock_last=None, diis_pos='both', diis_type=3):
        if dm is None: dm = self.make_rdm1()

        # DIIS was called inside super().get_fock. v_solvent, as a function of
        # dm, should be extrapolated as well. To enable it, v_solvent has to be
        # added to the fock matrix before DIIS was called.
        if getattr(vhf['e'], 'v_solvent', None) is None:
            vhf = self.get_veff(self.mol, dm)
        vhf_copy = copy.deepcopy(vhf)
        for t, comp in vhf_copy.items():
            vhf_copy[t] += vhf[t].v_solvent
        return super().get_fock(h1e, s1e, vhf_copy, vint, dm, cycle, diis,
                                diis_start_cycle, level_shift_factor, damp_factor,
                                fock_last, diis_pos, diis_type)

    def energy_elec(self, dm=None, h1e=None, vhf=None, vint=None):
        if dm is None:
            dm = self.make_rdm1()
        if getattr(vhf['e'], 'e_solvent', None) is None:
            vhf = self.get_veff(self.mol, dm)

        e_tot, e_coul = super().energy_elec(dm, h1e, vhf)
        e_solvent = vhf['e'].e_solvent
        e_tot += e_solvent
        self.scf_summary['e_solvent'] = vhf['e'].e_solvent.real

        if (hasattr(self.with_solvent, 'method') and
            self.with_solvent.method.upper() == 'SMD'):
            if self.with_solvent.e_cds is None:
                e_cds = self.with_solvent.get_cds()
                self.with_solvent.e_cds = e_cds
            else:
                e_cds = self.with_solvent.e_cds

            if isinstance(e_cds, numpy.ndarray):
                e_cds = e_cds[0]
            e_tot += e_cds
            self.scf_summary['e_cds'] = e_cds
            logger.info(self, f'CDS correction = {e_cds:.15f}')
        logger.info(self, 'Solvent Energy = %.15g', vhf['e'].e_solvent)
        return e_tot, e_coul

    def nuc_grad_method(self):
        grad_method = super().nuc_grad_method()
        return self.with_solvent.nuc_grad_method(grad_method)

    Gradients = nuc_grad_method

    def Hessian(self):
        hess_method = super().Hessian()
        return self.with_solvent.Hessian(hess_method)

    def gen_response(self, *args, **kwargs):
        vind = super().gen_response(*args, **kwargs)
        # * singlet=None is orbital hessian or CPHF type response function.
        # Except TDDFT, this is the default case for all response calculations
        # (such as stability analysis, SOSCF, polarizability and Hessian).
        # * In TDDFT, this setting only affect RHF wfn. The UHF wfn does not
        # depend on the setting of "singlet".
        # * For RHF reference, the triplet excitation does not change the total
        # electron density, thus does not lead to solvent response.
        singlet = kwargs.get('singlet', True)
        singlet = singlet or singlet is None
        def vind_with_solvent(dm1):
            v = vind(dm1)
            if self.with_solvent.equilibrium_solvation:
                v_solvent = self.with_solvent._B_dot_x(dm1)
                for t, comp in self.components.items():
                    if t == 'e' or t == 'p':
                        is_uhf = isinstance(comp, scf.uhf.UHF)
                        if is_uhf:
                            v[t] += v_solvent[t][0] + v_solvent[t][1]
                        elif singlet:
                            v[t] += v_solvent[t]
                        else:
                            # The response of electron density should be strictly zero
                            # for TDDFT triplet
                            pass
                    else:
                        # ??? (C)NEO-TDDFT? Should nuclei have solvent response?
                        # CNEO-TDDFT with frozen nuclei?
                        if singlet:
                            v[t] += v_solvent[t]
                        else:
                            raise NotImplementedError
            return v
        return vind_with_solvent

    def stability(self, *args, **kwargs):
        raise NotImplementedError

    def to_gpu(self):
        raise NotImplementedError
