from pyscf.neo import tdgrad
from pyscf.neo import df


class Gradients(tdgrad.Gradients):
    _keys = {'auxbasis_response'}

    auxbasis_response = True

    def check_sanity(self):
        assert isinstance(self.base._scf, df._DFNEO)

    def extra_force(self, atom_id, envs):
        if self.auxbasis_response:
            g_e = envs['td_grad_e']
            return g_e.extra_force(atom_id, envs)
        else:
            return 0
