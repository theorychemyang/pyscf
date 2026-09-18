from pyscf.grad import tdrks
from pyscf.df.grad import tdrhf as tdrhf_grad_df
from pyscf import df


class Gradients(tdrks.Gradients):
    _keys = {'with_df', 'auxbasis_response'}

    auxbasis_response = True

    def check_sanity(self):
        assert isinstance(self.base._scf, df.df_jk._DFHF)

    get_jk = tdrhf_grad_df.Gradients.get_jk
    get_j = tdrhf_grad_df.Gradients.get_j
    get_k = tdrhf_grad_df.Gradients.get_k

    def extra_force(self, atom_id, envs):
        if self.auxbasis_response:
            vhf = envs["vhf_aux"][:, :, atom_id]
            e1_aux = vhf[0][0] * 4  # ground state oo0,oo0
            e1_aux += (vhf[0][1] + vhf[1][0])   # dmz1doo, oo0
            e1_aux += vhf[2][2] * 2 # X+Y
            if vhf.shape[0] > 3:
                e1_aux -= vhf[3][3] * 2 # X-Y

            return e1_aux
        else:
            return 0

