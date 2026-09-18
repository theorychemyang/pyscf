from pyscf.grad import tduhf
from pyscf.df.grad import tdrhf as tdrhf_grad_df
from pyscf import df


class Gradients(tduhf.Gradients):
    _keys = {'with_df', 'auxbasis_response'}

    auxbasis_response = True

    def check_sanity(self):
        assert isinstance(self.base._scf, df.df_jk._DFHF)

    get_jk = tdrhf_grad_df.Gradients.get_jk
    get_j = tdrhf_grad_df.Gradients.get_j
    get_k = tdrhf_grad_df.Gradients.get_k

    def extra_force(self, atom_id, envs):
        if self.auxbasis_response:
            vj = envs['vj_aux'][:, :, atom_id]
            nset = vj.shape[0] // 2
            # Spin-major density sets: J couples both spins, K only equal spins.
            vhf = vj.reshape(2, nset, 2, nset, 3).sum((0, 2))
            if envs['vk_aux'] is not None:
                vk = envs['vk_aux'][:, :, atom_id].reshape(2, nset, 2, nset, 3)
                vhf -= vk[0, :, 0] + vk[1, :, 1]
            # Unrestricted relaxed densities carry 1/4, transition densities 1/2.
            e1_aux = vhf[0][0]
            e1_aux += (vhf[0][1] + vhf[1][0]) * .25
            e1_aux += vhf[2][2] * .5
            if nset > 3:
                e1_aux -= vhf[3][3] * .5
            return e1_aux
        else:
            return 0

Grad = Gradients
