from pyscf.grad import tduks
from pyscf.df.grad import tduhf as tduhf_grad_df
from pyscf import df


class Gradients(tduks.Gradients):
    _keys = {'with_df', 'auxbasis_response'}

    auxbasis_response = True

    def check_sanity(self):
        assert isinstance(self.base._scf, df.df_jk._DFHF)

    get_jk = tduhf_grad_df.Gradients.get_jk
    get_j = tduhf_grad_df.Gradients.get_j
    get_k = tduhf_grad_df.Gradients.get_k
    extra_force = tduhf_grad_df.Gradients.extra_force

Grad = Gradients
