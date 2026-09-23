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
    extra_force = tdrhf_grad_df.Gradients.extra_force

