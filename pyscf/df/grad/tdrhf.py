from pyscf.grad import tdrhf
from pyscf import df


class Gradients(tdrhf.Gradients):
    _keys = {'with_df', 'auxbasis_response'}

    auxbasis_response = True

    def __init__(self, td):
        tdrhf.Gradients.__init__(self, td)


    def check_sanity(self):
        assert isinstance(self.base._scf, df.df_jk._DFHF)

    def get_jk(self, mol=None, dm=None, hermi=0, with_j=True, with_k=True,
               omega=None):
        mf = self.base._scf
        mf_grad = mf.Gradients()
        return mf_grad.get_jk(mol, dm, hermi, with_j, with_k, omega)

    def get_j(self, mol=None, dm=None, hermi=0, omega=None):
        mf = self.base._scf
        mf_grad = mf.Gradients()
        return mf_grad.get_j(mol, dm, hermi, omega)

    def get_k(self, mol=None, dm=None, hermi=0, omega=None):
        mf = self.base._scf
        mf_grad = mf.Gradients()
        return mf_grad.get_k(mol, dm, hermi,omega)

    def extra_force(self, atom_id, envs):
        if self.auxbasis_response:
            vhf = envs['vhf_aux']
            e1_aux = vhf[0][0] * 4  # ground state oo0,oo0
            e1_aux += (vhf[0][1] + vhf[1][0])   # dmz1doo, oo0
            e1_aux += vhf[2][2] * 2 # X+Y
            e1_aux -= vhf[3][3] * 2 # X-Y

            return e1_aux[atom_id]
        else:
            return 0

