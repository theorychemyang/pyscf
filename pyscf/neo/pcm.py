import numpy
from pyscf import lib, gto, df
from pyscf.solvent import pcm
from pyscf.neo import _attach_solvent

@lib.with_doc(_attach_solvent._for_neo_scf.__doc__)
def pcm_for_neo_scf(mf, solvent_obj=None, dm=None):
    if solvent_obj is None:
        solvent_obj = PCM4NEO(mf.mol.components['e']) # Must be e_mol
    return _attach_solvent._for_neo_scf(mf, solvent_obj, dm)

# Inject PCM to other methods
from pyscf import neo
neo.hf.HF.PCM = pcm_for_neo_scf

def _get_charge_from_mol_comp(super_mol, symb):
    if symb == 'e':
        charge = 1
    elif symb == 'p':
        charge = -1
    elif symb.startswith('n'):
        atom_index = int(symb[1:])
        charge = -1 * super_mol.atom_charge(atom_index)
    else:
        raise RuntimeError(f'What is {symb=}?')
    return charge

class PCM4NEO(pcm.PCM):
    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol.components['e'] # Must be e_mol
        self._intermediates = None
        self.surface = None
        self.intopt = None
        return self

    def _get_vind(self, dms):
        if not self._intermediates:
            self.build()

        v_grids = self.v_grids_n
        super_mol = self.mol.super_mol
        for t, comp in super_mol.components.items():
            dms_comp = dms[t]
            nao = dms_comp.shape[-1]
            dms_comp = dms_comp.reshape(-1,nao,nao)
            if dms_comp.shape[0] == 2:
                dms_comp = (dms_comp[0] + dms_comp[1]).reshape(-1,nao,nao)

            v_grids_comp = self._get_v(dms_comp, comp)
            charge = _get_charge_from_mol_comp(super_mol, t)
            v_grids = v_grids - charge * v_grids_comp

        K = self._intermediates['K']
        R = self._intermediates['R']
        b = numpy.dot(R, v_grids.T)
        q = numpy.linalg.solve(K, b).T

        vK_1 = numpy.linalg.solve(K.T, v_grids.T)
        qt = numpy.dot(R.T, vK_1).T
        q_sym = (q + qt)/2.0

        vmat = {}
        for t, comp in super_mol.components.items():
            charge = _get_charge_from_mol_comp(super_mol, t)
            vmat[t] = charge * self._get_vmat(q_sym, comp)[0]
        epcm = 0.5 * numpy.dot(q_sym[0], v_grids[0])

        self._intermediates['q'] = q[0]
        self._intermediates['q_sym'] = q_sym[0]
        self._intermediates['v_grids'] = v_grids[0]
        self._intermediates['dm'] = dms
        return epcm, vmat

    def _get_v(self, dms, mol):
        '''
        return electrostatic potential on surface
        '''
        nao = dms.shape[-1]
        grid_coords = self.surface['grid_coords']
        exponents   = self.surface['charge_exp']
        ngrids = grid_coords.shape[0]
        nset = dms.shape[0]
        v_grids_e = numpy.empty([nset, ngrids])
        max_memory = self.max_memory - lib.current_memory()[0]
        blksize = int(max(max_memory*.9e6/8/nao**2, 400))
        int3c2e = mol._add_suffix('int3c2e')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e)
        for p0, p1 in lib.prange(0, ngrids, blksize):
            fakemol = gto.fakemol_for_charges(grid_coords[p0:p1], expnt=exponents[p0:p1]**2)
            v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e, aosym='s1', cintopt=cintopt)
            for i in range(nset):
                v_grids_e[i,p0:p1] = numpy.einsum('ijL,ij->L',v_nj, dms[i])

        return v_grids_e

    def _get_vmat(self, q, mol):
        nao = mol.nao
        grid_coords = self.surface['grid_coords']
        exponents   = self.surface['charge_exp']
        ngrids = grid_coords.shape[0]
        q = q.reshape([-1,ngrids])
        nset = q.shape[0]
        vmat = numpy.zeros([nset,nao,nao])
        max_memory = self.max_memory - lib.current_memory()[0]
        blksize = int(max(max_memory*.9e6/8/nao**2, 400))

        int3c2e = mol._add_suffix('int3c2e')
        cintopt = gto.moleintor.make_cintopt(mol._atm, mol._bas, mol._env, int3c2e)
        for p0, p1 in lib.prange(0, ngrids, blksize):
            fakemol = gto.fakemol_for_charges(grid_coords[p0:p1], expnt=exponents[p0:p1]**2)
            v_nj = df.incore.aux_e2(mol, fakemol, intor=int3c2e, aosym='s1', cintopt=cintopt)
            for i in range(nset):
                vmat[i] += -numpy.einsum('ijL,L->ij', v_nj, q[i,p0:p1])
        return vmat

    def nuc_grad_method(self, grad_method):
        from pyscf.neo import pcm_grad
        if self.frozen:
            raise RuntimeError('Frozen solvent model is not supported')
        from pyscf.neo import hf
        if isinstance(grad_method.base, hf.HF):
            return pcm_grad.make_grad_object(grad_method)
        else:
            raise NotImplementedError

    def Hessian(self, hess_method):
        pass
        from pyscf.neo import pcm_hess
        if self.frozen:
            raise RuntimeError('Frozen solvent model is not supported')
        from pyscf.neo import cdft
        if isinstance(hess_method.base, cdft.CDFT):
            return pcm_hess.make_hess_object(hess_method)
        else:
            raise NotImplementedError

    def _B_dot_x(self, dms):
        if not self._intermediates:
            self.build()
        vmat = {}
        super_mol = self.mol.super_mol
        for t, comp in super_mol.components.items():
            dms_comp = dms[t]
            out_shape = dms_comp.shape
            nao = dms_comp.shape[-1]
            dms_comp = dms_comp.reshape(-1,nao,nao)

            K = self._intermediates['K']
            R = self._intermediates['R']
            charge = _get_charge_from_mol_comp(super_mol, t)
            v_grids = -charge * self._get_v(dms_comp, comp)

            b = numpy.dot(R, v_grids.T)
            q = numpy.linalg.solve(K, b).T

            vK_1 = numpy.linalg.solve(K.T, v_grids.T)
            qt = numpy.dot(R.T, vK_1).T
            q_sym = (q + qt)/2.0

            vmat[t] = self._get_vmat(q_sym, comp).reshape(out_shape)
        return vmat
