'''
Analytical polarizability for CNEO
'''
from functools import reduce
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.neo import cphf
from pyscf.neo import _response_functions  


def polarizability(polobj,with_cphf=True):
    log = logger.new_logger(polobj)
    mf = polobj._scf
    mol = mf.mol
    mo_energy = mf.mo_energy
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    #orbv = mo_coeff[:,~occidx]

    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    charge_center = numpy.einsum('i,ix->x', charges, coords) / charges.sum()
    h1 = {}
    s1 = {}
    for t, comp in mf.components.items():
        with comp.mol.with_common_orig(charge_center):
            int1e_r = comp.mol.intor_symmetric('int1e_r', comp=3)
        occidx = mo_occ[t] > 0
        mocc = mo_coeff[t][:, occidx]
        h1[t] = lib.einsum('xuv, ui, vj -> xij', int1e_r, mo_coeff[t], mocc) * comp.charge
        s1[t] = numpy.zeros_like(h1[t])
    vind = polobj.gen_vind(mf, mo_coeff, mo_occ)
    if with_cphf:
        mo1 = cphf.solve(vind, mo_energy, mo_occ, h1, s1,with_f1=True,
                         max_cycle=polobj.max_cycle_cphf,tol=polobj.conv_tol,
                         verbose=log)[0]
    else:
        raise NotImplementedError('without cphf is not implemented yet')

    e2 = lib.einsum('xpi,ypi->xy', h1['e'], mo1['e'])
    # *-1 from the definition of dipole moment. *2 for double occupancy
    e2 = (e2 + e2.T) * -2

    if mf.verbose >= logger.INFO:
        xx, yy, zz = e2.diagonal()
        log.note('Isotropic polarizability %.12g', (xx+yy+zz)/3)
        log.note('Polarizability anisotropy %.12g',
                 (.5 * ((xx-yy)**2 + (yy-zz)**2 + (zz-xx)**2))**.5)
        log.debug('Static polarizability tensor\n%s', e2)
    return e2


def hyper_polarizability(polobj, with_cphf=True):
    return NotImplementedError('hyperpolarizability is not implemented yet')


def polarizability_with_freq(polobj, freq, with_cphf=True):
    return NotImplementedError('frequency-dependent polarizability is not implemented yet')


class Polarizability(lib.StreamObject):
    def __init__(self, mf):
        mol = mf.mol
        self.mol = mol
        self.verbose = mol.verbose
        self.stdout = mol.stdout
        self._scf = mf

        self.cphf = True
        self.max_cycle_cphf = 100
        ### Convergence tolerance for the CPHF equation
        ### default: 1e-4 is enough for polarizability(the same order of magnitude as the numerical error)
        self.conv_tol = 1e-6

        self._keys = set(self.__dict__.keys())  

    def gen_vind(self,mf, mo_coeff, mo_occ):
        nao = {}
        nmo = {}
        mocc = {}
        nocc = {}
        is_component_unrestricted = {}
        for t in mo_coeff.keys():
            mo_coeff[t] = numpy.asarray(mo_coeff[t])
            if mo_coeff[t].ndim > 2: # unrestricted
                assert not t.startswith('n')
                assert mo_coeff[t].shape[0] == 2
                is_component_unrestricted[t] = True
                nao[t], nmoa = mo_coeff[t][0].shape
                nmob = mo_coeff[t][1].shape[1]
                nmo[t] = (nmoa, nmob)
                mo_occ[t] = numpy.asarray(mo_occ[t])
                assert mo_occ[t].ndim > 1 and mo_occ[t].shape[0] == 2
                mocca = mo_coeff[t][0][:,mo_occ[t][0]>0]
                moccb = mo_coeff[t][1][:,mo_occ[t][1]>0]
                mocc[t] = (mocca, moccb)
                nocca = mocca.shape[1]
                noccb = moccb.shape[1]
                nocc[t] = (nocca, noccb)
            else: # restricted
                is_component_unrestricted[t] = False
                nao[t], nmo[t] = mo_coeff[t].shape
                mocc[t] = mo_coeff[t][:,mo_occ[t]>0]
                nocc[t] = mocc[t].shape[1]
        vresp = mf.gen_response(mo_coeff, mo_occ, hermi=1)
        def fx(mo1, f1=None):
            dm1 = {}
            for t, comp in mo1.items():
                if is_component_unrestricted[t]:
                    nmoa, nmob = nmo[t]
                    mocca, moccb = mocc[t]
                    nocca, noccb = nocc[t]
                    comp = comp.reshape(-1,nmoa*nocca+nmob*noccb)
                    nset = len(comp)
                    dm1[t] = numpy.empty((2,nset,nao[t],nao[t]))
                    for i, x in enumerate(comp):
                        xa = x[:nmoa*nocca].reshape(nmoa,nocca)
                        xb = x[nmoa*nocca:].reshape(nmob,noccb)
                        dma = reduce(numpy.dot, (mo_coeff[t][0], xa, mocca.T))
                        dmb = reduce(numpy.dot, (mo_coeff[t][1], xb, moccb.T))
                        dm1[t][0,i] = dma + dma.T
                        dm1[t][1,i] = dmb + dmb.T
                else:
                    comp = comp.reshape(-1,nmo[t],nocc[t])
                    nset = len(comp)
                    dm1[t] = numpy.empty((nset,nao[t],nao[t]))
                    for i, x in enumerate(comp):
                        if t.startswith('n'):
                            # quantum nuclei are always singly occupied
                            dm = reduce(numpy.dot, (mo_coeff[t], x, mocc[t].T))
                        else:
                            # *2 for double occupancy (RHF electrons)
                            dm = reduce(numpy.dot, (mo_coeff[t], x*2, mocc[t].T))
                        dm1[t][i] = dm + dm.T
            v1 = vresp(dm1)
            v1vo = {}
            if f1 is None:
                r1vo = None
            else:
                r1vo = {}
            for t, comp in mo1.items():
                if is_component_unrestricted[t]:
                    nmoa, nmob = nmo[t]
                    mocca, moccb = mocc[t]
                    nocca, noccb = nocc[t]
                    comp = comp.reshape(-1,nmoa*nocca+nmob*noccb)
                    nset = len(comp)
                    v1vo[t] = numpy.empty_like(comp)
                    for i in range(nset):
                        v1vo[t][i,:nmoa*nocca] = reduce(numpy.dot, (mo_coeff[t][0].T, v1[t][0,i], mocca)).ravel()
                        v1vo[t][i,nmoa*nocca:] = reduce(numpy.dot, (mo_coeff[t][1].T, v1[t][1,i], moccb)).ravel()
                else:
                    comp = comp.reshape(-1,nmo[t],nocc[t])
                    v1vo[t] = numpy.empty_like(comp)
                    for i, x in enumerate(v1[t]):
                        v1vo[t][i] = reduce(numpy.dot, (mo_coeff[t].T, x, mocc[t]))
                if f1 is not None and t in f1 and t.startswith('n'):
                    # DEBUG: Verify nuclear dm1 * int1e_r
                    # if debug:
                    #     position = numpy.einsum('aij,xij->ax', dm1[t], mf.components[t].int1e_r)
                    #     print(f'[DEBUG] norm(dm1 * int1e_r) for {t}: {numpy.linalg.norm(position)}')
                    rvo = numpy.empty((3,nmo[t],nocc[t]))
                    for i, x in enumerate(mf.components[t].int1e_r):
                        rvo[i] = reduce(numpy.dot, (mo_coeff[t].T, x, mocc[t]))
                    # Calculate f1 * r and add to nuclear Fock derivative
                    v1vo[t] += numpy.einsum('ax,xpi->api', f1[t], rvo)
                    # Store r * mo1, which will lead to equation r * mo1 = 0
                    r1vo[t] = numpy.einsum('api,xpi->ax', comp, rvo)
            return v1vo, r1vo
        return fx


    polarizability = polarizability
    polarizability_with_freq = polarizability_with_freq
    hyper_polarizability = hyper_polarizability

if __name__ == '__main__':
    from pyscf import gto, scf
    from pyscf import neo
    from pyscf.prop.polarizability.rks import Polarizability as PolarizabilityRKS
    import time
    atom0 = '''  H  ,  0.   0.   0.
                F  ,  0.5   0.5   .6 '''
    atom='''O      -0.238592051971541      3.392862580374811     -0.162250372572209
            H      -0.372155346926180      4.140860293438132      0.458993573797752
            O      -0.372179575211653      0.927817762829201      0.930787926659881
            H       0.558585632607203      0.942689527210511      0.537530265546420
            H      -0.569749794403297      2.564812770048602      0.313389233846506
            O       1.882940760094289      1.722608203620976     -0.321583375288254
            H       2.708145104394429      1.960187678129227      0.153839472332841
            H       1.320382834214582      2.560169548693901     -0.357958027550273
            H      -0.849540572154995      0.199065638210133      0.477693313245828'''
    mol=neo.M(
          atom = atom,
        basis = 'cc-pvdz',
        quantum_nuc = ['H'])
    mf = neo.CDFT(mol,xc='b3lyp')
    mf.run(conv_tol=1e-14,max_cycle=1000)
    polobj1 = Polarizability(mf)
    polobj1.verbose = 5
    t_0 = time.time()
    pol = polobj1.polarizability()
    t_1 = time.time()
    print('CNEO pol time:', t_1-t_0)
    # charges = mol.atom_charges()
    # coords  = mol.atom_coords()
    # charge_center = numpy.einsum('i,ix->x', charges, coords) / charges.sum()
    # ao_dip = {}
    # h1 = {}
    # def apply_E(E):
    #     hcore = {}
    #     for t, comp in mol.components.items():
    #         hcore[t] = mf.components[t].get_hcore(mol=comp).copy()
    #         hcore[t] -= numpy.einsum('x,xij->ij', E, comp.intor('int1e_r', comp=3)) * comp.charge
    #     def get_hcore(_mol=None):
    #         return hcore
    #     mf.get_hcore = get_hcore
    #     mf.conv_tol = 1e-14
    #     mf.max_cycle = 1000
    #     mf.kernel()
    #     return mf.dip_moment(mol, mf.make_rdm1(), unit='AU', verbose=0)
    # e1 = apply_E([ 0.0001, 0, 0])
    # e2 = apply_E([-0.0001, 0, 0])
    # a1 = (e1 - e2) / 0.0002
    # e1 = apply_E([0, 0.0001, 0])
    # e2 = apply_E([0,-0.0001, 0])
    # a2 = (e1 - e2) / 0.0002
    # e1 = apply_E([0, 0, 0.0001])
    # e2 = apply_E([0, 0,-0.0001])
    # a3 = (e1 - e2) / 0.0002
    # numpol = numpy.array([a1, a2, a3])
    # print('numerical CNEO polarizability\n',numpol)
    # print(numpy.allclose(numpol, Polarizability(mf).polarizability(),atol=1e-3)) 
    # print('max_diff',numpy.max(abs(numpol - Polarizability(mf).polarizability())))
    # print("new time:", t1-t0)
    ### not all close but comnpared with DFT the tolerance is comparable
    mol1=gto.M(
          atom = atom,
        basis = 'cc-pvdz',)
    mf1 = scf.RKS(mol1)
    mf1.xc='b3lyp'
    mf1.kernel()
    # mf1.verbose = 5
    polobj = PolarizabilityRKS(mf1)
    polobj.verbose = 5
    t_0 = time.time()
    pol1 = polobj.polarizability()
    t_1 = time.time()
    print('DFT pol time:', t_1-t_0)
    # with mol.with_common_orig(charge_center):
    #     ao_dip = mol.intor_symmetric('int1e_r', comp=3)
    # h1 = mf1.get_hcore()
    # def applyDFT(E):
    #     mf1.get_hcore = lambda *args, **kwargs: h1 + numpy.einsum('x,xij->ij', E, ao_dip)
    #     mf1.run(conv_tol=1e-14)
    #     return mf1.dip_moment(mol, mf1.make_rdm1(), unit='AU', verbose=0)
    # e1 = applyDFT([ 0.0001, 0, 0])
    # e2 = applyDFT([-0.0001, 0, 0])
    # a1 = (e1 - e2) / 0.0002
    # e1 = applyDFT([0, 0.0001, 0])
    # e2 = applyDFT([0,-0.0001, 0])
    # a2 = (e1 - e2) / 0.0002                 
    # e1 = applyDFT([0, 0, 0.0001])
    # e2 = applyDFT([0, 0,-0.0001])
    # a3 = (e1 - e2) / 0.0002
    # numpol = numpy.array([a1, a2, a3])
    # t0 = time.time()
    # print('analytical DFT polarizability\n',PolarizabilityRKS(mf1).polarizability())
    # t1 = time.time()
    # print('numerical DFT polarizability\n',numpol)
    # print("DFT time:", t1-t0)
    # print(numpy.allclose(numpol, PolarizabilityRKS(mf1).polarizability(),atol=1e-3)) 
    # print('max_diff',numpy.max(abs(numpol - PolarizabilityRKS(mf1).polarizability())))
##########################################
# analytical CNEO polarizability
#  [[2.45253419 0.81346612 0.97615912]
#  [0.81346612 2.45253419 0.97615912]
#  [0.97615912 0.97615912 2.81046011]]
# numerical CNEO polarizability
#  [[2.45253471 0.81346662 0.97615973]
#  [0.81347571 2.45254384 0.97616871]
#  [0.97616082 0.97616082 2.81046217]]
# True
# max_diff 0.0007018651113224195
# new time: 0.5280303955078125
# analytical DFT polarizability
#  [[2.42798567 0.79575737 0.95477746]
#  [0.79575737 2.42798567 0.95477746]
#  [0.95477746 0.95477746 2.77780492]]
# numerical DFT polarizability
#  [[2.42827663 0.79589134 0.95506939]
#  [0.79589016 2.42827593 0.95506797]
#  [0.95506    0.95506141 2.77846026]]
# DFT time: 0.23527979850769043
# True
# max_diff 0.0006553384576553078
############################################
mol=neo.M(
        atom = atom,
    basis = 'cc-pvdz',
    quantum_nuc = ['H'])
mf = neo.CDFT(mol,xc='b3lyp')
mf.kernel()
hess = mf.Hessian()
hess.verbose = 5
t_0 = time.time()
hessian = hess.kernel()
t_1 = time.time()
print('CNEO Hessian time:', t_1-t_0)
mol1=gto.M(
        atom = atom,
    basis = 'cc-pvdz',)
mf1 = scf.RKS(mol1)
mf1.xc='b3lyp'
mf1.kernel()

hess1 = mf1.Hessian()
hess1.verbose = 5
t_0 = time.time()
hessian = hess1.kernel()
t_1 = time.time()
print('DFT Hessian time:', t_1-t_0)