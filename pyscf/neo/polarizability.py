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
        self.conv_tol = 1e-4

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
