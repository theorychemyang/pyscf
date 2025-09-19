from pyscf import scf

def remove_linear_dep(mf, **kwargs):

    for t, comp in mf.components.items():
        mf.components[t] = scf.addons.remove_linear_dep(comp, **kwargs)

    return mf

