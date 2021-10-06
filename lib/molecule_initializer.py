from psi4 import SCFConvergenceError
import tequila as tq

__all__ = ['get_molecule_initializer']


def initialize_molecule(r, geometry, name, basis_set, active_orbitals, transformation, n_pno):
    try:
        if basis_set is None:
            return tq.Molecule(geometry=geometry.format(r=r),
                               basis_set=basis_set,
                               transformation=transformation,
                               name=name.format(r=r),
                               n_pno=n_pno)
        else:
            return tq.Molecule(geometry=geometry.format(r=r),
                               basis_set=basis_set,
                               active_orbitals=active_orbitals,
                               transformation=transformation,
                               name=name.format(r=r),
                               n_pno=n_pno)

    except SCFConvergenceError:
        print('WARNING! could not intialize molecule with bond distance {r}'.format(r=r))
        return None


def get_molecule_initializer(geometry, active_orbitals):
    def initializer(r, name, basis_set, transformation, n_pno):
        return initialize_molecule(r, geometry, name, basis_set, active_orbitals, transformation, n_pno)

    return initializer
