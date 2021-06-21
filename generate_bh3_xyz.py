import argparse
import os
import psi4

from constants import DATA_DIR

ZM_BH3 = """
        0 1
        B
        H       1        {R}
        H       1        {R}         2      120.00000
        H       1        {R}         2      120.00000     3      180.0
        """

parser = argparse.ArgumentParser()
parser.add_argument("--mol", type=str, default='bh3')
args = parser.parse_args()


def main():

    if args.mol != 'bh3':
        raise NotImplementedError('.xyz files cannot be generated for {}'.format(args.mol))

    bond_lengths_avail = _parse_bond_lengths(DATA_DIR, mol_str=args.mol)

    for bond_length in bond_lengths_avail:

        mol = psi4.geometry(ZM_BH3.format(R=bond_length))
        coord = mol.save_string_xyz_file()
        save_as = os.path.join(DATA_DIR, 'bh3_{R}.xyz'.format(R=bond_length))

        with open(save_as, 'w') as f:
            print(coord, file=f)

        print('saved as {}'.format(save_as))


def _parse_bond_lengths(data_dir, mol_str):
    bond_lengths = []
    for fn in os.listdir(data_dir):

        if mol_str in fn and "_htensor.npy" in fn:
            if len(fn.split(mol_str)[0]) == 0:
                bond_length = float(fn.split(f"{mol_str}_")[-1].split("_htensor.npy")[0])
                bond_lengths.append(bond_length)

    return bond_lengths


if __name__ == '__main__':
    main()
