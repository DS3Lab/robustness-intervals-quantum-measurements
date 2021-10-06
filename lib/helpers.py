import sys
from datetime import datetime as dt

import tequila as tq


def timestamp_human():
    return dt.now().strftime('%d-%m-%Y %H:%M:%S.%f')[:-3]


class Logger:
    def __init__(self, print_fp=None):
        self.terminal = sys.stdout
        self.log_file = "logfile.txt" if print_fp is None else print_fp
        self.encoding = sys.stdout.encoding

    def write(self, message):
        self.terminal.write(message)
        with open(self.log_file, "a") as log:
            log.write(message)

    def flush(self):
        pass


def print_summary(molecule, hamiltonian, ansatz, ansatz_name, use_grouping):
    pauli_terms_with_grouping = tq.ExpectationValue(H=hamiltonian, U=tq.QCircuit(),
                                                    optimize_measurements=True).count_expectationvalues()

    print(f"""---- molecule summary ----
molecule: {molecule}
    
n_orbitals\t: {molecule.n_orbitals}
n_electrons\t: {molecule.n_electrons}
    
Hamiltonian:
num_terms\t: {len(hamiltonian)}
num_qubits\t: {hamiltonian.n_qubits}
pauli_groups\t: {pauli_terms_with_grouping}
use_grouping\t: {use_grouping}
    
Ansatz:
name      \t: {ansatz_name}
num_params\t: {len(ansatz.extract_variables())}
    """)


if __name__ == '__main__':
    print(timestamp_human())
    '[04-10-2021 11:00:08.252]'
