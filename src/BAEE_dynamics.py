import math
import numpy as np
import os
import pickle
import time
import uuid
from itertools import combinations
from quspin.basis import spin_basis_1d  # Hilbert space spin basis
from quspin.operators import hamiltonian  # Hamiltonians and operators


def create_linear_logarithm_time(n1, n2, T1, T2):
    time = np.concatenate(
        (np.linspace(0.0, T1, n1 + 1), np.logspace(np.log10(T1), np.log10(T2), n2 + 1))
    )
    return time


def create_J_list(L, J, BC):
    bn = L if BC == "PBC" else L - 1
    return [[J, i, (i + 1) % L] for i in range(bn)]


def bipartitions_entanglement_entropy_dynamics(
    basis,
    psi0,
    time,
    fulle,
    fullv,
    sssi_list,
):
    EE = np.zeros((len(time), len(sssi_list)))
    utpsi = fullv.conj().T @ psi0.reshape(-1, 1)
    for i, t in enumerate(time):
        psi = np.exp(-1j * fulle * t) * utpsi.reshape(-1)
        psi = fullv @ psi.reshape(-1, 1)
        psi = psi.reshape(-1) / np.linalg.norm(psi)
        for ii, sssi in enumerate(sssi_list):
            EE[i, ii] = basis.ent_entropy(psi, sssi, density=False)["Sent_A"]
    return EE


L = 16
n01 = 200
n02 = 40
sss = L // 2  # subsystem size
tsn = 10  # total sample number

J_perp_t = 1.0
J_z_t = 0.5
LB = 2.0  # the logarithmic base to calculate entanglement entropy
T01 = 100.0
T02 = 1e4
W_t = 0.5

BC_t = "OBC"
pauli = False
pn = L // 2  # particle number


sd = math.comb(L, pn)  # sector dimension
T_thermal = create_linear_logarithm_time(n01, n02, T01, T02)


psi0_vector_index = np.random.randint(0, sd, size=tsn)
disorder_thermal = np.random.uniform(-W_t, W_t, size=(tsn, L))
entanglement_entropy_thermal = []
elapsed_time = 0
filename = os.path.splitext(os.path.basename(__file__))[0]
print("filename=", filename)
unique_id = uuid.uuid4()
print("unique_id=", unique_id)
filename = f"{filename}_{unique_id}"

variables_to_be_stored = [
    "L",
    "n01",
    "n02",
    "sss",
    "tsn",
    "J_perp_t",
    "J_z_t",
    "LB",
    "T01",
    "T02",
    "W_t",
    "BC_t",
    "pauli",
    "pn",
    "sd",
    "T_thermal",
    "psi0_vector_index",
    "disorder_thermal",
    "entanglement_entropy_thermal",
    "elapsed_time",
    "filename",
]
data = {name: globals()[name] for name in variables_to_be_stored}


wssi = list(range(L))
sssi_list = list(combinations(wssi, sss))
sssi_list = sssi_list[0 : len(sssi_list) // 2]  # subsystem spin index list
#
basis = spin_basis_1d(L, pauli=pauli, Nup=pn)
#
Jxt_list = Jyt_list = create_J_list(L, J_perp_t, BC_t)
Jzt_list = create_J_list(L, J_z_t, BC_t)
# data check
print("sd==basis.Ns:\n", sd == basis.Ns, "\n", sep="")


start_time = time.time()

for i in range(tsn):
    print(i)
    psi0_vector = np.zeros(basis.Ns)
    psi0_vector[psi0_vector_index[i]] = 1.0
    h_list = [[disorder_thermal[i, ii], ii] for ii in range(L)]
    h_hamiltonian = hamiltonian(
        [["xx", Jxt_list], ["yy", Jyt_list], ["zz", Jzt_list], ["z", h_list]],
        [],
        basis=basis,
        dtype=np.float64,
    )
    fulle, fullv = h_hamiltonian.eigh()
    EE_thermal = bipartitions_entanglement_entropy_dynamics(
        basis,
        psi0_vector,
        T_thermal,
        fulle,
        fullv,
        sssi_list,
    )
    entanglement_entropy_thermal.append(EE_thermal / np.log(LB))
    end_time = time.time()
    delta_time = end_time - start_time - elapsed_time
    elapsed_time = end_time - start_time
    data["elapsed_time"] = elapsed_time
    print(f"The running time of the {i+1}-th cycle: {delta_time:.2f} seconds")
    print(f"The running time of the previous {i+1} cycles: {elapsed_time:.2f} seconds")
    print("Based on this, it is estimated that the entire process will")
    print(f"be completed in approximately {elapsed_time/(i+1)*(tsn-i-1):.2f} seconds")
    with open(f"{filename}.txt", "wb") as f:
        pickle.dump(data, f)