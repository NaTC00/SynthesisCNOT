from collections import defaultdict
import csv
import gc
import os
import re
from pathlib import Path
from typing import Callable, Iterable
from qiskit import QuantumCircuit, QuantumRegister, transpile, ClassicalRegister
from qiskit.transpiler import CouplingMap, Layout
from qiskit_aer import Aer, StatevectorSimulator
from qiskit.quantum_info import Statevector, state_fidelity, DensityMatrix, entropy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit.library import StatePreparation
from qiskit.synthesis import generate_basic_approximations
from qiskit.transpiler.passes import SolovayKitaev
import numpy as np

CLIFFORD_T_BASIS = ['cx', 'h', 's', 'sdg', 't', 'tdg', 'x', 'y', 'z']
CLIFFORD_T_BASIS_U = ['cx', 'h', 's', 'sdg', 't', 'tdg', 'x', 'y', 'z', 'u']
_ALLOWED_IGNORED = {"barrier","measure","id"}  

def ensure_directories(folder) -> Path:
    folder = Path(folder)   # converte la stringa in Path
    folder.mkdir(parents=True, exist_ok=True)
    return folder


#crea il circuito dal file
def create_circuit_from_simple_file(filename: str) -> QuantumCircuit:
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

        num_qubits = int(lines[0])
        qc = QuantumCircuit(num_qubits)

        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 2:
                continue  # riga non valida

            gate_name = parts[0].upper()
            qubits = [int(q) for q in parts[1:]]

            if gate_name == 'H':
                qc.h(qubits[0])
            elif gate_name == 'X':
                qc.x(qubits[0])
            elif gate_name == 'Y':
                qc.y(qubits[0])
            elif gate_name == 'Z':
                qc.z(qubits[0])
            elif gate_name == 'S':
                qc.s(qubits[0])
            elif gate_name == 'S+':
                qc.sdg(qubits[0])
            elif gate_name == 'T':
                qc.t(qubits[0])
            elif gate_name == 'T+':
                qc.tdg(qubits[0])
            elif gate_name == 'CNOT':
                qc.cx(qubits[0], qubits[1])
            else:
                raise ValueError(f"Porta non supportata: {gate_name}")
    
    return qc


#partendo dal circuito crea un file contente le porte del circuito
def create_file_from_circuit(qc: QuantumCircuit, filepath: str, gatesSupported: list):
    with open(filepath, "w") as f:
        f.write(f"{qc.num_qubits}\n")
        for gate in qc.data:
            gateApplied = gate.operation.name.upper()
            qubitsApplied = gate.qubits

            if gateApplied == "TDG":
                gateApplied = "T+"
            elif gateApplied == "SDG":
                gateApplied = "S+"
            elif gateApplied == "CX":
                gateApplied = "CNOT"

            if gateApplied in gatesSupported:
                if len(qubitsApplied) == 1:
                    f.write(f"{gateApplied} {qubitsApplied[0]._index}\n")
                elif gateApplied == "CNOT":
                    f.write(f"{gateApplied} {qubitsApplied[0]._index} {qubitsApplied[1]._index}\n")


def load_adjacency_matrix_from_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
        size = int(lines[0].strip())
        adj_matrix = []

        for line in lines[1:size+1]:
            row = list(map(int, line.strip().split()))
            adj_matrix.append(row)

    return adj_matrix


def matrix_to_coupling_map(adj_matrix):
    edges = []
    size = len(adj_matrix)
    for i in range(size):
        for j in range(size):
            if adj_matrix[i][j] == 1:
                edges.append((i, j))
    return CouplingMap(edges)

 # --- CONTEGGIO CNOT ---
def count_cx_gates(circuit):
    return circuit.count_ops().get('cx', 0)

def compute_fidelity_qiskit(qc_qiskit) -> QuantumCircuit:
    layout_mapping = qc_qiskit.layout.final_layout
    
    if layout_mapping is not None:
        n = qc_qiskit.num_qubits
        dict = list(layout_mapping.get_physical_bits().keys())
        swap_list = []
        for i in range(n):
            for j in range(n - 1):
                if dict[j] > dict[j + 1]:  # Swap if elements are in the wrong order
                    dict[j], dict[j + 1] = dict[j + 1], dict[j]
                    swap_list.append((dict[j], dict[j+1]))
        for i in range(len(swap_list)):
            qc_qiskit.swap(swap_list[i][0],swap_list[i][1])
    
    return qc_qiskit


def simulate_and_compare(qc_original, qc_qiskit, qc_java):
    simulator = StatevectorSimulator()
    state_original = simulator.run(qc_original).result().get_statevector()
    state_qiskit = simulator.run(qc_qiskit).result().get_statevector()
    state_java = simulator.run(qc_java).result().get_statevector()

    fidelity_qiskit = state_fidelity(state_original, state_qiskit)
    fidelity_java = state_fidelity(state_original, state_java)

    return fidelity_qiskit, fidelity_java

def get_layout(qr):
    layout_dict = {}
    for i, qubit in enumerate(qr):
        layout_dict[qubit] = i  

    layout = Layout(layout_dict)
    return layout

def get_statevector(circuit):  
    simulator = StatevectorSimulator()
    result = simulator.run(circuit.decompose(reps=10)).result()
    statevector = result.get_statevector()
    return statevector



def qasm_to_clifford_and_t(qc, basic_approx_depth=10):
    qc = transpile(qc,basis_gates=["cx","u3"]) # You should transpile first to cx and u3, so it will deal with 2Q gates
    basis = ['h', 's', 'sdg', 'tdg', 't', 'x', 'y', 'z']
    approx = generate_basic_approximations(basis, depth=basic_approx_depth)
    skd = SolovayKitaev(recursion_degree=1, basic_approximations=approx)
    new_qc = skd(qc)
    return new_qc



# =========================
# --- Utils di filesystem
# =========================
def ensure_directories(folder) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    return folder

# --- base name: results_{adj_name}_{kind}
def csv_name_base(adj_name: str, kind: str) -> str:
    return f"results_{adj_name}_{kind}"

# --- regex: ^results_{adj_name}_{kind}_(?P<idx>\d+)\.csv$
def csv_regex_filename(adj_name: str, kind: str) -> str:
    base = csv_name_base(adj_name, kind)
    return rf"^{re.escape(base)}_(?P<idx>\d+)\.csv$"

def next_progressive_index(out_dir: Path, regex_filename: str, group_name: str = "idx") -> int:
    max_idx = 0
    for f in out_dir.iterdir():
        if f.is_file():
            m = re.match(regex_filename, f.name)
            if m:
                try:
                    max_idx = max(max_idx, int(m.group(group_name)))
                except (KeyError, ValueError):
                    pass
    return max_idx + 1


def next_progressive_index(out_dir: Path, regex_filename: str, group_name: str = "idx") -> int:
    """
    Scansiona la directory e trova il prossimo indice progressivo per i file
    che matchano la regex (deve includere (?P<idx>...)).
    """
    max_idx = 0
    for f in out_dir.iterdir():
        if f.is_file():
            m = re.match(regex_filename, f.name)
            if m:
                try:
                    idx = int(m.group(group_name))
                    if idx > max_idx:
                        max_idx = idx
                except (KeyError, ValueError):
                    pass
    return max_idx + 1




def create_circuit_from_file(filename: str) -> QuantumCircuit:
    with open(filename, 'r') as f:
       
        lines = [line.strip() for line in f.readlines() if line.strip()]

        num_qubits = int(lines[0])
        qc = QuantumCircuit(num_qubits)

        for line in lines[1:]:
            if '[' not in line or ']' not in line:
                continue #ignora righe malformate
            
            gate_name, qubit_str = line.split('[') # splitta la stringa in corrispondenza di [
            qubit_str = qubit_str.replace(']', '')

            qubits = [int(q.strip()) - 1 for q in qubit_str.split(',')]
            gate_name = gate_name.strip().upper()

            if gate_name == 'H':
                qc.h(qubits[0])
            elif gate_name == 'X':
                qc.x(qubits[0])
            elif gate_name == 'S':
                qc.s(qubits[0])
            elif gate_name == 'S+':
                qc.sdg(qubits[0])
            elif gate_name == 'T':
                qc.t(qubits[0])
            elif gate_name == 'T+':
                qc.tdg(qubits[0])
            elif gate_name == 'Y':
                qc.y(qubits[0])
            elif gate_name == 'Z':
                qc.z(qubits[0])
            elif gate_name == 'CNOT':
                qc.cx(qubits[0], qubits[1])
            else:
                raise ValueError(f"Porta non supportata: {gate_name}")
    return qc





def build_stateprep_from_circuit(qc: QuantumCircuit) -> QuantumCircuit:
    """Restituisce un nuovo circuito che prepara lo stesso stato finale di qc, partendo da |0...0>."""
    sv = Statevector.from_instruction(qc)
    n = qc.num_qubits
    prep = StatePreparation(sv.data)
    new_qc = QuantumCircuit(n)
    new_qc.append(prep, list(range(n)))
    return new_qc.decompose(reps=10)

def von_neumann_entropy(circuit: QuantumCircuit) -> float:
    
    state = Statevector.from_instruction(circuit)
    
    return entropy(state)

def is_clifford_t(qc: QuantumCircuit) -> bool:
    """True se tutti i gate del circuito sono in Clifford+T (ignorando barrier/measure/id)"""
    ops = {inst.name for inst,_,_ in qc.data}
    ops -= _ALLOWED_IGNORED
    return ops.issubset(set(CLIFFORD_T_BASIS))

def to_clifford_t_if_needed(qc: QuantumCircuit) -> QuantumCircuit:
    """Se qc non è già in Clifford+T, lo converto in quella base; altrimenti lo restituisco inalterato"""
    if is_clifford_t(qc):
        return qc
    else: 
        return qasm_to_clifford_and_t(qc)

def evaluate_over_levels(
    qc: QuantumCircuit,
    coupling_map,
    isStatePrep,
    *,
    levels: Iterable[int] = (0,1,2,3)
):
    if isStatePrep:
        basis_gates = CLIFFORD_T_BASIS_U
    else:
        basis_gates = CLIFFORD_T_BASIS
    """Transpilo qc ai vari livelli e ritorna liste di CNOT e entropie."""
    cnot_levels = []
    for lvl in levels:
        qct = transpile(
            qc,
            coupling_map=coupling_map,
            basis_gates=basis_gates,
            layout_method="trivial",
            optimization_level=lvl,
            seed_transpiler=123,
        )
        cnot_levels.append(count_cx_gates(qct))
        print(f"Traspilato livello {lvl}")
    return cnot_levels

def write_compare_results_csv(out_csv: Path, rows: list, headers: list):
    """
    Scrivo i risultati dettagliati per ciascun circuito in un CSV.
    """
   
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in rows:
            w.writerow(row)

def write_summary_csv(out_csv: Path, rows: list, headers: list):
    """
    Raggruppa le righe per CNOT Originale e calcola le medie di overhead, entropia e fidelity.
    Salva in un file `_summary.csv` accanto al CSV principale.
    """
    grouped = defaultdict(list)

    # indice della colonna "CNOT Originale"
    cnot_idx = headers.index("CNOT Originale")
    # indici delle metriche da mediare
    overhead_qc_idx = headers.index("Overhead medio (QC) (%)")
    overhead_sp_idx = headers.index("Overhead medio (StatePrep) (%)")
    fidelity_idx = headers.index("Fidelity (Original vs StatePrep)")
    fidelity_idx_ct  = headers.index("Fidelity (Original vs Clifford-T)") if "Fidelity (Original vs Clifford-T)" in headers else None

    for r in rows:
        grouped[r[cnot_idx]].append(r)

    summary_path = out_csv.with_name(out_csv.stem + "_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        hdr = [
            "CNOT Originale",
            "Overhead medio (QC) (%)",
            "Overhead medio (StatePrep) (%)",
            "Fidelity media (Original vs StatePrep)"
        ]
        if fidelity_idx_ct is not None:
            hdr.append("Fidelity media (Original vs Clifford-T)")
        w.writerow(hdr)


        for cnot_original, group in sorted(grouped.items()):
            avg_over_qc = sum(float(r[overhead_qc_idx]) for r in group) / len(group)
            avg_over_sp = sum(float(r[overhead_sp_idx]) for r in group) / len(group)
            avg_fid_sp  = sum(float(r[fidelity_idx]) for r in group) / len(group)
            row = [
                cnot_original,
                round(avg_over_qc, 2),
                round(avg_over_sp, 2),
                round(avg_fid_sp, 6)
            ]
            if fidelity_idx_ct is not None:
                avg_fid_ct = sum(float(r[fidelity_idx_ct]) for r in group) / len(group)
                row.append(round(avg_fid_ct, 6))
            w.writerow(row)
