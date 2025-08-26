from collections import defaultdict
import csv
import os
from pathlib import Path
import re
import sys
from typing import Sequence, Type
import numpy as np
from pytket import Circuit, Qubit
from pytket.circuit import OpType
from pytket.architecture import Architecture
from pytket.placement import place_with_map
from pytket.mapping import LexiRouteRoutingMethod
import networkx as nx
from typing import List, Union, Tuple
from pytket.circuit import StatePreparationBox, Node
from pytket.utils.results import compare_statevectors
from pytket.transform import Transform
from pytket.extensions.qiskit import AerStateBackend
from pytket.passes import (
    SequencePass, DecomposeBoxes, RemoveRedundancies, AutoRebase,
    SynthesiseTket, FullPeepholeOptimise,
    KAKDecomposition, CliffordSimp, RoutingPass, DecomposeSwapsToCXs, DecomposeMultiQubitsCX
)
"""def matrix_to_architecture(adj_matrix):
    edges = []
    size = len(adj_matrix)
    for i in range(size):
        for j in range(size):
            if adj_matrix[i][j] == 1:
                edges.append((i, j))
    return Architecture(edges)"""

def matrix_to_architecture(adj_matrix: Sequence[Sequence[int]], name: str = "node") -> Architecture:
    """
    Crea un'Architecture da una matrice di adiacenza
    """
    n = len(adj_matrix)
    if any(len(row) != n for row in adj_matrix):
        raise ValueError("La matrice deve essere quadrata.")

    nodes = [Node(name, i) for i in range(n)]
    edges = []

    # aggiungo ogni arco una sola volta (parte superiore della matrice)
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i][j] or adj_matrix[j][i]:
                edges.append((nodes[i], nodes[j]))

    return Architecture(edges)

def draw_graph(coupling_map: List[Union[Tuple[int, int], Tuple[Node, Node]]]):
    coupling_graph = nx.Graph(coupling_map)
    nx.draw(coupling_graph, labels={node: node for node in coupling_graph.nodes()})


def check_input_files(file):
    if not os.path.isfile(file):
        sys.exit(f"Errore: file non è stato trovato: {file}")

def check_input_folder(folder):
    if not os.path.isdir(folder):
        sys.exit(f"Errore: cartella non trovata: {folder}")


def load_adjacency_matrix_from_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
        size = int(lines[0].strip())
        adj_matrix = []

        for line in lines[1:size+1]:
            row = list(map(int, line.strip().split()))
            adj_matrix.append(row)

    return (adj_matrix, size)

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

def create_circuit_from_simple_file(filename: str) -> Circuit:
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

        num_qubits = int(lines[0])

        qc = Circuit(num_qubits)
       
        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 2:
                continue  # riga non valida
            gate_name = parts[0].upper()
            qubits = [(int(q)) for q in parts[1:]]

            if gate_name == 'H':
                qc.add_gate(OpType.H, [qubits[0]])
            elif gate_name == 'X':
                qc.add_gate(OpType.X, [qubits[0]])
            elif gate_name == 'Y':
                qc.add_gate(OpType.Y, [qubits[0]])
            elif gate_name == 'Z':
                qc.add_gate(OpType.Z, [qubits[0]])
            elif gate_name == 'S':
                qc.add_gate(OpType.S, [qubits[0]])
            elif gate_name == 'S+':
                qc.add_gate(OpType.Sdg, [qubits[0]])
            elif gate_name == 'T':
                qc.add_gate(OpType.T, [qubits[0]])
            elif gate_name == 'T+':
                qc.add_gate(OpType.Tdg, [qubits[0]])
            elif gate_name == 'CNOT':
                qc.add_gate(OpType.CX, [qubits[0], qubits[1]])
            else:
                raise ValueError(f"Porta non supportata: {gate_name}")
    return qc



def build_stateprep_from_circuit(c: Circuit) -> Circuit:
    # ottieni statevector finale
    sv = c.get_statevector()
    
    n = c.n_qubits
   
    # costruisci box di preparazione
    prep_box = StatePreparationBox(sv.data)

    # nuovo circuito che parte da |0...0>
    new_circ = Circuit(n).add_gate(prep_box, list(range(n)))

    DecomposeBoxes().apply(new_circ)
   
    return new_circ

def fidelity_statevectors(psi, phi) -> float:
    psi = np.asarray(psi, dtype=np.complex128).ravel()
    phi = np.asarray(phi, dtype=np.complex128).ravel()
    # F(|psi>,|phi>) = |<psi|phi>|^2
    return float(abs(np.vdot(psi, phi))**2)

def fidelity_between_circuits(c1: Circuit, c2: Circuit) -> float:
    sv1 = c1.get_statevector()
    sv2 = c2.get_statevector()
    return fidelity_statevectors(sv1, sv2)   

#Il circuito in pytket è rappresentato internamente come un DAG (grafo aciclico diretto), non come una lista sequenziale di comandi.
#Per questo l'ordine di stampa non è coerente con l'ordine di inserimento
def create_file_from_circuit(qc: Circuit):
    commands = qc.get_commands()
    num_qubit = qc.n_qubits
    print(f"{num_qubit}")
    for cmd in commands:
        if cmd.op.type == OpType.T:
            print(f"T {cmd.qubits[0].index[0]}")
        elif cmd.op.type == OpType.Tdg:
            print(f"T+ {cmd.qubits[0].index[0]}")
        elif cmd.op.type == OpType.S:
            print(f"S {cmd.qubits[0].index[0]}")
        elif cmd.op.type == OpType.Sdg:
            print(f"S+ {cmd.qubits[0].index[0]}")
        elif cmd.op.type == OpType.X:
            print(f"X {cmd.qubits[0].index[0]}")
        elif cmd.op.type == OpType.Y:
            print(f"Y {cmd.qubits[0].index[0]}")
        elif cmd.op.type == OpType.Z:
            print(f"Z {cmd.qubits[0].index[0]}")
        elif cmd.op.type == OpType.H:
            print(f"H {cmd.qubits[0].index[0]}")
        elif cmd.op.type == OpType.CX:
            control = cmd.qubits[0].index[0]
            target = cmd.qubits[1].index[0]
            print(f"CNOT {control} {target}")
    

    
def get_statevector(circ):
    backend = AerStateBackend()
    compiled_circ = backend.get_compiled_circuit(circ)

    state = backend.run_circuit(compiled_circ).get_state()

    return state

def evaluate_routing(
    circ: Circuit, 
    architecture: Architecture
):
   
   
     # Mapping 1:1 logico -> fisico
    qmap = {
        qubit: Node("node", i)
        for i, qubit in enumerate(circ.qubits)
    }

    # Applica il mapping
    place_with_map(circ, qmap)


    rebase_pass = AutoRebase({OpType.CX, OpType.U3, OpType.H, OpType.S, OpType.Sdg, OpType.T, OpType.Tdg, OpType.X, OpType.Y, OpType.Z})

    cnot = []
    """for i in range(4):
        circ_to_transpile = circ
        pass_seq = ibm_like_default_pass(architecture, rebase_pass, i)
        pass_seq.apply(circ_to_transpile)
        cnot.append(circ_to_transpile.n_gates_of_type(OpType.CX))"""
        
    # Definizione dei pass di compilazione
    pass_seq = SequencePass([
        RoutingPass(architecture),
        rebase_pass
    ])

    pass_seq.apply(circ)

    #DecomposeBoxes().apply(circ),
    #DecomposeMultiQubitsCX().apply(circ),
    #Transform.DecomposeSWAPtoCX(architecture).apply(circ) 


    #cnot_count_after_transpilation = float(np.mean(cnot)) #circ.n_gates_of_type(OpType.CX)
    return circ.n_gates_of_type(OpType.CX)

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
    overhead_qc_idx = headers.index("Overhead (QC) (%)")
    #overhead_sp_idx = headers.index("Overhead medio (StatePrep) (%)")
    #entropy_qc_idx = headers.index("Entropia media (QC-Transpiled)")
    #entropy_sp_idx = headers.index("Entropia media (StatePrep-Transpiled)")
    #fidelity_idx = headers.index("Fidelity (Original vs StatePrep)")
    #fidelity_idx_ct  = headers.index("Fidelity (Original vs Clifford-T)") if "Fidelity (Original vs Clifford-T)" in headers else None

    for r in rows:
        grouped[r[cnot_idx]].append(r)

    summary_path = out_csv.with_name(out_csv.stem + "_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        hdr = [
            "CNOT Originale",
            "Overhead (QC) (%)",
            #"Overhead medio (StatePrep) (%)",
            #"Entropia media (QC)",
            #"Entropia media (StatePrep)",
            #"Fidelity media (Original vs StatePrep)"
        ]
        #if fidelity_idx_ct is not None:
            #hdr.append("Fidelity media (Original vs Clifford-T)")
        w.writerow(hdr)


        for cnot_original, group in sorted(grouped.items()):
            avg_over_qc = sum(float(r[overhead_qc_idx]) for r in group) / len(group)
            #avg_over_sp = sum(float(r[overhead_sp_idx]) for r in group) / len(group)
            #avg_ent_qc  = sum(float(r[entropy_qc_idx]) for r in group) / len(group)
            #avg_ent_sp  = sum(float(r[entropy_sp_idx]) for r in group) / len(group)
            #avg_fid_sp  = sum(float(r[fidelity_idx]) for r in group) / len(group)
            row = [
                cnot_original,
                round(avg_over_qc, 2),
                #round(avg_over_sp, 2),
                #round(avg_ent_qc, 6),
                #round(avg_ent_sp, 6),
                #round(avg_fid_sp, 6)
            ]
            #if fidelity_idx_ct is not None:
                #avg_fid_ct = sum(float(r[fidelity_idx_ct]) for r in group) / len(group)
                #row.append(round(avg_fid_ct, 6))
            w.writerow(row)



def ibm_like_default_pass(architecture, setgates, optimisation_level):
    
    seq = [DecomposeBoxes()] 

    
    if optimisation_level == 0:
        seq += [setgates, LightSabre(), setgates, RemoveRedundancies()]
    elif optimisation_level == 1:
        seq += [SynthesiseTket(), SynthesiseTket(), setgates, RemoveRedundancies()]
    elif optimisation_level == 2:
        seq += [
            FullPeepholeOptimise(),
            KAKDecomposition(allow_swaps=False),
            CliffordSimp(allow_swaps=False),
            SynthesiseTket(), setgates, RemoveRedundancies()
        ]
    elif optimisation_level == 3:
        seq += [
            FullPeepholeOptimise(),
            KAKDecomposition(allow_swaps=True),
            CliffordSimp(allow_swaps=True),
            SynthesiseTket(), setgates, RemoveRedundancies()
        ]
   
    
    if architecture is not None:
        seq.insert(1, RoutingPass(architecture))
        

    return SequencePass(seq)