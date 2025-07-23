import os
import sys
from typing import Type
from pytket import Circuit, Qubit
from pytket.circuit import OpType
from pytket.architecture import Architecture
from pytket.architecture import Architecture
from pytket.passes import SequencePass, FullPeepholeOptimise, RoutingPass, PlacementPass
from pytket.placement import Placement, GraphPlacement
from pytket.placement import place_with_map
import networkx as nx
from typing import List, Union, Tuple
from pytket.circuit import Node

def matrix_to_architecture(adj_matrix):
    edges = []
    size = len(adj_matrix)
    for i in range(size):
        for j in range(size):
            if adj_matrix[i][j] == 1:
                edges.append((i, j))
    return Architecture(edges)

def draw_graph(coupling_map: List[Union[Tuple[int, int], Tuple[Node, Node]]]):
    coupling_graph = nx.Graph(coupling_map)
    nx.draw(coupling_graph, labels={node: node for node in coupling_graph.nodes()})


def check_input_files(file):
    if not os.path.isfile(file):
        sys.exit(f"Errore: file non è stato trovato: {file}")

def ensure_directories(*folders):
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

def load_adjacency_matrix_from_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
        size = int(lines[0].strip())
        adj_matrix = []

        for line in lines[1:size+1]:
            row = list(map(int, line.strip().split()))
            adj_matrix.append(row)

    return adj_matrix


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

def process_circuit_file_tket(filepath_input, filepath_adj, architecture):


    qc_original = create_circuit_from_simple_file(filepath_input)
    
    cnot_count_before_transpilation = qc_original.n_gates_of_type(OpType.CX)

   # Mapping 1:1 logico -> fisico
    qmap = {
        qubit: Node("node", i)
        for i, qubit in enumerate(qc_original.qubits)
    }

    # Applica il mapping
    place_with_map(qc_original, qmap)

    # Definizione dei pass di compilazione
    pass_seq = SequencePass([  
        RoutingPass(architecture),             # Inserisce SWAP dove necessario
        FullPeepholeOptimise()         # Ottimizza le porte
    ])

    # Applica la sequenza di pass
    pass_seq.apply(qc_original)

    cnot_count_after_transpilation = qc_original.n_gates_of_type(OpType.CX)
    return (
        os.path.basename(filepath_input),
        os.path.basename(filepath_adj),
        cnot_count_before_transpilation,
        cnot_count_after_transpilation
        
    )

   

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
        elif cmd.op.type == OpType.CX:
            control = cmd.qubits[0].index[0]
            target = cmd.qubits[1].index[0]
            print(f"CNOT {control} {target}")
    
        