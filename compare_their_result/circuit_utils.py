from qiskit import QuantumCircuit, QuantumRegister
from qiskit.transpiler import CouplingMap
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer, StatevectorSimulator
from qiskit.quantum_info import Statevector, state_fidelity


#crea il circuito dal file contente le porte   
def create_circuit_from_file(filename: str) -> QuantumCircuit:
    with open(filename, 'r') as f:
       #readlines(): legge tutte le righe, strip() rimuove gli spazi vuoti iniziali e finali, if line.strip() scarta le stringhe vuote 
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
def count_cnots(qc):
    return sum(1 for gate in qc.data if gate.operation.name.upper() in ["CX", "CNOT"])

def create_circuit(gatesSupported: list) -> QuantumCircuit:
    qc = QuantumCircuit(9)
    qc.tdg(0)
    qc.tdg(6)
    qc.t(3)
    qc.t(5)
    qc.t(7)
    qc.t(0)
    qc.cx(3, 8)
    qc.x(8)
    qc.tdg(6)
    qc.t(3)
    qc.tdg(0)
    qc.t(1)
    qc.h(3)
    qc.h(5)
    qc.x(2)
    qc.x(6)
    qc.t(3)
    qc.tdg(6)
    qc.h(6)
    qc.t(6)
    qc.cx(4, 0)
    qc.t(7)
    qc.tdg(2)
    qc.t(0)
    qc.tdg(5)
    qc.tdg(6)
    qc.x(4)
    qc.tdg(8)
    qc.tdg(1)
    qc.h(1)
    qc.tdg(5)
    qc.cx(8, 2)
    return qc


