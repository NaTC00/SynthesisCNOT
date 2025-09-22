from collections import defaultdict
import csv
import gc
import os
import re
from pathlib import Path
from typing import Callable, Iterable, List, Dict
from statistics import mean
from qiskit import QuantumCircuit, QuantumRegister, transpile, ClassicalRegister
from qiskit.transpiler import CouplingMap, Layout, PassManager
from qiskit.transpiler.passes import Decompose
from qiskit_aer import Aer, StatevectorSimulator
from qiskit.quantum_info import Statevector, state_fidelity, DensityMatrix, entropy
import numpy as np
from qiskit.circuit.library import StatePreparation
from qiskit.synthesis import generate_basic_approximations
from qiskit.transpiler.passes import SolovayKitaev
import numpy as np
from qiskit.circuit.library import CXGate


CLIFFORD_T_BASIS = ['cx', 'h', 's', 'sdg', 't', 'tdg', 'x', 'y', 'z', 'u']
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
    
    
    # Decompose tutte le porte
    pm = PassManager([Decompose()])

    # Applicare la pass ad un circuito
    new_circuit = pm.run(qc)

    #print(new_circuit)
    
    return new_circuit



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
        return circ_in_basis(qc) #qasm_to_clifford_and_t(qc)
    

def circ_in_basis(qc):
    # Solo traduzione alla base, niente routing, niente layout
    q_unrolled = transpile(
        qc,
        basis_gates=CLIFFORD_T_BASIS,
        coupling_map=None,        # no routing
        layout_method=None,       # nessun layout imposto
        optimization_level=0      # evita cancellazioni che confondono il confronto
    )
    return q_unrolled


def evaluate_cnot_means_over_layouts(
    qc: QuantumCircuit,
    coupling_map,
    *,
    levels: Iterable[int] = (0, 1, 2, 3),
    seed_transpiler: int = 123,
) -> Dict[str, float]:
    """
    Ritorna un dizionario con la media dei CNOT per ciascun layout:
    {
      "default": <media sui livelli>,
      "dense": <media sui livelli>,
      "trivial": <media sui livelli>,
      "sabre": <media sui livelli>,
    }
    """
    layout_methods = [
        ("default", None),
        ("dense", "dense"),
        ("trivial", "trivial"),
        ("sabre", "sabre"),
    ]

    means_per_layout: Dict[str, float] = {}

    for layout_name, layout_method in layout_methods:
        cnot_counts = []
        for lvl in levels:
            qct = transpile(
                qc,
                coupling_map=coupling_map,
                basis_gates=CLIFFORD_T_BASIS,
                layout_method=layout_method,   # None per 'default'
                optimization_level=lvl,
                seed_transpiler=seed_transpiler,
            )
            cxs = count_cx_gates(qct)
            cnot_counts.append(cxs)
            print(f"Traspilato layout {layout_name}, livello {lvl}: CNOT={cxs}")

        means_per_layout[layout_name] = float(mean(cnot_counts)) if cnot_counts else float("nan")
        print(f"Media CNOT layout {layout_name}: {means_per_layout[layout_name]:.2f}")

    return means_per_layout


def evaluate_over_levels(
    qc: QuantumCircuit,
    coupling_map,
    *,
    levels: Iterable[int] = (0,1,2,3)
):
    """Transpilo qc ai vari livelli e ritorna liste di CNOT e entropie."""
    cnot_levels = []
    for lvl in levels:
        qct = transpile(
            qc,
            coupling_map=coupling_map,
            basis_gates=CLIFFORD_T_BASIS,
            layout_method="trivial",
            optimization_level=lvl,
            seed_transpiler=123,
        )
        cnot_levels.append(count_cx_gates(qct))
        print(f"Traspilato livello {lvl}")
    return cnot_levels

def print_cx_pairs(circ: QuantumCircuit, show_physical=False):
    """
    Stampa le CX del circuito `circ` come (control -> target).
    Se show_physical=True e il layout è disponibile, mostra anche gli indici fisici.
    """
    # prova a recuperare il layout finale (compatibile con diverse versioni di Qiskit)
    layout = (getattr(circ, "layout", None)
              or getattr(circ, "_layout", None)
              or getattr(circ, "_final_layout", None))

    total = 0
    for inst, qargs, _ in circ.data:
        if isinstance(inst, CXGate) or inst.name.lower() in ("cx", "cnot"):
            # indici "logici" (posizione delle linee nel circuito transpiled)
            ctrl = circ.qubits.index(qargs[0])
            targ = circ.qubits.index(qargs[1])

            line = f"{total+1:3d}: cx q[{ctrl}] -> q[{targ}]"

            if show_physical and layout is not None:
                try:
                    p_ctrl = layout[circ.qubits[ctrl]]
                    p_targ = layout[circ.qubits[targ]]
                    # p_ctrl/p_targ possono essere int o Qubit; proviamo a estrarre l'indice
                    p_ctrl = p_ctrl if isinstance(p_ctrl, int) else getattr(p_ctrl, "index", str(p_ctrl))
                    p_targ = p_targ if isinstance(p_targ, int) else getattr(p_targ, "index", str(p_targ))
                    line += f"   (phys {p_ctrl} -> {p_targ})"
                except Exception:
                    pass

            print(line)
            total += 1

    print(f"Totale CX: {total}")





def save_qc_results(
    folder_results: str | Path,
    filepath_adj: str,
    rows: list[dict],          # <<-- lista di dict come sopra
    kind: str                  # "qc" oppure "stateprep"
):
    """
    Salva un CSV per 'kind' (qc/stateprep) con:
      - File Input, Matrice Adiacenza, CNOT Logici
      - CNOT medio (default/dense/trivial/sabre)
      - Overhead (default/dense/trivial/sabre) %
      - (opzionale) Fidelity (Original vs Clifford-T)

    Crea anche un file summary raggruppando per CNOT Logici e mediando tutte le colonne numeriche.
    """

    # --- prepara cartella output
    out_dir = ensure_directories(folder_results)

    # --- nome base del file
    adj_name = os.path.splitext(os.path.basename(filepath_adj))[0]

    # --- colonne (chiavi -> etichette leggibili)
    label_map = {
        "file_input": "File Input",
        "adj_name": "Matrice Adiacenza",
        "cnot_logical": "CNOT Logici",

        "cnot_default": "CNOT medio (default)",
        "cnot_dense": "CNOT medio (dense)",
        "cnot_trivial": "CNOT medio (trivial)",
        "cnot_sabre": "CNOT medio (sabre)",

        "overhead_default_pct": "Overhead (default) (%)",
        "overhead_dense_pct": "Overhead (dense) (%)",
        "overhead_trivial_pct": "Overhead (trivial) (%)",
        "overhead_sabre_pct": "Overhead (sabre) (%)",

        "fidelity_logical": "Fidelity (Original vs Clifford-T)",
    }

    # determina se c'è la fidelity nei dati
    has_fidelity = any("fidelity_logical" in r for r in rows)

    # --- ordine colonne
    col_order = [
        "file_input", "adj_name", "cnot_logical",
        "cnot_default", "cnot_dense", "cnot_trivial", "cnot_sabre",
        "overhead_default_pct", "overhead_dense_pct", "overhead_trivial_pct", "overhead_sabre_pct",
    ]
    if has_fidelity:
        col_order.append("fidelity_logical")

    # --- header leggibili
    headers = [label_map[k] for k in col_order]

    # --- calcolo nome file progressivo come facevi già
    regex = csv_regex_filename(adj_name, kind)
    next_idx = next_progressive_index(out_dir, regex)
    out_csv = out_dir / f"{csv_name_base(adj_name, kind)}_{next_idx}.csv"

    # --- scrittura CSV principale (righe flatten secondo col_order)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow([r.get(k, "") for k in col_order])

    write_summary_csv(out_csv, rows)


def write_summary_csv(out_csv: Path, rows: list[dict]):
    """
    Scrive un file di summary (_summary.csv) raggruppando per 'cnot_logical'
    e calcolando la media dei valori numerici per ciascun mapping.
    Le colonne hanno etichette fisse:
      CNOT Logici, CNOT medio (default/dense/trivial/sabre),
      Overhead (default/dense/trivial/sabre) (%)
    """
    # raggruppa per CNOT logici
    buckets = defaultdict(list)
    for r in rows:
        buckets[int(r["cnot_logical"])].append(r)

    mapping_order = ["default", "dense", "trivial", "sabre"]

    # header fisso
    summary_headers = [
        "CNOT Logici",
        "CNOT medio (default)",
        "CNOT medio (dense)",
        "CNOT medio (trivial)",
        "CNOT medio (sabre)",
        "Overhead (default) (%)",
        "Overhead (dense) (%)",
        "Overhead (trivial) (%)",
        "Overhead (sabre) (%)",
    ]

    out_summary = out_csv.with_name(out_csv.stem + "_summary.csv")

    def _safe_mean(iterable, ndigits=2):
        nums = []
        for v in iterable:
            try:
                nums.append(float(v))
            except (TypeError, ValueError):
                pass
        return round(sum(nums) / len(nums), ndigits) if nums else ""

    with open(out_summary, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(summary_headers)

        for cnot_init in sorted(buckets.keys()):
            rs = buckets[cnot_init]

            # medie CNOT
            cnot_means = [
                _safe_mean((r.get(f"cnot_{m}") for r in rs))
                for m in mapping_order
            ]
            # medie Overhead
            oh_means = [
                _safe_mean((r.get(f"overhead_{m}_pct") for r in rs))
                for m in mapping_order
            ]

            row_out = [cnot_init] + cnot_means + oh_means
            w.writerow(row_out)

    print(f"Salvato summary: {out_summary}")


def write_compare_results_csv(out_csv: Path, rows: list, headers: list):
    """
    Scrivo i risultati dettagliati per ciascun circuito in un CSV.
    """
   
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in rows:
            w.writerow(row)

def write_summary_csv_delete(out_csv: Path, rows: list, headers: list):
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
