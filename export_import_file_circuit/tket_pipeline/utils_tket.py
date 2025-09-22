from collections import defaultdict
import csv
import os
from pathlib import Path
import re
import sys
from typing import Optional, Sequence, Type
import numpy as np
from pytket import Circuit, Qubit
from pytket.circuit import OpType
from pytket.architecture import Architecture
from pytket.placement import place_with_map, LinePlacement, GraphPlacement
from pytket.mapping import MappingManager, LexiLabellingMethod, LexiRouteRoutingMethod
import networkx as nx
from typing import Union, Dict, Iterable, List, Tuple
from pytket.circuit import StatePreparationBox, Node
from pytket.utils.results import compare_statevectors
from pytket.transform import Transform
from pytket.extensions.qiskit import AerStateBackend

from pytket.passes import (
    SequencePass, DecomposeBoxes, RemoveRedundancies, AutoRebase,
    SynthesiseTket, FullPeepholeOptimise,
    KAKDecomposition, CliffordSimp, RoutingPass, DecomposeSwapsToCXs, DecomposeMultiQubitsCX
)
from statistics import mean
from pytket.passes import (
    PlacementPass, RoutingPass, DecomposeSwapsToCXs, SequencePass, AutoRebase,
    RemoveRedundancies, FullPeepholeOptimise, EulerAngleReduction, KAKDecomposition,
    PauliSimp, RepeatPass,
)
from pytket.transform import CXConfigType

# -------- base set richiesto --------
BASE_OPS = {
    OpType.CX, OpType.U3,  # 'u' di Qiskit corrisponde a U3 in pytket
    OpType.H, OpType.S, OpType.Sdg, OpType.T, OpType.Tdg,
    OpType.X, OpType.Y, OpType.Z,
}
REBASE = AutoRebase(BASE_OPS)

def _apply_if_pass(c: Circuit, p: Optional[SequencePass]) -> None:
    if p is not None:
        p.apply(c)

# -------- helpers --------
def _optim_pass(level: int) -> SequencePass:
    """
    Livelli di ottimizzazione 'ragionevoli' per pytket:
      0: nessuna ottimizzazione
      1: pulizia leggera (redundancy + euler su 1q)
      2: peephole completa + KAK per 2q
      3: come (2) + PauliSimp (configurabile) e fissaggio finale
    """
    if level <= 0:
        return None
    if level == 1:
        return SequencePass([
            RemoveRedundancies()
        ])
    if level == 2:
        return SequencePass([
            FullPeepholeOptimise(),  # include varie semplificazioni locali
            RemoveRedundancies(),
        ])
    # level >= 3 (più aggressivo)
    return SequencePass([
        FullPeepholeOptimise(),  # include varie semplificazioni locali
        KAKDecomposition(),      # decomposizioni 2q ottimali (~<=3 CX)
        RemoveRedundancies(),
    ])

def _count_cx(c: Circuit,  arch: Architecture) -> int:
    DecomposeSwapsToCXs(arc=arch, respect_direction=False).apply(c)  # traduci SWAP in CX
    
    cx = c.n_gates_of_type(OpType.CX)
   
    # (opz.) conta BRIDGE come 2 CX se presenti
    try:
        cx += 2 * c.n_gates_of_type(OpType.BRIDGE)
    except AttributeError:
        pass

    print(f"numero cnot: {cx}")
    return cx

def _route_default(circ: Circuit, arch: Architecture) -> Circuit:
    c = circ.copy()
    mm = MappingManager(arch)
    mm.route_circuit(c, [LexiLabellingMethod(), LexiRouteRoutingMethod(lookahead=10)])
    return c

def _route_identity(circ: Circuit, arch: Architecture) -> Circuit:
    c = circ.copy()
    nodes: List[Node] = sorted(list(arch.nodes), key=lambda n: n.index)
    if len(c.qubits) > len(nodes):
        raise ValueError("Circuito con più qubit dell'architettura.")
    place_with_map(c, {q: nodes[i] for i, q in enumerate(c.qubits)})
    RoutingPass(arch).apply(c)
    return c

def _route_line(circ: Circuit, arch: Architecture) -> Circuit:
    c = circ.copy()
    PlacementPass(LinePlacement(arch)).apply(c)
    RoutingPass(arch).apply(c)
    return c

def _route_graph(circ: Circuit, arch: Architecture) -> Circuit:
    c = circ.copy()
    PlacementPass(GraphPlacement(arch)).apply(c)
    RoutingPass(arch).apply(c)
    return c

# -------- funzione principale --------
def evaluate_cnot_means_over_placements(
    circ: Circuit,
    arch: Architecture,
    *,
    levels: Iterable[int] = (0, 1, 2, 3),
) -> Dict[str, float]:
    """
    Restituisce:
    {
      "default": media CNOT sui livelli,
      "identity": ...,
      "line": ...,
      "graph": ...
    }
    """
    placements: List[Tuple[str, callable]] = [
        ("default",  _route_default),
        ("identity", _route_identity),
        ("line",     _route_line),
        ("graph",    _route_graph),
    ]

    means: Dict[str, float] = {}

    for name, router in placements:
        counts: List[int] = []
        for lvl in levels:
            REBASE.apply(circ)                        # forza il tuo base set
            c = router(circ, arch)                 # routing (placement scelto)
            _apply_if_pass(c, _optim_pass(lvl))        # ottimizzazione (livello)
           
            counts.append(_count_cx(c, arch))            # conta i CNOT
        means[name] = float(mean(counts)) if counts else float("nan")
        print(f"media cnot {means[name]} per placment {name}")

    return means

def save_qc_results(
    folder_results: str | Path,
    filepath_adj: str,
    rows: List[Dict],          # lista di dict (una riga per circuito/test)
    kind: str                  # "qc" oppure "stateprep" o "ffqram" ecc.
):
    """
    Salva un CSV per 'kind' con:
      - File Input, Matrice Adiacenza, CNOT Logici
      - CNOT medio (default/identity/line/graph)
      - Overhead (default/identity/line/graph) %
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

        "cnot_default":  "CNOT medio (default)",
        "cnot_identity": "CNOT medio (identity)",
        "cnot_line":     "CNOT medio (line)",
        "cnot_graph":    "CNOT medio (graph)",

        "overhead_default_pct":  "Overhead (default) (%)",
        "overhead_identity_pct": "Overhead (identity) (%)",
        "overhead_line_pct":     "Overhead (line) (%)",
        "overhead_graph_pct":    "Overhead (graph) (%)",
    }

    # determina se c'è la fidelity nei dati
    has_fidelity = any("fidelity_logical" in r for r in rows)

    # --- ordine colonne
    col_order = [
        "file_input", "adj_name", "cnot_logical",
        "cnot_default", "cnot_identity", "cnot_line", "cnot_graph",
        "overhead_default_pct", "overhead_identity_pct", "overhead_line_pct", "overhead_graph_pct",
    ]


    # --- header leggibili
    headers = [label_map[k] for k in col_order]

    # --- calcolo nome file progressivo
    regex = csv_regex_filename(adj_name, kind)
    next_idx = next_progressive_index(out_dir, regex)
    out_csv = out_dir / f"{csv_name_base(adj_name, kind)}_{next_idx}.csv"

    # --- scrittura CSV principale
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            # riempi campi mancanti (es. se non hai calcolato tutti i placement)
            w.writerow([r.get(k, "") for k in col_order])

    write_summary_csv(out_csv, rows)


def write_summary_csv(out_csv: Path, rows: List[Dict]):
    """
    Scrive un file di summary (_summary.csv) raggruppando per 'cnot_logical'
    e calcolando la media dei valori numerici per ciascun mapping pytket:
    default, identity, line, graph.
    """
    # raggruppa per CNOT logici
    buckets = defaultdict(list)
    for r in rows:
        # ignora righe malformate
        try:
            buckets[int(r["cnot_logical"])].append(r)
        except Exception:
            continue

    mapping_order = ["default", "identity", "line", "graph"]

    # header fisso
    summary_headers = [
        "CNOT Logici",
        "CNOT medio (default)",
        "CNOT medio (identity)",
        "CNOT medio (line)",
        "CNOT medio (graph)",
        "Overhead (default) (%)",
        "Overhead (identity) (%)",
        "Overhead (line) (%)",
        "Overhead (graph) (%)",
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

            # medie CNOT per mapping pytket
            cnot_means = [
                _safe_mean((r.get(f"cnot_{m}") for r in rs))
                for m in mapping_order
            ]
            # medie Overhead per mapping pytket
            oh_means = [
                _safe_mean((r.get(f"overhead_{m}_pct") for r in rs))
                for m in mapping_order
            ]

            row_out = [cnot_init] + cnot_means + oh_means
            w.writerow(row_out)

    print(f"Salvato summary: {out_summary}")

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

def write_summary_csv_old(out_csv: Path, rows: list, headers: list):
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