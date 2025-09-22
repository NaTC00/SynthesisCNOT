from locale import normalize
from pathlib import Path
import sys
import os
from typing import Iterable
from pytket.architecture import Architecture
from collections import defaultdict
import csv
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, normalize as sk_normalize
from pytket.circuit import Circuit, OpType
from pytket.utils import Graph
from pytket.circuit import Circuit, OpType, Qubit, Op, QControlBox
from pytket.passes import DecomposeBoxes, DecomposeMultiQubitsCX
from pytket.circuit.display import render_circuit_jupyter as draw
from pytket.passes import (
    SequencePass, DecomposeBoxes, RemoveRedundancies, AutoRebase,
    SynthesiseTket, FullPeepholeOptimise,
    KAKDecomposition, CliffordSimp, RoutingPass, DecomposeSwapsToCXs, DecomposeMultiQubitsCX
)
from pytket.transform import Transform 
from utils_tket import (
   
    load_adjacency_matrix_from_file,
    ensure_directories,
    check_input_files,
    matrix_to_architecture,
    create_circuit_from_simple_file,
    check_input_folder,
    build_stateprep_from_circuit,
    evaluate_routing,
    write_compare_results_csv,
    write_summary_csv,
    csv_regex_filename,
    next_progressive_index,
    csv_name_base,
    fidelity_between_circuits,
    evaluate_cnot_means_over_placements,
    save_qc_results
    
)

from utils_ffqram import(
    generate_normalized_dataset,
    FFQRAM_tk,
    build_tk_from_script
)


#salvo i risultati in un file .csv
def save_qc_vs_stateprep_results(
    folder_results: str | Path,
    filepath_adj: str,
    rows: list,                      
    kind: str        
):
    
    # headers standard
    headers = [
        "File Input",
        "Matrice Adiacenza",
        "CNOT Originale",
        "CNOT StatePrep",
        "CNOT (QC-Transpiled)",
        "CNOT (StatePrep-Transpiled)",
        "Overhead (QC) (%)",
        "Overhead (StatePrep) (%)",
        "Fidelity (Original vs StatePrep)"
    ]
    #if kind == "ffqram":
        #headers.append("Fidelity (Original vs Clifford-T)")

    # prepara cartella
    out_dir = ensure_directories(folder_results)

    # nome base del file
    adj_name = os.path.splitext(os.path.basename(filepath_adj))[0]

    regex = csv_regex_filename(adj_name, kind)
    next_idx = next_progressive_index(out_dir, regex)
    out_csv = out_dir / f"{csv_name_base(adj_name, kind)}_{next_idx}.csv"

    # salva CSV e summary
    write_compare_results_csv(out_csv, rows, headers)
    write_summary_csv(out_csv, rows, headers)

    print(f"Salvati: {out_csv} e {out_csv.with_name(out_csv.stem + '_summary.csv')}")

def evaluate_qc_old(
    circ: Circuit,
    filepath_adj: str,   
    filepath_circ: str,                   
    architecture: Architecture):

    adj_name = os.path.splitext(os.path.basename(filepath_adj))[0]
    file_input = os.path.basename(filepath_circ) if filepath_circ else None

     # --- CNOT logici
    cnot_original_logical = circ.n_gates_of_type(OpType.CX)
    print(cnot_original_logical)

    # --- transpile su coupling map con base Clifford+T
    c_cx_traspiled = evaluate_routing(circ, architecture)
    print(c_cx_traspiled)

    # --- costruisco StatePreparation dallo stato di qc_ffqram
    c_stateprep = build_stateprep_from_circuit(circ)

    cnot_original_stateprep = c_stateprep.n_gates_of_type(OpType.CX)
   
    fidelity = fidelity_between_circuits(circ, c_stateprep)

    sp_cx_traspiled = evaluate_routing(c_stateprep, architecture)

   
    #cnot_sp_avg = float(np.mean(sp_cx_levels))
    overhead_qc_avg_pct = 0.0 if cnot_original_logical == 0 else (c_cx_traspiled - cnot_original_logical) / cnot_original_logical * 100.0
    overhead_sp_avg_pct = 0.0 if cnot_original_stateprep == 0 else (sp_cx_traspiled - cnot_original_stateprep) / cnot_original_stateprep * 100.0

    

    row = [
        file_input,
        adj_name,
        cnot_original_logical,
        cnot_original_stateprep,
        c_cx_traspiled,
        sp_cx_traspiled,
        round(overhead_qc_avg_pct, 2),
        round(overhead_sp_avg_pct, 2),
        round(fidelity, 6),
    ]

    return row


def evaluate_qc(
    qc: Circuit,
    filepath_adj: str,
    filepath_circ: str,
    arch: Architecture,
    levels: Iterable[int] = (0, 1, 2, 3)
):
    adj_name = os.path.splitext(os.path.basename(filepath_adj))[0]
    file_input = os.path.basename(filepath_circ) if filepath_circ else None

   
   

    # CNOT logici (prima del mapping)
    cnot_original_logical = qc.n_gates_of_type(OpType.CX)
    print(cnot_original_logical)

   
    # Medie CNOT per mapping
    means_per_layout = evaluate_cnot_means_over_placements(
        qc,
        arch,
        levels=levels
    )

    # Overhead per mapping
    if cnot_original_logical == 0:
        overhead_per_layout_pct = {k: 0.0 for k in means_per_layout}
    else:
        overhead_per_layout_pct = {
            name: (avg_cx - cnot_original_logical) / cnot_original_logical * 100.0
            for name, avg_cx in means_per_layout.items()
        }

    # --- Costruisco dizionario finale per output tabellare
    result = {
        "file_input": file_input,
        "adj_name": adj_name,
        "cnot_logical": cnot_original_logical,
    }

    # aggiungo colonne per CNOT medi
    for name, avg_cx in means_per_layout.items():
        result[f"cnot_{name}"] = round(avg_cx, 2)

    # aggiungo colonne per overhead %
    for name, overhead in overhead_per_layout_pct.items():
        result[f"overhead_{name}_pct"] = round(overhead, 2)


    return result

def process_folder_qc(
    folder_input: str | Path,
    filepath_adj: str,
    folder_results: str | Path,
    architecture: Architecture
    
):
    rows = []

    for filename in os.listdir(folder_input):
        if not filename.endswith(".txt"):  
            continue

        filepath_circ = os.path.join(folder_input, filename)
        print(filepath_circ)
        qc = create_circuit_from_simple_file(filepath_circ) 

        

        row = evaluate_qc(qc, filepath_adj, filepath_circ, architecture)
        rows.append(row)

    save_qc_results(folder_results, filepath_adj, rows, kind="qc")
    #save_qc_vs_stateprep_results(folder_results, filepath_adj, rows, kind="circuit")



def run_ffqram_experiments_on_random_datasets(
    filepath_adj: str,            
    folder_results: str | Path,   
    architecture,       
    num_qubits: int,      
    seeds=(101, 202, 303)
          
):
   
    # --------- calcolo dimensione dataset 2^k x 2^k ----------
    rows_qc = []
    rows_stateprep = []
    for seed in seeds:

        circ_ffqram = build_tk_from_script(num_qubits, seed)

        rec_qc = evaluate_qc(circ_ffqram, filepath_adj, None, architecture, levels=(0,1,2,3))
        rows_qc.append(rec_qc)

        # --- costruisco StatePreparation dallo stato di qc_ffqram
        qc_stateprep = build_stateprep_from_circuit(circ_ffqram)

        rec_sp = evaluate_qc(qc_stateprep, filepath_adj, None, architecture, levels=(0,1,2,3))
        rows_stateprep.append(rec_sp)

    

    save_qc_results(folder_results, filepath_adj, rows_qc, kind="ffqram")
    save_qc_results(folder_results, filepath_adj, rows_stateprep, kind="stateprep")

def main():


    if len(sys.argv) != 3:
        print("Uso: <matrice_adiacenza.txt> <cartella_results>")
        sys.exit(1)

    filepath_adj, folder_results = sys.argv[1:]

    check_input_files(filepath_adj)
    ensure_directories(folder_results)

    adj_matrix, num_qubits = load_adjacency_matrix_from_file(filepath_adj)
    architecture = matrix_to_architecture(adj_matrix)

    print("\nScegli la modalità di sperimentazione:")
    print("1. Circuiti randomici da file")
    print("2. FF-QRAM su dataset generato internamente")
    choice = input("Inserisci 1 o 2: ").strip()


    if choice == '1':
        
        folder_input = input("Inserisci il path della cartella contenente i circuiti: ").strip()
        
        check_input_folder(folder_input)

        process_folder_qc(folder_input, filepath_adj, folder_results, architecture)

        #transpile_and_evaluate_all_random_circuits_tket(filepath_adj, folder_input, folder_results, architecture)

    elif choice == '2':
        
        
        run_ffqram_experiments_on_random_datasets(filepath_adj, folder_results, architecture, num_qubits, seeds=(101, 202, 303))
        

    else:
        print("Scelta non valida. Uscita.")
        sys.exit(1)

if __name__ == "__main__":
    main()
