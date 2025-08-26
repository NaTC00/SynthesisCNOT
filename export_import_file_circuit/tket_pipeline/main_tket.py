from locale import normalize
from pathlib import Path
import sys
import os
from typing import Iterable
from pytket.architecture import Architecture
from collections import defaultdict
import csv
from pytket.circuit import OpType
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, normalize as sk_normalize
from pytket.circuit import Circuit

from utils_tket import (
   
    load_adjacency_matrix_from_file,
    ensure_directories,
    check_input_files,
    matrix_to_architecture,
    create_circuit_from_simple_file,
    check_input_folder,
    build_stateprep_from_circuit,
    fidelity,
    get_statevector,
    evaluate_routing,
    write_compare_results_csv,
    write_summary_csv,
    csv_regex_filename,
    next_progressive_index,
    csv_name_base,
    fidelity_between_circuits
    
)

from utils_ffqram import(
    generate_normalized_dataset,
    FFQRAM_tk
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

def evaluate_qc_vs_stateprep(
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

def process_folder_qc_vs_stateprep(
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

        row = evaluate_qc_vs_stateprep(qc, filepath_adj, filepath_circ, architecture)
        rows.append(row)

    save_qc_vs_stateprep_results(folder_results, filepath_adj, rows, kind="circuit")



def run_ffqram_experiments_on_random_datasets(
    filepath_adj: str,            
    folder_results: str | Path,   
    architecture,       
    num_qubits: int,      
    seeds=(101, 202, 303)
          
):
   
    # --------- calcolo dimensione dataset 2^k x 2^k ----------
    rows = []
    for seed in seeds:
        # 1) genera dataset normalizzato 2^k x 2^k con seed diverso
        #N, M = adjust_dataset_for_hw(num_qubits)
        bits_needed = bits_needed = int(np.ceil(np.log2(num_qubits)))
        dataset = generate_normalized_dataset(pow(2, bits_needed), pow(2, bits_needed), seed=seed)

        # 2) costruisco circuito FF-QRAM
        circ_ffqram = FFQRAM_tk(dataset)

        # 4) raccolgo i dati
        row = evaluate_qc_vs_stateprep(
            circ=circ_ffqram,
            filepath_adj=filepath_adj,
            filepath_circ=None,
            architecture=architecture
        )
        rows.append(row)

    save_qc_vs_stateprep_results(folder_results, filepath_adj, rows, kind="ffqram")

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

        process_folder_qc_vs_stateprep(folder_input, filepath_adj, folder_results, architecture)

        #transpile_and_evaluate_all_random_circuits_tket(filepath_adj, folder_input, folder_results, architecture)

    elif choice == '2':
        
       run_ffqram_experiments_on_random_datasets(filepath_adj, folder_results, architecture, num_qubits, seeds=(101, 202, 303))

    else:
        print("Scelta non valida. Uscita.")
        sys.exit(1)

if __name__ == "__main__":
    main()
