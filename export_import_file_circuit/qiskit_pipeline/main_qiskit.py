from pathlib import Path
import sys
from typing import Callable, Iterable
from qiskit.transpiler import CouplingMap
import os
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import GenericBackendV2
from sklearn import datasets
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import normalize
from qiskit.visualization import circuit_drawer
import csv
import networkx as nx
from collections import defaultdict
from qiskit.quantum_info import Statevector, state_fidelity, Statevector, DensityMatrix, entropy

from utils_quiskit import (
    create_circuit_from_file,
    create_file_from_circuit,
    create_circuit_from_simple_file,
    load_adjacency_matrix_from_file,
    matrix_to_coupling_map,
    count_cx_gates,
    qasm_to_clifford_and_t,
    simulate_and_compare,
    get_layout,
    compute_fidelity_qiskit,
    ensure_directories,
    next_progressive_index,
    csv_name_base,
    csv_regex_filename,
    get_statevector,
    von_neumann_entropy,
    to_clifford_t_if_needed,
    build_stateprep_from_circuit,
    evaluate_over_levels,
    write_compare_results_csv,
    write_summary_csv
    
)
from utils_ffqram import(
    FFQRAM,
    generate_normalized_dataset,
    adjust_dataset_for_hw
)
import numpy as np
from cnot_synth import run_java_cnotsynth




def check_input_files(file):
    if not os.path.isfile(file):
        sys.exit(f"Errore: file non è stato trovato: {file}")

def check_input_folder(folder):
    if not os.path.isdir(folder):
        sys.exit(f"Errore: cartella non trovata: {folder}")


def java_better_stats(results):
    better = sum(1 for row in results if row[4] < row[3])
    return better, len(results)












def run_ffqram_synthesis(filepath_adj, filepath_jar, folder_input, folder_output, folder_results, coupling_map):

    existing_files = os.listdir(folder_input)
    input_file_number = len([f for f in existing_files if f.endswith('_Input.txt')]) + 1 
    input_file_name = f"{input_file_number}_Input.txt"
    filepath_input = os.path.join(folder_input, input_file_name)
    output_filename = f"{input_file_number}_Output.txt"
    filepath_output = os.path.join(folder_output, output_filename)

    df = load_and_preprocess_data()

    # Creazione del circuito FFQRAM
    qc = FFQRAM(df)
    #qc.draw('mpl')

    # decomposizione del circuito
    discretized = qasm_to_clifford_and_t(qc)
    #discretized.draw('mpl')

    results = []
    result = process_circuit(
            create_circuit_from_simple_file(filepath_input),
            filepath_output,
            filepath_adj,
            coupling_map,
            "2",
            discretized
        )
    results.append(result)
    print(f"file: {filepath_input} fidelity_qiskit: {result[5]} | fidelity_java: {result[6]}")
    print(f"cnot_original: {result[2]} | cnot_qiskit: {result[3]} | cnot_java: {result[4]}\n")

    results_file = os.path.join(folder_results, f"{input_file_number}_Result.txt")
    save_results(results, results_file)

# Funzione per caricare e pre-elaborare i dati Iris
def load_and_preprocess_data():
    """Carica e pre-elabora il dataset Iris."""
    df = datasets.load_iris(as_frame=True).frame
    df.columns = ["f0", "f1", "f2", "f3", "class"]
    df = df.drop('class', axis=1)
    df = df.sample(n=2, random_state=123)
    df.reset_index(drop=True, inplace=True)

    # Standardizzazione e normalizzazione
    scaler = StandardScaler()
    df.loc[:, :] = scaler.fit_transform(df.loc[:, :])
    df.loc[:, :] = normalize(df.loc[:, :])
    
    return df.to_numpy()



    
def evaluate_qc_vs_stateprep(
    qc: QuantumCircuit,
    filepath_adj: str,   
    filepath_circ: str,                   
    coupling_map: CouplingMap,
    levels: Iterable[int] = (0,1,2,3)):

    adj_name = os.path.splitext(os.path.basename(filepath_adj))[0]
    file_input = os.path.basename(filepath_circ) if filepath_circ else None

    # --- porto il circuito originale in Clifford+T (se necessario)
    qc_ct = to_clifford_t_if_needed(qc)

    # --- transpile su coupling map con base Clifford+T
    qc_cx_levels = evaluate_over_levels(
        qc_ct, coupling_map,
        levels=levels
    )
    # --- costruisco StatePreparation dallo stato di qc_ffqram
    qc_stateprep = build_stateprep_from_circuit(qc)

    # --- porto il circuito della StatePreparation in Clifford+T (se necessario)
    qc_stateprep_ct = to_clifford_t_if_needed(qc_stateprep)

    sp_cx_levels = evaluate_over_levels(
        qc_stateprep_ct, coupling_map,
        levels=levels
    )

    fidelity_stateprep = float(state_fidelity(get_statevector(qc),get_statevector(qc_stateprep)))
 

     # --- CNOT logici
    cnot_original_logical = count_cx_gates(qc_ct)

    cnot_original_stateprep = count_cx_gates(qc_stateprep_ct)

    fidelity_logical = None
    if file_input is None:
        fidelity_logical = float(state_fidelity(get_statevector(qc),get_statevector(qc_ct)))
    
   
   

    # --- medie
    cnot_qc_avg = float(np.mean(qc_cx_levels))
    cnot_sp_avg = float(np.mean(sp_cx_levels))
    overhead_qc_avg_pct = 0.0 if cnot_original_logical == 0 else (cnot_qc_avg - cnot_original_logical) / cnot_original_logical * 100.0
    overhead_sp_avg_pct = 0.0 if cnot_original_stateprep == 0 else (cnot_sp_avg - cnot_original_stateprep) / cnot_original_stateprep * 100.0


    

    row = [
        file_input,
        adj_name,
        cnot_original_logical,
        cnot_original_stateprep,
        round(cnot_qc_avg, 2),
        round(cnot_sp_avg, 2),
        round(overhead_qc_avg_pct, 2),
        round(overhead_sp_avg_pct, 2),
        round(fidelity_stateprep, 6),
    ]

    # aggiungi fidelity_logical solo se disponibile
    if fidelity_logical is not None:
        row.append(round(fidelity_logical, 6))
    return row
# ----------- funzione che scorre la cartella degli input -----------
def process_folder_qc_vs_stateprep(
    folder_input: str | Path,
    filepath_adj: str,
    folder_results: str | Path,
    coupling_map: CouplingMap
    
):
    rows = []

    for filename in os.listdir(folder_input):
        if not filename.endswith(".txt"):  
            continue

        filepath_circ = os.path.join(folder_input, filename)
        print(filepath_circ)
        qc = create_circuit_from_simple_file(filepath_circ) 

        row = evaluate_qc_vs_stateprep(qc, filepath_adj, filepath_circ, coupling_map, levels=(0,1,2,3))
        rows.append(row)

    save_qc_vs_stateprep_results(folder_results, filepath_adj, rows, kind="circuit")

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
        "CNOT medio (QC-Transpiled)",
        "CNOT medio (StatePrep-Transpiled)",
        "Overhead medio (QC) (%)",
        "Overhead medio (StatePrep) (%)",
        "Fidelity (Original vs StatePrep)"
    ]
    if kind == "ffqram":
        headers.append("Fidelity (Original vs Clifford-T)")

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

def run_ffqram_experiments_on_random_datasets(
    filepath_adj: str,            
    folder_results: str | Path,   
    coupling_map,                 
    levels=(0,1,2,3),
    seeds=(101, 202, 303)         
):
   
    # --------- calcolo dimensione dataset 2^k x 2^k ----------
    rows = []
    for seed in seeds:

        # 1) genera dataset normalizzato 2^k x 2^k con seed diverso
        dataset = generate_normalized_dataset(pow(2, K), pow(2, K), seed=seed)

        # 2) costruisco circuito FF-QRAM
        qc = FFQRAM(dataset)

        # 4) raccolgo i dati
        row = evaluate_qc_vs_stateprep(
            qc=qc,
            filepath_adj=filepath_adj,
            filepath_circ=None,
            coupling_map=coupling_map,
            levels=levels
        )
        rows.append(row)

    save_qc_vs_stateprep_results(folder_results, filepath_adj, rows, kind="ffqram")


def main():

    if len(sys.argv) != 3:
        print("Uso: <matrice_adiacenza.txt> <cartella_results>")
        sys.exit(1)

    filepath_adj, folder_results = sys.argv[1:]

    check_input_files(filepath_adj)
    

    adj_matrix = load_adjacency_matrix_from_file(filepath_adj)
    coupling_map = matrix_to_coupling_map(adj_matrix)

    print("\nScegli la modalità di sperimentazione:")
    print("1. Circuiti randomici da file")
    print("2. FF-QRAM su dataset generato internamente")
    choice = input("Inserisci 1 o 2: ").strip()

    adj_name = os.path.splitext(os.path.basename(filepath_adj))[0]

    if choice == '1':
        
        folder_input = input("Inserisci il path della cartella contenente i circuiti: ").strip()
        
        check_input_folder(folder_input)

        process_folder_qc_vs_stateprep(folder_input, filepath_adj, folder_results, coupling_map)


        """transpile_and_evaluate_random_circuits(
            adj_name, folder_input, folder_results, coupling_map
        )"""

    elif choice == '2':
        num_qubits = coupling_map.size()
        run_ffqram_experiments_on_random_datasets(filepath_adj, folder_results, coupling_map, levels=(0,1,2,3), seeds=(101, 202, 303))
        
       

    else:
        print("Scelta non valida. Uscita.")
        sys.exit(1)



    
    


if __name__ == "__main__":
    main()
