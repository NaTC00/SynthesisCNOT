import sys
import os
from pytket.architecture import Architecture
from collections import defaultdict
import csv

from utils_tket import (
   
    load_adjacency_matrix_from_file,
    ensure_directories,
    check_input_files,
    matrix_to_architecture,
    process_circuit_file_tket,
    draw_graph
    
)

def save_results_tket(results, results_file):
    # Raggruppa gli overhead in base al numero di CNOT originali
    overhead_by_cnot_original = defaultdict(list)

    # Intestazioni per il file CSV
    headers = [
        "File Input",             # Nome del file sorgente
        "Matrice Adiacenza",      # Nome del file della matrice
        "CNOT Originale",         # CNOT nel circuito originale
        "CNOT Tket Transpile",    # CNOT dopo traspilazione TKET
        "Overhead (%)"            # Aumento percentuale
    ]

    rows_to_write = []

    for row in results:
        file_input = row[0]
        adj = row[1]
        cnot_original = row[2]
        cnot_tket = row[3]

        # Calcolo dell’overhead percentuale
        overhead = ((cnot_tket - cnot_original) / cnot_original) * 100 if cnot_original > 0 else 0.0
        overhead_by_cnot_original[cnot_original].append(overhead)

        rows_to_write.append([
            file_input,
            adj,
            cnot_original,
            cnot_tket,
            round(overhead, 2)
        ])

    # Ordina per numero di CNOT originale
    rows_to_write.sort(key=lambda row: row[2])

    # Scrivi il CSV principale
    with open(results_file, mode="w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows_to_write)

    # Scrivi il file riassuntivo
    summary_file = results_file.replace(".csv", "_summary.csv")
    with open(summary_file, mode="w", newline='') as summary_csv:
        writer = csv.writer(summary_csv)
        writer.writerow(["CNOT Originale", "Overhead Medio (%)"])

        for cnot_original in sorted(overhead_by_cnot_original.keys()):
            overhead_list = overhead_by_cnot_original[cnot_original]
            avg_overhead = sum(overhead_list) / len(overhead_list)
            writer.writerow([cnot_original, round(avg_overhead, 2)])

def transpile_and_evaluate_all_circuits_tket(filepath_adj, folder_input, folder_results, architecture: Architecture):
    results = []

    for filename in os.listdir(folder_input):
        filepath_input = os.path.join(folder_input, filename)
        if not filename.endswith(".txt"):
            continue
        result = process_circuit_file_tket(
            filepath_input,
            filepath_adj,
            architecture
        )
        results.append(result)
    adj_name = os.path.splitext(os.path.basename(filepath_adj))[0]
    results_file = os.path.join(folder_results, f"my_results_{adj_name}.csv")
    save_results_tket(results, results_file)

    print(f"Processo completato! Risultati salvati in {results_file}")
        

def main():

    if len(sys.argv) != 4:
        print("Uso: <matrice_adiacenza.txt> <cartella_input> <cartella_results>")
        sys.exit(1)

    for file in sys.argv[1:2]:
        check_input_files(file)

    filepath_adj, folder_input, folder_results = sys.argv[1:]

    ensure_directories(folder_input, folder_results)

    adj_matrix = load_adjacency_matrix_from_file(filepath_adj)
    architecture = matrix_to_architecture(adj_matrix)

    draw_graph(architecture.coupling)

    transpile_and_evaluate_all_circuits_tket(filepath_adj, folder_input, folder_results, architecture)


if __name__ == "__main__":
    main()
