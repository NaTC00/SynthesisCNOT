from locale import normalize
import sys
import os
from pytket.architecture import Architecture
from collections import defaultdict
import csv
from pytket.circuit import OpType
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, normalize as sk_normalize


from utils_tket import (
   
    load_adjacency_matrix_from_file,
    ensure_directories,
    check_input_files,
    matrix_to_architecture,
    process_circuit_file_tket,
    create_circuit_from_simple_file,
    check_input_folder
    
)

from utils_ffqram import(
    generate_normalized_dataset,
    FFQRAM_tk
)

def save_results_tket(results, results_file,  include_file_input):
    # Raggruppa gli overhead in base al numero di CNOT originali
    overhead_by_cnot_original = defaultdict(list)

    # Intestazioni per il file CSV
    headers = []
    if include_file_input:
        headers.append("File Input")  # Nome del file sorgente
    headers += [
        "Matrice Adiacenza",      # Nome del file della matrice
        "CNOT Originale",         # CNOT nel circuito originale
        "CNOT Tket Transpile",    # CNOT dopo traspilazione TKET
        "Overhead (%)"            # Aumento percentuale
    ]

    rows_to_write = []

    for row in results:
        if include_file_input:
            file_input = row[0]
            adj = row[1]
            cnot_original = row[2][0]
            cnot_tket = row[2][1]
        else:
            adj = row[0]
            cnot_original = row[1][0]
            cnot_tket = row[1][1]
        

        # Calcolo dell’overhead percentuale
        if cnot_original > 0:
            overhead = ((cnot_tket - cnot_original) / cnot_original) * 100.0
        else:
           
            overhead = 0.0
    
        overhead_by_cnot_original[cnot_original].append(overhead)

        row_data = []
        if include_file_input:
            row_data.append(file_input)
        row_data += [
            adj,
            cnot_original,
            cnot_tket,
            round(overhead, 2)
        ]

        rows_to_write.append(row_data)

    # Ordina per numero di CNOT originale
    rows_to_write.sort(key=lambda row: row[2])

   
    with open(results_file, mode="w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows_to_write)

   
    summary_file = results_file.replace(".csv", "_summary.csv")
    with open(summary_file, mode="w", newline='') as summary_csv:
        writer = csv.writer(summary_csv)
        writer.writerow(["CNOT Originale", "Overhead Medio (%)"])

        for cnot_original in sorted(overhead_by_cnot_original.keys()):
            overhead_list = overhead_by_cnot_original[cnot_original]
            avg_overhead = sum(overhead_list) / len(overhead_list)
            writer.writerow([cnot_original, round(avg_overhead, 2)])

def transpile_and_evaluate_all_random_circuits_tket(filepath_adj, folder_input, folder_results, architecture: Architecture):
    results = []

    for filename in os.listdir(folder_input):
        filepath_input = os.path.join(folder_input, filename)
        if not filename.endswith(".txt"):
            continue

         
       
       
        result = process_circuit_file_tket(
            create_circuit_from_simple_file(filepath_input),
            architecture
        )
        results.append((os.path.basename(filepath_input), os.path.basename(filepath_adj), result))
    adj_name = os.path.splitext(os.path.basename(filepath_adj))[0]
    results_file = os.path.join(folder_results, f"my_results_{adj_name}.csv")
    save_results_tket(results, results_file, True)

    print(f"Processo completato! Risultati salvati in {results_file}")

def transpile_and_evaluate_ffqram_circuit(circ, filepath_adj, folder_results, architecture: Architecture):

    results = []
    result = process_circuit_file_tket(
            circ,
            architecture
        )
    results.append((os.path.basename(filepath_adj), result))
    adj_name = os.path.splitext(os.path.basename(filepath_adj))[0]
    results_file = os.path.join(folder_results, f"my_results_{adj_name}.csv")
    save_results_tket(results, results_file, False)

    print(f"Processo completato! Risultati salvati in {results_file}")

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

        transpile_and_evaluate_all_random_circuits_tket(filepath_adj, folder_input, folder_results, architecture)

    elif choice == '2':
        
        bits_needed = bits_needed = int(np.ceil(np.log2(num_qubits)))
        dataset = generate_normalized_dataset(pow(2, bits_needed), pow(2, bits_needed))

        df = datasets.load_iris(as_frame=True).frame
        # rename columns
        df.columns = ["f0","f1","f2","f3","class"]

        # drop class column
        df = df.drop('class', axis=1)

        df = df.sample(n=2, random_state=123)
        df.reset_index(drop=True, inplace=True)
    
        # dataset random 16x16

        scaler = StandardScaler()
        df.loc[:,:] = scaler.fit_transform(df.loc[:,:])
        df.loc[:,:] = sk_normalize(df.loc[:,:])
        df = df.to_numpy()
        
        ffqram_qc = FFQRAM_tk(df)

        print("FFQRAM CREATA")
       
       # print(ffqram_qc.n_gates_of_type(OpType.CX))
        transpile_and_evaluate_ffqram_circuit(ffqram_qc, filepath_adj, folder_results, architecture)


    else:
        print("Scelta non valida. Uscita.")
        sys.exit(1)

if __name__ == "__main__":
    main()
