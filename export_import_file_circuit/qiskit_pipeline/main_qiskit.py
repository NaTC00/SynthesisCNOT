import sys
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
    FFQRAM,
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
    generate_normalized_dataset
)
import numpy as np
from cnot_synth import run_java_cnotsynth


def ensure_directories(*folders):
    for folder in folders:
        os.makedirs(folder, exist_ok=True)


def check_input_files(file):
    if not os.path.isfile(file):
        sys.exit(f"Errore: file non è stato trovato: {file}")

def check_input_folder(folder):
    if not os.path.isdir(folder):
        sys.exit(f"Errore: cartella non trovata: {folder}")

def process_circuit(qc_original, coupling_map, include_file_input=True):


    transpiler_result = []
    num_cnot_list = []

    for i in range(4):
  
        if include_file_input:
            qc_transpiled = transpile(
                qc_original,
                basis_gates=['cx', 'h', 's', 'sdg', 't', 'tdg', 'x', 'y', 'z'],
                coupling_map=coupling_map,
                layout_method="trivial",
                seed_transpiler=123,
                optimization_level=i
            )
        else:
            qc_transpiled = transpile(
                qc_original,
                basis_gates=['cx', 'h', 's', 'sdg', 't', 'tdg', 'x', 'y', 'z', 'u'],
                coupling_map=coupling_map,
                layout_method="trivial",
                seed_transpiler=123,
                optimization_level=i
            )
            print(count_cx_gates(qc_transpiled))
        print(f"traspilato circuito livello {i}")
        transpiler_result.append(qc_transpiled)
    
    for qc_transpiled in transpiler_result:
        num_cnot = count_cx_gates(qc_transpiled)
        num_cnot_list.append(num_cnot)


   
    num_cnot_qc_original = count_cx_gates(qc_original)
    



    return (
        num_cnot_qc_original,
        num_cnot_list   
    )

def java_better_stats(results):
    better = sum(1 for row in results if row[4] < row[3])
    return better, len(results)


def save_results_traspiler(results, results_file, include_file_input=True):
    # Dizionario per raggruppare gli overhead in base al numero di CNOT originali
    overhead_by_cnot_original = defaultdict(list)

    # Definizione delle intestazioni del file CSV, inclusa la colonna dell'overhead
    # Intestazioni CSV
    headers = []
    if include_file_input:
        headers.append("File Input")  # Nome del file sorgente
    headers += [
        "Matrice Adiacenza",    # Rappresentazione della topologia hardware
        "CNOT Originale",       # Numero di CNOT nel circuito originale       
        "CNOT Qiskit_Level0",   # Numero di CNOT dopo il transpiler Qiskit livello 0
        "CNOT Qiskit_Level1",   # Idem per livello 1
        "CNOT Qiskit_Level2",   # Idem per livello 2
        "CNOT Qiskit_Level3",   # Idem per livello 3
        "Overhead (%)"         # Overhead percentuale medio rispetto al circuito originale
    ]
    
    # Lista temporanea dove accumuliamo tutte le righe da scrivere nel CSV
    rows_to_write = []

    # Iterazione su ciascun risultato nella lista "results"
    for row in results:

        if include_file_input:
            file_input = row[0]
            adj = row[1]
            cont_original = row[2][0]
            cnot_levels = row[2][1]
        else:
            file_input = None
            adj = row[0]
            cont_original = row[1]
            cnot_levels = row[2]
       
        # Calcolo della media dei CNOT ottenuti dopo transpiling con livelli 0-3
        avg_final_count = sum(cnot_levels) / len(cnot_levels)

        # Calcolo dell'overhead in percentuale rispetto al circuito originale
        overhead = ((avg_final_count - cont_original) / cont_original) * 100

        # Salvataggio dell'overhead nel dizionario, raggruppato per CNOT originali
        overhead_by_cnot_original[cont_original].append(overhead)

        # Aggiunta della riga da scrivere nel file, includendo l'overhead arrotondato a 2 decimali
        row_data = []
        if include_file_input:
            row_data.append(file_input)
        row_data += [
            adj,
            cont_original,
            *cnot_levels,
            round(overhead, 2)
        ]

        rows_to_write.append(row_data)

    # Ordina i risultati per numero di CNOT Originale (colonna 2)
    rows_to_write.sort(key=lambda row: row[2 if include_file_input else 1])

    # Scrittura finale su file CSV
    with open(results_file, mode="w", newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Scrive la riga di intestazione
        writer.writerow(headers)

        # Scrive tutte le righe accumulate
        writer.writerows(rows_to_write)     

    # Scrittura nuova tabella riassuntiva
    summary_file = results_file.replace(".csv", "_summary.csv")
    with open(summary_file, mode="w", newline='') as summary_csv:
        writer = csv.writer(summary_csv)
        writer.writerow(["CNOT Originale", "Overhead Medio (%)"])

        for cnot_original in sorted(overhead_by_cnot_original.keys()):
            overhead_list = overhead_by_cnot_original[cnot_original]
            avg_overhead = sum(overhead_list) / len(overhead_list)
            writer.writerow([cnot_original, round(avg_overhead, 2)])


def save_results(results, results_file):
    java_better, total = java_better_stats(results)
    with open(results_file, "w") as f:
        f.write(f"{'File Input':<25} {'Matrice Adiacenza':<25} {'CNOT Originale':<15} {'CNOT Qiskit':<15} {'CNOT Java':<15} {'Fidelity Qiskit':<20} {'Fidelity Java':<20}\n")
        f.write("=" * 140 + "\n")
        for row in results:
            f.write(f"{row[0]:<25} {row[1]:<25} {row[2]:<15} {row[3]:<15} {row[4]:<15} {row[5]:<20.6f} {row[6]:<20.6f}\n")
        # ---------- riepilogo finale ----------
        f.write("\n")
        f.write(f"Java con meno CNOT di Qiskit: {java_better}/{total}\n")

def transpile_and_evaluate_random_circuits(filepath_adj, folder_input, folder_results, coupling_map):
    
    results = []

    # Itera su tutti i file nella cartella input
    for filename in os.listdir(folder_input):
        filepath_input = os.path.join(folder_input, filename)
        if not filename.endswith(".txt"):
            print("file ignored")
            continue  # Ignora file non di testo

        filepath_input = os.path.join(folder_input, filename)

        print(os.path.basename(filepath_input))
      
        result =process_circuit(
            create_circuit_from_simple_file(filepath_input),
            coupling_map
        )
        #entropy = von_neumann_entropy(circuit)
        results.append((
        os.path.basename(filepath_input),
        os.path.basename(filepath_adj),
        result
        ))

       

       
    adj_name = os.path.splitext(os.path.basename(filepath_adj))[0]
    results_file = os.path.join(folder_results, f"results_{adj_name}.csv")
    save_results_traspiler(results, results_file)

    print(f"Processo completato! Risultati salvati in {results_file}")

def transpile_and_evaluate_ffqram_circuit(filepath_adj, qc_ffqram, folder_results, coupling_map):
    results = []
    num_cnot_qc_original, num_cnot_qc_trnspiled =process_circuit(
            qc_ffqram,
            coupling_map,
            False
        )
    #entropy = von_neumann_entropy(qc_ffqram)
    results.append((os.path.basename(filepath_adj), num_cnot_qc_original, num_cnot_qc_trnspiled))
    results_file = os.path.join(folder_results, f"ffqram_results.csv")
    save_results_traspiler(results, results_file, include_file_input=False)

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


def von_neumann_entropy(circuit: QuantumCircuit) -> float:
    # Step 1: recupero lo state vector dal circuito
    state = Statevector.from_instruction(circuit)
    
    # Step 2: converto lo state vector nella matrice densità
    rho = DensityMatrix(state)
    
    # Step 3: calcolo l'entropia
    return entropy(rho)
    


def main():

    if len(sys.argv) != 3:
        print("Uso: <matrice_adiacenza.txt> <cartella_results>")
        sys.exit(1)

    filepath_adj, folder_results = sys.argv[1:]

    check_input_files(filepath_adj)
    ensure_directories(folder_results)

    adj_matrix = load_adjacency_matrix_from_file(filepath_adj)
    coupling_map = matrix_to_coupling_map(adj_matrix)

    print("\nScegli la modalità di sperimentazione:")
    print("1. Circuiti randomici da file")
    print("2. FF-QRAM su dataset generato internamente")
    choice = input("Inserisci 1 o 2: ").strip()


    if choice == '1':
        
        folder_input = input("Inserisci il path della cartella contenente i circuiti: ").strip()
        
        check_input_folder(folder_input)

        transpile_and_evaluate_random_circuits(
            filepath_adj, folder_input, folder_results, coupling_map
        )

    elif choice == '2':
        num_qubits = coupling_map.size()
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
        df.loc[:,:] = normalize(df.loc[:,:])
        df = df.to_numpy()

        ffqram_qc = FFQRAM(dataset)
        print(ffqram_qc.count_ops())
        print(count_cx_gates(ffqram_qc))
        transpile_and_evaluate_ffqram_circuit(filepath_adj, ffqram_qc, folder_results, coupling_map)


    else:
        print("Scelta non valida. Uscita.")
        sys.exit(1)



    
    


if __name__ == "__main__":
    main()
