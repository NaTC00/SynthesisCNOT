import sys
import os
from matplotlib import pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import GenericBackendV2
from sklearn import datasets
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import normalize
from qiskit.visualization import circuit_drawer
from collections import defaultdict
from circuit_utils import (
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
    compute_fidelity_qiskit
)
import numpy as np
from cnot_synth import run_java_cnotsynth


def ensure_directories(*folders):
    for folder in folders:
        os.makedirs(folder, exist_ok=True)


def check_input_files(file):
    if not os.path.isfile(file):
        sys.exit(f"Errore: file non è stato trovato: {file}")


def process_circuit_file(filepath_input, filepath_output, filepath_adj, filepath_jar, coupling_map, choice, circuit=None):
    
    if choice == "1":
        qc_original = create_circuit_from_simple_file(filepath_input)
        create_file_from_circuit(qc_original, filepath_input, ['CNOT', 'H', 'S', 'S+', 'T', 'T+', 'X', 'Y', 'Z'])

   
    if choice == "2":
        qc_original = QuantumCircuit(circuit.num_qubits)
        qc_original = qc_original.compose(circuit)

    transpiler_result = []
    num_cnot_list = []

    for i in range(4):
  
        qc_transpiled = transpile(
        qc_original,
        basis_gates=['cx', 'h', 's', 'sdg', 't', 'tdg', 'x', 'y', 'z', 'u'],
        coupling_map=coupling_map,
        layout_method="trivial",
        seed_transpiler=123,
        optimization_level=i
        )
        transpiler_result.append(qc_transpiled)
    
    for qc_transpiled in transpiler_result:
        num_cnot = count_cx_gates(qc_transpiled)
        num_cnot_list.append(num_cnot)


   
    num_cnot_qc_original = count_cx_gates(qc_original)
    



    return (
        os.path.basename(filepath_input),
        os.path.basename(filepath_adj),
        num_cnot_qc_original,
        num_cnot_list
        
    )

def java_better_stats(results):
    better = sum(1 for row in results if row[4] < row[3])
    return better, len(results)

def save_results_traspiler(results, results_file):
    overhead_by_cnot_original = defaultdict(list)
    with open(results_file, "w") as f:
        # Intestazione della tabella
        f.write(f"{'File Input':<25} {'Matrice Adiacenza':<25} {'CNOT Originale':<20} {'CNOT Qiskit_Level0':<25} {'CNOT Qiskit_Level1':<25} {'CNOT Qiskit_Level2':<25} {'CNOT Qiskit_Level3':<25}\n")
        f.write("=" * 170 + "\n")
        
        # Scrittura dei dati
        for row in results:
            file_input = row[0]
            adj = row[1]
            cont_original = row[2]
            cnot_levels = row[3]
            f.write(f"{file_input:<25} {adj:<25} {cont_original:<20}")
            for num_cnot in cnot_levels:
                f.write(f"{num_cnot:<25}")
            f.write("\n")

            # Calcolo della media dei CNOT finali dei livelli di ottimizzazione di qiskit
            avg_final_count = sum(cnot_levels) / len(cnot_levels)

            #calcolo overhead
            overhead = ((avg_final_count - cont_original)/cont_original) * 100

            #inserisco l'overehead nel dizionario
            overhead_by_cnot_original[cont_original].append(overhead)
        
        f.write("\n\n====== Overhead Medio per CNOT Originale (sulla media dei livelli 0-3) ======\n")
        for cnot_original in sorted(overhead_by_cnot_original.keys()):

            #recupero la lista con gli overhead raggruppati per numero di cnot
            overhead_list = overhead_by_cnot_original[cnot_original]

            #calcolo la media dell'ovrehead per quel gruppo di cnot iniziali
            avg_overhead = sum(overhead_list) / len(overhead_list)
            f.write(f"Overhead medio con {cnot_original} CNOT: {avg_overhead:.2f}%\n")
        


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

def run_random_circuit_synth(filepath_adj, filepath_jar, folder_input, folder_output, folder_results, coupling_map):
    
    results = []

    # Itera su tutti i file nella cartella input
    for filename in os.listdir(folder_input):
        filepath_input = os.path.join(folder_input, filename)
        if not filename.endswith(".txt"):
            continue  # Ignora file non di testo

        filepath_input = os.path.join(folder_input, filename)
        output_filename = f"{filename.replace('.txt', '')}_Output.txt"
        filepath_output = os.path.join(folder_output, output_filename)

        result = process_circuit_file(
            filepath_input,
            filepath_output,
            filepath_adj,
            filepath_jar,
            coupling_map,
            "1"
        )
        results.append(result)
       # print(f"file: {filepath_input} fidelity_qiskit: {result[5]} | fidelity_java: {result[6]}")
       # print(f"cnot_original: {result[2]} | cnot_qiskit: {result[3]} | cnot_java: {result[4]}\n")

    adj_name = os.path.splitext(os.path.basename(filepath_adj))[0]
    results_file = os.path.join(folder_results, f"my_results_{adj_name}.txt")
    save_results_traspiler(results, results_file)

    print(f"Processo completato! Risultati salvati in {results_file}")

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
    result = process_circuit_file(
            filepath_input,
            filepath_output,
            filepath_adj,
            filepath_jar,
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


def main():

    if len(sys.argv) != 6:
        print("Uso: <matrice_adiacenza.txt> <cnotsynth.jar> <cartella_input> <cartella_output> <cartella_results>")
        sys.exit(1)

    for file in sys.argv[1:2]:
        check_input_files(file)


    filepath_adj, filepath_jar, folder_input, folder_output, folder_results = sys.argv[1:]

    ensure_directories(folder_input, folder_output, folder_results)

    adj_matrix = load_adjacency_matrix_from_file(filepath_adj)
    coupling_map = matrix_to_coupling_map(adj_matrix)

    print("Seleziona un'opzione:")
    print("1. Sperimentazione sintesi CNOT su circuiti randomici")
    print("2. Sperimentazione sintesi CNOT per FFQRAM")

    choice = input("Inserisci il numero dell'opzione (1 o 2): ")

    if choice.strip() == "1":
        run_random_circuit_synth(filepath_adj, filepath_jar, folder_input, folder_output, folder_results, coupling_map)
    elif choice.strip() == "2":
        run_ffqram_synthesis(filepath_adj, filepath_jar, folder_input, folder_output, folder_results, coupling_map)

    


if __name__ == "__main__":
    main()
