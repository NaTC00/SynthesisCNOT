import sys
import os
from qiskit_aer import StatevectorSimulator
from qiskit.quantum_info import state_fidelity
from qiskit import transpile
from circuit_utils import (
    create_circuit_from_file,
    create_circuit_from_simple_file,
    load_adjacency_matrix_from_file,
    matrix_to_coupling_map,
    count_cnots
)

def main():
    if len(sys.argv) != 5:
        print("Uso: python main.py <matrice_adiacenza.txt> <cartella_input> <cartella_output> <cartella_results>")
        sys.exit(1)

    filepath_adj = sys.argv[1]
    folder_input = sys.argv[2]   # Es. Input/9CNOT/
    folder_output = sys.argv[3]  # Es. Output/9CNOT/
    folder_results = sys.argv[4] # Es. Results/

    os.makedirs(folder_output, exist_ok=True)
    os.makedirs(folder_results, exist_ok=True)

    if not os.path.isfile(filepath_adj):
        print(f"Errore: file matrice adiacenza non trovato: {filepath_adj}")
        sys.exit(1)

    if not os.path.isdir(folder_input):
        print(f"Errore: cartella di input non trovata: {folder_input}")
        sys.exit(1)

    # Carica la matrice di adiacenza e crea la coupling map
    adj_matrix = load_adjacency_matrix_from_file(filepath_adj)
    coupling_map = matrix_to_coupling_map(adj_matrix)

    results = []

    for filename in os.listdir(folder_input):
        if not filename.endswith(".txt"):
            continue  

        filepath_input = os.path.join(folder_input, filename)

        
        # Carica il circuito originale e conta le CNOT
        qc_original = create_circuit_from_simple_file(filepath_input)
        cnot_original = count_cnots(qc_original)

        # Costruisce il percorso corretto del file output
        if qc_original.num_qubits == 9:
            output_filename = f"Results_{filename}"
            filepath_output = os.path.join(folder_output, output_filename)
        elif  qc_original.num_qubits == 16:
            output_filename = f"Results_16Rigetti_{filename}"
            filepath_output = os.path.join(folder_output, output_filename)

        if not os.path.isfile(filepath_output):
            print(f"⚠️  File output non trovato per {filename}, saltato.")
            continue  


        # Transpile con Qiskit rispettando la topologia
        layout = list(range(qc_original.num_qubits))
        qc_qiskit = transpile(
            qc_original,
            coupling_map=coupling_map,
            basis_gates=['cx', 'h', 's', 'sdg', 'tdg', 't', 'x', 'y', 'z'],
            initial_layout=layout,
            optimization_level=0
        )

        # Carica il circuito sintetizzato dal file di output
        qc_java = create_circuit_from_file(filepath_output)

        # Simulazione degli stati quantistici
        simulator = StatevectorSimulator()
        state_original = simulator.run(qc_original).result().get_statevector()
        state_qiskit = simulator.run(qc_qiskit).result().get_statevector()
        state_java = simulator.run(qc_java).result().get_statevector()

        # Calcolo della fidelity
        fidelity_qiskit = state_fidelity(state_original, state_qiskit)
        fidelity_java = state_fidelity(state_original, state_java)

        # Conta le CNOT dopo la traslazione
        cnot_qiskit = count_cnots(qc_qiskit)
        cnot_java = count_cnots(qc_java)

        # Salva i risultati
        results.append((filename, output_filename, cnot_original, cnot_qiskit, cnot_java, fidelity_qiskit, fidelity_java))

    # Scrittura dei risultati nel file di output
    results_file = os.path.join(folder_results, "their_result_statefidelity.txt")
    with open(results_file, "w") as f:
        f.write(f"{'File Input':<25} {'File Output':<25} {'CNOT Originale':<15} {'CNOT Qiskit':<15} {'CNOT Java':<15} {'Fidelity Qiskit':<20} {'Fidelity Java':<20}\n")
        f.write("=" * 140 + "\n")
        for row in results:
            f.write(f"{row[0]:<25} {row[1]:<25} {row[2]:<15} {row[3]:<15} {row[4]:<15} {row[5]:<20.6f} {row[6]:<20.6f}\n")

    print(f"✅ Processo completato! Risultati salvati in {results_file}")

if __name__ == "__main__":
    main()
