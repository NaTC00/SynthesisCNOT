import subprocess
import os

def run_java_cnotsynth(adj_matrix_path: str, cicuit_inputh_path: str, circuit_output_path: str, jar_path: str):
   
    script_dir = os.path.dirname(os.path.abspath(__file__))
    jar_path = os.path.abspath(os.path.join(script_dir, jar_path))

    if not os.path.isfile(jar_path):
        print(f"Errore: il file JAR non esiste: {jar_path}")
        return

    result = subprocess.run(
           ["java", "-jar", jar_path, adj_matrix_path, cicuit_inputh_path, circuit_output_path],
           capture_output=True, text=True
           )
    if result.returncode != 0:
        print("Errore nell'esecuzione del programma Java:")
        print(result.stderr)
    else:
        print("Programma cnotsynth eseguito con successo")
