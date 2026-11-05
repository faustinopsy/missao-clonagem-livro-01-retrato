# cap-01_check_env.py
# Checagem da maleta da Agente Íris: Python, bibliotecas e placa de vídeo.

import importlib   # importa uma biblioteca a partir do nome dela (texto)
import os          # informações do sistema operacional
import shutil      # procura programas instalados no computador
import subprocess  # executa outro programa e lê a resposta dele
import sys         # informações sobre o Python que está rodando

EXPECTED_PYTHON = (3, 12)                      # versão usada no livro
LIBRARIES = ["cv2", "mediapipe", "numpy"]      # cv2 é o nome do OpenCV no código


def check_python():
    """Mostra a versão do Python e se estamos dentro da maleta (.venv)."""
    version = sys.version.split()[0]
    print(f"Python: {version}")
    print(f"Executável: {sys.executable}")

    if sys.version_info[:2] == EXPECTED_PYTHON:
        print("✔ Versão do Python correta para o livro.")
    else:
        print("✘ Versão diferente da do livro (esperado: 3.12).")

    # Dentro da maleta, sys.prefix aponta para a pasta .venv
    # e sys.base_prefix aponta para o Python original do computador.
    if sys.prefix != sys.base_prefix:
        print("✔ Maleta (.venv) ativa.")
    else:
        print("✘ Maleta desativada. Rode: .\\.venv\\Scripts\\Activate.ps1")


def check_libraries():
    """Tenta importar cada biblioteca do Livro 1 e mostra a versão."""
    for name in LIBRARIES:
        try:
            module = importlib.import_module(name)
            print(f"✔ {name}: {module.__version__}")
        except ImportError:
            print(f"✘ {name}: não encontrada. Rode: python -m pip install -r requirements.txt")


def check_hardware():
    """Mostra os núcleos da CPU e procura uma placa NVIDIA."""
    print(f"Núcleos lógicos da CPU: {os.cpu_count()}")

    # Sem o programa nvidia-smi, não há placa NVIDIA (ou falta o driver).
    if shutil.which("nvidia-smi") is None:
        print("Placa NVIDIA não encontrada → Rota CPU (tudo certo para o Livro 1).")
        return

    # Pede ao nvidia-smi só três informações, separadas por vírgula.
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    print(f"Placa NVIDIA: {result.stdout.strip()}")
    print("→ Rota GPU disponível (será usada a partir do Livro 3).")


def main():
    print("=== Checagem da maleta — Agente Íris ===")
    check_python()
    check_libraries()
    check_hardware()


if __name__ == "__main__":
    main()