# ambiente.py — verificações do computador para a aba "Ambiente" do Estúdio.
# Cada verificação devolve um dicionário: item, situação ("ok", "erro" ou "info") e detalhe.

import hashlib            # calcula o hash SHA-256 de um arquivo
import importlib          # importa uma biblioteca a partir do nome (texto)
import os                 # número de núcleos da CPU
import shutil             # procura programas instalados (nvidia-smi)
import subprocess         # executa outro programa e lê a resposta
import sys                # informações sobre o Python que está rodando
from pathlib import Path

ESTUDIO_FOLDER = Path(__file__).resolve().parent         # a pasta estudio (independente do livro)
EXPECTED_PYTHON = (3, 12)

# Nome no código -> nome do pacote no pip
LIBRARIES = {
    "cv2": "opencv-contrib-python",
    "mediapipe": "mediapipe",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "joblib": "joblib",
    "PIL": "pillow",
    "matplotlib": "matplotlib",     # gráfico da aba Limiar (modo Identidade)
}

from instalar import MODELS        # a lista de modelos fica num lugar só: o instalar.py


def check(item, status, detail):
    """Monta o resultado de uma verificação."""
    return {"item": item, "status": status, "detail": detail}


IS_EXE = getattr(sys, "frozen", False)   # True quando roda como Estudio.exe (PyInstaller)


def check_python():
    version = sys.version.split()[0]
    if sys.version_info[:2] == EXPECTED_PYTHON:
        return check("Python", "ok", f"{version}")
    return check("Python", "erro", f"{version} (o Estúdio usa 3.12)")


def check_suitcase():
    if IS_EXE:
        return check("Maleta (.venv)", "info", "não precisa: o Estudio.exe já traz tudo dentro")
    # Dentro da maleta, sys.prefix é diferente de sys.base_prefix (Capítulo 1)
    if sys.prefix != sys.base_prefix:
        return check("Maleta (.venv)", "ok", sys.prefix)
    return check("Maleta (.venv)", "erro", "não está ativa")


def check_libraries():
    results = []
    for module_name, package_name in LIBRARIES.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "?")
            results.append(check(package_name, "ok", version))
        except ImportError:
            results.append(check(package_name, "erro", "não instalada"))
    return results


def file_hash(path):
    """Calcula o SHA-256 de um arquivo (em maiúsculas, como o PowerShell)."""
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def check_models():
    results = []
    for file_name, (url, expected_hash) in MODELS.items():
        path = ESTUDIO_FOLDER / "modelos" / file_name
        if not path.exists():
            results.append(check(f"Modelo {file_name}", "erro", "arquivo não encontrado (rode o iniciar_estudio.bat)"))
        elif file_hash(path) == expected_hash:
            results.append(check(f"Modelo {file_name}", "ok", "hash confere"))
        else:
            results.append(check(f"Modelo {file_name}", "erro", "hash diferente do livro (rode o iniciar_estudio.bat)"))
    return results


def check_camera(camera_index=0):
    import cv2   # importado aqui: se o OpenCV faltar, só esta verificação falha
    camera = cv2.VideoCapture(camera_index)
    ok, frame = camera.read()
    camera.release()
    if ok:
        height, width = frame.shape[:2]
        return check("Câmera", "ok", f"{width}x{height}")
    return check("Câmera", "erro", "não consegui ler a câmera (ligada? outro programa usando?)")


def check_project_folder(folder):
    """A pasta do projeto existe (ou pode ser criada) e aceita arquivos?"""
    folder = Path(folder)
    try:
        folder.mkdir(parents=True, exist_ok=True)
        test_file = folder / ".teste_escrita"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink()                     # apaga o arquivo de teste
        return check("Pasta do projeto", "ok", str(folder))
    except OSError as error:
        return check("Pasta do projeto", "erro", f"sem permissão: {error}")


def check_hardware():
    results = [check("CPU", "info", f"{os.cpu_count()} núcleos lógicos")]
    if shutil.which("nvidia-smi") is None:
        results.append(check("GPU NVIDIA", "info", "não encontrada → Rota CPU (tudo certo para o Livro 1)"))
    else:
        # CREATE_NO_WINDOW: no Windows, não pisca uma janela preta (importante no .exe)
        no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        answer = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, stdin=subprocess.DEVNULL, creationflags=no_window,
        )
        results.append(check("GPU NVIDIA", "info", answer.stdout.strip()))
    return results


def run_all_checks(project_folder):
    """Roda todas as verificações e devolve a lista de resultados."""
    results = [check_python(), check_suitcase()]
    results += check_libraries()
    results += check_models()
    results.append(check_project_folder(project_folder))
    results.append(check_camera())
    results += check_hardware()
    return results
