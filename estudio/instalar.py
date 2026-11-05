# instalar.py — prepara o Estúdio: instala as bibliotecas e baixa os modelos.
#
# Quem chama este arquivo é o iniciar_estudio.bat, já com o Python da maleta (.venv).
# Ele usa só a biblioteca padrão do Python, porque roda ANTES de as outras existirem.
# Pode rodar quantas vezes quiser: o que já está certo é pulado.

import hashlib                     # confere o hash SHA-256 dos modelos
import subprocess                  # chama o pip
import sys
import urllib.request              # baixa arquivos da internet
from importlib import metadata     # descobre a versão instalada de cada biblioteca
from pathlib import Path

ESTUDIO_FOLDER = Path(__file__).resolve().parent
REQUIREMENTS_FILE = ESTUDIO_FOLDER / "requirements.txt"
MODELS_FOLDER = ESTUDIO_FOLDER / "modelos"

# Modelos: arquivo -> (endereço de download, hash SHA-256 de referência do Capítulo 4)
MODELS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        "64184E229B263107BC2B804C6625DB1341FF2BB731874B0BCC2FE6544E0BC9FF",
    ),
    # Capítulo 7 (modo Identidade): YuNet do Capítulo 3 e SFace
    "face_detection_yunet_2023mar.onnx": (
        "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
        "8F2383E4DD3CFBB4553EA8718107FC0423210DC964F9F4280604804ED2552FA4",
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
        "0BA9FBFA01B5270C96627C4EF784DA859931E02F04419C829E83484087C34E79",
    ),
}


def file_hash(path):
    """SHA-256 de um arquivo, em maiúsculas (como o Get-FileHash do PowerShell)."""
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_requirements():
    """Lê o requirements.txt: devolve um dicionário pacote -> versão."""
    pins = {}
    for line in REQUIREMENTS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line == "" or line.startswith("#"):
            continue
        name, version = line.split("==")
        pins[name] = version
    return pins


def missing_libraries():
    """Quais bibliotecas faltam ou estão numa versão diferente da lista?"""
    missing = []
    for name, version in read_requirements().items():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = None
        if installed != version:
            missing.append(f"{name} (pedida {version}, instalada {installed or 'nenhuma'})")
    return missing


def install_libraries():
    missing = missing_libraries()
    if len(missing) == 0:
        print("[ok] Bibliotecas: todas instaladas, nas versões da lista.")
        return True
    print("[..] Instalando bibliotecas (a primeira vez pode levar alguns minutos):")
    for item in missing:
        print("     -", item)
    answer = subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)])
    if answer.returncode != 0:
        print("[ERRO] O pip não conseguiu instalar. Confira a internet e leia a mensagem acima.")
        return False
    print("[ok] Bibliotecas instaladas.")
    return True


def download_models():
    MODELS_FOLDER.mkdir(exist_ok=True)
    all_ok = True
    for file_name, (url, expected_hash) in MODELS.items():
        path = MODELS_FOLDER / file_name
        if path.exists() and file_hash(path) == expected_hash:
            print(f"[ok] Modelo {file_name}: já existe e o hash confere.")
            continue
        print(f"[..] Baixando {file_name}…")
        partial = path.with_suffix(".parcial")       # baixa com outro nome; só renomeia se estiver certo
        try:
            urllib.request.urlretrieve(url, partial)
        except OSError as error:
            print(f"[ERRO] Não consegui baixar {file_name}: {error}")
            all_ok = False
            continue
        if file_hash(partial) != expected_hash:
            print(f"[ERRO] {file_name} baixou, mas o hash não confere. Arquivo descartado.")
            partial.unlink()
            all_ok = False
            continue
        partial.replace(path)
        print(f"[ok] Modelo {file_name}: baixado e conferido.")
    return all_ok


def main():
    print("=== Estúdio de Treino: preparando o ambiente ===")
    print(f"Python: {sys.version.split()[0]}  ({sys.executable})")
    if sys.version_info[:2] != (3, 12):
        print("[ERRO] O Estúdio precisa do Python 3.12.")
        return 1
    if not install_libraries():
        return 1
    if not download_models():
        return 1
    print("=== Tudo pronto. Abrindo o Estúdio… ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())                # o número volta para o .bat: 0 = deu certo
