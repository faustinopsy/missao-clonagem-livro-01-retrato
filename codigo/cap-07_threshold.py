# cap-07_threshold.py
# Missão: comparar TODAS as fotos entre si e escolher o limiar com os SEUS dados,
# contando os falsos aceites e as falsas rejeições.
#
# Pasta das fotos: uma subpasta por pessoa (só o código, nunca o nome):
#   dados-pessoais/identidade/P01/*.jpg
#   dados-pessoais/identidade/P02/*.jpg  ...

from itertools import combinations   # todos os pares possíveis de uma lista
from pathlib import Path

import cv2
import matplotlib.pyplot as plt       # gráficos (veio junto com o MediaPipe)
import numpy as np

PHOTOS_FOLDER = Path("dados-pessoais/identidade")
CHART_FILE = PHOTOS_FOLDER / "similaridades.png"
DETECTOR_FILE = "modelos/face_detection_yunet_2023mar.onnx"
RECOGNIZER_FILE = "modelos/face_recognition_sface_2021dec.onnx"

SCORE_THRESHOLD = 0.7        # o mesmo do Passo 3 (fotos com o fundo escondido)
NMS_THRESHOLD = 0.3
TOP_K = 5000
DOC_THRESHOLD = 0.363        # limiar da documentação do OpenCV
THRESHOLDS = [0.2, 0.3, 0.363, 0.4, 0.5, 0.6, 0.7]   # limiares que vamos testar
PHOTO_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def read_photo(path):
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        return None


def load_descriptors(detector, recognizer):
    """Percorre as subpastas. Devolve uma lista de (pessoa, nome da foto, descritor)."""
    samples = []
    for person_folder in sorted(PHOTOS_FOLDER.iterdir()):
        if not person_folder.is_dir():
            continue
        person = person_folder.name
        for path in sorted(person_folder.iterdir()):
            if path.suffix.lower() not in PHOTO_EXTENSIONS:
                continue
            image = read_photo(path)
            if image is None:
                print(f"  ✘ não abriu: {person}/{path.name}")
                continue
            height, width = image.shape[:2]
            detector.setInputSize((width, height))
            _, faces = detector.detect(image)
            if faces is None:
                print(f"  ✘ sem rosto: {person}/{path.name}")
                continue
            descriptor = recognizer.feature(recognizer.alignCrop(image, faces[0]))
            samples.append((person, path.name, descriptor))
    return samples


detector = cv2.FaceDetectorYN.create(DETECTOR_FILE, "", (320, 320), SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K)
recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_FILE, "")

# 1. Um descritor por foto
print(f"Lendo as fotos de {PHOTOS_FOLDER}…")
samples = load_descriptors(detector, recognizer)
people = sorted(set(person for person, _, _ in samples))
print(f"{len(samples)} fotos com rosto, {len(people)} pessoas:")
for person in people:
    count = sum(1 for p, _, _ in samples if p == person)
    print(f"  {person}: {count} fotos")

# 2. Todos os pares: mesma pessoa (genuínos) ou pessoas diferentes (impostores)
genuine = []
impostor = []
impostor_pairs = []        # (similaridade, "P02/foto.jpg x P03/foto.jpg"), para ver os mais parecidos
genuine_pairs = []
for (person_a, name_a, desc_a), (person_b, name_b, desc_b) in combinations(samples, 2):
    similarity = recognizer.match(desc_a, desc_b, cv2.FaceRecognizerSF_FR_COSINE)
    pair = f"{person_a}/{name_a} x {person_b}/{name_b}"
    if person_a == person_b:
        genuine.append(similarity)
        genuine_pairs.append((similarity, pair))
    else:
        impostor.append(similarity)
        impostor_pairs.append((similarity, pair))
genuine = np.array(genuine)
impostor = np.array(impostor)

print(f"\nPares da mesma pessoa:      {len(genuine):4}  similaridade de {genuine.min():.3f} a {genuine.max():.3f} (média {genuine.mean():.3f})")
print(f"Pares de pessoas diferentes: {len(impostor):4}  similaridade de {impostor.min():.3f} a {impostor.max():.3f} (média {impostor.mean():.3f})")

# Os casos-limite: os impostores MAIS parecidos e os pares da mesma pessoa MENOS parecidos
print("\nPessoas diferentes mais parecidas:")
for similarity, pair in sorted(impostor_pairs, reverse=True)[:3]:
    print(f"  {similarity:.3f}  {pair}")
print("Mesma pessoa menos parecida:")
for similarity, pair in sorted(genuine_pairs)[:3]:
    print(f"  {similarity:.3f}  {pair}")

# 3. Para cada limiar: quantos erros de cada tipo?
print("\nlimiar   falsos aceites          falsas rejeições")
for threshold in THRESHOLDS:
    false_accepts = np.sum(impostor >= threshold)    # impostor com similaridade alta: entrou
    false_rejects = np.sum(genuine < threshold)      # dono com similaridade baixa: barrado
    mark = "  <- documentação" if threshold == DOC_THRESHOLD else ""
    print(f"{threshold:6.3f}   {false_accepts:4} de {len(impostor):4} ({false_accepts / len(impostor):6.1%})"
          f"   {false_rejects:4} de {len(genuine):4} ({false_rejects / len(genuine):6.1%}){mark}")

# 4. Há uma "folga" entre os dois grupos?
if impostor.max() < genuine.min():
    print(f"\nFolga: qualquer limiar entre {impostor.max():.3f} e {genuine.min():.3f} separa tudo, sem erro.")
else:
    print("\nSem folga: os dois grupos se misturam; qualquer limiar vai errar de algum lado.")

# 5. Gráfico: as duas "montanhas" de similaridade e o limiar da documentação
plt.figure(figsize=(9, 4.5))
bins = np.linspace(-0.3, 1.0, 53)
plt.hist(impostor, bins=bins, alpha=0.6, color="tab:red", label=f"pessoas diferentes ({len(impostor)} pares)")
plt.hist(genuine, bins=bins, alpha=0.6, color="tab:green", label=f"mesma pessoa ({len(genuine)} pares)")
plt.axvline(DOC_THRESHOLD, color="black", linestyle="--", label=f"limiar da documentação ({DOC_THRESHOLD})")
plt.xlabel("similaridade de cosseno")
plt.ylabel("quantidade de pares")
plt.title("Mesma pessoa × pessoas diferentes")
plt.legend()
plt.tight_layout()
plt.savefig(CHART_FILE, dpi=120)
print(f"\nGráfico salvo em {CHART_FILE}")
plt.show()                                       # abre a janela do gráfico; feche para terminar
