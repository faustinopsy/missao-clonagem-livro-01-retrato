# cap-08_evaluate.py
# Missão: MEDIR a porta antes de confiar nela.
# As fotos de dados-pessoais/identidade/<código>/ fazem o papel de "visitantes" e passam pela
# porta, sem câmera. Dois testes:
#   A) com todos na galeria: a porta abre para a pessoa CERTA?
#   B) com a pessoa FORA da galeria (mundo aberto): a porta diz "desconhecido"?

from pathlib import Path

import cv2
import numpy as np

VISITORS_FOLDER = Path("dados-pessoais/identidade")
GALLERY_FOLDER = Path("dados-pessoais/galeria")
DETECTOR_FILE = "modelos/face_detection_yunet_2023mar.onnx"
RECOGNIZER_FILE = "modelos/face_recognition_sface_2021dec.onnx"

PHOTO_SCORE = 0.7
NMS_THRESHOLD = 0.3
TOP_K = 5000
THRESHOLDS = [0.4, 0.5, 0.6]          # 0.5 é o limiar da porta; os outros, para comparar
AMBIGUITY_MARGIN = 0.05
PHOTO_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def valid_code(name):
    return name.startswith("P") and name[1:].isdigit()


def load_gallery():
    """{código: {pose: descritor}} — guardamos o nome da pose para não comparar uma foto com ela mesma."""
    gallery = {}
    for person_folder in sorted(GALLERY_FOLDER.iterdir()):
        if person_folder.is_dir() and valid_code(person_folder.name):
            poses = {}
            for path in sorted(person_folder.glob("*.npy")):
                poses[path.stem] = np.load(path)
            gallery[person_folder.name] = poses
    return gallery


def load_visitors(detector, recognizer):
    """Lista de (código verdadeiro, nome da foto, descritor) para as fotos com rosto."""
    visitors = []
    for person_folder in sorted(VISITORS_FOLDER.iterdir()):
        if not (person_folder.is_dir() and valid_code(person_folder.name)):
            continue
        for path in sorted(person_folder.iterdir()):
            if path.suffix.lower() not in PHOTO_EXTENSIONS:
                continue
            image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            height, width = image.shape[:2]
            detector.setInputSize((width, height))
            _, faces = detector.detect(image)
            if faces is None:
                continue
            descriptor = recognizer.feature(recognizer.alignCrop(image, faces[0]))
            visitors.append((person_folder.name, path.stem, descriptor))
    return visitors


def ranking_for(recognizer, query, query_name, gallery, left_out=None):
    """Ranking 1:N. Pula o agente 'left_out' (teste B) e a pose com o MESMO nome da foto
    (quando a foto do visitante também foi usada no cadastro, ela não pode se comparar consigo)."""
    ranking = []
    for code, poses in gallery.items():
        if code == left_out:
            continue
        best = -1.0
        for pose_name, pose in poses.items():
            if pose_name == query_name:
                continue
            best = max(best, recognizer.match(query, pose, cv2.FaceRecognizerSF_FR_COSINE))
        if best > -1.0:
            ranking.append((best, code))
    return sorted(ranking, reverse=True)


def decide(ranking, threshold):
    """O mesmo decide() da porta: (resposta, código)."""
    best_value, best_code = ranking[0]
    if best_value < threshold:
        return "FECHADA", "desconhecido"
    if len(ranking) > 1:
        second_value, second_code = ranking[1]
        if second_value >= threshold and best_value - second_value < AMBIGUITY_MARGIN:
            return "TENTE DE NOVO", best_code
    return "ABERTA", best_code


detector = cv2.FaceDetectorYN.create(DETECTOR_FILE, "", (320, 320), PHOTO_SCORE, NMS_THRESHOLD, TOP_K)
recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_FILE, "")
gallery = load_gallery()
visitors = load_visitors(detector, recognizer)
print(f"Galeria: {', '.join(gallery)}  |  visitantes: {len(visitors)} fotos com rosto\n")

# Os rankings não dependem do limiar: calculamos uma vez e reaproveitamos
rankings_a = []          # teste A: todos na galeria
rankings_b = []          # teste B: o visitante fora da galeria
for true_code, name, descriptor in visitors:
    rankings_a.append(ranking_for(recognizer, descriptor, name, gallery))
    rankings_b.append(ranking_for(recognizer, descriptor, name, gallery, left_out=true_code))

print("A) TODOS NA GALERIA — a porta abre para a pessoa certa?")
print("limiar  certos  falsas rejeições  confusões  tente de novo")
for threshold in THRESHOLDS:
    correct = rejected = confused = retry = 0
    for (true_code, _, _), ranking in zip(visitors, rankings_a):
        answer, code = decide(ranking, threshold)
        if answer == "FECHADA":
            rejected += 1
        elif answer == "TENTE DE NOVO":
            retry += 1
        elif code == true_code:
            correct += 1
        else:
            confused += 1                       # abriu para OUTRO agente
    print(f"{threshold:6.2f}  {correct:6}  {rejected:16}  {confused:9}  {retry:13}")

print("\nB) A PESSOA FORA DA GALERIA (mundo aberto) — a porta diz \"desconhecido\"?")
print("limiar  desconhecido (certo)  FALSOS ACEITES (abriu para outro)")
for threshold in THRESHOLDS:
    unknown = accepted = 0
    for (true_code, _, _), ranking in zip(visitors, rankings_b):
        answer, _ = decide(ranking, threshold)
        if answer == "FECHADA":
            unknown += 1
        else:
            accepted += 1
    print(f"{threshold:6.2f}  {unknown:20}  {accepted:10}")

print("\nPor pessoa, com o limiar 0.5:")
print("pessoa  fotos  A: certos  A: rejeitadas  B: maior similaridade com OUTRO agente")
for code in gallery:
    photos = correct = rejected = 0
    highest_other = -1.0
    for i, (true_code, _, _) in enumerate(visitors):
        if true_code != code:
            continue
        photos += 1
        answer, chosen = decide(rankings_a[i], 0.5)
        if answer == "ABERTA" and chosen == code:
            correct += 1
        if answer == "FECHADA":
            rejected += 1
        highest_other = max(highest_other, rankings_b[i][0][0])   # o "impostor" mais parecido
    if photos > 0:
        print(f"{code:6}  {photos:5}  {correct:9}  {rejected:13}  {highest_other:.3f}")
