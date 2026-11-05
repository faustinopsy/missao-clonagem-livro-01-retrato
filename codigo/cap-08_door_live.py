# cap-08_door_live.py
# Missão: a PORTA da Divisão — identificação 1:N ao vivo.
# Compara o rosto com TODOS os agentes da galeria, escolhe o mais parecido,
# aplica o limiar e anota cada tentativa no registro de acesso (sem imagens).

import csv
import time
from datetime import datetime          # data e hora para o registro
from pathlib import Path

import cv2
import numpy as np

CAMERA_INDEX = 0
GALLERY_FOLDER = Path("dados-pessoais/galeria")
LOG_FILE = Path("dados-pessoais/porta/registro.csv")
DETECTOR_FILE = "modelos/face_detection_yunet_2023mar.onnx"
RECOGNIZER_FILE = "modelos/face_recognition_sface_2021dec.onnx"
WINDOW_NAME = "Agente Iris - porta da Divisao"
QUIT_KEY = "q"

VIDEO_SCORE = 0.9
NMS_THRESHOLD = 0.3
TOP_K = 5000
SMALL_WIDTH, SMALL_HEIGHT = 640, 360   # analisar no tamanho da tela (lição do Cap. 7)
DESCRIBE_EVERY = 3                     # descritor a cada 3 quadros
THRESHOLD = 0.5                        # limiar escolhido no Cap. 7
THRESHOLD_STEP = 0.05
AMBIGUITY_MARGIN = 0.05                # dois agentes acima do limiar e quase empatados: "tente de novo"
CONFIRM_COUNT = 5                      # a mesma resposta em 5 descritores seguidos (~0,5 s) para valer
UNKNOWN = "desconhecido"

# Modo privacidade (Capítulo 3)
BLUR_SIZE = 31
BLUE_ALPHA = 0.4
MARGIN = 0.4
BLUE_BGR = (255, 0, 0)

PANEL_WIDTH = 300                      # painel da porta, à direita do vídeo
GREEN = (0, 200, 0)
RED = (0, 0, 230)
YELLOW = (0, 200, 230)
GRAY = (128, 128, 128)
WHITE = (255, 255, 255)
DOOR_BROWN = (40, 70, 120)


def load_gallery():
    """Lê a galeria: {código: array (n_poses, 128)}. Só pastas com código válido (P01, P02…)."""
    gallery = {}
    for person_folder in sorted(GALLERY_FOLDER.iterdir()):
        code = person_folder.name
        if not (person_folder.is_dir() and code.startswith("P") and code[1:].isdigit()):
            continue
        descriptors = []
        for path in sorted(person_folder.glob("*.npy")):
            descriptors.append(np.load(path))
        if len(descriptors) > 0:
            gallery[code] = np.vstack(descriptors)     # uma linha por pose
    return gallery


def identify(recognizer, query, gallery):
    """1:N. Para cada agente, a MAIOR similaridade entre as poses dele.
    Devolve a lista [(similaridade, código), ...] do mais parecido para o menos parecido."""
    ranking = []
    for code, poses in gallery.items():
        best = -1.0
        for pose in poses:
            value = recognizer.match(query, pose.reshape(1, -1), cv2.FaceRecognizerSF_FR_COSINE)
            best = max(best, value)
        ranking.append((best, code))
    return sorted(ranking, reverse=True)


def decide(ranking, threshold):
    """Primeiro o mais parecido; depois, o limiar. Devolve (resposta, código, cor)."""
    best_value, best_code = ranking[0]
    if best_value < threshold:
        return "PORTA FECHADA", UNKNOWN, RED
    if len(ranking) > 1:
        second_value, second_code = ranking[1]
        if second_value >= threshold and best_value - second_value < AMBIGUITY_MARGIN:
            return "TENTE DE NOVO", f"{best_code}?{second_code}?", YELLOW
    return "PORTA ABERTA", best_code, GREEN


def write_log(code, value, answer):
    """Uma linha no registro de acesso: data e hora, código, similaridade e resposta."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    new_file = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if new_file:
            writer.writerow(["data_hora", "codigo", "similaridade", "resposta"])
        writer.writerow([datetime.now().isoformat(sep=" ", timespec="seconds"), code, f"{value:.3f}", answer])


def hide_background(image, faces):
    background = cv2.GaussianBlur(image, (BLUR_SIZE, BLUR_SIZE), 0)
    tint = background.copy()
    tint[:] = BLUE_BGR
    background = cv2.addWeighted(background, 1 - BLUE_ALPHA, tint, BLUE_ALPHA, 0)
    height, width = image.shape[:2]
    for face in faces:
        x, y, w, h = face[0:4].astype(int)
        left = max(x - int(w * MARGIN), 0)
        top = max(y - int(h * MARGIN), 0)
        right = min(x + w + int(w * MARGIN), width)
        bottom = min(y + h + int(h * MARGIN), height)
        background[top:bottom, left:right] = image[top:bottom, left:right]
    return background


def draw_door(panel, answer, code, color, ranking, threshold):
    """O "gadget": a porta (fechada ou aberta), a luz e o ranking dos agentes."""
    panel[:] = (30, 30, 30)
    cv2.circle(panel, (PANEL_WIDTH // 2, 30), 18, color, -1)                      # a luz
    cv2.rectangle(panel, (70, 60), (230, 260), (90, 90, 90), 4)                    # o batente
    if answer == "PORTA ABERTA":
        door = np.array([[74, 64], [120, 84], [120, 240], [74, 256]], dtype=np.int32)   # porta girada
        cv2.fillPoly(panel, [door], DOOR_BROWN)
    else:
        cv2.rectangle(panel, (74, 64), (226, 256), DOOR_BROWN, -1)                 # porta fechada
        cv2.circle(panel, (205, 160), 6, (0, 200, 230), -1)                        # maçaneta
    cv2.putText(panel, answer, (20, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(panel, code, (20, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
    cv2.putText(panel, f"limiar {threshold:.2f} (+/-)", (20, 352), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GRAY, 1)
    for row, (value, agent) in enumerate(ranking[:4]):                             # os 4 mais parecidos
        cv2.putText(panel, f"{agent}  {value:.2f}", (20, 385 + row * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 1)


recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_FILE, "")
gallery = load_gallery()
if len(gallery) == 0:
    print(f"Galeria vazia em {GALLERY_FOLDER}. Faça o Passo 1 primeiro.")
    raise SystemExit
print("Galeria:")
for code, poses in gallery.items():
    print(f"  {code}: {len(poses)} poses")

detector = cv2.FaceDetectorYN.create(DETECTOR_FILE, "", (SMALL_WIDTH, SMALL_HEIGHT),
                                     VIDEO_SCORE, NMS_THRESHOLD, TOP_K)
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, SMALL_WIDTH + PANEL_WIDTH, SMALL_HEIGHT + 120)
cv2.moveWindow(WINDOW_NAME, 0, 0)

camera = cv2.VideoCapture(CAMERA_INDEX)
threshold = THRESHOLD
frame_number = 0
ranking = []
candidate = None                      # resposta que está sendo confirmada
candidate_count = 0                   # quantos descritores seguidos deram essa resposta
answer, code, color = "SEM ROSTO", "-", GRAY   # a resposta CONFIRMADA (é a que a porta mostra)
previous_time = time.perf_counter()
print(f"Porta pronta. Limiar {threshold}. Teclas: + e - mudam o limiar; '{QUIT_KEY}' sai.")

while True:
    ok, frame = camera.read()
    if not ok:
        print("Não consegui ler a câmera.")
        break
    small = cv2.resize(frame, (SMALL_WIDTH, SMALL_HEIGHT))
    scale = frame.shape[1] / SMALL_WIDTH
    _, faces = detector.detect(small)
    faces = [] if faces is None else faces

    frame_number += 1
    progress = ""
    if len(faces) == 0:
        # Sem rosto: a porta fecha NA HORA (errar para o lado seguro) e a confirmação recomeça
        ranking = []
        answer, code, color = "SEM ROSTO", "-", GRAY
        candidate, candidate_count = None, 0
    elif len(ranking) == 0 or frame_number % DESCRIBE_EVERY == 0:
        face = faces[0].copy()
        face[0:14] = face[0:14] * scale                   # caixa e pontos no tamanho original
        query = recognizer.feature(recognizer.alignCrop(frame, face))
        ranking = identify(recognizer, query, gallery)
        new_answer = decide(ranking, threshold)
        # Confirmação: a resposta só vale depois de se repetir CONFIRM_COUNT vezes seguidas
        if candidate is not None and new_answer[:2] == candidate[:2]:
            candidate_count += 1
        else:
            candidate, candidate_count = new_answer, 1
        if candidate_count >= CONFIRM_COUNT and candidate[:2] != (answer, code):
            answer, code, color = candidate                # muda a porta e registra
            write_log(code, ranking[0][0], answer)
            print(f"{datetime.now():%H:%M:%S}  {answer:14} {code:14} similaridade {ranking[0][0]:.3f}")
    if candidate is not None and candidate[:2] != (answer, code):
        progress = f"confirmando {candidate[1]}: {candidate_count}/{CONFIRM_COUNT}"

    shown = hide_background(small, faces)
    for face in faces:
        x, y, w, h = face[0:4].astype(int)
        cv2.rectangle(shown, (x, y), (x + w, y + h), color, 2)
    now = time.perf_counter()
    fps = 1 / (now - previous_time)
    previous_time = now
    cv2.putText(shown, f"FPS: {fps:.0f}", (15, SMALL_HEIGHT - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
    if progress:
        cv2.putText(shown, progress, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, YELLOW, 2)

    panel = np.zeros((SMALL_HEIGHT + 120, PANEL_WIDTH, 3), dtype=np.uint8)
    draw_door(panel, answer, code, color, ranking, threshold)
    video = np.zeros((SMALL_HEIGHT + 120, SMALL_WIDTH, 3), dtype=np.uint8)
    video[:SMALL_HEIGHT] = shown
    cv2.imshow(WINDOW_NAME, np.hstack([video, panel]))

    key = cv2.waitKey(1)
    if key == ord(QUIT_KEY):
        break
    if key == ord("+") or key == ord("="):
        threshold = min(threshold + THRESHOLD_STEP, 0.95)
    if key == ord("-"):
        threshold = max(threshold - THRESHOLD_STEP, 0.05)

camera.release()
cv2.destroyAllWindows()
