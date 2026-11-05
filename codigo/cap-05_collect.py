# cap-05_collect.py
# Missão: gravar exemplos das SUAS expressões (pessoa, origem, classe e os 52 blendshapes).

import csv                # ler e gravar arquivos de tabela (CSV)
import os                 # pastas e arquivos do sistema
import time               # relógio: carimbo de tempo do modo vídeo

import cv2                # OpenCV: câmera, imagem e desenho
import mediapipe as mp    # MediaPipe: pontos faciais e blendshapes
import numpy as np        # NumPy: máscara e lista de pontos

CAMERA_INDEX = 0
MODEL_FILE = "modelos/face_landmarker.task"
DATA_FOLDER = "dados-pessoais/expressoes"
DATA_FILE = "dados-pessoais/expressoes/amostras.csv"
PERSON_CODE = "P01"        # código da pessoa (nunca o nome): P01, P02...
SOURCE = "camera"          # origem da amostra: "camera" ou "foto"
WINDOW_NAME = "Agente Iris - coleta de expressoes"
QUIT_KEY = "q"
STOP_KEY = "0"

# Tecla -> classe. A classe "neutro" ensina o modelo a não ver expressão onde não há.
KEY_TO_CLASS = {
    "1": "neutro",
    "2": "sorriso",
    "3": "surpresa",
    "4": "bravo",
}
SAMPLES_PER_CLASS = 60    # meta de amostras por classe
SAMPLE_EVERY = 3          # grava 1 a cada 3 quadros (evita amostras quase iguais)

# Modo privacidade: esconde tudo fora do contorno do rosto
HIDE_BACKGROUND = True
BLUR_SIZE = 61
BLUE_ALPHA = 0.4
BLUE_BGR = (255, 0, 0)

TEXT_COLOR = (255, 255, 255)   # branco
REC_COLOR = (0, 0, 255)        # vermelho
DONE_COLOR = (0, 255, 0)       # verde

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


def to_pixels(face, width, height):
    """Converte os 478 pontos normalizados (0 a 1) em uma lista de pixels (x, y)."""
    pixel_points = []
    for point in face:
        pixel_points.append((int(point.x * width), int(point.y * height)))
    return pixel_points


def hide_background(image, pixel_points):
    """Devolve a imagem com tudo borrado e azulado, menos o contorno do rosto."""
    background = cv2.GaussianBlur(image, (BLUR_SIZE, BLUR_SIZE), 0)
    tint = background.copy()
    tint[:] = BLUE_BGR
    background = cv2.addWeighted(background, 1 - BLUE_ALPHA, tint, BLUE_ALPHA, 0)

    if len(pixel_points) == 0:
        return background

    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    hull = cv2.convexHull(np.array(pixel_points, dtype=np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    return cv2.copyTo(image, mask, background)


def count_samples():
    """Conta quantas amostras de cada classe já existem no arquivo."""
    counts = {}
    for class_name in KEY_TO_CLASS.values():
        counts[class_name] = 0
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if row["classe"] in counts:
                    counts[row["classe"]] += 1
    return counts


def save_sample(class_name, blendshapes):
    """Acrescenta uma linha ao arquivo: pessoa, origem, classe e os 52 escores."""
    new_file = not os.path.exists(DATA_FILE)
    with open(DATA_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if new_file:   # primeira vez: escreve o cabeçalho com os nomes das colunas
            header = ["pessoa", "origem", "classe"]
            for category in blendshapes:
                header.append(category.category_name)
            writer.writerow(header)
        row = [PERSON_CODE, SOURCE, class_name]
        for category in blendshapes:
            row.append(round(category.score, 4))
        writer.writerow(row)


os.makedirs(DATA_FOLDER, exist_ok=True)   # cria a pasta, se ainda não existir
counts = count_samples()

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_FILE),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=True,
)
landmarker = FaceLandmarker.create_from_options(options)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 640, 360)
cv2.moveWindow(WINDOW_NAME, 0, 0)

camera = cv2.VideoCapture(CAMERA_INDEX)
recording = None          # classe sendo gravada agora (None = nenhuma)
frame_number = 0
print("Teclas: 1 neutro, 2 sorriso, 3 surpresa, 4 bravo, 0 para, q sai.")

while True:
    ok, frame = camera.read()
    if not ok:
        print("Não consegui ler a câmera.")
        break
    height, width = frame.shape[:2]
    frame_number += 1

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, int(time.perf_counter() * 1000))

    pixel_points = []
    if len(result.face_landmarks) > 0:
        pixel_points = to_pixels(result.face_landmarks[0], width, height)

    # Grava 1 amostra a cada SAMPLE_EVERY quadros, só se houver rosto
    if recording is not None and len(result.face_blendshapes) > 0:
        if frame_number % SAMPLE_EVERY == 0:
            save_sample(recording, result.face_blendshapes[0])
            counts[recording] += 1
            if counts[recording] >= SAMPLES_PER_CLASS:
                print(f"Meta atingida para '{recording}'.")
                recording = None

    if HIDE_BACKGROUND:
        frame = hide_background(frame, pixel_points)

    # Painel: uma linha por classe, com a contagem
    font = cv2.FONT_HERSHEY_SIMPLEX
    row = 0
    for key, class_name in KEY_TO_CLASS.items():
        color = TEXT_COLOR
        if counts[class_name] >= SAMPLES_PER_CLASS:
            color = DONE_COLOR
        if class_name == recording:
            color = REC_COLOR
        text = f"[{key}] {class_name}: {counts[class_name]}/{SAMPLES_PER_CLASS}"
        cv2.putText(frame, text, (20, 40 + row * 35), font, 0.9, color, 2)
        row += 1
    if recording is not None:
        cv2.putText(frame, f"GRAVANDO: {recording}", (20, 40 + row * 35), font, 0.9, REC_COLOR, 2)

    cv2.imshow(WINDOW_NAME, frame)
    key = cv2.waitKey(1)
    if key == ord(QUIT_KEY):
        break
    elif key == ord(STOP_KEY):
        recording = None
    else:
        for key_text, class_name in KEY_TO_CLASS.items():
            if key == ord(key_text):
                if counts[class_name] >= SAMPLES_PER_CLASS:
                    print(f"'{class_name}' já tem {counts[class_name]} amostras (meta: {SAMPLES_PER_CLASS}). Nada gravado.")
                else:
                    recording = class_name
                    print(f"Gravando '{class_name}'... faça a expressão e mexa um pouco a cabeça.")

camera.release()
cv2.destroyAllWindows()
landmarker.close()
print("Amostras no arquivo:", counts)
