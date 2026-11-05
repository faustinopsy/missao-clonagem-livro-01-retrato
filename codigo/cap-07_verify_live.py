# cap-07_verify_live.py
# Missão: verificação 1:1 ao vivo — "a pessoa na frente da câmera é o P01?"
# A referência é a MÉDIA dos descritores das fotos do P01 (Passo 4), salva num arquivo .npy.

import time
from pathlib import Path

import cv2
import numpy as np

CAMERA_INDEX = 0
PERSON = "P01"                                         # quem a porta deve reconhecer
PHOTOS_FOLDER = Path("dados-pessoais/identidade") / PERSON
REFERENCE_FILE = Path("dados-pessoais/identidade") / f"{PERSON}.npy"
DETECTOR_FILE = "modelos/face_detection_yunet_2023mar.onnx"
RECOGNIZER_FILE = "modelos/face_recognition_sface_2021dec.onnx"
WINDOW_NAME = "Agente Iris - verificacao ao vivo"
QUIT_KEY = "q"

VIDEO_SCORE_THRESHOLD = 0.9   # detector no vídeo (rosto inteiro, sem fundo escondido)
PHOTO_SCORE_THRESHOLD = 0.7   # detector nas fotos com o fundo escondido (Passo 3)
NMS_THRESHOLD = 0.3
TOP_K = 5000
THRESHOLD = 0.5               # limiar escolhido no Passo 4 (meio da folga 0.403–0.576)
THRESHOLD_STEP = 0.05         # teclas + e - mudam o limiar ao vivo

# Modo privacidade (o mesmo do Capítulo 3)
HIDE_BACKGROUND = True
BLUR_SIZE = 61
BLUE_ALPHA = 0.4
MARGIN = 0.4
BLUE_BGR = (255, 0, 0)

GREEN = (0, 200, 0)
RED = (0, 0, 230)
GRAY = (128, 128, 128)
WHITE = (255, 255, 255)


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


def build_reference(recognizer):
    """Média dos descritores das fotos da pessoa. Cada descritor é normalizado antes
    (comprimento 1), para que todas as fotos pesem igual na média."""
    detector = cv2.FaceDetectorYN.create(DETECTOR_FILE, "", (320, 320),
                                         PHOTO_SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K)
    descriptors = []
    for path in sorted(PHOTOS_FOLDER.glob("*.jpg")):
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        detector.setInputSize((width, height))
        _, faces = detector.detect(image)
        if faces is None:
            continue
        descriptor = recognizer.feature(recognizer.alignCrop(image, faces[0]))
        descriptors.append(descriptor / np.linalg.norm(descriptor))
    if len(descriptors) == 0:
        return None, 0
    stacked = np.vstack(descriptors)                    # uma linha por foto: (n, 128)
    reference = np.mean(stacked, axis=0, keepdims=True) # média de cada coluna: (1, 128)
    return reference.astype(np.float32), len(descriptors)


def draw_panel(image, similarity, threshold):
    """O "gadget": uma luz verde (é o P01) ou vermelha (não é), com o número embaixo."""
    if similarity is None:
        color, text = GRAY, "sem rosto"
    elif similarity >= threshold:
        color, text = GREEN, f"{PERSON} confirmado"
    else:
        color, text = RED, f"nao e o {PERSON}"
    cv2.rectangle(image, (0, 0), (image.shape[1], 110), (40, 40, 40), -1)
    cv2.circle(image, (60, 55), 38, color, -1)
    cv2.putText(image, text, (120, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
    detail = f"limiar {threshold:.2f}  (+/- muda)"
    if similarity is not None:
        detail = f"similaridade {similarity:.2f} | " + detail
    cv2.putText(image, detail, (120, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)


recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_FILE, "")

# 1. A referência: lê do .npy se já existir; senão, calcula a partir das fotos e salva
if REFERENCE_FILE.exists():
    reference = np.load(REFERENCE_FILE)
    print(f"Referência carregada de {REFERENCE_FILE}")
else:
    reference, count = build_reference(recognizer)
    if reference is None:
        print(f"Nenhuma foto com rosto em {PHOTOS_FOLDER}. Faça o Passo 4 primeiro.")
        raise SystemExit
    np.save(REFERENCE_FILE, reference)
    print(f"Referência do {PERSON}: média de {count} fotos, salva em {REFERENCE_FILE}")
print(f"Formato da referência: {reference.shape}")

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 640, 360)
cv2.moveWindow(WINDOW_NAME, 0, 0)

camera = cv2.VideoCapture(CAMERA_INDEX)
ok, frame = camera.read()
if not ok:
    print("Não consegui ler a câmera.")
else:
    height, width = frame.shape[:2]
    detector = cv2.FaceDetectorYN.create(DETECTOR_FILE, "", (width, height),
                                         VIDEO_SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K)
    threshold = THRESHOLD
    previous_time = time.perf_counter()
    print(f"Verificando o {PERSON}. Teclas: + e - mudam o limiar; '{QUIT_KEY}' sai.")

    while True:
        ok, frame = camera.read()
        if not ok:
            print("Não consegui ler a câmera.")
            break

        _, faces = detector.detect(frame)
        if faces is None:
            faces = []

        # 2. Descritor do rosto mais confiável e comparação com a referência
        similarity = None
        if len(faces) > 0:
            descriptor = recognizer.feature(recognizer.alignCrop(frame, faces[0]))
            similarity = recognizer.match(descriptor, reference, cv2.FaceRecognizerSF_FR_COSINE)

        shown = hide_background(frame, faces) if HIDE_BACKGROUND else frame.copy()
        draw_panel(shown, similarity, threshold)

        now = time.perf_counter()
        fps = 1 / (now - previous_time)
        previous_time = now
        cv2.putText(shown, f"FPS: {fps:.0f}", (20, shown.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)
        cv2.imshow(WINDOW_NAME, shown)

        key = cv2.waitKey(1)
        if key == ord(QUIT_KEY):
            break
        if key == ord("+") or key == ord("="):        # "=" é a mesma tecla do "+" sem o Shift
            threshold = min(threshold + THRESHOLD_STEP, 0.95)
        if key == ord("-"):
            threshold = max(threshold - THRESHOLD_STEP, 0.05)

camera.release()
cv2.destroyAllWindows()
