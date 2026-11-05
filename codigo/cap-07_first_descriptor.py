# cap-07_first_descriptor.py
# Missão: transformar o seu rosto num DESCRITOR (uma lista de números) e ver o rosto alinhado.

import time  # relógio, para medir quanto o descritor leva

import cv2           # OpenCV: câmera, YuNet (detector) e SFace (descritor)
import numpy as np   # NumPy: contas com o vetor

CAMERA_INDEX = 0
DETECTOR_FILE = "modelos/face_detection_yunet_2023mar.onnx"
RECOGNIZER_FILE = "modelos/face_recognition_sface_2021dec.onnx"
WINDOW_NAME = "Agente Iris - descritor"
ALIGNED_WINDOW = "Rosto alinhado"
CAPTURE_KEY = " "    # barra de espaço: calcula o descritor do quadro atual
QUIT_KEY = "q"

SCORE_THRESHOLD = 0.9
NMS_THRESHOLD = 0.3
TOP_K = 5000
ALIGNED_ZOOM = 3      # o recorte alinhado é pequeno: mostramos 3 vezes maior

# Modo privacidade (o mesmo do Capítulo 3)
HIDE_BACKGROUND = True
BLUR_SIZE = 61
BLUE_ALPHA = 0.4
MARGIN = 0.4

BOX_COLOR = (0, 255, 0)      # verde
POINT_COLOR = (0, 0, 255)    # vermelho
BLUE_BGR = (255, 0, 0)       # azul do fundo
TEXT_COLOR = (255, 255, 255) # branco


def hide_background(image, faces):
    """Fundo borrado e azulado; só a região em volta de cada rosto fica nítida."""
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


def draw_face(image, face):
    """Caixa e os 5 pontos que o alinhamento usa (olhos, nariz, cantos da boca)."""
    x, y, w, h = face[0:4].astype(int)
    cv2.rectangle(image, (x, y), (x + w, y + h), BOX_COLOR, 2)
    points = face[4:14].reshape(5, 2)
    for px, py in points:
        cv2.circle(image, (int(px), int(py)), 4, POINT_COLOR, -1)


def show_descriptor(recognizer, frame, face):
    """Alinha o rosto, calcula o descritor e mostra tudo no terminal e numa janela."""
    aligned = recognizer.alignCrop(frame, face)    # 1. alinha e recorta
    start = time.perf_counter()
    descriptor = recognizer.feature(aligned)       # 2. rosto alinhado -> descritor
    elapsed_ms = (time.perf_counter() - start) * 1000

    height, width = aligned.shape[:2]
    print("\n--- descritor calculado ---")
    print(f"Recorte alinhado: {width}x{height} pixels")
    print(f"Formato do descritor: {descriptor.shape}  (tipo {descriptor.dtype})")
    print(f"Primeiros 8 números: {np.round(descriptor[0, :8], 3)}")
    print(f"Comprimento do vetor (norma): {np.linalg.norm(descriptor):.2f}")
    print(f"Tempo do descritor: {elapsed_ms:.1f} ms")

    big = cv2.resize(aligned, (width * ALIGNED_ZOOM, height * ALIGNED_ZOOM),
                     interpolation=cv2.INTER_NEAREST)   # NEAREST: mostra os pixels como são
    cv2.imshow(ALIGNED_WINDOW, big)


# Janelas ajustáveis, lado a lado no topo da tela
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 640, 360)
cv2.moveWindow(WINDOW_NAME, 0, 0)
cv2.namedWindow(ALIGNED_WINDOW, cv2.WINDOW_AUTOSIZE)
cv2.moveWindow(ALIGNED_WINDOW, 660, 0)

camera = cv2.VideoCapture(CAMERA_INDEX)
ok, frame = camera.read()
if not ok:
    print("Não consegui ler a câmera.")
else:
    height, width = frame.shape[:2]
    detector = cv2.FaceDetectorYN.create(
        DETECTOR_FILE, "", (width, height), SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K
    )
    recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_FILE, "")   # cria o SFace UMA vez
    print(f"Câmera em {width}x{height}.")
    print("Aperte ESPAÇO na janela para calcular o descritor; 'q' para sair.")

    while True:
        ok, frame = camera.read()
        if not ok:
            print("Não consegui ler a câmera.")
            break

        _, faces = detector.detect(frame)
        if faces is None:
            faces = []

        shown = hide_background(frame, faces) if HIDE_BACKGROUND else frame.copy()
        for face in faces:
            draw_face(shown, face)
        info = f"rostos: {len(faces)} | ESPACO = descritor"
        cv2.putText(shown, info, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, TEXT_COLOR, 2)
        cv2.imshow(WINDOW_NAME, shown)

        key = cv2.waitKey(1)
        if key == ord(QUIT_KEY):
            break
        if key == ord(CAPTURE_KEY):
            if len(faces) == 0:
                print("Nenhum rosto no quadro. Olhe para a câmera e tente de novo.")
            else:
                # O descritor usa o quadro ORIGINAL (sem o fundo borrado) e o rosto mais confiável
                show_descriptor(recognizer, frame, faces[0])

camera.release()
cv2.destroyAllWindows()
