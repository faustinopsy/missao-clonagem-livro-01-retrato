# cap-03_detection_lab.py
# Missão: testar o limiar, o tamanho máximo do rosto e o tempo de detecção.

import time  # relógio, para medir o tempo da detecção

import cv2   # OpenCV: biblioteca de visão computacional

CAMERA_INDEX = 0
MODEL_FILE = "modelos/face_detection_yunet_2023mar.onnx"
WINDOW_NAME = "Agente Iris - laboratorio"

QUIT_KEY = "q"
UP_KEY = "s"        # Subir o limiar
DOWN_KEY = "d"      # Descer o limiar
THRESHOLD_STEP = 0.05

NMS_THRESHOLD = 0.3
TOP_K = 5000

# Modo privacidade: esconde tudo, menos a região do rosto
HIDE_BACKGROUND = True
BLUR_SIZE = 61
BLUE_ALPHA = 0.4
MARGIN = 0.4

# Cores na ordem do OpenCV: (Azul, Verde, Vermelho)
BOX_COLOR = (0, 255, 0)
POINT_COLOR = (0, 0, 255)
BLUE_BGR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)


def hide_background(image, faces):
    """Devolve uma cópia da imagem com o fundo borrado e azulado,
    mantendo nítida só a região em volta de cada rosto."""
    background = cv2.GaussianBlur(image, (BLUR_SIZE, BLUR_SIZE), 0)
    tint = background.copy()
    tint[:] = BLUE_BGR
    background = cv2.addWeighted(background, 1 - BLUE_ALPHA, tint, BLUE_ALPHA, 0)

    height, width = image.shape[:2]
    for face in faces:
        x, y, w, h = face[0:4].astype(int)   # converte para inteiro (dtype int)
        margin_x = int(w * MARGIN)
        margin_y = int(h * MARGIN)
        left = max(x - margin_x, 0)
        top = max(y - margin_y, 0)
        right = min(x + w + margin_x, width)
        bottom = min(y + h + margin_y, height)
        background[top:bottom, left:right] = image[top:bottom, left:right]

    return background


def draw_face(image, face):
    """Desenha a caixa, os 5 pontos, a confiança e a altura de UM rosto."""
    x, y, w, h = face[0:4].astype(int)   # converte para inteiro (dtype int)
    score = face[14]
    cv2.rectangle(image, (x, y), (x + w, y + h), BOX_COLOR, 2)
    points = face[4:14].reshape(5, 2)
    for px, py in points:
        cv2.circle(image, (int(px), int(py)), 4, POINT_COLOR, -1)
    label = f"rosto {score:.2f} | altura {h}px"
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BOX_COLOR, 2)


cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 640, 360)
cv2.moveWindow(WINDOW_NAME, 0, 0)

score_threshold = 0.9   # começa igual aos passos anteriores

camera = cv2.VideoCapture(CAMERA_INDEX)
ok, frame = camera.read()
if not ok:
    print("Não consegui ler a câmera.")
else:
    height, width = frame.shape[:2]
    detector = cv2.FaceDetectorYN.create(
        MODEL_FILE, "", (width, height), score_threshold, NMS_THRESHOLD, TOP_K
    )
    print(f"'{UP_KEY}' sobe o limiar, '{DOWN_KEY}' desce, '{QUIT_KEY}' sai.")

    while True:
        ok, frame = camera.read()
        if not ok:
            print("Não consegui ler a câmera.")
            break

        # Cronometra SÓ a detecção (sem câmera, sem desenho)
        start = time.perf_counter()
        _, faces = detector.detect(frame)
        detect_ms = (time.perf_counter() - start) * 1000   # segundos -> milissegundos

        if faces is None:
            faces = []

        if HIDE_BACKGROUND:
            frame = hide_background(frame, faces)
        for face in faces:
            draw_face(frame, face)

        info = f"limiar: {score_threshold:.2f} | deteccao: {detect_ms:.0f} ms | rostos: {len(faces)}"
        cv2.putText(frame, info, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_COLOR, 2)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1)
        if key == ord(QUIT_KEY):
            break
        elif key == ord(UP_KEY):
            score_threshold = min(score_threshold + THRESHOLD_STEP, 0.99)
            detector.setScoreThreshold(score_threshold)
        elif key == ord(DOWN_KEY):
            score_threshold = max(score_threshold - THRESHOLD_STEP, 0.05)
            detector.setScoreThreshold(score_threshold)

camera.release()
cv2.destroyAllWindows()