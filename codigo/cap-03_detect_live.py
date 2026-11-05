# cap-03_detect_live.py
# Missão: detectar rostos ao vivo, com modo privacidade e contador de FPS.

import time  # relógio, para medir os quadros por segundo

import cv2   # OpenCV: biblioteca de visão computacional

CAMERA_INDEX = 0
MODEL_FILE = "modelos/face_detection_yunet_2023mar.onnx"
WINDOW_NAME = "Agente Iris - deteccao ao vivo"
QUIT_KEY = "q"

SCORE_THRESHOLD = 0.9
NMS_THRESHOLD = 0.3
TOP_K = 5000

# Modo privacidade: esconde tudo, menos a região do rosto
HIDE_BACKGROUND = True
BLUR_SIZE = 61
BLUE_ALPHA = 0.4
MARGIN = 0.4

# Cores na ordem do OpenCV: (Azul, Verde, Vermelho)
BOX_COLOR = (0, 255, 0)      # verde
POINT_COLOR = (0, 0, 255)    # vermelho
BLUE_BGR = (255, 0, 0)       # azul do fundo
TEXT_COLOR = (255, 255, 255) # branco


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
    """Desenha a caixa, os 5 pontos e a confiança de UM rosto."""
    x, y, w, h = face[0:4].astype(int)   # converte para inteiro (dtype int)
    score = face[14]
    cv2.rectangle(image, (x, y), (x + w, y + h), BOX_COLOR, 2)
    points = face[4:14].reshape(5, 2)
    for px, py in points:
        cv2.circle(image, (int(px), int(py)), 4, POINT_COLOR, -1)
    label = f"rosto {score:.2f}"
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BOX_COLOR, 2)


# Janela ajustável, menor que a imagem, no canto superior esquerdo da tela
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 640, 360)
cv2.moveWindow(WINDOW_NAME, 0, 0)

camera = cv2.VideoCapture(CAMERA_INDEX)
ok, frame = camera.read()
if not ok:
    print("Não consegui ler a câmera.")
else:
    # Cria o detector UMA vez, já com o tamanho do quadro
    height, width = frame.shape[:2]
    detector = cv2.FaceDetectorYN.create(
        MODEL_FILE, "", (width, height), SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K
    )
    detector.setInputSize((width, height))
    print(f"Detectando em {width}x{height}. Aperte '{QUIT_KEY}' na janela para sair.")

    previous_time = time.perf_counter()

    while True:
        ok, frame = camera.read()
        if not ok:
            print("Não consegui ler a câmera.")
            break

        _, faces = detector.detect(frame)
        if faces is None:
            faces = []

        if HIDE_BACKGROUND:
            frame = hide_background(frame, faces)
        for face in faces:
            draw_face(frame, face)

        # FPS: 1 dividido pelo tempo que este quadro levou
        now = time.perf_counter()
        fps = 1 / (now - previous_time)
        previous_time = now
        info = f"FPS: {fps:.0f} | rostos: {len(faces)}"
        cv2.putText(frame, info, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, TEXT_COLOR, 2)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1)
        if key == ord(QUIT_KEY):
            break

camera.release()
cv2.destroyAllWindows()