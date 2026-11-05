# cap-04_landmarks_live.py
# Missão: desenhar os 478 pontos ao vivo, escondendo tudo fora do contorno do rosto.

import time               # relógio: FPS e carimbo de tempo do modo vídeo

import cv2                # OpenCV: câmera, imagem e desenho
import mediapipe as mp    # MediaPipe: pontos faciais
import numpy as np        # NumPy: matrizes (máscara e lista de pontos)

CAMERA_INDEX = 0
MODEL_FILE = "modelos/face_landmarker.task"
WINDOW_NAME = "Agente Iris - mapa do rosto"
QUIT_KEY = "q"

# Modo privacidade: esconde tudo fora do contorno do rosto
HIDE_BACKGROUND = True
BLUR_SIZE = 61
BLUE_ALPHA = 0.4

# Cores na ordem do OpenCV: (Azul, Verde, Vermelho)
MESH_COLOR = (0, 255, 0)       # verde: pontos da malha
IRIS_COLOR = (0, 255, 255)     # amarelo: pontos das íris
BLUE_BGR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)

FIRST_IRIS_POINT = 468         # os pontos 468 a 477 são das íris

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
        return background             # sem rosto: esconde tudo

    # Máscara: imagem preta de 1 canal; o contorno do rosto é pintado de branco
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    hull = cv2.convexHull(np.array(pixel_points, dtype=np.int32))
    cv2.fillConvexPoly(mask, hull, 255)

    # Copia a imagem original para o fundo, só onde a máscara é branca
    return cv2.copyTo(image, mask, background)


def draw_landmarks(image, pixel_points):
    """Desenha os pontos da malha (verde) e das íris (amarelo)."""
    for px, py in pixel_points[:FIRST_IRIS_POINT]:
        cv2.circle(image, (px, py), 1, MESH_COLOR, -1)
    for px, py in pixel_points[FIRST_IRIS_POINT:]:
        cv2.circle(image, (px, py), 2, IRIS_COLOR, -1)


options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_FILE),
    running_mode=RunningMode.VIDEO,   # VIDEO = quadros em sequência, com rastreamento
    num_faces=1,
)
landmarker = FaceLandmarker.create_from_options(options)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 640, 360)
cv2.moveWindow(WINDOW_NAME, 0, 0)

camera = cv2.VideoCapture(CAMERA_INDEX)
print(f"Aperte '{QUIT_KEY}' na janela para sair.")
previous_time = time.perf_counter()

while True:
    ok, frame = camera.read()
    if not ok:
        print("Não consegui ler a câmera.")
        break
    height, width = frame.shape[:2]

    # BGR -> RGB e carimbo de tempo em milissegundos (sempre crescente)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_ms = int(time.perf_counter() * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    pixel_points = []
    if len(result.face_landmarks) > 0:
        pixel_points = to_pixels(result.face_landmarks[0], width, height)

    if HIDE_BACKGROUND:
        frame = hide_background(frame, pixel_points)
    draw_landmarks(frame, pixel_points)

    now = time.perf_counter()
    fps = 1 / (now - previous_time)
    previous_time = now
    info = f"FPS: {fps:.0f} | pontos: {len(pixel_points)}"
    cv2.putText(frame, info, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, TEXT_COLOR, 2)

    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) == ord(QUIT_KEY):
        break

camera.release()
cv2.destroyAllWindows()
landmarker.close()