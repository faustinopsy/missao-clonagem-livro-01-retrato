# cap-04_face_regions.py
# Missão: desenhar o mapa do rosto por regiões e mostrar o número de pontos-chave.

import time               # relógio: carimbo de tempo do modo vídeo

import cv2                # OpenCV: câmera, imagem e desenho
import mediapipe as mp    # MediaPipe: pontos faciais
import numpy as np        # NumPy: máscara e lista de pontos

CAMERA_INDEX = 0
MODEL_FILE = "modelos/face_landmarker.task"
WINDOW_NAME = "Agente Iris - regioes do rosto"
QUIT_KEY = "q"
MESH_KEY = "m"            # liga/desliga a malha de triângulos

# Modo privacidade: esconde tudo fora do contorno do rosto
HIDE_BACKGROUND = True
BLUR_SIZE = 61
BLUE_ALPHA = 0.4
BLUE_BGR = (255, 0, 0)

Connections = mp.tasks.vision.FaceLandmarksConnections

# Cada região: (ligações entre pontos, cor BGR)
# "LEFT" e "RIGHT" do MediaPipe são o lado esquerdo e direito DA PESSOA
REGIONS = {
    "contorno": (Connections.FACE_LANDMARKS_FACE_OVAL, (255, 255, 255)),     # branco
    "sobrancelhas": (Connections.FACE_LANDMARKS_LEFT_EYEBROW
                     + Connections.FACE_LANDMARKS_RIGHT_EYEBROW, (0, 165, 255)),  # laranja
    "olhos": (Connections.FACE_LANDMARKS_LEFT_EYE
              + Connections.FACE_LANDMARKS_RIGHT_EYE, (0, 255, 0)),          # verde
    "iris": (Connections.FACE_LANDMARKS_LEFT_IRIS
             + Connections.FACE_LANDMARKS_RIGHT_IRIS, (0, 255, 255)),        # amarelo
    "nariz": (Connections.FACE_LANDMARKS_NOSE, (255, 0, 255)),               # rosa
    "labios": (Connections.FACE_LANDMARKS_LIPS, (0, 0, 255)),                # vermelho
}
MESH = Connections.FACE_LANDMARKS_TESSELATION
MESH_COLOR = (200, 200, 200)

# Pontos-chave que ganham o número escrito ao lado
KEY_POINTS = [10, 4, 152, 33, 263, 61, 291, 468, 473]
LABEL_COLOR = (255, 255, 255)

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


def draw_connections(image, pixel_points, connections, color, thickness):
    """Desenha uma linha para cada ligação (ponto inicial -> ponto final)."""
    for connection in connections:
        start = pixel_points[connection.start]
        end = pixel_points[connection.end]
        cv2.line(image, start, end, color, thickness)


def draw_key_points(image, pixel_points):
    """Escreve o número (índice) ao lado de cada ponto-chave."""
    for index in KEY_POINTS:
        px, py = pixel_points[index]
        cv2.circle(image, (px, py), 3, LABEL_COLOR, -1)
        cv2.putText(image, str(index), (px + 5, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, LABEL_COLOR, 1)


options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_FILE),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
)
landmarker = FaceLandmarker.create_from_options(options)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 640, 360)
cv2.moveWindow(WINDOW_NAME, 0, 0)

camera = cv2.VideoCapture(CAMERA_INDEX)
show_mesh = False
print(f"'{MESH_KEY}' liga/desliga a malha, '{QUIT_KEY}' sai.")

while True:
    ok, frame = camera.read()
    if not ok:
        print("Não consegui ler a câmera.")
        break
    height, width = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, int(time.perf_counter() * 1000))

    pixel_points = []
    if len(result.face_landmarks) > 0:
        pixel_points = to_pixels(result.face_landmarks[0], width, height)

    if HIDE_BACKGROUND:
        frame = hide_background(frame, pixel_points)

    if len(pixel_points) > 0:
        if show_mesh:
            draw_connections(frame, pixel_points, MESH, MESH_COLOR, 1)
        for name, (connections, color) in REGIONS.items():
            draw_connections(frame, pixel_points, connections, color, 2)
        draw_key_points(frame, pixel_points)

    cv2.imshow(WINDOW_NAME, frame)
    key = cv2.waitKey(1)
    if key == ord(QUIT_KEY):
        break
    elif key == ord(MESH_KEY):
        show_mesh = not show_mesh          # inverte: True vira False e vice-versa

camera.release()
cv2.destroyAllWindows()
landmarker.close()