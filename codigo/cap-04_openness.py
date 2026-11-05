# cap-04_openness.py
# Missão: medir a abertura dos olhos e da boca e contar piscadas.

import math               # matemática: distância entre dois pontos
import time               # relógio: carimbo de tempo do modo vídeo

import cv2                # OpenCV: câmera, imagem e desenho
import mediapipe as mp    # MediaPipe: pontos faciais
import numpy as np        # NumPy: máscara e lista de pontos

CAMERA_INDEX = 0
MODEL_FILE = "modelos/face_landmarker.task"
WINDOW_NAME = "Agente Iris - abertura"
QUIT_KEY = "q"

# Modo privacidade: esconde tudo fora do contorno do rosto
HIDE_BACKGROUND = True
BLUR_SIZE = 61
BLUE_ALPHA = 0.4
BLUE_BGR = (255, 0, 0)

# Pontos de cada medida: (cima, baixo, canto, canto) — lados DA PESSOA
RIGHT_EYE = (159, 145, 33, 133)
LEFT_EYE = (386, 374, 263, 362)
MOUTH = (13, 14, 61, 291)

# Limiares (ajuste com os números que aparecerem na sua tela)
EYE_CLOSED_THRESHOLD = 0.15    # abaixo disto, o olho está fechado
MOUTH_OPEN_THRESHOLD = 0.30    # acima disto, a boca está aberta

LINE_COLOR = (0, 255, 255)     # amarelo
TEXT_COLOR = (255, 255, 255)   # branco
ALERT_COLOR = (0, 0, 255)      # vermelho

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


def openness(pixel_points, indexes):
    """Abertura = altura / largura. Dividir pela largura faz a medida
    não depender da distância até a câmera."""
    top, bottom, corner_a, corner_b = indexes
    height = math.dist(pixel_points[top], pixel_points[bottom])
    width = math.dist(pixel_points[corner_a], pixel_points[corner_b])
    return height / width


def draw_measure(image, pixel_points, indexes):
    """Desenha as duas linhas da medida: altura e largura."""
    top, bottom, corner_a, corner_b = indexes
    cv2.line(image, pixel_points[top], pixel_points[bottom], LINE_COLOR, 2)
    cv2.line(image, pixel_points[corner_a], pixel_points[corner_b], LINE_COLOR, 1)


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
blink_count = 0          # quantas piscadas já contamos
eyes_were_closed = False # bandeira: no quadro anterior, os olhos estavam fechados?
print(f"Pisque, abra a boca. Aperte '{QUIT_KEY}' na janela para sair.")

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
        right_eye = openness(pixel_points, RIGHT_EYE)
        left_eye = openness(pixel_points, LEFT_EYE)
        eyes = (right_eye + left_eye) / 2          # média dos dois olhos
        mouth = openness(pixel_points, MOUTH)

        # Piscada: os olhos FECHAM e depois ABREM de novo
        eyes_closed = eyes < EYE_CLOSED_THRESHOLD
        if eyes_were_closed and not eyes_closed:
            blink_count += 1                       # soma 1 ao contador
        eyes_were_closed = eyes_closed

        for indexes in (RIGHT_EYE, LEFT_EYE, MOUTH):
            draw_measure(frame, pixel_points, indexes)

        # "valor_se_sim if condição else valor_se_não"
        mouth_open = mouth > MOUTH_OPEN_THRESHOLD
        eye_text = "FECHADOS" if eyes_closed else "abertos"
        eye_color = ALERT_COLOR if eyes_closed else TEXT_COLOR
        mouth_text = "ABERTA" if mouth_open else "fechada"
        mouth_color = ALERT_COLOR if mouth_open else TEXT_COLOR

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"olhos: {eyes:.2f} ({eye_text})", (20, 40), font, 0.9, eye_color, 2)
        cv2.putText(frame, f"boca:  {mouth:.2f} ({mouth_text})", (20, 75), font, 0.9, mouth_color, 2)
        cv2.putText(frame, f"piscadas: {blink_count}", (20, 110), font, 0.9, TEXT_COLOR, 2)
    else:
        cv2.putText(frame, "sem rosto", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_COLOR, 2)

    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) == ord(QUIT_KEY):
        break

camera.release()
cv2.destroyAllWindows()
landmarker.close()
