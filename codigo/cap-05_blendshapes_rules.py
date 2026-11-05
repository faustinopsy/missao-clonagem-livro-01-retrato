# cap-05_blendshapes_rules.py
# Missão: ver os 52 movimentos do rosto ao vivo e reconhecer expressões com REGRAS.

import time               # relógio: carimbo de tempo do modo vídeo

import cv2                # OpenCV: câmera, imagem e desenho
import mediapipe as mp    # MediaPipe: pontos faciais e blendshapes
import numpy as np        # NumPy: máscara e lista de pontos

CAMERA_INDEX = 0
MODEL_FILE = "modelos/face_landmarker.task"
WINDOW_NAME = "Agente Iris - expressoes por regras"
QUIT_KEY = "q"
TOP_N = 5                 # quantos movimentos mais fortes mostrar

# Modo privacidade: esconde tudo fora do contorno do rosto
HIDE_BACKGROUND = True
BLUR_SIZE = 61
BLUE_ALPHA = 0.4
BLUE_BGR = (255, 0, 0)

# Limiares das regras (ajuste com os números da sua tela)
SMILE_THRESHOLD = 0.5
SURPRISE_BROW_THRESHOLD = 0.5
SURPRISE_JAW_THRESHOLD = 0.3
FROWN_THRESHOLD = 0.4

TEXT_COLOR = (255, 255, 255)   # branco
BAR_COLOR = (0, 255, 255)      # amarelo
RESULT_COLOR = (0, 255, 0)     # verde

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


def to_dictionary(blendshapes):
    """Transforma a lista de blendshapes do MediaPipe em um dicionário nome -> escore."""
    scores = {}
    for category in blendshapes:
        scores[category.category_name] = category.score
    return scores


def by_score(item):
    """Usada pelo sorted(): devolve o escore de um par (nome, escore)."""
    return item[1]


def recognize_by_rules(scores):
    """Decide a expressão com regras escritas por nós."""
    smile = (scores["mouthSmileLeft"] + scores["mouthSmileRight"]) / 2
    frown = (scores["browDownLeft"] + scores["browDownRight"]) / 2

    if smile > SMILE_THRESHOLD:
        return "sorriso"
    elif scores["browInnerUp"] > SURPRISE_BROW_THRESHOLD and scores["jawOpen"] > SURPRISE_JAW_THRESHOLD:
        return "surpresa"
    elif frown > FROWN_THRESHOLD:
        return "testa franzida"
    else:
        return "neutro"


def draw_top_scores(image, scores):
    """Desenha os TOP_N movimentos mais fortes como barras."""
    ranking = sorted(scores.items(), key=by_score, reverse=True)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for row, (name, score) in enumerate(ranking[:TOP_N]):
        y = 90 + row * 35
        cv2.rectangle(image, (20, y - 20), (20 + int(score * 200), y), BAR_COLOR, -1)
        cv2.putText(image, f"{name} {score:.2f}", (230, y - 2), font, 0.7, TEXT_COLOR, 2)


options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_FILE),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=True,     # liga os 52 blendshapes
)
landmarker = FaceLandmarker.create_from_options(options)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 640, 360)
cv2.moveWindow(WINDOW_NAME, 0, 0)

camera = cv2.VideoCapture(CAMERA_INDEX)
print(f"Faça expressões. Aperte '{QUIT_KEY}' na janela para sair.")

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

    font = cv2.FONT_HERSHEY_SIMPLEX
    if len(result.face_blendshapes) > 0:
        scores = to_dictionary(result.face_blendshapes[0])
        expression = recognize_by_rules(scores)
        cv2.putText(frame, f"regra: {expression}", (20, 45), font, 1.1, RESULT_COLOR, 3)
        draw_top_scores(frame, scores)
    else:
        cv2.putText(frame, "sem rosto", (20, 45), font, 1.1, TEXT_COLOR, 2)

    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) == ord(QUIT_KEY):
        break

camera.release()
cv2.destroyAllWindows()
landmarker.close()
