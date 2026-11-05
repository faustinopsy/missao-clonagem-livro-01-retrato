# cap-05_live_model.py
# Missão: usar o SEU modelo treinado para reconhecer expressões ao vivo.

import time               # relógio: carimbo de tempo do modo vídeo

import cv2                # OpenCV: câmera, imagem e desenho
import joblib             # carregar o modelo salvo no Passo 4
import mediapipe as mp    # MediaPipe: pontos faciais e blendshapes
import numpy as np        # NumPy: máscara e lista de pontos

CAMERA_INDEX = 0
LANDMARKER_FILE = "modelos/face_landmarker.task"
MODEL_FILE = "dados-pessoais/expressoes/modelo_knn.joblib"
WINDOW_NAME = "Agente Iris - meu modelo ao vivo"
QUIT_KEY = "q"
MIN_CONFIDENCE = 0.6      # abaixo disto, o modelo está em dúvida

# Modo privacidade: esconde tudo fora do contorno do rosto
HIDE_BACKGROUND = True
BLUR_SIZE = 61
BLUE_ALPHA = 0.4
BLUE_BGR = (255, 0, 0)

TEXT_COLOR = (255, 255, 255)   # branco
BAR_COLOR = (0, 255, 255)      # amarelo
SURE_COLOR = (0, 255, 0)       # verde
UNSURE_COLOR = (0, 165, 255)   # laranja

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


def to_feature_vector(blendshapes, feature_names):
    """Monta a lista de 52 números NA MESMA ORDEM usada no treino."""
    scores = {}
    for category in blendshapes:
        scores[category.category_name] = category.score
    vector = []
    for name in feature_names:
        vector.append(scores[name])
    return vector


# Carrega o modelo e a ordem das características salvos no Passo 4
saved = joblib.load(MODEL_FILE)
model = saved["model"]
feature_names = saved["features"]
print(f"Modelo carregado. Classes: {', '.join(model.classes_)}")

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=LANDMARKER_FILE),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=True,
)
landmarker = FaceLandmarker.create_from_options(options)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 640, 360)
cv2.moveWindow(WINDOW_NAME, 0, 0)

camera = cv2.VideoCapture(CAMERA_INDEX)
print(f"Faça as suas expressões. Aperte '{QUIT_KEY}' na janela para sair.")

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
        vector = to_feature_vector(result.face_blendshapes[0], feature_names)

        # predict_proba devolve a "confiança" de cada classe (proporção de votos)
        probabilities = model.predict_proba([vector])[0]
        best = int(np.argmax(probabilities))           # posição da maior confiança
        best_class = model.classes_[best]
        confidence = probabilities[best]

        if confidence >= MIN_CONFIDENCE:
            cv2.putText(frame, f"{best_class} ({confidence:.2f})", (20, 45), font, 1.1, SURE_COLOR, 3)
        else:
            cv2.putText(frame, f"em duvida ({confidence:.2f})", (20, 45), font, 1.1, UNSURE_COLOR, 3)

        # Uma barra por classe
        y = 90
        for class_name, probability in zip(model.classes_, probabilities):
            cv2.rectangle(frame, (20, y - 20), (20 + int(probability * 200), y), BAR_COLOR, -1)
            cv2.putText(frame, f"{class_name} {probability:.2f}", (230, y - 2), font, 0.7, TEXT_COLOR, 2)
            y += 35
    else:
        cv2.putText(frame, "sem rosto", (20, 45), font, 1.1, TEXT_COLOR, 2)

    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) == ord(QUIT_KEY):
        break

camera.release()
cv2.destroyAllWindows()
landmarker.close()
