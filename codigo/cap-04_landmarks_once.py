# cap-04_landmarks_once.py
# Missão: tirar UMA foto e ler os números de alguns dos 478 pontos do rosto.

import cv2               # OpenCV: câmera e imagem
import mediapipe as mp   # MediaPipe: pontos faciais

CAMERA_INDEX = 0
MODEL_FILE = "modelos/face_landmarker.task"

# Alguns pontos-chave e o número (índice) de cada um nos 478 pontos do MediaPipe
KEY_POINTS = {
    "testa (topo)": 10,
    "ponta do nariz": 4,
    "queixo": 152,
    "canto do olho 33": 33,
    "canto do olho 263": 263,
    "canto da boca 61": 61,
    "canto da boca 291": 291,
    "centro da íris 468": 468,
    "centro da íris 473": 473,
}

# Nomes curtos para as peças do MediaPipe que vamos usar
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_FILE),
    running_mode=RunningMode.IMAGE,   # IMAGE = uma foto por vez
    num_faces=1,                      # procurar no máximo 1 rosto
)
landmarker = FaceLandmarker.create_from_options(options)

camera = cv2.VideoCapture(CAMERA_INDEX)
ok, frame = camera.read()
camera.release()

if ok:
    height, width = frame.shape[:2]

    # O MediaPipe espera RGB; o OpenCV entrega BGR
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect(mp_image)
    print(f"Rostos encontrados: {len(result.face_landmarks)}")

    if len(result.face_landmarks) > 0:
        face = result.face_landmarks[0]          # o primeiro (e único) rosto
        print(f"Pontos neste rosto: {len(face)}")
        print()
        for name, index in KEY_POINTS.items():
            point = face[index]
            pixel_x = int(point.x * width)       # normalizado -> pixel
            pixel_y = int(point.y * height)
            print(f"{name:20} x={point.x:.3f}  y={point.y:.3f}  z={point.z:+.3f}  ->  pixel ({pixel_x}, {pixel_y})")
else:
    print("Não consegui ler a câmera.")

landmarker.close()