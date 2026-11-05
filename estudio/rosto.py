# rosto.py — a classe FaceAnalyzer: MediaPipe (pontos e blendshapes) e modo privacidade.

import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ESTUDIO_FOLDER = Path(__file__).resolve().parent      # a pasta estudio
LANDMARKER_FILE = ESTUDIO_FOLDER / "modelos" / "face_landmarker.task"

BLUR_SIZE = 61
BLUE_ALPHA = 0.4
BLUE_BGR = (255, 0, 0)


class FaceResult:
    """O que o analisador descobriu num quadro."""

    def __init__(self, pixel_points, blendshapes):
        self.pixel_points = pixel_points      # lista de (x, y) — vazia se não há rosto
        self.blendshapes = blendshapes        # lista de categorias do MediaPipe — vazia se não há rosto

    def has_face(self):
        return len(self.pixel_points) > 0


class FaceAnalyzer:
    """Cria o Face Landmarker uma vez e analisa quadros de vídeo ou fotos."""

    def __init__(self):
        vision = mp.tasks.vision
        base = mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER_FILE))
        # Dois marcadores: um para vídeo (com rastreamento) e outro para fotos soltas
        self.video_landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(base_options=base, running_mode=vision.RunningMode.VIDEO,
                                         num_faces=1, output_face_blendshapes=True))
        self.image_landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(base_options=base, running_mode=vision.RunningMode.IMAGE,
                                         num_faces=1, output_face_blendshapes=True))
        self.last_timestamp = 0

    def analyze_video_frame(self, frame):
        """Analisa um quadro da câmera (modo vídeo)."""
        timestamp = max(int(time.perf_counter() * 1000), self.last_timestamp + 1)  # sempre crescente
        self.last_timestamp = timestamp
        result = self.video_landmarker.detect_for_video(self.to_mp_image(frame), timestamp)
        return self.to_face_result(result, frame)

    def analyze_photo(self, image):
        """Analisa uma foto solta (modo imagem)."""
        result = self.image_landmarker.detect(self.to_mp_image(image))
        return self.to_face_result(result, image)

    def to_mp_image(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)      # o MediaPipe espera RGB
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    def to_face_result(self, result, frame):
        if len(result.face_landmarks) == 0:
            return FaceResult([], [])
        height, width = frame.shape[:2]
        pixel_points = []
        for point in result.face_landmarks[0]:
            pixel_points.append((int(point.x * width), int(point.y * height)))
        return FaceResult(pixel_points, result.face_blendshapes[0])

    def hide_background(self, image, face_result):
        """Fundo borrado e azulado; só o contorno do rosto fica nítido."""
        background = cv2.GaussianBlur(image, (BLUR_SIZE, BLUR_SIZE), 0)
        tint = background.copy()
        tint[:] = BLUE_BGR
        background = cv2.addWeighted(background, 1 - BLUE_ALPHA, tint, BLUE_ALPHA, 0)
        if not face_result.has_face():
            return background
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        hull = cv2.convexHull(np.array(face_result.pixel_points, dtype=np.int32))
        cv2.fillConvexPoly(mask, hull, 255)
        return cv2.copyTo(image, mask, background)

    def close(self):
        self.video_landmarker.close()
        self.image_landmarker.close()
