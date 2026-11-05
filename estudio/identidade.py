# identidade.py — a classe FaceIdentity: YuNet (detecta e dá os 5 pontos) + SFace (descritor).
# Reúne o que os Capítulos 7 e 8 fizeram nos scripts do terminal: descrever, comparar, medir o
# limiar, montar a referência de uma pessoa e, no Cap. 8, a GALERIA e a identificação 1:N.

from itertools import combinations   # todos os pares de uma lista
from pathlib import Path

import cv2
import numpy as np

ESTUDIO_FOLDER = Path(__file__).resolve().parent
DETECTOR_FILE = ESTUDIO_FOLDER / "modelos" / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_FILE = ESTUDIO_FOLDER / "modelos" / "face_recognition_sface_2021dec.onnx"

VIDEO_SCORE = 0.9          # confiança mínima do detector no vídeo
PHOTO_SCORE = 0.7          # nas fotos com o fundo escondido, a confiança cai (Passo 3)
NMS_THRESHOLD = 0.3
TOP_K = 5000
DOC_THRESHOLD = 0.363      # limiar da documentação do OpenCV para o SFace
THRESHOLDS = [0.2, 0.3, 0.363, 0.4, 0.5, 0.6, 0.7]
PHOTO_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
MIN_FACE_WIDTH = 100       # cadastro: o rosto precisa ter pelo menos 100 pixels de largura
AMBIGUITY_MARGIN = 0.05    # porta: dois agentes acima do limiar e quase empatados = "tente de novo"
UNKNOWN = "desconhecido"

# Modo privacidade do Capítulo 3: fundo borrado e azulado, retângulo do rosto nítido
BLUR_SIZE = 61
BLUE_ALPHA = 0.4
BLUE_BGR = (255, 0, 0)
MARGIN = 0.4


def read_photo(path):
    """Lê uma foto (funciona com acentos no caminho). Devolve None se não der."""
    try:
        return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except OSError:
        return None


class FaceIdentity:
    """Cria o YuNet e o SFace UMA vez e responde: "estes rostos são da mesma pessoa?"."""

    def __init__(self):
        self.detector = cv2.FaceDetectorYN.create(str(DETECTOR_FILE), "", (320, 320),
                                                  VIDEO_SCORE, NMS_THRESHOLD, TOP_K)
        self.recognizer = cv2.FaceRecognizerSF.create(str(RECOGNIZER_FILE), "")

    # ---------- um rosto ----------
    def detect(self, image, score):
        """Devolve a lista de rostos (cada um com caixa, 5 pontos e confiança)."""
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        self.detector.setScoreThreshold(score)
        _, faces = self.detector.detect(image)
        return [] if faces is None else faces

    def describe(self, image, face):
        """Rosto -> (recorte alinhado 112x112, descritor de 128 números)."""
        aligned = self.recognizer.alignCrop(image, face)
        return aligned, self.recognizer.feature(aligned)

    def describe_photo(self, image):
        """Foto inteira -> (recorte, descritor) do rosto mais confiável, ou (None, None)."""
        faces = self.detect(image, PHOTO_SCORE)
        if len(faces) == 0:
            return None, None
        return self.describe(image, faces[0])

    def similarity(self, descriptor_a, descriptor_b):
        """Similaridade de cosseno: quanto maior, mais parecidos (não é porcentagem)."""
        return self.recognizer.match(descriptor_a, descriptor_b, cv2.FaceRecognizerSF_FR_COSINE)

    def distance(self, descriptor_a, descriptor_b):
        """Distância L2: quanto menor, mais parecidos."""
        return self.recognizer.match(descriptor_a, descriptor_b, cv2.FaceRecognizerSF_FR_NORM_L2)

    # ---------- uma pasta com uma subpasta por pessoa ----------
    def load_folder(self, folder):
        """Devolve (amostras, problemas). Amostra = (pessoa, nome da foto, descritor)."""
        samples = []
        problems = []
        for person_folder in sorted(Path(folder).iterdir()):
            if not person_folder.is_dir() or person_folder.name.startswith("_"):
                continue
            for path in sorted(person_folder.iterdir()):
                if path.suffix.lower() not in PHOTO_EXTENSIONS:
                    continue
                image = read_photo(path)
                if image is None:
                    problems.append(f"não abriu: {person_folder.name}/{path.name}")
                    continue
                _, descriptor = self.describe_photo(image)
                if descriptor is None:
                    problems.append(f"sem rosto: {person_folder.name}/{path.name}")
                    continue
                samples.append((person_folder.name, path.name, descriptor))
        return samples, problems

    def evaluate(self, samples):
        """Compara todos os pares e conta os erros de cada limiar (o Passo 4 inteiro)."""
        genuine, impostor = [], []
        genuine_pairs, impostor_pairs = [], []
        for (person_a, name_a, desc_a), (person_b, name_b, desc_b) in combinations(samples, 2):
            value = self.similarity(desc_a, desc_b)
            pair = f"{person_a}/{name_a} x {person_b}/{name_b}"
            if person_a == person_b:
                genuine.append(value)
                genuine_pairs.append((value, pair))
            else:
                impostor.append(value)
                impostor_pairs.append((value, pair))
        genuine = np.array(genuine)
        impostor = np.array(impostor)

        table = []
        for threshold in THRESHOLDS:
            table.append({
                "threshold": threshold,
                "false_accepts": int(np.sum(impostor >= threshold)),
                "false_rejects": int(np.sum(genuine < threshold)),
            })
        gap = None                                    # a "folga" entre os dois grupos
        if len(genuine) > 0 and len(impostor) > 0 and impostor.max() < genuine.min():
            gap = (float(impostor.max()), float(genuine.min()))
        return {
            "genuine": genuine,
            "impostor": impostor,
            "table": table,
            "gap": gap,
            "closest_impostors": sorted(impostor_pairs, reverse=True)[:3],
            "farthest_genuine": sorted(genuine_pairs)[:3],
        }

    def suggest_threshold(self, result):
        """O meio da folga; sem folga, o limiar da tabela com menos erros no total."""
        if result["gap"] is not None:
            low, high = result["gap"]
            return round((low + high) / 2, 3)
        best = min(result["table"], key=lambda row: row["false_accepts"] + row["false_rejects"])
        return best["threshold"]

    # ---------- a referência de uma pessoa ----------
    def build_reference(self, person_folder):
        """Média dos descritores normalizados das fotos da pessoa. Devolve (referência, nº de fotos)."""
        descriptors = []
        for path in sorted(Path(person_folder).iterdir()):
            if path.suffix.lower() not in PHOTO_EXTENSIONS:
                continue
            image = read_photo(path)
            if image is None:
                continue
            _, descriptor = self.describe_photo(image)
            if descriptor is not None:
                descriptors.append(descriptor / np.linalg.norm(descriptor))
        if len(descriptors) == 0:
            return None, 0
        reference = np.mean(np.vstack(descriptors), axis=0, keepdims=True)
        return reference.astype(np.float32), len(descriptors)

    # ---------- Capítulo 8: galeria e identificação 1:N ----------
    def check_quality(self, faces, scale=1.0):
        """Conferência do cadastro. 'scale' converte a largura da imagem pequena para a original."""
        if len(faces) == 0:
            return False, "nenhum rosto"
        if len(faces) > 1:
            return False, f"{len(faces)} rostos: só a pessoa cadastrada deve aparecer"
        width = int(faces[0][2] * scale)
        if width < MIN_FACE_WIDTH:
            return False, f"rosto pequeno ({width} px): chegue mais perto"
        return True, f"ok (rosto com {width} px)"

    def identify(self, query, gallery):
        """1:N: para cada agente, a MAIOR similaridade entre as poses dele. Ranking do maior ao menor."""
        ranking = []
        for code, poses in gallery.items():
            best = -1.0
            for pose in poses.values():
                best = max(best, self.similarity(query, pose))
            ranking.append((best, code))
        return sorted(ranking, reverse=True)

    # ---------- vídeo ----------
    def hide_background(self, image, faces):
        """Modo privacidade do Capítulo 3 (retângulo em volta de cada rosto)."""
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


def load_gallery(folder):
    """{código: {pose: descritor}} a partir de <galeria>/<código>/<pose>.npy (não precisa dos modelos)."""
    gallery = {}
    folder = Path(folder)
    if not folder.is_dir():
        return gallery
    for person_folder in sorted(folder.iterdir()):
        code = person_folder.name
        if person_folder.is_dir() and code.startswith("P") and code[1:].isdigit():
            poses = {}
            for path in sorted(person_folder.glob("*.npy")):
                poses[path.stem] = np.load(path)
            if len(poses) > 0:
                gallery[code] = poses
    return gallery


def decide(ranking, threshold):
    """Primeiro o mais parecido; depois, o limiar. Devolve (resposta, código)."""
    best_value, best_code = ranking[0]
    if best_value < threshold:
        return "PORTA FECHADA", UNKNOWN
    if len(ranking) > 1:
        second_value, second_code = ranking[1]
        if second_value >= threshold and best_value - second_value < AMBIGUITY_MARGIN:
            return "TENTE DE NOVO", f"{best_code}?{second_code}?"
    return "PORTA ABERTA", best_code
