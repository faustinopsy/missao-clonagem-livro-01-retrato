# cap-07_compare.py
# Missão: comparar DUAS fotos e responder "é a mesma pessoa?", com a similaridade de cosseno.
#
# Uso:  python codigo\cap-07_compare.py                      (escolhe as 2 fotos numa janela)
#       python codigo\cap-07_compare.py foto1.jpg foto2.jpg  (passa as 2 fotos no comando)

import sys                                 # lê os nomes das fotos digitados no comando
from pathlib import Path
from tkinter import Tk, filedialog         # a janela do Windows para escolher arquivos (Cap. 6)

import cv2
import numpy as np

DETECTOR_FILE = "modelos/face_detection_yunet_2023mar.onnx"
RECOGNIZER_FILE = "modelos/face_recognition_sface_2021dec.onnx"
WINDOW_NAME = "Agente Iris - comparacao"

SCORE_THRESHOLD = 0.7        # mais baixo que no vídeo (0.9): fotos com o fundo escondido
                             # perdem a testa e o queixo, e a confiança do YuNet cai para 0.76-0.83
NMS_THRESHOLD = 0.3
TOP_K = 5000
MAX_WIDTH = 1280             # fotos maiores são reduzidas antes de detectar
COSINE_THRESHOLD = 0.363     # limiar da documentação do OpenCV para o SFace (mesma pessoa se >=)
L2_THRESHOLD = 1.128         # o mesmo limiar, na régua da distância L2 (mesma pessoa se <=)
CROP_ZOOM = 2                # os recortes de 112x112 aparecem 2 vezes maiores

SAME_COLOR = (0, 200, 0)     # verde
DIFFERENT_COLOR = (0, 0, 230)  # vermelho
PHOTO_TYPES = [("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Todos os arquivos", "*.*")]


def choose_photos():
    """Abre a janela do Windows duas vezes e devolve os dois caminhos (ou None)."""
    root = Tk()
    root.withdraw()                        # esconde a janela principal do Tkinter
    first = filedialog.askopenfilename(title="Escolha a 1ª foto", filetypes=PHOTO_TYPES)
    second = filedialog.askopenfilename(title="Escolha a 2ª foto", filetypes=PHOTO_TYPES)
    root.destroy()
    if not first or not second:
        return None
    return first, second


def read_photo(path):
    """Lê a foto (funciona com acentos no caminho) e reduz se for muito grande."""
    try:
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except OSError:                        # arquivo não existe ou não deu para ler
        return None
    if image is None:
        return None
    height, width = image.shape[:2]
    if width > MAX_WIDTH:
        scale = MAX_WIDTH / width
        image = cv2.resize(image, (MAX_WIDTH, int(height * scale)))
    return image


def describe(detector, recognizer, image):
    """Foto -> (rosto alinhado, descritor). Devolve (None, None) se não achar rosto."""
    height, width = image.shape[:2]
    detector.setInputSize((width, height))    # cada foto pode ter um tamanho diferente
    _, faces = detector.detect(image)
    if faces is None:
        return None, None
    aligned = recognizer.alignCrop(image, faces[0])
    return aligned, recognizer.feature(aligned)


def cosine_by_hand(a, b):
    """A similaridade de cosseno feita "na mão", com NumPy, para conferir o OpenCV."""
    a = a.flatten()
    b = b.flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# 1. Quais fotos? Do comando ou da janela de escolher arquivos
if len(sys.argv) == 3:
    paths = sys.argv[1], sys.argv[2]
else:
    paths = choose_photos()
if paths is None:
    print("Nenhuma foto escolhida.")
    sys.exit()

detector = cv2.FaceDetectorYN.create(DETECTOR_FILE, "", (320, 320), SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K)
recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_FILE, "")

# 2. Um descritor para cada foto
crops = []
descriptors = []
for path in paths:
    image = read_photo(path)
    if image is None:
        print(f"Não consegui abrir: {Path(path).name}")
        sys.exit()
    aligned, descriptor = describe(detector, recognizer, image)
    if descriptor is None:
        print(f"Nenhum rosto encontrado em: {Path(path).name}")
        sys.exit()
    crops.append(aligned)
    descriptors.append(descriptor)

# 3. Comparar: similaridade de cosseno (régua principal) e distância L2
similarity = recognizer.match(descriptors[0], descriptors[1], cv2.FaceRecognizerSF_FR_COSINE)
distance = recognizer.match(descriptors[0], descriptors[1], cv2.FaceRecognizerSF_FR_NORM_L2)
same_person = similarity >= COSINE_THRESHOLD

print(f"Foto 1: {Path(paths[0]).name}")
print(f"Foto 2: {Path(paths[1]).name}")
print(f"Similaridade de cosseno: {similarity:.3f}   (na mão, com NumPy: {cosine_by_hand(*descriptors):.3f})")
print(f"Distância L2:            {distance:.3f}")
verdict = "MESMA PESSOA" if same_person else "PESSOA DIFERENTE"
print(f"Resposta com o limiar {COSINE_THRESHOLD}: {verdict}")

# 4. Mostra os dois recortes lado a lado, com a resposta embaixo
side_by_side = np.hstack(crops)                         # junta as duas imagens na horizontal
height, width = side_by_side.shape[:2]
side_by_side = cv2.resize(side_by_side, (width * CROP_ZOOM, height * CROP_ZOOM))
bar = np.zeros((60, side_by_side.shape[1], 3), dtype=np.uint8)
bar[:] = SAME_COLOR if same_person else DIFFERENT_COLOR
cv2.putText(bar, f"similaridade {similarity:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
result = np.vstack([side_by_side, bar])                 # junta na vertical: recortes + barra

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
cv2.moveWindow(WINDOW_NAME, 0, 0)
cv2.imshow(WINDOW_NAME, result)
print("Aperte qualquer tecla na janela para sair.")
cv2.waitKey(0)
cv2.destroyAllWindows()
