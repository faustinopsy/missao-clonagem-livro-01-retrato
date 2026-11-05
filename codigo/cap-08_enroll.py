# cap-08_enroll.py
# Missão: cadastrar um agente na GALERIA da porta, com várias poses.
#
# Uso:  python codigo\cap-08_enroll.py P01                      (câmera, cadastro guiado por poses)
#       python codigo\cap-08_enroll.py P02 pasta\com\fotos      (a partir de fotos já existentes)
#       python codigo\cap-08_enroll.py dados-pessoais\identidade  (pasta com UMA SUBPASTA POR PESSOA:
#                                                               cadastra todas de uma vez)
#
# Galeria: dados-pessoais/galeria/<código>/<pose>.npy  — só os descritores, nenhuma imagem.

import sys
from pathlib import Path

import cv2
import numpy as np

CAMERA_INDEX = 0
GALLERY_FOLDER = Path("dados-pessoais/galeria")
DETECTOR_FILE = "modelos/face_detection_yunet_2023mar.onnx"
RECOGNIZER_FILE = "modelos/face_recognition_sface_2021dec.onnx"
WINDOW_NAME = "Agente Iris - cadastro"
CAPTURE_KEY = " "      # ESPAÇO: captura a pose
SKIP_KEY = "p"         # pula a pose
QUIT_KEY = "q"

VIDEO_SCORE = 0.9      # detector no vídeo
PHOTO_SCORE = 0.7      # detector em fotos com o fundo escondido (Cap. 7)
NMS_THRESHOLD = 0.3
TOP_K = 5000
MIN_FACE_WIDTH = 100   # qualidade: o rosto precisa ter pelo menos 100 pixels de largura
CHECK_THRESHOLD = 0.5  # limiar do Cap. 7: abaixo dele, a pose parece de OUTRA pessoa
PHOTO_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# As poses do credenciamento: (nome do arquivo, instrução na tela)
POSES = [
    ("frente", "De frente, expressao neutra"),
    ("esquerda", "Vire um pouco o rosto para a SUA esquerda"),
    ("direita", "Vire um pouco o rosto para a SUA direita"),
    ("cima", "Olhe um pouco para cima"),
    ("baixo", "Olhe um pouco para baixo"),
    ("sorriso", "De frente, sorrindo"),
    ("perto", "Chegue mais perto da camera"),
    ("longe", "Afaste-se um pouco da camera"),
]

# Modo privacidade (Capítulo 3)
BLUR_SIZE = 61
BLUE_ALPHA = 0.4
MARGIN = 0.4
BLUE_BGR = (255, 0, 0)
GREEN = (0, 200, 0)
RED = (0, 0, 230)
WHITE = (255, 255, 255)


def valid_code(text):
    """Aceita P + números (P01, p2...). Devolve o código em maiúsculas ou None."""
    code = text.strip().upper()
    if code.startswith("P") and code[1:].isdigit():
        return code
    return None


def hide_background(image, faces):
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


def check_quality(faces):
    """Conferência de qualidade do cadastro. Devolve (ok, mensagem)."""
    if len(faces) == 0:
        return False, "nenhum rosto"
    if len(faces) > 1:
        return False, f"{len(faces)} rostos: so a pessoa cadastrada deve aparecer"
    width = int(faces[0][2])
    if width < MIN_FACE_WIDTH:
        return False, f"rosto pequeno ({width} px): chegue mais perto"
    return True, f"ok (rosto com {width} px)"


def detect(detector, image, score):
    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    detector.setScoreThreshold(score)
    _, faces = detector.detect(image)
    return [] if faces is None else faces


def save_pose(person_folder, pose, descriptor):
    person_folder.mkdir(parents=True, exist_ok=True)
    np.save(person_folder / f"{pose}.npy", descriptor)


def enroll_from_camera(detector, recognizer, person_folder):
    """Cadastro guiado: uma pose por vez, só aceita com exatamente um rosto grande o bastante."""
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 640, 360)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    camera = cv2.VideoCapture(CAMERA_INDEX)
    saved = {}
    feedback = ""
    pose_index = 0
    while pose_index < len(POSES):
        ok, frame = camera.read()
        if not ok:
            print("Não consegui ler a câmera.")
            break
        pose, instruction = POSES[pose_index]
        faces = detect(detector, frame, VIDEO_SCORE)
        quality_ok, quality_text = check_quality(faces)

        shown = hide_background(frame, faces)
        for face in faces:
            x, y, w, h = face[0:4].astype(int)
            cv2.rectangle(shown, (x, y), (x + w, y + h), GREEN if quality_ok else RED, 3)
        cv2.rectangle(shown, (0, 0), (shown.shape[1], 120), (40, 40, 40), -1)
        cv2.putText(shown, f"Pose {pose_index + 1}/{len(POSES)}: {instruction}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, WHITE, 2)
        cv2.putText(shown, f"ESPACO = capturar | p = pular | q = sair | {quality_text}", (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN if quality_ok else RED, 2)
        if feedback:
            cv2.putText(shown, feedback, (20, shown.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
        cv2.imshow(WINDOW_NAME, shown)

        key = cv2.waitKey(1)
        if key == ord(QUIT_KEY):
            break
        if key == ord(SKIP_KEY):
            print(f"  - {pose}: pulada")
            pose_index += 1
        if key == ord(CAPTURE_KEY):
            if not quality_ok:
                feedback = f"Pose '{pose}' recusada: {quality_text}"
                print(f"  ✘ {pose}: {quality_text}")
                continue
            descriptor = recognizer.feature(recognizer.alignCrop(frame, faces[0]))
            save_pose(person_folder, pose, descriptor)
            saved[pose] = descriptor
            feedback = f"Pose '{pose}' cadastrada!"
            print(f"  ✔ {pose}: {quality_text}")
            pose_index += 1
    camera.release()
    cv2.destroyAllWindows()
    return saved


def enroll_from_photos(detector, recognizer, person_folder, photos_folder):
    """Cadastro a partir de fotos: cada foto com um rosto bom vira uma "pose" (o nome do arquivo)."""
    saved = {}
    for path in sorted(Path(photos_folder).iterdir()):
        if path.suffix.lower() not in PHOTO_EXTENSIONS:
            continue
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print(f"  ✘ {path.name}: não abriu")
            continue
        faces = detect(detector, image, PHOTO_SCORE)
        quality_ok, quality_text = check_quality(faces)
        if not quality_ok:
            print(f"  ✘ {path.name}: {quality_text}")
            continue
        descriptor = recognizer.feature(recognizer.alignCrop(image, faces[0]))
        save_pose(person_folder, path.stem, descriptor)
        saved[path.stem] = descriptor
        print(f"  ✔ {path.name}: {quality_text}")
    return saved


def prepare_folder(code):
    """Se o agente já existe, pergunta se refaz. Devolve a pasta dele ou None (não refazer)."""
    person_folder = GALLERY_FOLDER / code
    if person_folder.exists():
        answer = input(f"{code} já está cadastrado. Refazer o cadastro? (s/n): ")
        if answer.strip().lower() != "s":
            return None
        for old_file in person_folder.glob("*.npy"):  # descadastrar = apagar os descritores antigos
            old_file.unlink()
    return person_folder


def show_summary(recognizer, code, saved):
    """Quantas poses e o quanto cada uma se parece com a primeira."""
    print(f"{code}: {len(saved)} poses na galeria.")
    if len(saved) < 2:
        return
    names = list(saved)
    first = names[0]
    print(f"Similaridade de cada pose com '{first}':")
    for name in names[1:]:
        value = recognizer.match(saved[first], saved[name], cv2.FaceRecognizerSF_FR_COSINE)
        warning = "   <- abaixo de 0.5: confira (outra pessoa? foto ruim?)" if value < CHECK_THRESHOLD else ""
        print(f"  {name:12} {value:.3f}{warning}")


detector = cv2.FaceDetectorYN.create(DETECTOR_FILE, "", (320, 320), VIDEO_SCORE, NMS_THRESHOLD, TOP_K)
recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_FILE, "")
first_argument = sys.argv[1] if len(sys.argv) >= 2 else input("Código do agente (P01, P02…) ou pasta: ")

if Path(first_argument).is_dir():
    # MODO PASTA: uma subpasta por pessoa (P01, P02…). Cadastra todas de uma vez.
    print(f"Lendo {first_argument}: uma subpasta por pessoa.")
    for subfolder in sorted(Path(first_argument).iterdir()):
        code = valid_code(subfolder.name) if subfolder.is_dir() else None
        if code is None:
            continue                                    # arquivos e pastas sem código são ignorados
        person_folder = prepare_folder(code)
        if person_folder is None:
            print(f"{code}: mantido como estava.\n")
            continue
        print(f"Cadastrando {code} a partir de {subfolder}")
        saved = enroll_from_photos(detector, recognizer, person_folder, subfolder)
        show_summary(recognizer, code, saved)
        print()
else:
    # MODO UMA PESSOA: pela câmera (só o código) ou por fotos (código + pasta)
    code = valid_code(first_argument)
    if code is None:
        print("Use um código como P01, P02… (nunca o nome), ou uma pasta com uma subpasta por pessoa.")
        raise SystemExit
    person_folder = prepare_folder(code)
    if person_folder is None:
        raise SystemExit
    print(f"Cadastrando {code} em {person_folder}")
    if len(sys.argv) >= 3:
        saved = enroll_from_photos(detector, recognizer, person_folder, sys.argv[2])
    else:
        saved = enroll_from_camera(detector, recognizer, person_folder)
    print()
    show_summary(recognizer, code, saved)
