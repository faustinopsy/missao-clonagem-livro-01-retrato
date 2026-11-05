# cap-03_draw_detection.py
# Missão: desenhar a caixa, os 5 pontos e a confiança de cada rosto detectado,
# escondendo o fundo (borrado + azul) para proteger a privacidade.

import cv2  # OpenCV: biblioteca de visão computacional

CAMERA_INDEX = 0
MODEL_FILE = "modelos/face_detection_yunet_2023mar.onnx"
OUTPUT_FILE = "dados-pessoais/cap-03_deteccao.png"   # pasta que não vai para o GitHub
WINDOW_NAME = "Agente Iris - deteccao"

SCORE_THRESHOLD = 0.9
NMS_THRESHOLD = 0.3
TOP_K = 5000

# Modo privacidade: esconde tudo, menos a região do rosto
HIDE_BACKGROUND = True   # mude para False para ver a imagem original
BLUR_SIZE = 61           # tamanho do borrão (precisa ser ímpar; maior = mais borrado)
BLUE_ALPHA = 0.4         # quanto de azul misturar (0 = nada, 1 = azul total)
MARGIN = 0.4             # margem em volta do rosto (0.4 = 40% do tamanho da caixa)

# Cores na ordem do OpenCV: (Azul, Verde, Vermelho)
BOX_COLOR = (0, 255, 0)      # verde
POINT_COLOR = (0, 0, 255)    # vermelho
BLUE_BGR = (255, 0, 0)       # azul do fundo


def hide_background(image, faces):
    """Devolve uma cópia da imagem com o fundo borrado e azulado,
    mantendo nítida só a região em volta de cada rosto."""
    # 1. Borra a imagem inteira
    background = cv2.GaussianBlur(image, (BLUR_SIZE, BLUR_SIZE), 0)

    # 2. Mistura um azul semitransparente: (1 - BLUE_ALPHA) de imagem + BLUE_ALPHA de azul
    tint = background.copy()
    tint[:] = BLUE_BGR
    background = cv2.addWeighted(background, 1 - BLUE_ALPHA, tint, BLUE_ALPHA, 0)

    # 3. Devolve a imagem original na região de cada rosto (com margem)
    height, width = image.shape[:2]
    for face in faces:
        x, y, w, h = face[0:4].astype(int)   # converte para inteiro (dtype int)
        margin_x = int(w * MARGIN)
        margin_y = int(h * MARGIN)
        left = max(x - margin_x, 0)            # não deixa sair pela esquerda
        top = max(y - margin_y, 0)             # nem por cima
        right = min(x + w + margin_x, width)   # nem pela direita
        bottom = min(y + h + margin_y, height) # nem por baixo
        background[top:bottom, left:right] = image[top:bottom, left:right]

    return background


def draw_face(image, face):
    """Desenha a caixa, os 5 pontos e a confiança de UM rosto."""
    x, y, w, h = face[0:4].astype(int)   # converte para inteiro (dtype int)
    score = face[14]

    # Caixa: do canto superior esquerdo (x, y) ao inferior direito (x + w, y + h)
    cv2.rectangle(image, (x, y), (x + w, y + h), BOX_COLOR, 2)

    # 5 pontos: olho direito, olho esquerdo, nariz, canto direito e canto esquerdo da boca
    points = face[4:14].reshape(5, 2)
    for px, py in points:
        cv2.circle(image, (int(px), int(py)), 4, POINT_COLOR, -1)

    # Confiança, escrita logo acima da caixa
    label = f"rosto {score:.2f}"
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BOX_COLOR, 2)


camera = cv2.VideoCapture(CAMERA_INDEX)
ok, frame = camera.read()
camera.release()

if ok:
    height, width = frame.shape[:2]
    detector = cv2.FaceDetectorYN.create(
        MODEL_FILE, "", (width, height), SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K
    )
    _, faces = detector.detect(frame)

    if faces is None:
        print("Nenhum rosto detectado.")
        faces = []          # lista vazia: nada para desenhar
    else:
        print(f"Rostos detectados: {len(faces)}")

    # Primeiro esconde o fundo, depois desenha por cima
    if HIDE_BACKGROUND:
        frame = hide_background(frame, faces)

    for face in faces:
        draw_face(frame, face)

    saved = cv2.imwrite(OUTPUT_FILE, frame)
    print(f"Imagem salva em {OUTPUT_FILE}: {saved}")

    # Mostra o resultado numa janela ajustável e espera uma tecla
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 640, 360)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.imshow(WINDOW_NAME, frame)
    print("Aperte qualquer tecla na janela da imagem para fechar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Não consegui ler a câmera.")