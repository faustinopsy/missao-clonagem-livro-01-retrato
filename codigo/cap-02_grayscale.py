# cap-02_grayscale.py
# Missão: converter o vídeo para tons de cinza e comparar os formatos.

import cv2  # OpenCV: biblioteca de visão computacional

CAMERA_INDEX = 0
QUIT_KEY = "q"

COLOR_WINDOW = "Agente Iris - colorido"
GRAY_WINDOW = "Agente Iris - cinza"

# Tamanho das janelas na tela (a imagem por dentro continua do tamanho da câmera)
PREVIEW_WIDTH, PREVIEW_HEIGHT = 640, 360   # 16:9, mesma proporção de 1280x720

# Cria as janelas com tamanho ajustável (dá para redimensionar com o mouse)
cv2.namedWindow(COLOR_WINDOW, cv2.WINDOW_NORMAL)
cv2.namedWindow(GRAY_WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(COLOR_WINDOW, PREVIEW_WIDTH, PREVIEW_HEIGHT)
cv2.resizeWindow(GRAY_WINDOW, PREVIEW_WIDTH, PREVIEW_HEIGHT)

# Posiciona as janelas: colorida em cima, cinza logo abaixo (lado esquerdo da tela)
cv2.moveWindow(COLOR_WINDOW, 0, 0)
cv2.moveWindow(GRAY_WINDOW, 0, PREVIEW_HEIGHT + 40)   # +40 = espaço da barra de título

camera = cv2.VideoCapture(CAMERA_INDEX)
print(f"Aperte '{QUIT_KEY}' em uma das janelas para sair.")

shape_printed = False   # bandeira: já mostramos os formatos?

while True:
    ok, frame = camera.read()
    if not ok:
        print("Não consegui ler a câmera.")
        break

    # Converte de colorido (BGR) para tons de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if not shape_printed:
        print(f"Colorido  -> shape: {frame.shape}")
        print(f"Cinza     -> shape: {gray.shape}")
        print(f"Números no colorido: {frame.size}")
        print(f"Números no cinza:    {gray.size}")
        shape_printed = True

    cv2.imshow(COLOR_WINDOW, frame)
    cv2.imshow(GRAY_WINDOW, gray)

    key = cv2.waitKey(1)
    if key == ord(QUIT_KEY):
        break

camera.release()
cv2.destroyAllWindows()