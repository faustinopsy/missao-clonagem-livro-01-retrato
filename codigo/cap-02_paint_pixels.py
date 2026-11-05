# cap-02_paint_pixels.py
# Missão: provar que a imagem é uma matriz, trocando os números de uma região.

import cv2  # OpenCV: biblioteca de visão computacional

CAMERA_INDEX = 0
WINDOW_NAME = "Agente Iris - pintando pixels"
QUIT_KEY = "q"

# Região a pintar: LINHAS (altura, eixo y) primeiro, COLUNAS (largura, eixo x) depois
TOP, BOTTOM = 50, 150      # linhas 50 a 149  -> 100 pixels de altura
LEFT, RIGHT = 100, 300     # colunas 100 a 299 -> 200 pixels de largura

# Cor na ordem do OpenCV: (Azul, Verde, Vermelho)
RED_BGR = (0, 0, 255)

# Janela ajustável, menor que a imagem, no canto superior esquerdo da tela
PREVIEW_WIDTH, PREVIEW_HEIGHT = 640, 360   # 16:9, mesma proporção de 1280x720
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, PREVIEW_WIDTH, PREVIEW_HEIGHT)
cv2.moveWindow(WINDOW_NAME, 0, 0)

camera = cv2.VideoCapture(CAMERA_INDEX)
print(f"Aperte '{QUIT_KEY}' na janela do vídeo para sair.")

while True:
    ok, frame = camera.read()
    if not ok:
        print("Não consegui ler a câmera.")
        break

    # Troca todos os pixels da região pela cor escolhida
    frame[TOP:BOTTOM, LEFT:RIGHT] = RED_BGR

    cv2.imshow(WINDOW_NAME, frame)

    key = cv2.waitKey(1)
    if key == ord(QUIT_KEY):
        break

camera.release()
cv2.destroyAllWindows()