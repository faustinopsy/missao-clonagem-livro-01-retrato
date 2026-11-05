# cap-02_channels.py
# Missão: separar a imagem colorida em seus três canais (B, G, R).

import cv2  # OpenCV: biblioteca de visão computacional

CAMERA_INDEX = 0
QUIT_KEY = "q"

# Tamanho de cada janela na tela (4 janelas em grade 2x2)
PREVIEW_WIDTH, PREVIEW_HEIGHT = 480, 270   # 16:9
GAP = 40                                   # espaço para a barra de título


def create_window(name, x, y):
    """Cria uma janela ajustável, com tamanho reduzido, na posição (x, y)."""
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, PREVIEW_WIDTH, PREVIEW_HEIGHT)
    cv2.moveWindow(name, x, y)


# Grade 2x2: original e azul em cima; verde e vermelho embaixo
create_window("Original", 0, 0)
create_window("Canal azul (B)", PREVIEW_WIDTH, 0)
create_window("Canal verde (G)", 0, PREVIEW_HEIGHT + GAP)
create_window("Canal vermelho (R)", PREVIEW_WIDTH, PREVIEW_HEIGHT + GAP)

camera = cv2.VideoCapture(CAMERA_INDEX)
print(f"Aperte '{QUIT_KEY}' em uma das janelas para sair.")

shape_printed = False

while True:
    ok, frame = camera.read()
    if not ok:
        print("Não consegui ler a câmera.")
        break

    # Separa as três matrizes, na ordem do OpenCV: azul, verde, vermelho
    blue, green, red = cv2.split(frame)

    if not shape_printed:
        print(f"Original -> shape: {frame.shape}")
        print(f"Cada canal -> shape: {blue.shape}")
        shape_printed = True

    cv2.imshow("Original", frame)
    cv2.imshow("Canal azul (B)", blue)
    cv2.imshow("Canal verde (G)", green)
    cv2.imshow("Canal vermelho (R)", red)

    key = cv2.waitKey(1)
    if key == ord(QUIT_KEY):
        break

camera.release()
cv2.destroyAllWindows()