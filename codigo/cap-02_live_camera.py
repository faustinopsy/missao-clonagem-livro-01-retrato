# cap-02_live_camera.py
# Missão: mostrar a webcam ao vivo, quadro a quadro, até apertar "q".

import cv2  # OpenCV: biblioteca de visão computacional

CAMERA_INDEX = 0                    # 0 = primeira câmera do computador
WINDOW_NAME = "Agente Iris - camera ao vivo"
QUIT_KEY = "q"                      # tecla para encerrar

# Janela ajustável, menor que a imagem, no canto superior esquerdo da tela
PREVIEW_WIDTH, PREVIEW_HEIGHT = 640, 360   # 16:9, mesma proporção de 1280x720
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, PREVIEW_WIDTH, PREVIEW_HEIGHT)
cv2.moveWindow(WINDOW_NAME, 0, 0)

camera = cv2.VideoCapture(CAMERA_INDEX)   # liga a câmera
print(f"Câmera ligada. Aperte '{QUIT_KEY}' na janela do vídeo para sair.")

while True:
    ok, frame = camera.read()             # pega o próximo quadro
    if not ok:
        print("Não consegui ler a câmera.")
        break

    cv2.imshow(WINDOW_NAME, frame)        # mostra o quadro na janela

    # Espera 1 milissegundo por uma tecla e devolve o número dela
    key = cv2.waitKey(1)
    if key == ord(QUIT_KEY):
        break

camera.release()                          # desliga a câmera
cv2.destroyAllWindows()                   # fecha as janelas do OpenCV
print("Câmera desligada.")