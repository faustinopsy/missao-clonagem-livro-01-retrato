# cap-02_first_frame.py
# Missão: abrir a webcam, capturar UM quadro e descobrir o que ele é por dentro.

import cv2  # OpenCV: biblioteca de visão computacional

CAMERA_INDEX = 0                                   # 0 = primeira câmera do computador
OUTPUT_FILE = "dados-pessoais/cap-02_quadro.png"   # pasta que não vai para o GitHub

camera = cv2.VideoCapture(CAMERA_INDEX)            # liga a câmera
ok, frame = camera.read()                          # tira uma "foto": um quadro
camera.release()                                   # desliga a câmera

if ok:
    print("Quadro capturado!")
    print(f"Tipo do objeto: {type(frame)}")
    print(f"Formato (shape): {frame.shape}")
    print(f"Tipo de cada número (dtype): {frame.dtype}")
    print(f"Pixel do canto superior esquerdo [0, 0] (B, G, R): {frame[0, 0]}")
    saved = cv2.imwrite(OUTPUT_FILE, frame)        # salva o quadro como imagem
    print(f"Imagem salva em {OUTPUT_FILE}: {saved}")
else:
    print("Não consegui ler a câmera. Ela está ligada? Outro programa está usando?")