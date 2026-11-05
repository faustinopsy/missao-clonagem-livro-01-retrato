# cap-03_detect_once.py
# Missão: tirar UMA foto e ver os números que o detector YuNet devolve.

import cv2  # OpenCV: biblioteca de visão computacional

CAMERA_INDEX = 0
MODEL_FILE = "modelos/face_detection_yunet_2023mar.onnx"

SCORE_THRESHOLD = 0.9   # limiar: só aceita caixas com confiança >= 0.9
NMS_THRESHOLD = 0.3     # NMS: o quanto duas caixas podem se sobrepor antes de uma ser descartada
TOP_K = 5000            # quantas caixas candidatas o detector analisa antes da NMS

camera = cv2.VideoCapture(CAMERA_INDEX)
ok, frame = camera.read()
camera.release()

if ok:
    # O shape é (altura, largura, canais); pegamos só os dois primeiros
    height, width = frame.shape[:2]
    print(f"Tamanho do quadro: largura={width}, altura={height}")

    # Cria o detector já ajustado para o tamanho do quadro (largura, altura)
    detector = cv2.FaceDetectorYN.create(
        MODEL_FILE, "", (width, height), SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K
    )

    _, faces = detector.detect(frame)

    if faces is None:
        print("Nenhum rosto detectado.")
    else:
        print(f"Rostos detectados: {len(faces)}")
        print(f"Formato do resultado: {faces.shape}")
        for face in faces:
            x, y, w, h = face[0:4]      # números 0 a 3: a caixa
            score = face[14]            # número 14: a confiança
            print(f"Caixa: x={x:.0f}, y={y:.0f}, largura={w:.0f}, altura={h:.0f} | confiança={score:.2f}")
else:
    print("Não consegui ler a câmera.")