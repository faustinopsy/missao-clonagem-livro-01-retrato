# camera.py — a classe Camera: liga, lê quadros e desliga a webcam.

import cv2   # OpenCV: câmera


class Camera:
    """Embrulha o cv2.VideoCapture com mensagens claras e um 'está ligada?'."""

    def __init__(self, index=0):
        self.index = index
        self.capture = None          # None = desligada

    def is_on(self):
        return self.capture is not None

    def turn_on(self):
        """Liga a câmera. Devolve True se conseguiu ler o primeiro quadro."""
        self.capture = cv2.VideoCapture(self.index)
        ok, _ = self.capture.read()
        if not ok:
            self.turn_off()
        return ok

    def read(self):
        """Devolve o próximo quadro (array BGR) ou None se não deu para ler."""
        if not self.is_on():
            return None
        ok, frame = self.capture.read()
        if not ok:
            return None
        return frame

    def turn_off(self):
        if self.capture is not None:
            self.capture.release()
        self.capture = None
