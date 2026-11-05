# Modelos pré-treinados

Os modelos não ficam no repositório. Baixe cada um para esta pasta, no capítulo em que ele aparece, e confira o SHA-256 com:

```powershell
Get-FileHash modelos\<arquivo> -Algorithm SHA256
```

| Arquivo | Capítulo | Licença | SHA-256 |
|---|---|---|---|
| `face_detection_yunet_2023mar.onnx` (YuNet, OpenCV Zoo) | 3 | MIT | `8F2383E4DD3CFBB4553EA8718107FC0423210DC964F9F4280604804ED2552FA4` |
| `face_landmarker.task` (MediaPipe Face Landmarker, Google) | 4 | Apache 2.0 | `64184E229B263107BC2B804C6625DB1341FF2BB731874B0BCC2FE6544E0BC9FF` |
| `face_recognition_sface_2021dec.onnx` (SFace, OpenCV Zoo) | 7 | Apache 2.0 | `0BA9FBFA01B5270C96627C4EF784DA859931E02F04419C829E83484087C34E79` |

Endereços de download:
- YuNet: https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx
- Face Landmarker: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
- SFace: https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx

O Estúdio de Treino baixa e confere os três sozinho, pelo `estudio\iniciar_estudio.bat`.
