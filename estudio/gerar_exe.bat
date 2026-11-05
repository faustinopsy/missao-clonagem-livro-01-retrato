@echo off
rem gerar_exe.bat - transforma o Estudio de Treino em dist\Estudio\Estudio.exe (PyInstaller)
rem Rode DEPOIS do iniciar_estudio.bat (ele cria a maleta .venv e baixa o modelo).

title Estudio de Treino - gerar .exe
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERRO] Rode primeiro o iniciar_estudio.bat: falta a maleta .venv.
    pause
    exit /b 1
)
if not exist "modelos\face_recognition_sface_2021dec.onnx" (
    echo [ERRO] Rode primeiro o iniciar_estudio.bat: falta o modelo em modelos\.
    pause
    exit /b 1
)

echo === 1/2 Instalando o PyInstaller na maleta ===
".venv\Scripts\python.exe" -m pip install pyinstaller==6.22.3
if errorlevel 1 goto erro

echo === 2/2 Gerando o .exe (leva alguns minutos) ===
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean --windowed --name Estudio ^
    --add-data "modelos\face_landmarker.task;modelos" ^
    --add-data "modelos\face_detection_yunet_2023mar.onnx;modelos" ^
    --add-data "modelos\face_recognition_sface_2021dec.onnx;modelos" ^
    --add-data "LICENCAS.txt;." ^
    --collect-all mediapipe ^
    app.py
if errorlevel 1 goto erro

copy /y LICENCAS.txt dist\Estudio\ >nul
echo.
echo Pronto: dist\Estudio\Estudio.exe
echo Para entregar, compacte a pasta dist\Estudio inteira (.zip).
pause
exit /b 0

:erro
echo.
echo [ERRO] Algo deu errado. Leia a mensagem acima (ou tire um print dela).
pause
exit /b 1
