@echo off
rem iniciar_estudio.bat - prepara e abre o Estudio de Treino (Missao Clonagem, Livro 1, Cap. 6)
rem Duplo clique neste arquivo. Na primeira vez ele instala tudo; depois, so abre o Estudio.

title Estudio de Treino - iniciador
cd /d "%~dp0"

echo === 1/4 Python 3.12 ===
py -V:3.12 -c "pass" >nul 2>&1
if not errorlevel 1 goto python_ok

echo Python 3.12 nao encontrado. Tentando instalar...
where py >nul 2>&1
if not errorlevel 1 py install 3.12
py -V:3.12 -c "pass" >nul 2>&1
if not errorlevel 1 goto python_ok

where winget >nul 2>&1
if not errorlevel 1 winget install --id Python.Python.3.12 -e --accept-package-agreements --accept-source-agreements
py -V:3.12 -c "pass" >nul 2>&1
if not errorlevel 1 goto python_ok

echo.
echo [ERRO] Nao consegui instalar o Python 3.12 sozinho.
echo  - Se a instalacao acabou de terminar, FECHE esta janela e abra o iniciador de novo.
echo  - Senao, instale pelo site https://www.python.org/downloads/ (versao 3.12)
echo    e rode este arquivo de novo.
pause
exit /b 1

:python_ok
py -V:3.12 --version

echo === 2/4 Maleta (.venv) ===
if exist ".venv\Scripts\python.exe" goto venv_ok
echo Criando a maleta .venv (so na primeira vez)...
py -V:3.12 -m venv .venv
if errorlevel 1 goto erro
:venv_ok
echo Maleta: %CD%\.venv

echo === 3/4 Bibliotecas e modelos ===
".venv\Scripts\python.exe" instalar.py
if errorlevel 1 goto erro

echo === 4/4 Abrindo o Estudio ===
".venv\Scripts\python.exe" app.py
if errorlevel 1 goto erro
exit /b 0

:erro
echo.
echo [ERRO] Algo deu errado. Leia a mensagem acima (ou tire um print dela).
pause
exit /b 1
