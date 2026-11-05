# Missão Clonagem · Livro 1: Retrato

Código do livro **Missão Clonagem · Livro 1: Retrato**, sobre reconhecimento facial para iniciantes, em Python.

Este repositório acompanha o livro. A explicação de cada linha, as saídas reais e os erros comuns estão nos capítulos; aqui ficam os scripts prontos, para você copiar, comparar e corrigir os seus.

## Baixar o código
Na página do repositório, clique em **Code → Download ZIP** e extraia a pasta. Se você usa Git:

```powershell
git clone https://github.com/faustinopsy/missao-clonagem-livro-01-retrato.git
cd missao-clonagem-livro-01-retrato
```

## O que você vai precisar
- Windows (o livro foi escrito e testado no Windows 11).
- Uma webcam (a do notebook serve).
- Python **3.12** (o livro usa o 3.12.10).
- Placa de vídeo: **não precisa**. Tudo roda no processador (Rota CPU).

## Preparar a maleta (Capítulo 1)
No PowerShell, dentro da pasta deste repositório:

```powershell
# 1. Instala o Python 3.12 ao lado de outras versões (não desinstala nada)
py install 3.12

# 2. Cria a maleta (ambiente virtual) com o Python 3.12
py -V:3.12 -m venv .venv

# 3. Ativa a maleta: "(.venv)" aparece no começo da linha
.\.venv\Scripts\Activate.ps1

# 4. Instala as bibliotecas nas versões exatas do livro
python -m pip install -r requirements.txt

# 5. Confere tudo
python codigo\cap-01_check_env.py
```

Toda vez que abrir um PowerShell novo, entre na pasta e ative a maleta de novo (passo 3).

## Os modelos
Os modelos pré-treinados **não ficam neste repositório**. Cada capítulo mostra como baixá-los para a pasta `modelos\` e como conferir o SHA-256. Veja a lista em [`modelos/LEIA-ME.md`](modelos/LEIA-ME.md).

## Os scripts, capítulo por capítulo
Rode cada script a partir da pasta do repositório, com a maleta ativa. Exemplo: `python codigo\cap-02_live_camera.py`.

| Capítulo | Script | O que faz |
|---|---|---|
| 1 | `cap-01_check_env.py` | Confere Python, maleta, bibliotecas e placa de vídeo. |
| 2 | `cap-02_first_frame.py` | Abre a webcam, captura um quadro e mostra o que ele é por dentro. |
| 2 | `cap-02_live_camera.py` | Mostra a webcam ao vivo, até apertar "q". |
| 2 | `cap-02_paint_pixels.py` | Prova que a imagem é uma matriz, pintando uma região. |
| 2 | `cap-02_grayscale.py` | Converte o vídeo para tons de cinza e compara os formatos. |
| 2 | `cap-02_channels.py` | Separa a imagem colorida nos canais B, G e R. |
| 3 | `cap-03_detect_once.py` | Tira uma foto e mostra os números que o detector YuNet devolve. |
| 3 | `cap-03_draw_detection.py` | Desenha a caixa, os 5 pontos e a confiança de cada rosto. |
| 3 | `cap-03_detect_live.py` | Detecta rostos ao vivo, com modo privacidade e FPS. |
| 3 | `cap-03_detection_lab.py` | Testa o limiar, o tamanho do rosto e o tempo de detecção. |
| 4 | `cap-04_landmarks_once.py` | Tira uma foto e lê alguns dos 478 pontos do rosto. |
| 4 | `cap-04_landmarks_live.py` | Desenha os 478 pontos ao vivo. |
| 4 | `cap-04_face_regions.py` | Desenha o mapa do rosto por regiões. |
| 4 | `cap-04_openness.py` | Mede a abertura dos olhos e da boca e conta piscadas. |
| 5 | `cap-05_blendshapes_rules.py` | Mostra os 52 movimentos do rosto e reconhece expressões com regras. |
| 5 | `cap-05_collect.py` | Grava exemplos das suas expressões num CSV. |
| 5 | `cap-05_train.py` | Treina um classificador k-NN com as suas amostras e mede o resultado. |
| 5 | `cap-05_live_model.py` | Usa o seu modelo treinado para reconhecer expressões ao vivo. |
| 6 | `estudio\` | O Estúdio de Treino (veja abaixo). |
| 7 | `cap-07_first_descriptor.py` | Transforma o seu rosto num descritor de 128 números. |
| 7 | `cap-07_compare.py` | Compara duas fotos: "é a mesma pessoa?". |
| 7 | `cap-07_threshold.py` | Compara todas as fotos entre si e ajuda a escolher o limiar. |
| 7 | `cap-07_verify_live.py` | Verificação 1:1 ao vivo: "é o P01?". |
| 8 | `cap-08_enroll.py` | Cadastra um agente na galeria da porta, com várias poses. |
| 8 | `cap-08_door_live.py` | A porta da Divisão: identificação 1:N ao vivo, com registro. |
| 8 | `cap-08_evaluate.py` | Mede a porta antes de confiar nela. |
| 9 | (sem código) | Limites, vieses e LGPD. |

## O Estúdio de Treino (Capítulos 6, 7 e 8)
Um programa com janelas e botões que reúne tudo o que o livro faz no terminal.

- **Jeito fácil:** dê duplo clique em `estudio\iniciar_estudio.bat`. Na primeira vez, ele cria a maleta do Estúdio, instala as bibliotecas e baixa os modelos, conferindo o SHA-256 de cada um.
- **Gerar o `.exe`:** depois de rodar o iniciador uma vez, dê duplo clique em `estudio\gerar_exe.bat`. O programa fica em `estudio\dist\Estudio\`.

As licenças dos modelos e das bibliotecas incluídas no `.exe` estão em [`estudio/LICENCAS.txt`](estudio/LICENCAS.txt).

## Privacidade
Os scripts guardam fotos, amostras, descritores e registros na pasta `dados-pessoais\`. Essa pasta **nunca vai para o GitHub** (está no `.gitignore`). Use só o seu rosto, ou o de pessoas que autorizaram. O Capítulo 9 explica por que isso importa e o que a LGPD exige.

## Licença
<!-- [DECIDIR pelo autor] licença do código (ex.: MIT) antes de tornar o repositório público. -->
A definir.
