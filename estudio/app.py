# app.py — Estúdio de Treino da Divisão Espelho
# Capítulo 7: dois modos, Expressões (Cap. 6) e Identidade (Cap. 7): tela inicial com dois botões
#             grandes, botão na barra superior para trocar e o menu "Modo".

from pathlib import Path          # caminhos de pastas e arquivos
import tkinter as tk              # Tkinter: interface gráfica que vem com o Python
from tkinter import filedialog, messagebox, ttk   # escolher pasta, caixas de aviso e componentes
import tkinter.font as tkfont     # fontes (tipos e tamanhos de letra) do Tkinter

import json                       # ler e gravar o config.json
import sys
import traceback                  # o texto completo de um erro (para o estudio.log)
import time

import cv2
import numpy as np
from PIL import Image, ImageTk    # Pillow: converte a imagem do OpenCV para o Tkinter

import ambiente                   # nosso módulo: verificações do computador
from camera import Camera         # nosso módulo: a webcam
from rosto import FaceAnalyzer    # nosso módulo: MediaPipe e modo privacidade
from dados import Dataset         # nosso módulo: classes, amostras e imagens
import treino                     # nosso módulo: treinar, avaliar e salvar o modelo
from modo_identidade import IdentityMode   # nosso módulo: as abas do modo Identidade
from modo_porta import DoorMode            # nosso módulo: Cadastrar e Porta (Capítulo 8)

APP_TITLE = "Estúdio de Treino — Divisão Espelho"
ESTUDIO_FOLDER = Path(__file__).resolve().parent       # a pasta estudio
# Rodando como .exe (PyInstaller), sys.frozen existe. Aí os arquivos que o Estúdio GRAVA
# ficam ao lado do Estudio.exe, e não na pasta interna (_internal) do PyInstaller.
IS_EXE = getattr(sys, "frozen", False)
APP_FOLDER = Path(sys.executable).parent if IS_EXE else ESTUDIO_FOLDER
DEFAULT_PROJECT = APP_FOLDER / "projeto"                # pasta do projeto na primeira vez
CONFIG_FILE = APP_FOLDER / "config.json"                # guarda a última pasta e a meta
LOG_FILE = APP_FOLDER / "estudio.log"                   # erros inesperados ficam registrados aqui

# Tamanho do texto: começa maior que o padrão e pode ir de 8 a 24
DEFAULT_FONT_SIZE = 12
MIN_FONT_SIZE = 8
MAX_FONT_SIZE = 24
# Fontes "com nome" que o Tkinter usa em todos os componentes
FONT_NAMES = ["TkDefaultFont", "TkTextFont", "TkHeadingFont", "TkMenuFont", "TkFixedFont"]

PREVIEW_WIDTH, PREVIEW_HEIGHT = 640, 360   # tamanho do vídeo dentro da janela (16:9)
BLANK_GRAY = (70, 70, 70)                  # cor do quadro vazio, com a câmera desligada
VIDEO_DELAY_MS = 15                        # intervalo entre um quadro e outro
DEFAULT_GOAL = 60                          # meta de amostras por classe
SAVE_EVERY = 3                             # grava 1 amostra a cada 3 quadros
MAX_PHOTO_WIDTH = 1280                     # fotos maiores são reduzidas antes de analisar
DEFAULT_NEIGHBORS = 5                      # k do k-NN (o mesmo do Cap. 5)
DEFAULT_TEST_PERCENT = 20                  # % das amostras guardadas para a prova
DEFAULT_MIN_CONFIDENCE = 0.6               # abaixo disto, o modelo está "em dúvida" (Cap. 5)
PHOTO_TYPES = [("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Todos os arquivos", "*.*")]


class EstudioApp:
    """A janela principal do Estúdio: barra superior + abas."""

    def __init__(self, root):
        # __init__ roda uma vez, quando o objeto é criado: monta a janela
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1280x800")
        self.root.minsize(1000, 650)

        # Variáveis do Tkinter: quando mudam, os componentes ligados a elas se atualizam
        config = load_config()            # o que ficou guardado da última vez (ou {} na primeira)
        self.project_folder = tk.StringVar(value=config.get("pasta_do_projeto", str(DEFAULT_PROJECT)))
        self.person_code = tk.StringVar(value="P01")
        self.camera_status = tk.StringVar(value="Câmera: desligada")
        self.font_size = tk.IntVar(value=DEFAULT_FONT_SIZE)
        self.result_font = tkfont.Font(weight="bold")   # letra grande do resultado na aba Testar
        self.big_font = tkfont.Font(weight="bold")      # letra dos botões grandes da tela inicial

        self.apply_font_size()

        self.camera = Camera(0)
        self.analyzer = None              # o MediaPipe só é carregado ao ligar a câmera
        self.hide_background = tk.BooleanVar(value=True)
        self.previous_time = time.perf_counter()
        self.dataset = Dataset(self.project_folder.get())
        self.recording_class = None       # classe sendo gravada agora (None = nenhuma)
        self.frame_count = 0
        self.goal = tk.IntVar(value=config.get("meta", DEFAULT_GOAL))
        self.save_images = tk.BooleanVar(value=False)
        self.new_class_name = tk.StringVar()
        self.message = tk.StringVar(value="Crie uma classe, selecione-a e clique em Gravar amostras.")
        self.neighbors = tk.IntVar(value=DEFAULT_NEIGHBORS)
        self.test_percent = tk.IntVar(value=DEFAULT_TEST_PERCENT)
        self.trained = None               # dados do último treino (para o botão Salvar modelo)
        self.test_model = None            # modelo carregado na aba Testar
        self.min_confidence = tk.DoubleVar(value=DEFAULT_MIN_CONFIDENCE)
        self.result_text = tk.StringVar(value="—")
        self.model_status = tk.StringVar(value="Nenhum modelo carregado.")
        self.mode = tk.StringVar(value="")             # "" = ainda na tela inicial
        self.environment_tables = []      # uma tabela de Ambiente em cada modo
        self.identity_mode = IdentityMode(self, config, save_setting)
        self.door_mode = DoorMode(self, self.identity_mode, config, save_setting)

        # Imagem cinza do tamanho do vídeo: segura o lugar enquanto a câmera está desligada
        self.blank_video = ImageTk.PhotoImage(Image.new("RGB", (PREVIEW_WIDTH, PREVIEW_HEIGHT), BLANK_GRAY))
        self.build_menu()
        self.build_top_bar()
        self.build_tabs()
        self.build_environment_tab(self.tabs["1. Ambiente"])
        self.build_environment_tab(self.id_tabs["1. Ambiente"])
        self.build_collect_tab()
        self.build_train_tab()
        self.build_test_tab()
        self.identity_mode.build(self.id_tabs)
        self.door_mode.build(self.id_tabs)
        self.build_welcome()
        self.show_mode()                  # começa na tela inicial, com os dois botões grandes

        # Ao fechar a janela no X, desliga a câmera antes de sair
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def build_menu(self):
        """Barra de menu do Windows, com o menu Modo: Expressões ou Identidade."""
        menu_bar = tk.Menu(self.root)
        mode_menu = tk.Menu(menu_bar, tearoff=0)        # tearoff=0: sem a linha pontilhada
        for mode in ("Expressões", "Identidade"):
            mode_menu.add_radiobutton(label=mode, variable=self.mode, value=mode, command=self.show_mode)
        menu_bar.add_cascade(label="Modo", menu=mode_menu)
        self.root.config(menu=menu_bar)

    def build_welcome(self):
        """Tela inicial: dois botões grandes, um para cada modo."""
        self.welcome = ttk.Frame(self.root, padding=40)
        self.welcome.grid(row=1, column=0, sticky="nsew")
        ttk.Label(self.welcome, text="O que vamos treinar hoje?", font=self.result_font).grid(
            row=0, column=0, columnspan=2, pady=(20, 40))
        buttons = {
            "Expressões": "Expressões\n\no que o rosto está fazendo\n(feliz, raiva, neutro…)",
            "Identidade": "Identidade\n\nde quem é o rosto\n(é a mesma pessoa?)",
        }
        for column, (mode, text) in enumerate(buttons.items()):
            ttk.Button(self.welcome, text=text, style="Big.TButton",
                       command=lambda m=mode: self.choose_mode(m)).grid(
                row=1, column=column, padx=30, sticky="nsew")
            self.welcome.columnconfigure(column, weight=1)

    def choose_mode(self, mode):
        self.mode.set(mode)
        self.show_mode()

    def switch_mode(self):
        """O botão da barra superior: vai para o OUTRO modo."""
        self.choose_mode("Identidade" if self.mode.get() == "Expressões" else "Expressões")

    def show_mode(self):
        """Mostra a tela inicial (sem modo) ou as abas do modo escolhido, escondendo o resto."""
        mode = self.mode.get()
        if mode == "":
            self.welcome.grid()
        else:
            self.welcome.grid_remove()
        for name, notebook in self.notebooks.items():
            if name == mode:
                notebook.grid()                     # volta a aparecer, no mesmo lugar
            else:
                notebook.grid_remove()              # some, mas guarda o lugar
        self.notebook = self.notebooks.get(mode)    # None na tela inicial
        if mode == "":
            self.switch_button.config(text="Escolha um modo abaixo", state="disabled")
        else:
            other = "Identidade" if mode == "Expressões" else "Expressões"
            self.switch_button.config(text=f"Ir para {other} ⇄", state="normal")

    def build_top_bar(self):
        """Barra superior em 2 linhas: (1) pasta do projeto; (2) pessoa, câmera e tamanho do texto.
        Cada linha é um painel (Frame) próprio, para uma não empurrar a outra."""
        bar = ttk.Frame(self.root, padding=8)
        bar.grid(row=0, column=0, sticky="ew")

        # Linha 1: o botão vem antes do caminho, para nunca sumir da tela
        line1 = ttk.Frame(bar)
        line1.grid(row=0, column=0, sticky="w")
        ttk.Button(line1, text="Escolher pasta do projeto…", command=self.choose_folder).grid(row=0, column=0)
        ttk.Label(line1, text="Pasta do projeto:").grid(row=0, column=1, padx=(8, 4))
        ttk.Label(line1, textvariable=self.project_folder).grid(row=0, column=2)

        # Linha 2: pessoa, câmera e tamanho do texto (acessibilidade)
        line2 = ttk.Frame(bar)
        line2.grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Label(line2, text="Modo:").grid(row=0, column=7, padx=(24, 4))
        ttk.Label(line2, textvariable=self.mode, font="TkHeadingFont").grid(row=0, column=8)
        self.switch_button = ttk.Button(line2, command=self.switch_mode)
        self.switch_button.grid(row=0, column=9, padx=(12, 0))
        ttk.Label(line2, text="Pessoa (código):").grid(row=0, column=0)
        ttk.Entry(line2, textvariable=self.person_code, width=6).grid(row=0, column=1, padx=(4, 0))
        ttk.Label(line2, textvariable=self.camera_status).grid(row=0, column=2, padx=24)
        ttk.Label(line2, text="Texto:").grid(row=0, column=3, padx=(0, 4))
        ttk.Button(line2, text="A-", width=3, command=lambda: self.change_font_size(-1)).grid(row=0, column=4)
        ttk.Label(line2, textvariable=self.font_size, width=3, anchor="center").grid(row=0, column=5)
        ttk.Button(line2, text="A+", width=3, command=lambda: self.change_font_size(+1)).grid(row=0, column=6)

    def build_tabs(self):
        """Um conjunto de abas (Notebook) para cada modo, no mesmo lugar da janela."""
        # A linha 1 e a coluna 0 crescem quando a janela aumenta
        self.root.rowconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)

        expression_tabs = {
            "1. Ambiente": "Verificação do computador: Python, maleta, bibliotecas, modelos, câmera.",
            "2. Coletar": "Câmera, classes, gravação de amostras e importação de fotos.",
            "3. Treinar": "Treino do modelo, acurácia e matriz de confusão.",
            "4. Testar": "Reconhecimento ao vivo com o modelo treinado.",
        }
        identity_tabs = {
            "1. Ambiente": "Verificação do computador: Python, maleta, bibliotecas, modelos, câmera.",
            "2. Comparar": "Duas fotos: são da mesma pessoa? (similaridade de cosseno)",
            "3. Limiar": "Todas as fotos contra todas: falsos aceites, falsas rejeições e o limiar.",
            "4. Verificar": "Verificação 1:1 ao vivo: a pessoa na câmera é quem diz ser?",
            "5. Cadastrar": "Cadastro dos agentes na galeria, com poses guiadas (Capítulo 8).",
            "6. Porta": "A porta da Divisão: quem é, entre todos os agentes? (identificação 1:N)",
        }
        self.notebooks = {}
        self.tabs = self.make_notebook("Expressões", expression_tabs)
        self.id_tabs = self.make_notebook("Identidade", identity_tabs)

    def make_notebook(self, mode, tabs):
        """Cria um Notebook com as abas pedidas e devolve {título: painel}."""
        notebook = ttk.Notebook(self.root)
        notebook.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        self.notebooks[mode] = notebook
        frames = {}
        for title, description in tabs.items():
            frame = ttk.Frame(notebook, padding=16)
            ttk.Label(frame, text=description).grid(row=0, column=0, columnspan=2, sticky="w")
            notebook.add(frame, text=title)
            frames[title] = frame
        return frames

    def change_font_size(self, step):
        """Soma 'step' ao tamanho do texto, sem passar dos limites."""
        new_size = self.font_size.get() + step
        new_size = max(MIN_FONT_SIZE, min(new_size, MAX_FONT_SIZE))
        self.font_size.set(new_size)
        self.apply_font_size()

    def apply_font_size(self):
        """Muda o tamanho de todas as fontes do Tkinter de uma vez."""
        for name in FONT_NAMES:
            tkfont.nametofont(name).configure(size=self.font_size.get())
        # As linhas das tabelas (Treeview) têm altura fixa: crescem junto com a letra
        ttk.Style().configure("Treeview", rowheight=int(self.font_size.get() * 2.2))
        self.result_font.configure(size=self.font_size.get() * 2)
        self.big_font.configure(size=int(self.font_size.get() * 1.6))
        ttk.Style().configure("Big.TButton", font=self.big_font, padding=(30, 40), justify="center")

    def build_environment_tab(self, frame):
        """Aba 1 (nos dois modos): tabela com as verificações e o botão para verificar de novo."""
        ttk.Button(frame, text="Verificar de novo", command=self.refresh_environment).grid(
            row=1, column=0, sticky="w", pady=8)

        columns = ("situacao", "detalhe")
        table = ttk.Treeview(frame, columns=columns, height=14)
        table.heading("#0", text="Item")
        table.heading("situacao", text="Situação")
        table.heading("detalhe", text="Detalhe")
        table.column("#0", width=260)
        table.column("situacao", width=110, anchor="center")
        table.column("detalhe", width=560)
        table.grid(row=2, column=0, sticky="nsew")
        frame.rowconfigure(2, weight=1)
        frame.columnconfigure(0, weight=1)

        # Cores por situação
        table.tag_configure("ok", foreground="#1a7f37")
        table.tag_configure("erro", foreground="#cf222e")
        table.tag_configure("info", foreground="#0550ae")

        self.environment_tables.append(table)
        self.refresh_environment()

    def refresh_environment(self):
        """Roda todas as verificações UMA vez e preenche as tabelas dos dois modos."""
        symbols = {"ok": "✔ ok", "erro": "✘ erro", "info": "ℹ info"}
        results = ambiente.run_all_checks(self.project_folder.get())
        for table in self.environment_tables:
            for row in table.get_children():
                table.delete(row)
            for result in results:
                table.insert("", "end", text=result["item"],
                             values=(symbols[result["status"]], result["detail"]),
                             tags=(result["status"],))

    def build_collect_tab(self):
        """Aba 2 (por enquanto): o vídeo ao vivo e o botão de ligar a câmera."""
        frame = self.tabs["2. Coletar"]
        self.video_label = ttk.Label(frame, image=self.blank_video)   # é aqui que cada quadro vai aparecer
        self.video_label.grid(row=1, column=0, pady=8, sticky="nw")

        controls = ttk.Frame(frame)
        controls.grid(row=2, column=0, sticky="w")
        self.camera_button = ttk.Button(controls, text="Ligar câmera", command=self.toggle_camera)
        self.camera_button.grid(row=0, column=0)
        self.camera_buttons = [self.camera_button]   # a aba Testar acrescenta o botão dela
        ttk.Checkbutton(controls, text="Esconder o fundo (modo privacidade)",
                        variable=self.hide_background).grid(row=0, column=1, padx=16)

        self.build_classes_panel(frame)

    def build_classes_panel(self, frame):
        """Painel à direita do vídeo: classes, meta, gravação e progresso."""
        panel = ttk.Frame(frame, padding=(16, 0))
        panel.grid(row=1, column=1, rowspan=2, sticky="nsew")
        frame.columnconfigure(1, weight=1)

        ttk.Label(panel, text="Classes").grid(row=0, column=0, columnspan=4, sticky="w")
        self.classes_table = ttk.Treeview(panel, columns=("amostras", "pessoas"), height=8,
                                          selectmode="browse")
        self.classes_table.heading("#0", text="Classe")
        self.classes_table.heading("amostras", text="Amostras")
        self.classes_table.heading("pessoas", text="Pessoas")
        self.classes_table.column("#0", width=140)
        self.classes_table.column("amostras", width=90, anchor="center")
        self.classes_table.column("pessoas", width=110, anchor="center")
        self.classes_table.grid(row=1, column=0, columnspan=4, sticky="nsew")
        self.classes_table.tag_configure("meta", foreground="#1a7f37")
        # Ao clicar numa classe, a barra de progresso mostra a situação dela
        self.classes_table.bind("<<TreeviewSelect>>", lambda event: self.update_progress())

        ttk.Entry(panel, textvariable=self.new_class_name, width=14).grid(row=2, column=0, pady=6, sticky="w")
        ttk.Button(panel, text="Nova", command=self.add_class).grid(row=2, column=1, padx=2)
        ttk.Button(panel, text="Renomear", command=self.rename_class).grid(row=2, column=2, padx=2)
        ttk.Button(panel, text="Excluir", command=self.delete_class).grid(row=2, column=3, padx=2)

        ttk.Label(panel, text="Meta por classe:").grid(row=3, column=0, pady=(10, 0), sticky="w")
        ttk.Spinbox(panel, from_=10, to=500, increment=10, textvariable=self.goal, width=6).grid(
            row=3, column=1, pady=(10, 0), sticky="w")
        # trace_add: chama goal_changed sempre que a meta mudar (setas OU digitando)
        self.goal.trace_add("write", lambda *args: self.goal_changed())

        self.record_button = ttk.Button(panel, text="Gravar amostras", command=self.toggle_recording)
        self.record_button.grid(row=4, column=0, columnspan=2, pady=10, sticky="w")
        self.progress = ttk.Progressbar(panel, length=260, mode="determinate")
        self.progress.grid(row=4, column=2, columnspan=2, pady=10, sticky="w")
        ttk.Label(panel, textvariable=self.message, wraplength=380).grid(
            row=5, column=0, columnspan=4, sticky="w")
        ttk.Checkbutton(panel, text="Salvar imagens (com fundo escondido)",
                        variable=self.save_images).grid(row=6, column=0, columnspan=4, pady=(10, 0), sticky="w")
        ttk.Button(panel, text="Importar fotos…", command=self.import_photos).grid(
            row=7, column=0, columnspan=2, pady=(10, 0), sticky="w")
        ttk.Label(panel, text="(fotos da classe selecionada, da pessoa do código acima)").grid(
            row=8, column=0, columnspan=4, sticky="w")

        self.refresh_classes()

    # ---------- classes ----------
    def selected_class(self):
        """Devolve o nome da classe selecionada na tabela (ou None)."""
        selection = self.classes_table.selection()
        if len(selection) == 0:
            return None
        return self.classes_table.item(selection[0])["text"]

    def refresh_classes(self):
        """Redesenha a tabela de classes com as contagens atuais."""
        selected = self.selected_class()
        table = self.classes_table
        for row in table.get_children():
            table.delete(row)
        goal = self.goal.get()
        for name, info in self.dataset.summary().items():
            people = ", ".join(sorted(info["pessoas"])) if info["pessoas"] else "—"
            tags = ("meta",) if info["amostras"] >= goal else ()
            item = table.insert("", "end", text=name, values=(f"{info['amostras']}/{goal}", people), tags=tags)
            if name == selected:
                table.selection_set(item)
        self.update_progress()

    def update_class_row(self, name):
        """Atualiza só a linha de uma classe na tabela (rápido, dá para fazer a cada amostra)."""
        info = self.dataset.summary()[name]
        goal = self.goal.get()
        people = ", ".join(sorted(info["pessoas"])) if info["pessoas"] else "—"
        for item in self.classes_table.get_children():
            if self.classes_table.item(item)["text"] == name:
                tags = ("meta",) if info["amostras"] >= goal else ()
                self.classes_table.item(item, values=(f"{info['amostras']}/{goal}", people), tags=tags)

    def add_class(self):
        if self.dataset.add_class(self.new_class_name.get()):
            self.new_class_name.set("")
            self.refresh_classes()
        else:
            messagebox.showwarning("Nova classe", "Digite um nome que ainda não exista.")

    def rename_class(self):
        old_name = self.selected_class()
        if old_name is None:
            messagebox.showinfo("Renomear", "Selecione uma classe na tabela.")
        elif self.dataset.rename_class(old_name, self.new_class_name.get()):
            self.new_class_name.set("")
            self.refresh_classes()
        else:
            messagebox.showwarning("Renomear", "Digite o novo nome no campo ao lado de 'Nova'.")

    def delete_class(self):
        name = self.selected_class()
        if name is None:
            messagebox.showinfo("Excluir", "Selecione uma classe na tabela.")
            return
        sure = messagebox.askyesno("Excluir", f"Excluir a classe '{name}' e TODAS as amostras dela?")
        if sure:
            self.dataset.delete_class(name)
            self.refresh_classes()

    # ---------- gravação ----------
    def toggle_recording(self):
        """Começa ou para a gravação da classe selecionada."""
        if self.recording_class is not None:
            self.stop_recording("Gravação interrompida.")
            return
        name = self.selected_class()
        if name is None:
            messagebox.showinfo("Gravar", "Selecione uma classe na tabela.")
        elif not self.camera.is_on():
            messagebox.showinfo("Gravar", "Ligue a câmera primeiro.")
        elif self.valid_person_code() is not None:
            self.recording_class = name
            self.record_button.config(text="Parar")
            self.message.set(f"Gravando '{name}'... faça a expressão e mexa um pouco a cabeça.")

    # ---------- importar fotos ----------
    def valid_person_code(self):
        """Confere o código da pessoa (P01, P02...). Devolve o código ou None."""
        person = self.person_code.get().strip().upper()
        if person.startswith("P") and person[1:].isdigit():
            self.person_code.set(person)
            return person
        messagebox.showwarning("Pessoa", "Use um código de pessoa como P01, P02... (nunca o nome).")
        return None

    def read_photo(self, path):
        """Lê uma foto do disco. Devolve o array BGR ou None.
        np.fromfile + cv2.imdecode funciona mesmo com acentos no caminho (o cv2.imread falha no Windows)."""
        try:
            data = np.fromfile(path, dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except OSError:
            return None
        if image is None:
            return None
        height, width = image.shape[:2]
        if width > MAX_PHOTO_WIDTH:              # foto de celular gigante: reduz para analisar mais rápido
            scale = MAX_PHOTO_WIDTH / width
            image = cv2.resize(image, (MAX_PHOTO_WIDTH, int(height * scale)))
        return image

    def import_photos(self):
        """Escolhe várias fotos e grava uma amostra por foto em que houver rosto."""
        name = self.selected_class()
        if name is None:
            messagebox.showinfo("Importar fotos", "Selecione uma classe na tabela.")
            return
        person = self.valid_person_code()
        if person is None:
            return
        paths = filedialog.askopenfilenames(title=f"Fotos de '{name}' — pessoa {person}",
                                            filetypes=PHOTO_TYPES)
        if not paths:                             # cancelou
            return
        if self.analyzer is None:
            self.message.set("Carregando o MediaPipe…")
            self.root.update()
            self.analyzer = FaceAnalyzer()

        imported, no_face, unreadable = 0, 0, 0
        self.progress["maximum"] = len(paths)
        for number, path in enumerate(paths, start=1):
            image = self.read_photo(path)
            if image is None:
                unreadable += 1
            else:
                face = self.analyzer.analyze_photo(image)
                if face.has_face():
                    self.dataset.save_sample(person, "foto", name, face.blendshapes)
                    if self.save_images.get():
                        self.dataset.save_image(name, self.analyzer.hide_background(image, face))
                    imported += 1
                else:
                    no_face += 1
            self.progress["value"] = number
            self.message.set(f"Importando {number}/{len(paths)}…")
            self.root.update_idletasks()          # redesenha a barra durante a importação

        self.refresh_classes()
        report = (f"Classe '{name}', pessoa {person}:\n"
                  f"✔ {imported} foto(s) importada(s)\n"
                  f"✘ {no_face} sem rosto encontrado\n"
                  f"✘ {unreadable} que não deu para abrir")
        self.message.set(report.replace("\n", " · "))
        messagebox.showinfo("Importar fotos", report)

    def stop_recording(self, text):
        self.recording_class = None
        self.record_button.config(text="Gravar amostras")
        self.message.set(text)
        self.refresh_classes()

    def record_if_needed(self, face, shown_frame):
        """Chamada a cada quadro: grava 1 amostra a cada SAVE_EVERY quadros, se houver rosto."""
        if self.recording_class is None or not face.has_face():
            return
        self.frame_count += 1
        if self.frame_count % SAVE_EVERY != 0:
            return
        name = self.recording_class
        self.dataset.save_sample(self.person_code.get(), "camera", name, face.blendshapes)
        if self.save_images.get():
            self.dataset.save_image(name, shown_frame)
        count = self.dataset.summary()[name]["amostras"]
        self.update_class_row(name)
        self.update_progress(count)
        if count >= self.goal.get():
            self.stop_recording(f"Meta atingida para '{name}' ({count} amostras).")

    def update_progress(self, count=None):
        """Barra de progresso da classe selecionada até a meta."""
        name = self.recording_class or self.selected_class()
        if name is None:
            self.progress["value"] = 0
            return
        if count is None:
            count = self.dataset.summary().get(name, {"amostras": 0})["amostras"]
        self.progress["maximum"] = self.goal.get()
        self.progress["value"] = min(count, self.goal.get())

    def toggle_camera(self):
        """Liga ou desliga a câmera (o mesmo botão faz as duas coisas)."""
        if self.camera.is_on():
            self.camera.turn_off()
            for button in self.camera_buttons:
                button.config(text="Ligar câmera")
            self.camera_status.set("Câmera: desligada")
            labels = [self.video_label, self.test_video_label, self.identity_mode.video_label]
            for label in labels + list(self.door_mode.video_labels.values()):
                label.config(image=self.blank_video)
            return

        if self.camera.turn_on():
            for button in self.camera_buttons:
                button.config(text="Desligar câmera")
            self.camera_status.set("Câmera: ligada")
            self.update_video()                      # começa o ciclo do vídeo
        else:
            self.camera_status.set("Câmera: erro (ligada? outro programa usando?)")

    def update_video(self):
        """Um quadro: ler → analisar → esconder o fundo → mostrar → agendar o próximo."""
        if not self.camera.is_on():
            return                                   # câmera desligada: o ciclo para aqui
        frame = self.camera.read()
        if frame is not None:
            if self.mode.get() == "Identidade" and self.current_tab() in ("5. Cadastrar", "6. Porta"):
                frame, status = self.door_mode.process_frame(frame, self.current_tab())   # Cap. 8
            elif self.mode.get() == "Identidade":
                frame, status = self.identity_mode.process_frame(frame)   # YuNet + SFace
            else:
                frame, status = self.process_expression_frame(frame)      # MediaPipe

            now = time.perf_counter()
            fps = 1 / (now - self.previous_time)
            self.previous_time = now
            self.camera_status.set(f"Câmera: ligada · {fps:.0f} FPS · {status}")

            self.show_frame(frame)
        self.root.after(VIDEO_DELAY_MS, self.update_video)   # "me chame de novo daqui a 15 ms"

    def process_expression_frame(self, frame):
        """Um quadro do modo Expressões (Cap. 6). Devolve (quadro para mostrar, status)."""
        if self.analyzer is None:
            self.camera_status.set("Câmera: carregando o MediaPipe…")
            self.root.update()                       # mostra a mensagem antes de carregar
            self.analyzer = FaceAnalyzer()
        face = self.analyzer.analyze_video_frame(frame)
        if self.hide_background.get():
            frame = self.analyzer.hide_background(frame, face)
        self.record_if_needed(face, frame)
        if self.current_tab() == "4. Testar":
            self.predict(face)
        return frame, "rosto encontrado" if face.has_face() else "sem rosto"

    def show_frame(self, frame):
        """Converte o quadro (BGR do OpenCV) para uma imagem que o Tkinter mostra."""
        small = cv2.resize(frame, (PREVIEW_WIDTH, PREVIEW_HEIGHT))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        # O vídeo aparece na aba que estiver aberta (Coletar, Testar ou Verificar)
        if self.mode.get() == "Identidade" and self.current_tab() in self.door_mode.video_labels:
            label = self.door_mode.video_labels[self.current_tab()]
        elif self.mode.get() == "Identidade":
            label = self.identity_mode.video_label
        elif self.current_tab() == "4. Testar":
            label = self.test_video_label
        else:
            label = self.video_label
        label.config(image=photo)
        label.image = photo              # guarda uma referência, senão a imagem some

    def current_tab(self):
        """O título da aba aberta agora, por exemplo "4. Testar" ("" na tela inicial)."""
        if self.notebook is None:
            return ""
        return self.notebook.tab(self.notebook.select(), "text")

    # ---------- aba 3: Treinar ----------
    def build_train_tab(self):
        """Aba Treinar: ajustes (k e % de teste), botões e o relatório do treino."""
        frame = self.tabs["3. Treinar"]

        settings = ttk.Frame(frame)
        settings.grid(row=1, column=0, sticky="w", pady=(12, 8))
        ttk.Label(settings, text="k (vizinhos):").grid(row=0, column=0)
        ttk.Spinbox(settings, from_=1, to=15, textvariable=self.neighbors, width=4).grid(
            row=0, column=1, padx=(4, 16))
        ttk.Label(settings, text="% para teste:").grid(row=0, column=2)
        ttk.Spinbox(settings, from_=10, to=50, increment=5, textvariable=self.test_percent, width=4).grid(
            row=0, column=3, padx=(4, 16))
        ttk.Button(settings, text="Treinar", command=self.train_model).grid(row=0, column=4)
        self.save_button = ttk.Button(settings, text="Salvar modelo", command=self.save_trained_model,
                                      state="disabled")
        self.save_button.grid(row=0, column=5, padx=(8, 0))

        # Relatório: uma caixa de texto com letra de largura fixa, para as colunas alinharem
        self.report = tk.Text(frame, font="TkFixedFont", wrap="none", height=20)
        self.report.grid(row=2, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(frame, orient="vertical", command=self.report.yview)
        scroll.grid(row=2, column=1, sticky="ns")
        self.report.configure(yscrollcommand=scroll.set)
        frame.rowconfigure(2, weight=1)
        frame.columnconfigure(0, weight=1)
        self.clear_training()

    def show_report(self, text):
        """Troca o texto do relatório (a caixa fica só para leitura)."""
        self.report.configure(state="normal")
        self.report.delete("1.0", "end")
        self.report.insert("1.0", text)
        self.report.configure(state="disabled")

    def clear_training(self):
        self.trained = None
        self.save_button.configure(state="disabled")
        self.show_report("Ajuste k e o % para teste e clique em Treinar.\n"
                         "O relatório mostra as notas, a matriz de confusão e, com 2 ou mais pessoas,\n"
                         "a prova deixando uma pessoa de fora.")

    def train_model(self):
        """Lê as amostras, confere, treina, avalia e mostra o relatório."""
        csv_path = self.dataset.samples_path
        if not csv_path.exists():
            messagebox.showwarning("Sem amostras", "Grave ou importe amostras na aba Coletar primeiro.")
            return
        try:
            neighbors = self.neighbors.get()
            test_percent = self.test_percent.get()
        except tk.TclError:                       # alguém digitou letras no lugar do número
            messagebox.showwarning("Ajustes", "k e % para teste precisam ser números inteiros.")
            return

        feature_names, X, y, people = treino.load_samples(csv_path)
        try:
            treino.check_data(y, neighbors)
            result = treino.split_and_evaluate(X, y, neighbors, test_percent)
        except ValueError as error:               # os erros de check_data e split_and_evaluate
            messagebox.showwarning("Não deu para treinar", str(error))
            return

        people_results = None
        if len(set(people)) >= 2:
            people_results = treino.leave_one_person_out(X, y, people, neighbors)

        self.show_report(self.build_report(y, people, neighbors, test_percent, result, people_results))
        self.trained = {"features": feature_names, "X": X, "y": y, "neighbors": neighbors,
                        "info": {"samples": len(X), "people": sorted(str(p) for p in set(people)),
                                 "test_accuracy": result["test_accuracy"]}}
        self.save_button.configure(state="normal")

    def build_report(self, y, people, neighbors, test_percent, result, people_results):
        """Monta o texto do relatório, parte por parte."""
        lines = []
        lines.append("DADOS")
        lines.append(f"  {len(y)} amostras · pessoas: {', '.join(sorted(str(p) for p in set(people)))}")
        lines.append(f"  {'classe':12}{'amostras':>9}   pessoas")
        for name in result["classes"]:
            in_class = y == name
            names = ", ".join(sorted(str(p) for p in set(people[in_class])))
            lines.append(f"  {str(name):12}{int(in_class.sum()):>9}   {names}")

        lines.append("")
        lines.append(f"SEPARAÇÃO ALEATÓRIA ({test_percent}% para teste, k = {neighbors})")
        lines.append(f"  Treino: {result['train_size']} amostras | Teste: {result['test_size']} amostras")
        lines.append(f"  Acurácia no treino: {result['train_accuracy']:.3f}")
        lines.append(f"  Acurácia no teste:  {result['test_accuracy']:.3f}")

        lines.append("")
        lines.append("MATRIZ DE CONFUSÃO (linha = verdadeiro, coluna = palpite)")
        header = " " * 14
        for name in result["classes"]:
            header += f"{str(name):>10}"
        lines.append(header)
        for name, row in zip(result["classes"], result["matrix"]):
            text = f"  {str(name):12}"
            for value in row:
                text += f"{value:>10}"
            lines.append(text)

        lines.append("")
        if people_results is None:
            lines.append("PESSOA DE FORA: precisa de 2 ou mais pessoas (códigos P01, P02...).")
        else:
            lines.append("PESSOA DE FORA (treina SEM a pessoa e testa SÓ nela)")
            lines.append(f"  {'pessoa':8}{'amostras':>9}{'acertos':>9}{'acurácia':>10}")
            total = 0
            for item in people_results:
                note = f"   (k = {item['k']})" if item["k"] != neighbors else ""
                lines.append(f"  {str(item['person']):8}{item['samples']:>9}{item['hits']:>9}"
                             f"{item['accuracy']:>10.3f}{note}")
                total += item["accuracy"]
            lines.append(f"  Média por pessoa: {total / len(people_results):.3f}")
            lines.append("  Nota bem menor que a do teste aleatório? O modelo reconhece as pessoas")
            lines.append("  que já viu, mas ainda não generaliza para um rosto novo.")
        return "\n".join(lines)

    def save_trained_model(self):
        """Treina de novo com TODAS as amostras e salva em <pasta>/modelo.joblib."""
        if self.trained is None:
            return
        data = self.trained
        model = treino.train_final(data["X"], data["y"], data["neighbors"])
        info = dict(data["info"])
        info["neighbors"] = data["neighbors"]
        path = Path(self.project_folder.get()) / treino.MODEL_FILE
        treino.save_model(path, model, data["features"], info)
        self.load_test_model()                       # a aba Testar já passa a usar o modelo novo
        messagebox.showinfo("Modelo salvo",
                            f"Modelo salvo em:\n{path}\n\n"
                            f"Treinado com todas as {info['samples']} amostras, k = {info['neighbors']}.\n"
                            f"Classes: {', '.join(str(c) for c in model.classes_)}")

    # ---------- aba 4: Testar ----------
    def build_test_tab(self):
        """Aba Testar: vídeo à esquerda; à direita, o modelo, o resultado, o limiar e as barras."""
        frame = self.tabs["4. Testar"]
        self.test_video_label = ttk.Label(frame, image=self.blank_video)
        self.test_video_label.grid(row=1, column=0, pady=8, sticky="nw")
        test_camera_button = ttk.Button(frame, text="Ligar câmera", command=self.toggle_camera)
        test_camera_button.grid(row=2, column=0, sticky="w")
        self.camera_buttons.append(test_camera_button)

        panel = ttk.Frame(frame, padding=(16, 0))
        panel.grid(row=1, column=1, rowspan=2, sticky="nsew")
        frame.columnconfigure(1, weight=1)

        ttk.Button(panel, text="Carregar modelo", command=self.load_test_model).grid(row=0, column=0, sticky="w")
        ttk.Label(panel, textvariable=self.model_status, wraplength=420).grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(4, 12))

        ttk.Label(panel, textvariable=self.result_text, font=self.result_font).grid(
            row=2, column=0, columnspan=3, sticky="w", pady=(0, 12))

        # Limiar de dúvida: um controle deslizante de 0.30 a 0.95
        ttk.Label(panel, text="Limiar de dúvida:").grid(row=3, column=0, sticky="w")
        ttk.Scale(panel, from_=0.3, to=0.95, variable=self.min_confidence, length=200).grid(
            row=3, column=1, sticky="w", padx=8)
        self.threshold_text = ttk.Label(panel, width=5)
        self.threshold_text.grid(row=3, column=2, sticky="w")
        self.min_confidence.trace_add("write", lambda *args: self.show_threshold())
        self.show_threshold()

        self.bars_panel = ttk.Frame(panel)          # as barras são criadas ao carregar o modelo
        self.bars_panel.grid(row=4, column=0, columnspan=3, sticky="w", pady=(16, 0))
        self.bars = {}
        self.load_test_model(silent=True)           # se já existe um modelo salvo, carrega

    def show_threshold(self):
        self.threshold_text.config(text=f"{self.min_confidence.get():.2f}")

    def load_test_model(self, silent=False):
        """Lê <pasta>/modelo.joblib e cria uma barra para cada classe."""
        path = Path(self.project_folder.get()) / treino.MODEL_FILE
        if not path.exists():
            self.test_model = None
            self.model_status.set("Nenhum modelo nesta pasta. Treine e salve na aba Treinar.")
            if not silent:
                messagebox.showwarning("Sem modelo", f"Não encontrei {path}.")
            return
        self.test_model = treino.load_model(path)
        info = self.test_model["info"]
        self.model_status.set(f"Modelo: {info['samples']} amostras, k = {info['neighbors']}, "
                              f"pessoas {', '.join(info['people'])} · salvo em {info['saved_at']}")

        for widget in self.bars_panel.winfo_children():   # apaga as barras do modelo anterior
            widget.destroy()
        self.bars = {}
        for row, name in enumerate(self.test_model["classes"]):
            ttk.Label(self.bars_panel, text=name).grid(row=row, column=0, sticky="w", pady=2)
            bar = ttk.Progressbar(self.bars_panel, length=220, maximum=1.0)
            bar.grid(row=row, column=1, padx=8)
            value = ttk.Label(self.bars_panel, text="0.00", width=5)
            value.grid(row=row, column=2)
            self.bars[name] = (bar, value)
        self.result_text.set("—")

    def predict(self, face):
        """Classifica o quadro: blendshapes na ordem do treino → predict_proba → barras."""
        if self.test_model is None:
            return
        if not face.has_face():
            self.result_text.set("sem rosto")
            for bar, value in self.bars.values():
                bar.config(value=0)
                value.config(text="0.00")
            return

        scores = {}
        for category in face.blendshapes:
            scores[category.category_name] = category.score
        vector = []
        for name in self.test_model["features"]:     # a MESMA ordem usada no treino
            vector.append(scores[name])

        probabilities = self.test_model["model"].predict_proba([vector])[0]
        best = int(np.argmax(probabilities))
        best_name = self.test_model["classes"][best]
        confidence = probabilities[best]
        if confidence >= self.min_confidence.get():
            self.result_text.set(f"{best_name} ({confidence:.2f})")
        else:
            self.result_text.set(f"em dúvida ({confidence:.2f})")

        for name, probability in zip(self.test_model["classes"], probabilities):
            bar, value = self.bars[name]
            bar.config(value=probability)
            value.config(text=f"{probability:.2f}")

    def on_close(self):
        """Fecha com cuidado: desliga a câmera e o MediaPipe, depois a janela."""
        self.camera.turn_off()
        if self.analyzer is not None:
            self.analyzer.close()
        self.root.destroy()

    def goal_changed(self):
        """A meta mudou: redesenha a tabela e guarda o valor no config.json."""
        try:
            goal = self.goal.get()
        except tk.TclError:          # campo vazio ou com letras enquanto a pessoa digita
            return
        self.refresh_classes()
        save_setting("meta", goal)

    def choose_folder(self):
        """Abre a janela do Windows para escolher a pasta do projeto."""
        folder = filedialog.askdirectory(title="Escolha a pasta do projeto",
                                         initialdir=self.project_folder.get())
        if folder:   # se o usuário cancelar, vem um texto vazio
            self.project_folder.set(folder)
            save_setting("pasta_do_projeto", folder)  # na próxima vez, o Estúdio já abre nela
            self.dataset = Dataset(folder)          # carrega as classes e amostras da nova pasta
            self.refresh_classes()
            self.refresh_environment()
            self.clear_training()                   # o treino anterior era de outra pasta
            self.load_test_model()                  # e o modelo também
            self.identity_mode.refresh_people()     # e a lista de pessoas do modo Identidade


def load_config():
    """Lê o config.json como dicionário (vazio se o arquivo não existe ou está quebrado)."""
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_setting(key, value):
    """Guarda UMA configuração no config.json, mantendo as outras."""
    config = load_config()
    config[key] = value
    CONFIG_FILE.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def report_error(error_type, error, error_traceback):
    """Erro inesperado num botão ou no vídeo: grava no estudio.log e avisa numa caixa."""
    text = "".join(traceback.format_exception(error_type, error, error_traceback))
    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(time.strftime("\n=== %Y-%m-%d %H:%M:%S ===\n") + text)
    messagebox.showerror("Erro inesperado", f"{error}\n\nDetalhes em:\n{LOG_FILE}")


def main():
    if IS_EXE:
        # Sem terminal, não há onde imprimir: as mensagens vão para o estudio.log
        log = open(LOG_FILE, "a", encoding="utf-8")
        sys.stdout = log
        sys.stderr = log
    root = tk.Tk()          # cria a janela principal
    root.report_callback_exception = report_error   # erros dentro da janela vão para report_error
    EstudioApp(root)        # monta o Estúdio dentro dela
    root.mainloop()         # laço de eventos: espera cliques até a janela ser fechada


if __name__ == "__main__":
    main()
