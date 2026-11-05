# modo_identidade.py — o modo "Identidade" do Estúdio (Capítulo 7): abas Comparar, Limiar e Verificar.
# A aba Ambiente é a mesma do modo Expressões e é montada pelo app.py.

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from matplotlib.figure import Figure                                # gráfico...
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg     # ...dentro de uma janela Tkinter

import identidade
from identidade import FaceIdentity

CROP_SIZE = 224                 # o recorte de 112x112 aparece 2 vezes maior
LIGHT_SIZE = 90                 # tamanho da "luz" da aba Verificar
GREEN, RED, GRAY = "#1a9c3a", "#cf222e", "#8c8c8c"
BOX_COLOR = (0, 255, 0)
VIDEO_WIDTH, VIDEO_HEIGHT = 640, 360   # o vídeo é analisado no tamanho em que aparece na tela
DESCRIBE_EVERY = 3                     # descritor a cada 3 quadros (entre eles, repete o último)
PHOTO_TYPES = [("Imagens", "*.jpg *.jpeg *.png *.bmp *.webp"), ("Todos os arquivos", "*.*")]


class IdentityMode:
    """As abas do modo Identidade. Recebe o EstudioApp para usar a câmera, as fontes e o config."""

    def __init__(self, app, config, save_setting):
        self.app = app
        self.save_setting = save_setting             # função do app.py que grava no config.json
        self.identity = None                         # YuNet + SFace: carregados na primeira vez
        default_folder = Path(app.project_folder.get()) / "identidade"
        self.folder = tk.StringVar(value=config.get("pasta_identidade", str(default_folder)))
        self.threshold = tk.DoubleVar(value=config.get("limiar_identidade", identidade.DOC_THRESHOLD))
        self.threshold.trace_add("write", lambda *args: self.threshold_changed())
        self.person = tk.StringVar()
        self.reference = None                        # descritor médio da pessoa escolhida
        self.compare_descriptors = [None, None]
        self.suggested = None                        # limiar sugerido pela aba Limiar
        self.frame_number = 0
        self.last_similarity = None                  # último resultado do descritor (vídeo)

    def get_identity(self):
        """Carrega os modelos só quando forem usados pela primeira vez."""
        if self.identity is None:
            previous = self.app.camera_status.get()
            self.app.camera_status.set("Carregando o YuNet e o SFace…")
            self.app.root.update()
            self.identity = FaceIdentity()
            self.app.camera_status.set(previous)     # volta o texto que estava antes
        return self.identity

    def threshold_changed(self):
        try:
            value = round(self.threshold.get(), 3)
        except tk.TclError:
            return
        self.threshold_text.config(text=f"{value:.2f}")
        self.show_current_threshold()
        self.save_setting("limiar_identidade", value)

    def build(self, tabs):
        self.build_compare_tab(tabs["2. Comparar"])
        self.build_threshold_tab(tabs["3. Limiar"])
        self.build_verify_tab(tabs["4. Verificar"])

    # ---------- aba 2: Comparar ----------
    def build_compare_tab(self, frame):
        self.crop_labels = []
        self.name_labels = []
        for column in range(2):
            ttk.Button(frame, text=f"Escolher foto {column + 1}…",
                       command=lambda c=column: self.choose_compare_photo(c)).grid(
                row=1, column=column, pady=8, padx=8, sticky="w")
            crop = ttk.Label(frame)
            crop.grid(row=2, column=column, padx=8, sticky="nw")
            self.crop_labels.append(crop)
            name = ttk.Label(frame, text="(nenhuma foto)")
            name.grid(row=3, column=column, padx=8, sticky="w")
            self.name_labels.append(name)
        self.compare_result = ttk.Label(frame, text="—", font=self.app.result_font)
        self.compare_result.grid(row=4, column=0, columnspan=2, padx=8, pady=(16, 4), sticky="w")
        self.compare_detail = ttk.Label(frame, text="Escolha as duas fotos.")
        self.compare_detail.grid(row=5, column=0, columnspan=2, padx=8, sticky="w")

    def choose_compare_photo(self, column):
        path = filedialog.askopenfilename(title=f"Foto {column + 1}", filetypes=PHOTO_TYPES)
        if not path:
            return
        image = identidade.read_photo(path)
        if image is None:
            messagebox.showwarning("Comparar", "Não consegui abrir essa foto.")
            return
        aligned, descriptor = self.get_identity().describe_photo(image)
        if descriptor is None:
            messagebox.showwarning("Comparar", "Nenhum rosto encontrado nessa foto.")
            return
        self.compare_descriptors[column] = descriptor
        show_image(self.crop_labels[column], cv2.resize(aligned, (CROP_SIZE, CROP_SIZE)))
        self.name_labels[column].config(text=Path(path).name)
        self.update_comparison()

    def update_comparison(self):
        first, second = self.compare_descriptors
        if first is None or second is None:
            return
        identity = self.get_identity()
        value = identity.similarity(first, second)
        distance = identity.distance(first, second)
        threshold = self.threshold.get()
        if value >= threshold:
            self.compare_result.config(text="MESMA PESSOA", foreground=GREEN)
        else:
            self.compare_result.config(text="PESSOA DIFERENTE", foreground=RED)
        self.compare_detail.config(text=f"similaridade de cosseno {value:.3f} · distância L2 {distance:.3f}"
                                        f" · limiar {threshold:.3f}")

    # ---------- aba 3: Limiar ----------
    def build_threshold_tab(self, frame):
        line = ttk.Frame(frame)
        line.grid(row=1, column=0, columnspan=2, sticky="w", pady=8)
        ttk.Button(line, text="Escolher pasta das pessoas…", command=self.choose_identity_folder).grid(row=0, column=0)
        ttk.Label(line, text="Pasta com uma subpasta por pessoa:").grid(row=0, column=1, padx=(8, 4))
        ttk.Label(line, textvariable=self.folder).grid(row=0, column=2)

        buttons = ttk.Frame(frame)
        buttons.grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Button(buttons, text="Calcular", command=self.calculate_threshold).grid(row=0, column=0)
        self.use_button = ttk.Button(buttons, text="Usar este limiar", state="disabled",
                                     command=self.use_suggested_threshold)
        self.use_button.grid(row=0, column=1, padx=8)
        self.current_threshold = ttk.Label(buttons)
        self.current_threshold.grid(row=0, column=2, padx=8)
        self.show_current_threshold()

        self.threshold_report = tk.Text(frame, font="TkFixedFont", wrap="none", width=62, height=22)
        self.threshold_report.grid(row=3, column=0, sticky="nsew", pady=8)
        self.figure = Figure(figsize=(5.5, 3.6), dpi=100)
        self.chart = FigureCanvasTkAgg(self.figure, master=frame)
        self.chart.get_tk_widget().grid(row=3, column=1, sticky="nsew", padx=8, pady=8)
        frame.rowconfigure(3, weight=1)
        frame.columnconfigure(1, weight=1)
        write_text(self.threshold_report, "Escolha a pasta com UMA SUBPASTA POR PESSOA (P01, P02…)\n"
                                          "e clique em Calcular.")

    def show_current_threshold(self):
        self.current_threshold.config(text=f"Limiar atual: {self.threshold.get():.3f}")

    def choose_identity_folder(self):
        folder = filedialog.askdirectory(title="Pasta com uma subpasta por pessoa",
                                         initialdir=self.folder.get())
        if not folder:
            return
        folder = Path(folder)
        if count_photos(folder) > 0 and count_people(folder) == 0 and count_people(folder.parent) >= 2:
            # Escolheu a pasta de UMA pessoa (ex.: P02): usamos a pasta de cima, que tem P01, P02…
            messagebox.showinfo("Pasta", f"'{folder.name}' é a pasta de uma pessoa só.\n"
                                         f"Vou usar a pasta de cima, com todas as pessoas:\n{folder.parent}")
            folder = folder.parent
        self.folder.set(str(folder))
        self.save_setting("pasta_identidade", str(folder))
        self.refresh_people()

    def calculate_threshold(self):
        folder = Path(self.folder.get())
        if not folder.is_dir():
            messagebox.showwarning("Limiar", f"A pasta não existe:\n{folder}")
            return
        write_text(self.threshold_report, "Calculando os descritores…")
        self.app.root.update()
        identity = self.get_identity()
        samples, problems = identity.load_folder(folder)
        people = sorted(set(person for person, _, _ in samples))
        if len(people) < 2:
            text = ("Não deu para calcular: são precisas fotos de pelo menos 2 pessoas,\n"
                    "cada uma na sua subpasta (P01, P02…).\n\n"
                    f"Pessoas encontradas em {folder.name}: {', '.join(people) or 'nenhuma'}")
            write_text(self.threshold_report, text)
            messagebox.showwarning("Limiar", text)
            return
        result = identity.evaluate(samples)
        self.suggested = identity.suggest_threshold(result)
        write_text(self.threshold_report, self.build_report(samples, problems, people, result))
        self.draw_chart(result)
        self.use_button.config(state="normal", text=f"Usar este limiar ({self.suggested:.3f})")

    def build_report(self, samples, problems, people, result):
        genuine, impostor = result["genuine"], result["impostor"]
        lines = [f"{len(samples)} fotos com rosto, {len(people)} pessoas:"]
        for person in people:
            count = sum(1 for p, _, _ in samples if p == person)
            lines.append(f"  {person}: {count} fotos")
        for problem in problems:
            lines.append(f"  ✘ {problem}")
        lines.append("")
        if len(genuine) > 0:
            lines.append(f"Mesma pessoa:     {len(genuine):4} pares, de {genuine.min():.3f} a {genuine.max():.3f}")
        lines.append(f"Pessoas difer.:   {len(impostor):4} pares, de {impostor.min():.3f} a {impostor.max():.3f}")
        lines.append("")
        lines.append("limiar  falsos aceites    falsas rejeições")
        for row in result["table"]:
            fa = row["false_accepts"]
            fr = row["false_rejects"]
            mark = "  <- doc." if row["threshold"] == identidade.DOC_THRESHOLD else ""
            fa_rate = fa / len(impostor) if len(impostor) else 0
            fr_rate = fr / len(genuine) if len(genuine) else 0
            lines.append(f"{row['threshold']:6.3f}  {fa:4} ({fa_rate:6.1%})    {fr:4} ({fr_rate:6.1%}){mark}")
        lines.append("")
        if result["gap"] is not None:
            low, high = result["gap"]
            lines.append(f"Folga: de {low:.3f} a {high:.3f} (sem erros).")
        else:
            lines.append("Sem folga: os grupos se misturam.")
        lines.append(f"Sugestão: {self.suggested:.3f}")
        lines.append("")
        lines.append("Pessoas diferentes mais parecidas:")
        for value, pair in result["closest_impostors"]:
            lines.append(f"  {value:.3f}  {pair}")
        lines.append("Mesma pessoa menos parecida:")
        for value, pair in result["farthest_genuine"]:
            lines.append(f"  {value:.3f}  {pair}")
        return "\n".join(lines)

    def draw_chart(self, result):
        """As duas "montanhas" de similaridade, como no Passo 4, agora dentro da janela."""
        self.figure.clear()
        axes = self.figure.add_subplot()
        bins = np.linspace(-0.3, 1.0, 53)
        axes.hist(result["impostor"], bins=bins, alpha=0.6, color="tab:red", label="pessoas diferentes")
        axes.hist(result["genuine"], bins=bins, alpha=0.6, color="tab:green", label="mesma pessoa")
        axes.axvline(identidade.DOC_THRESHOLD, color="black", linestyle="--", label="documentação")
        axes.axvline(self.suggested, color="tab:blue", linestyle="-", label="sugestão")
        axes.set_xlabel("similaridade de cosseno")
        axes.set_ylabel("pares")
        axes.legend(fontsize=8)
        self.figure.tight_layout()
        self.chart.draw()

    def use_suggested_threshold(self):
        self.threshold.set(self.suggested)
        self.show_current_threshold()
        messagebox.showinfo("Limiar", f"Limiar de identidade agora é {self.suggested:.3f}.")

    # ---------- aba 4: Verificar ----------
    def build_verify_tab(self, frame):
        self.video_label = ttk.Label(frame, image=self.app.blank_video)
        self.video_label.grid(row=1, column=0, pady=8, sticky="nw")
        camera_button = ttk.Button(frame, text="Ligar câmera", command=self.app.toggle_camera)
        camera_button.grid(row=2, column=0, sticky="w")
        self.app.camera_buttons.append(camera_button)

        panel = ttk.Frame(frame, padding=(16, 0))
        panel.grid(row=1, column=1, rowspan=2, sticky="nsew")
        frame.columnconfigure(1, weight=1)

        ttk.Label(panel, text="Pessoa:").grid(row=0, column=0, sticky="w")
        self.person_box = ttk.Combobox(panel, textvariable=self.person, state="readonly", width=10)
        self.person_box.grid(row=0, column=1, sticky="w", padx=4)
        self.person_box.bind("<<ComboboxSelected>>", lambda event: self.load_reference())
        ttk.Button(panel, text="Gerar referência", command=self.make_reference).grid(row=0, column=2, padx=4)
        self.reference_status = ttk.Label(panel, text="", wraplength=420)
        self.reference_status.grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 12))

        self.light = tk.Canvas(panel, width=LIGHT_SIZE, height=LIGHT_SIZE, highlightthickness=0)
        self.light.grid(row=2, column=0, sticky="w")
        self.light_circle = self.light.create_oval(4, 4, LIGHT_SIZE - 4, LIGHT_SIZE - 4, fill=GRAY, outline="")
        self.verify_result = ttk.Label(panel, text="—", font=self.app.result_font)
        self.verify_result.grid(row=2, column=1, columnspan=2, sticky="w", padx=8)
        self.verify_detail = ttk.Label(panel, text="")
        self.verify_detail.grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 12))

        ttk.Label(panel, text="Limiar:").grid(row=4, column=0, sticky="w")
        ttk.Scale(panel, from_=0.05, to=0.95, variable=self.threshold, length=200).grid(
            row=4, column=1, sticky="w")
        self.threshold_text = ttk.Label(panel, text=f"{self.threshold.get():.2f}", width=5)
        self.threshold_text.grid(row=4, column=2, sticky="w")
        self.refresh_people()

    def refresh_people(self):
        """A lista de pessoas = as subpastas da pasta de identidade."""
        folder = Path(self.folder.get())
        people = []
        if folder.is_dir():
            for item in sorted(folder.iterdir()):
                if item.is_dir() and not item.name.startswith("_"):
                    people.append(item.name)
        self.person_box.config(values=people)
        if self.person.get() not in people:
            self.person.set(people[0] if people else "")
        self.load_reference()

    def reference_path(self):
        return Path(self.folder.get()) / f"{self.person.get()}.npy"

    def load_reference(self):
        self.reference = None
        self.last_similarity = None
        if self.person.get() == "":
            self.reference_status.config(text="Nenhuma pessoa: escolha a pasta na aba Limiar.")
        elif self.reference_path().exists():
            self.reference = np.load(self.reference_path())
            self.reference_status.config(text=f"Referência carregada: {self.reference_path().name}")
        else:
            self.reference_status.config(text="Sem referência: clique em Gerar referência.")

    def make_reference(self):
        if self.person.get() == "":
            return
        self.reference_status.config(text="Calculando a média das fotos…")
        self.app.root.update()
        person_folder = Path(self.folder.get()) / self.person.get()
        reference, count = self.get_identity().build_reference(person_folder)
        if reference is None:
            messagebox.showwarning("Referência", f"Nenhuma foto com rosto em {person_folder}.")
            return
        np.save(self.reference_path(), reference)
        self.reference = reference
        self.reference_status.config(text=f"Referência: média de {count} fotos, salva em {self.reference_path().name}")

    def process_frame(self, frame):
        """Um quadro do vídeo no modo Identidade. Devolve (quadro para mostrar, texto do status).
        Para ficar rápido: detecta numa cópia PEQUENA do quadro (a mesma que vai para a tela)
        e só calcula o descritor a cada DESCRIBE_EVERY quadros, no quadro grande."""
        identity = self.get_identity()
        small = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
        scale = frame.shape[1] / VIDEO_WIDTH                 # 1280 / 640 = 2
        faces = identity.detect(small, identidade.VIDEO_SCORE)
        shown = identity.hide_background(small, faces) if self.app.hide_background.get() else small.copy()
        for face in faces:
            x, y, w, h = face[0:4].astype(int)
            cv2.rectangle(shown, (x, y), (x + w, y + h), BOX_COLOR, 2)

        self.frame_number += 1
        if len(faces) == 0:
            self.last_similarity = None
            self.set_light(GRAY, "sem rosto", "")
        elif self.reference is None:
            self.set_light(GRAY, "sem referência", "Gere a referência da pessoa.")
        else:
            if self.last_similarity is None or self.frame_number % DESCRIBE_EVERY == 0:
                face = faces[0].copy()
                face[0:14] = face[0:14] * scale              # caixa e 5 pontos de volta ao tamanho original
                _, descriptor = identity.describe(frame, face)
                self.last_similarity = identity.similarity(descriptor, self.reference)
            value = self.last_similarity
            threshold = self.threshold.get()
            detail = f"similaridade {value:.2f} · limiar {threshold:.2f}"
            if value >= threshold:
                self.set_light(GREEN, f"{self.person.get()} confirmado", detail)
            else:
                self.set_light(RED, f"não é o {self.person.get()}", detail)
        status = "rosto encontrado" if len(faces) > 0 else "sem rosto"
        return shown, status

    def set_light(self, color, text, detail):
        self.light.itemconfig(self.light_circle, fill=color)
        self.verify_result.config(text=text, foreground=color)
        self.verify_detail.config(text=detail)


def count_photos(folder):
    """Quantas fotos há diretamente nesta pasta (sem olhar as subpastas)."""
    return sum(1 for path in folder.iterdir() if path.suffix.lower() in identidade.PHOTO_EXTENSIONS)


def count_people(folder):
    """Quantas subpastas de pessoa há nesta pasta (as que começam com "_" não contam)."""
    return sum(1 for path in folder.iterdir() if path.is_dir() and not path.name.startswith("_"))


def show_image(label, image_bgr):
    """Mostra uma imagem do OpenCV num rótulo do Tkinter (e guarda a referência)."""
    photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)))
    label.config(image=photo)
    label.image = photo


def write_text(box, text):
    box.configure(state="normal")
    box.delete("1.0", "end")
    box.insert("1.0", text)
    box.configure(state="disabled")
