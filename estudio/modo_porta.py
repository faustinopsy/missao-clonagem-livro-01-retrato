# modo_porta.py — Capítulo 8: as abas "5. Cadastrar" e "6. Porta" do modo Identidade.
# Cadastrar: poses guiadas pela câmera (ou fotos) -> descritores na galeria (sem imagens).
# Porta: identificação 1:N ao vivo, com confirmação e registro de acesso (sem imagens).

import csv
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

import identidade

VIDEO_WIDTH, VIDEO_HEIGHT = 640, 360   # analisar no tamanho da tela (lição do Cap. 7)
DESCRIBE_EVERY = 3                     # descritor a cada 3 quadros
CONFIRM_COUNT = 5                      # a mesma resposta em 5 descritores seguidos para valer
LOG_ROWS = 8                           # quantas tentativas aparecem na tabela do registro
POSES = [
    ("frente", "De frente, expressão neutra"),
    ("esquerda", "Vire um pouco o rosto para a SUA esquerda"),
    ("direita", "Vire um pouco o rosto para a SUA direita"),
    ("cima", "Olhe um pouco para cima"),
    ("baixo", "Olhe um pouco para baixo"),
    ("sorriso", "De frente, sorrindo"),
    ("perto", "Chegue mais perto da câmera"),
    ("longe", "Afaste-se um pouco da câmera"),
]
GREEN, RED, YELLOW, GRAY = "#1a9c3a", "#cf222e", "#c99a00", "#8c8c8c"
BOX_GREEN, BOX_RED = (0, 200, 0), (0, 0, 230)
DOOR_BROWN, FRAME_GRAY = "#7a4a28", "#5a5a5a"


class DoorMode:
    """As abas Cadastrar e Porta. Usa a câmera do EstudioApp e o FaceIdentity do modo Identidade."""

    def __init__(self, app, identity_mode, config, save_setting):
        self.app = app
        self.identity_mode = identity_mode          # para usar o mesmo FaceIdentity e o mesmo limiar
        self.save_setting = save_setting
        default_folder = Path(app.project_folder.get()) / "galeria"
        self.gallery_folder = tk.StringVar(value=config.get("pasta_galeria", str(default_folder)))
        self.gallery = {}
        self.pose_index = 0
        self.last_frame = None                      # último quadro e rostos, para o botão Capturar
        self.last_faces = []
        self.last_quality = (False, "")
        self.frame_number = 0
        self.ranking = []
        self.candidate = None
        self.candidate_count = 0
        self.answer = ("SEM ROSTO", "-")
        self.video_labels = {}
        self.enroll_frames = 0
        self.feedback_until = 0                     # até que quadro a mensagem de captura fica na tela

    def build(self, tabs):
        self.build_enroll_tab(tabs["5. Cadastrar"])
        self.build_door_tab(tabs["6. Porta"])
        self.reload_gallery()

    def add_video(self, frame, tab):
        """Rótulo do vídeo + botão da câmera (o mesmo jeito das outras abas)."""
        label = ttk.Label(frame, image=self.app.blank_video)
        label.grid(row=1, column=0, pady=8, sticky="nw")
        self.video_labels[tab] = label
        button = ttk.Button(frame, text="Ligar câmera", command=self.app.toggle_camera)
        button.grid(row=2, column=0, sticky="nw")
        self.app.camera_buttons.append(button)
        panel = ttk.Frame(frame, padding=(16, 0))
        panel.grid(row=1, column=1, rowspan=2, sticky="nsew")
        frame.columnconfigure(1, weight=1)
        return panel

    # ---------- galeria (pasta) ----------
    def reload_gallery(self):
        self.gallery = identidade.load_gallery(self.gallery_folder.get())
        table = self.gallery_table
        for row in table.get_children():
            table.delete(row)
        for code, poses in self.gallery.items():
            table.insert("", "end", text=code, values=(len(poses), ", ".join(poses)))

    def choose_gallery_folder(self):
        folder = filedialog.askdirectory(title="Pasta da galeria", initialdir=self.gallery_folder.get())
        if folder:
            self.gallery_folder.set(folder)
            self.save_setting("pasta_galeria", folder)
            self.reload_gallery()

    def log_path(self):
        """O registro fica ao lado da galeria: <pasta de cima>/porta/registro.csv."""
        return Path(self.gallery_folder.get()).parent / "porta" / "registro.csv"

    # ---------- aba 5: Cadastrar ----------
    def build_enroll_tab(self, frame):
        panel = self.add_video(frame, "5. Cadastrar")
        line = ttk.Frame(panel)
        line.grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Button(line, text="Escolher pasta da galeria…", command=self.choose_gallery_folder).grid(
            row=0, column=0, sticky="w")
        ttk.Label(line, textvariable=self.gallery_folder, wraplength=420).grid(row=1, column=0, sticky="w")
        ttk.Label(panel, text="Agente = o código da barra superior (Pessoa).").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(8, 4))

        self.pose_text = ttk.Label(panel, text="", font=self.app.result_font, wraplength=520)
        self.pose_text.grid(row=2, column=0, columnspan=3, sticky="w", pady=4)
        self.quality_text = ttk.Label(panel, text="Ligue a câmera.")
        self.quality_text.grid(row=3, column=0, columnspan=3, sticky="w")
        buttons = ttk.Frame(panel)
        buttons.grid(row=4, column=0, columnspan=3, sticky="w", pady=8)
        ttk.Button(buttons, text="Capturar pose", command=self.capture_pose).grid(row=0, column=0)
        ttk.Button(buttons, text="Pular", command=self.skip_pose).grid(row=0, column=1, padx=6)
        ttk.Button(buttons, text="Recomeçar", command=self.restart_poses).grid(row=0, column=2)
        ttk.Button(buttons, text="Cadastrar de uma pasta de fotos…", command=self.enroll_from_photos).grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self.poses_status = ttk.Label(panel, text="", font="TkFixedFont")
        self.poses_status.grid(row=5, column=0, columnspan=3, sticky="w", pady=(0, 10))

        ttk.Label(panel, text="Galeria (só descritores, nenhuma imagem):").grid(row=6, column=0, sticky="w")
        self.gallery_table = ttk.Treeview(panel, columns=("poses", "nomes"), height=6)
        self.gallery_table.heading("#0", text="Agente")
        self.gallery_table.heading("poses", text="Poses")
        self.gallery_table.heading("nomes", text="Nomes das poses")
        self.gallery_table.column("#0", width=90)
        self.gallery_table.column("poses", width=60, anchor="center")
        self.gallery_table.column("nomes", width=300)
        self.gallery_table.grid(row=7, column=0, columnspan=3, sticky="nsew")
        ttk.Button(panel, text="Descadastrar agente selecionado", command=self.unenroll).grid(
            row=8, column=0, sticky="w", pady=6)
        self.show_pose()

    def show_pose(self):
        if self.pose_index < len(POSES):
            name, instruction = POSES[self.pose_index]
            self.pose_text.config(text=f"Pose {self.pose_index + 1}/{len(POSES)}: {instruction}")
        else:
            self.pose_text.config(text="Cadastro completo! ✔")
        lines = []
        for number, (name, _) in enumerate(POSES):
            mark = "✔" if number < self.pose_index else ("➜" if number == self.pose_index else "·")
            lines.append(f"{mark} {name}")
        # 4 poses por linha, para caber no painel
        self.poses_status.config(text="   ".join(lines[:4]) + "\n" + "   ".join(lines[4:]))

    def agent_folder(self):
        """Pasta do agente na galeria, a partir do código da barra superior (ou None)."""
        code = self.app.valid_person_code()
        if code is None:
            return None
        return Path(self.gallery_folder.get()) / code

    def capture_pose(self):
        if not self.app.camera.is_on():
            messagebox.showinfo("Cadastrar", "Ligue a câmera primeiro.")
            return
        if self.pose_index >= len(POSES):
            messagebox.showinfo("Cadastrar", "Todas as poses já foram capturadas. Use Recomeçar para refazer.")
            return
        folder = self.agent_folder()
        if folder is None:
            return
        ok, text = self.last_quality
        if not ok:
            self.show_feedback(f"Pose recusada: {text}", RED)
            return
        if self.pose_index == 0 and any(folder.glob("*.npy")):
            if not messagebox.askyesno("Cadastrar", f"{folder.name} já está na galeria. Refazer o cadastro?\n"
                                                    "(os descritores antigos serão apagados)"):
                return
            for old_file in folder.glob("*.npy"):
                old_file.unlink()
        _, descriptor = self.identity_mode.get_identity().describe(self.last_frame, self.last_faces[0])
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / f"{POSES[self.pose_index][0]}.npy", descriptor)
        self.show_feedback(f"Pose '{POSES[self.pose_index][0]}' cadastrada! ✔", GREEN)
        self.pose_index += 1
        self.show_pose()
        self.reload_gallery()

    def show_feedback(self, text, color):
        """Mensagem da captura: fica na tela por uns 45 quadros (~1,5 s)."""
        self.quality_text.config(text=text, foreground=color)
        self.feedback_until = self.enroll_frames + 45

    def skip_pose(self):
        if self.pose_index < len(POSES):
            self.pose_index += 1
            self.show_pose()

    def restart_poses(self):
        self.pose_index = 0
        self.show_pose()

    def enroll_from_photos(self):
        folder = self.agent_folder()
        if folder is None:
            return
        photos = filedialog.askdirectory(title=f"Fotos do agente {folder.name}")
        if not photos:
            return
        if any(folder.glob("*.npy")) and not messagebox.askyesno(
                "Cadastrar", f"{folder.name} já está na galeria. Refazer com estas fotos?"):
            return
        for old_file in folder.glob("*.npy"):
            old_file.unlink()
        identity = self.identity_mode.get_identity()
        accepted, refused = 0, []
        for path in sorted(Path(photos).iterdir()):
            if path.suffix.lower() not in identidade.PHOTO_EXTENSIONS:
                continue
            image = identidade.read_photo(path)
            faces = [] if image is None else identity.detect(image, identidade.PHOTO_SCORE)
            ok, text = identity.check_quality(faces)
            if not ok:
                refused.append(f"{path.name}: {text}")
                continue
            _, descriptor = identity.describe(image, faces[0])
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder / f"{path.stem}.npy", descriptor)
            accepted += 1
        self.reload_gallery()
        report = f"{folder.name}: {accepted} foto(s) cadastrada(s)."
        if refused:
            report += "\n\nRecusadas:\n" + "\n".join(refused)
        messagebox.showinfo("Cadastrar", report)

    def unenroll(self):
        """Descadastrar = apagar os descritores do agente de verdade (LGPD)."""
        selection = self.gallery_table.selection()
        if len(selection) == 0:
            messagebox.showinfo("Descadastrar", "Selecione um agente na tabela.")
            return
        code = self.gallery_table.item(selection[0])["text"]
        if not messagebox.askyesno("Descadastrar", f"Apagar TODOS os descritores de {code}?"):
            return
        folder = Path(self.gallery_folder.get()) / code
        for old_file in folder.glob("*.npy"):
            old_file.unlink()
        if not any(folder.iterdir()):
            folder.rmdir()                                  # pasta vazia: apaga também
        self.reload_gallery()

    def process_enroll_frame(self, frame):
        identity = self.identity_mode.get_identity()
        small = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
        scale = frame.shape[1] / VIDEO_WIDTH
        faces = identity.detect(small, identidade.VIDEO_SCORE)
        self.last_quality = identity.check_quality(faces, scale)
        self.last_frame = frame
        self.last_faces = []
        for face in faces:
            big = face.copy()
            big[0:14] = big[0:14] * scale                    # caixa e pontos no tamanho original
            self.last_faces.append(big)
        shown = identity.hide_background(small, faces) if self.app.hide_background.get() else small.copy()
        ok, text = self.last_quality
        for face in faces:
            x, y, w, h = face[0:4].astype(int)
            cv2.rectangle(shown, (x, y), (x + w, y + h), BOX_GREEN if ok else BOX_RED, 2)
        self.enroll_frames += 1
        if self.enroll_frames > self.feedback_until:
            self.quality_text.config(text=text, foreground=GREEN if ok else RED)
        return shown, "rosto encontrado" if len(faces) > 0 else "sem rosto"

    # ---------- aba 6: Porta ----------
    def build_door_tab(self, frame):
        panel = self.add_video(frame, "6. Porta")
        self.door = tk.Canvas(panel, width=220, height=260, highlightthickness=0, background="#1e1e1e")
        self.door.grid(row=0, column=0, rowspan=4, sticky="nw")
        info = ttk.Frame(panel, padding=(16, 0))
        info.grid(row=0, column=1, sticky="nw")
        self.door_answer = ttk.Label(info, text="SEM ROSTO", font=self.app.result_font, foreground=GRAY)
        self.door_answer.grid(row=0, column=0, sticky="w")
        self.door_code = ttk.Label(info, text="-", font="TkHeadingFont")
        self.door_code.grid(row=1, column=0, sticky="w")
        self.door_progress = ttk.Label(info, text="", foreground=YELLOW)
        self.door_progress.grid(row=2, column=0, sticky="w", pady=(4, 8))
        self.door_ranking = ttk.Label(info, text="", font="TkFixedFont")
        self.door_ranking.grid(row=3, column=0, sticky="w")
        limiar = ttk.Frame(info)
        limiar.grid(row=4, column=0, sticky="w", pady=8)
        ttk.Label(limiar, text="Limiar:").grid(row=0, column=0)
        ttk.Scale(limiar, from_=0.05, to=0.95, variable=self.identity_mode.threshold, length=160).grid(
            row=0, column=1, padx=6)
        self.threshold_text = ttk.Label(limiar, width=5)        # o valor do limiar, ao lado do controle
        self.threshold_text.grid(row=0, column=2)
        self.identity_mode.threshold.trace_add("write", lambda *args: self.show_threshold())
        self.show_threshold()
        ttk.Button(info, text="Recarregar galeria", command=self.reload_gallery).grid(row=5, column=0, sticky="w")

        ttk.Label(panel, text="Registro de acesso (sem imagens):").grid(row=4, column=0, columnspan=2,
                                                                       sticky="w", pady=(12, 0))
        self.log_table = ttk.Treeview(panel, columns=("codigo", "similaridade", "resposta"), height=LOG_ROWS)
        for column, title, width in (("#0", "Data e hora", 190), ("codigo", "Código", 110),
                                     ("similaridade", "Similaridade", 100), ("resposta", "Resposta", 140)):
            self.log_table.heading(column, text=title)
            self.log_table.column(column, width=width)
        self.log_table.grid(row=5, column=0, columnspan=2, sticky="nsew")
        self.draw_door("SEM ROSTO")

    def show_threshold(self):
        try:
            self.threshold_text.config(text=f"{self.identity_mode.threshold.get():.2f}")
        except tk.TclError:
            pass

    def draw_door(self, answer):
        """A porta desenhada no Canvas: fechada (com maçaneta) ou aberta, e a luz."""
        canvas = self.door
        canvas.delete("all")
        color = {"PORTA ABERTA": GREEN, "PORTA FECHADA": RED, "TENTE DE NOVO": YELLOW}.get(answer, GRAY)
        canvas.create_oval(95, 6, 125, 36, fill=color, outline="")
        canvas.create_rectangle(40, 46, 180, 250, outline=FRAME_GRAY, width=5)
        if answer == "PORTA ABERTA":
            canvas.create_polygon(44, 50, 85, 68, 85, 232, 44, 246, fill=DOOR_BROWN)
        else:
            canvas.create_rectangle(44, 50, 176, 246, fill=DOOR_BROWN, outline="")
            canvas.create_oval(155, 142, 165, 152, fill="#e6c200", outline="")

    def write_log(self, code, value, answer):
        path = self.log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists()
        moment = datetime.now().isoformat(sep=" ", timespec="seconds")
        with open(path, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if new_file:
                writer.writerow(["data_hora", "codigo", "similaridade", "resposta"])
            writer.writerow([moment, code, f"{value:.3f}", answer])
        self.log_table.insert("", 0, text=moment, values=(code, f"{value:.3f}", answer))   # 0 = no topo
        rows = self.log_table.get_children()
        if len(rows) > LOG_ROWS:
            self.log_table.delete(rows[-1])

    def process_door_frame(self, frame):
        identity = self.identity_mode.get_identity()
        small = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
        scale = frame.shape[1] / VIDEO_WIDTH
        faces = identity.detect(small, identidade.VIDEO_SCORE)
        self.frame_number += 1
        if len(self.gallery) == 0:
            self.door_progress.config(text="Galeria vazia: cadastre agentes na aba 5.")
        elif len(faces) == 0:
            self.ranking = []
            self.candidate, self.candidate_count = None, 0
            self.answer = ("SEM ROSTO", "-")                 # sem rosto: fecha na hora
        elif len(self.ranking) == 0 or self.frame_number % DESCRIBE_EVERY == 0:
            face = faces[0].copy()
            face[0:14] = face[0:14] * scale
            _, query = identity.describe(frame, face)
            self.ranking = identity.identify(query, self.gallery)
            new_answer = identidade.decide(self.ranking, self.identity_mode.threshold.get())
            if self.candidate == new_answer:
                self.candidate_count += 1
            else:
                self.candidate, self.candidate_count = new_answer, 1
            if self.candidate_count >= CONFIRM_COUNT and self.candidate != self.answer:
                self.answer = self.candidate
                self.write_log(self.answer[1], self.ranking[0][0], self.answer[0])
        self.show_door_state()

        shown = identity.hide_background(small, faces) if self.app.hide_background.get() else small.copy()
        box = BOX_GREEN if self.answer[0] == "PORTA ABERTA" else BOX_RED
        for face in faces:
            x, y, w, h = face[0:4].astype(int)
            cv2.rectangle(shown, (x, y), (x + w, y + h), box, 2)
        return shown, "rosto encontrado" if len(faces) > 0 else "sem rosto"

    def show_door_state(self):
        answer, code = self.answer
        color = {"PORTA ABERTA": GREEN, "PORTA FECHADA": RED, "TENTE DE NOVO": YELLOW}.get(answer, GRAY)
        self.door_answer.config(text=answer, foreground=color)
        self.door_code.config(text=code)
        self.draw_door(answer)
        if self.candidate is not None and self.candidate != self.answer:
            self.door_progress.config(text=f"confirmando {self.candidate[1]}: {self.candidate_count}/{CONFIRM_COUNT}")
        elif len(self.gallery) > 0:
            self.door_progress.config(text="")
        lines = []
        for value, agent in self.ranking[:4]:
            lines.append(f"{agent:6} {value:5.2f}")
        self.door_ranking.config(text="\n".join(lines))

    def process_frame(self, frame, tab):
        if tab == "5. Cadastrar":
            return self.process_enroll_frame(frame)
        return self.process_door_frame(frame)
