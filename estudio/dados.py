# dados.py — a classe Dataset: classes, amostras (CSV) e imagens de um projeto do Estúdio.
#
# Pasta do projeto:
#   amostras.csv      pessoa, origem, classe + 52 blendshapes (mesmo formato do Capítulo 5)
#   classes.json      lista de classes (inclusive as que ainda não têm amostras)
#   imagens/<classe>/ imagens salvas, já com o fundo escondido (opcional)
#   imagens/_excluidas/ pastas de classes excluídas (nada é apagado de vez)

import csv
import json                       # ler e gravar a lista de classes
import shutil                     # mover pastas inteiras
import time
from pathlib import Path

import cv2

SAMPLES_FILE = "amostras.csv"
CLASSES_FILE = "classes.json"
INFO_COLUMNS = ["pessoa", "origem", "classe"]


class Dataset:
    """Tudo o que fica guardado na pasta de um projeto."""

    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.samples_path = self.folder / SAMPLES_FILE
        self.classes_path = self.folder / CLASSES_FILE
        self.classes = self.load_classes()
        self.recount()      # contagens guardadas na memória (ler o CSV a cada quadro deixaria o vídeo lento)

    # ---------- classes ----------
    def load_classes(self):
        """Lê o classes.json; se não existir, descobre as classes pelo CSV."""
        if self.classes_path.exists():
            return json.loads(self.classes_path.read_text(encoding="utf-8"))
        found = []
        for row in self.read_rows():
            if row["classe"] not in found:
                found.append(row["classe"])
        return found

    def save_classes(self):
        self.classes_path.write_text(json.dumps(self.classes, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_class(self, name):
        name = name.strip()
        if name == "" or name in self.classes:
            return False
        self.classes.append(name)
        self.save_classes()
        return True

    def rename_class(self, old_name, new_name):
        """Troca o nome da classe na lista E em todas as amostras do CSV."""
        new_name = new_name.strip()
        if new_name == "" or new_name in self.classes:
            return False
        rows = self.read_rows()
        for row in rows:
            if row["classe"] == old_name:
                row["classe"] = new_name
        self.write_rows(rows)
        old_folder = self.folder / "imagens" / old_name
        if old_folder.exists():                        # as imagens acompanham o novo nome
            self.move_to_trash(new_name)               # sobra antiga com o novo nome? vai para a lixeira
            shutil.move(str(old_folder), str(self.folder / "imagens" / new_name))
        self.classes[self.classes.index(old_name)] = new_name
        self.save_classes()
        self.recount()
        return True

    def delete_class(self, name):
        """Remove a classe e as amostras dela do CSV; as imagens vão para imagens/_excluidas/."""
        rows = []
        for row in self.read_rows():
            if row["classe"] != name:
                rows.append(row)
        self.write_rows(rows)
        self.move_to_trash(name)
        self.classes.remove(name)
        self.save_classes()
        self.recount()

    def move_to_trash(self, name):
        """Move imagens/<classe>/ para imagens/_excluidas/<classe>_<data-hora>/."""
        folder = self.folder / "imagens" / name
        if not folder.exists():
            return None
        trash = self.folder / "imagens" / "_excluidas"
        trash.mkdir(parents=True, exist_ok=True)
        target = trash / (name + "_" + time.strftime("%Y%m%d-%H%M%S"))
        shutil.move(str(folder), str(target))
        return target

    # ---------- amostras ----------
    def read_rows(self):
        """Lê o CSV inteiro como uma lista de dicionários (vazia se não existir)."""
        if not self.samples_path.exists():
            return []
        with open(self.samples_path, newline="", encoding="utf-8") as file:
            return list(csv.DictReader(file))

    def write_rows(self, rows):
        """Regrava o CSV inteiro (usado ao renomear ou excluir classes)."""
        if len(rows) == 0:
            if self.samples_path.exists():
                self.samples_path.unlink()
            return
        with open(self.samples_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def save_sample(self, person, source, class_name, blendshapes):
        """Acrescenta uma amostra ao CSV (cria o cabeçalho na primeira vez)."""
        new_file = not self.samples_path.exists()
        with open(self.samples_path, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if new_file:
                header = list(INFO_COLUMNS)
                for category in blendshapes:
                    header.append(category.category_name)
                writer.writerow(header)
            row = [person, source, class_name]
            for category in blendshapes:
                row.append(round(category.score, 4))
            writer.writerow(row)
        # Atualiza as contagens na memória, sem reler o arquivo
        if class_name not in self.counts:
            self.counts[class_name] = {"amostras": 0, "pessoas": set()}
        self.counts[class_name]["amostras"] += 1
        self.counts[class_name]["pessoas"].add(person)

    def save_image(self, class_name, image):
        """Salva uma imagem (já com o fundo escondido) em imagens/<classe>/."""
        folder = self.folder / "imagens" / class_name
        folder.mkdir(parents=True, exist_ok=True)
        file_name = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}.jpg"
        cv2.imwrite(str(folder / file_name), image)

    def recount(self):
        """Lê o CSV UMA vez e guarda, para cada classe, quantas amostras e quais pessoas."""
        self.counts = self.count_from_file()

    def summary(self):
        """Contagens atuais (da memória): para cada classe, amostras e pessoas."""
        for name in self.classes:
            if name not in self.counts:
                self.counts[name] = {"amostras": 0, "pessoas": set()}
        return self.counts

    def count_from_file(self):
        """Para cada classe: quantas amostras e quais pessoas (lendo o CSV)."""
        result = {}
        for name in self.classes:
            result[name] = {"amostras": 0, "pessoas": set()}
        for row in self.read_rows():
            name = row["classe"]
            if name not in result:            # classe que está no CSV mas não na lista
                result[name] = {"amostras": 0, "pessoas": set()}
            result[name]["amostras"] += 1
            result[name]["pessoas"].add(row["pessoa"])
        return result
