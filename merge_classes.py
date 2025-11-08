import glob
import os
import yaml

from config import output_folder, classes as new_classes


LABELS_DIR = os.path.join(output_folder, "labels")
YAML_PATH = os.path.join(output_folder, "dataset.yaml")

# classe antiga -> classe nova
mapping = {
    0: 0,  # D00 -> D00
    1: 1,  # D10 -> D10
    2: 2,  # D20 -> D20
    3: 6,  # D30 -> other
    4: 3,  # D40 -> D40
    5: 6,  # D50 -> other
    6: 4,  # D60 -> D60
    7: 5,  # D70 -> D70
    8: 6,  # D80 -> other
    9: 6,  # D90 -> other
}

# --------------------
#  atualizar labels 
# --------------------
def merge_yolo_labels():
    print(f"Procurando labels em: {os.path.abspath(LABELS_DIR)}")
    label_files = glob.glob(os.path.join(LABELS_DIR, "**", "*.txt"), recursive=True)
    
    if not label_files:
        print(f"Aviso: Nenhum arquivo .txt encontrado em {LABELS_DIR}. Verifique o caminho.")
        return

    for path in label_files:
        new_lines = []
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cls = int(parts[0])
                    new_cls = mapping.get(cls, cls)
                    parts[0] = str(new_cls)
                    new_lines.append(" ".join(parts))
                except ValueError:
                    print(f"Aviso: Linha mal formatada em {path}: '{line.strip()}'")
        
        with open(path, "w") as f:
            f.write("\n".join(new_lines))
    
    print(f"[OK] Atualizadas {len(label_files)} labels com o novo mapeamento.")


# --------------------
# atualizar YAML 
# --------------------
def update_yaml():
    if not os.path.exists(YAML_PATH):
        print(f"Erro: Arquivo YAML não encontrado em {YAML_PATH}.")
        print("Rode o script voc2yolo.py primeiro.")
        return

    with open(YAML_PATH, "r") as f:
        data = yaml.safe_load(f)

    data["nc"] = len(new_classes)
    data["names"] = new_classes

    with open(YAML_PATH, "w") as f:
        yaml.safe_dump(data, f)

    print(f"[OK] YAML atualizado: {YAML_PATH}")
    print(f"Novas classes: {new_classes}")


if __name__ == "__main__":
    merge_yolo_labels()
    update_yaml()
    print("\n[FINALIZADO] Classes fundidas e YAML atualizado.")