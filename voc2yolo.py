import os
import xml.etree.ElementTree as ET
import shutil
import yaml
from sklearn.model_selection import train_test_split

from config import output_folder, original_classes, countries, val_ratio, test_ratio


def convert_bbox(size, box):
    """
    Converte bbox Pascal VOC para YOLO
    
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def process_files(folder, file_list, split, target_class=None):
    jumped_files = 0

    if not file_list:
        print(f" {split} - Nenhum arquivo para processar {'para a classe '+target_class if target_class else''}")
        return

    for xml_file in file_list:
        if not xml_file:
            continue

        tree = ET.parse(os.path.join(annotations_folder, xml_file))
        root = tree.getroot()

        objs = root.findall("object")
        if target_class is not None:
            objs = [obj for obj in objs if obj.find("name").text == target_class]

        if not objs:
            jumped_files += 1
            continue

        filename = root.find("filename").text
        img_path = os.path.join(folder, filename)

        if not os.path.exists(img_path):
            print(f"Aviso: Imagem {img_path} não encontrada, pulando arquivo {xml_file}")
            jumped_files += 1
            continue


        out_img_dir = os.path.join(output_folder, "images", split)
        out_label_dir = os.path.join(output_folder, "labels", split)

        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_label_dir, exist_ok=True)

        shutil.copy(img_path, os.path.join(out_img_dir, filename))

        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)

        txt_file = os.path.join(out_label_dir, filename.replace(".jpg", ".txt"))
        with open(txt_file, "w") as f:
            for obj in objs:
                cls = obj.find("name").text

                # Lida com classes não esperadas
                if cls not in original_classes:
                    print(f"Aviso: Classe '{cls}' em {xml_file} não está em 'original_classes'. Pulando objeto.")
                    continue

                cls_id = original_classes.index(cls)
                bndbox = obj.find("bndbox")
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)
                x, y, w_box, h_box = convert_bbox((w,h), (xmin, xmax, ymin, ymax))
                f.write(f"{cls_id} {x:.6f} {y:.6f} {w_box:.6f} {h_box:.6f}\n")

    print(f" {split} - Processados. {jumped_files} arquivos pulados {'da classe '+target_class if target_class else''}")


if __name__ == '__main__':
    # -------------------------
    # Cria estrutura de pastas
    # -------------------------
    print(f"Classes Originais: {original_classes}")

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_folder, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_folder, "labels", split), exist_ok=True)

    for country in countries:
        base_path = f"../raw/train/{country}"
        images_folder = f"{base_path}/images"
        annotations_folder = f"{base_path}/annotations"

        if not os.path.isdir(annotations_folder):
            print(f"Aviso: Pasta de anotações não encontrada: {annotations_folder}. Pulando país {country}.")
            continue

        xml_files = [f for f in os.listdir(annotations_folder) if f.endswith(".xml")]

        if not xml_files:
            print(f"Aviso: Nenhuma anotação .xml encontrada em {annotations_folder}. Pulando país {country}.")
            continue

        # split para train/val/test
        temp_size = val_ratio + test_ratio

        if temp_size > 0 and temp_size < 1.0:
            train_files, temp_files = train_test_split(xml_files, test_size=temp_size, random_state=42)
            
            # Evita divisão por zero se temp_size for > 0 mas um dos ratios for 0
            if temp_size == 0:
                 test_split_ratio = 0
            else:
                 test_split_ratio = test_ratio / temp_size

            if test_split_ratio > 0 and test_split_ratio < 1.0:
                val_files, test_files = train_test_split(temp_files, test_size=test_split_ratio, random_state=42)
            elif test_split_ratio == 1.0:
                val_files = []
                test_files = temp_files
            else:
                val_files = temp_files
                test_files = []

        elif temp_size == 1.0:
            train_files = []
            test_split_ratio = test_ratio / temp_size
            if test_split_ratio > 0 and test_split_ratio < 1.0:
                val_files, test_files = train_test_split(xml_files, test_size=test_split_ratio, random_state=42)
            elif test_split_ratio == 1.0:
                val_files = []
                test_files = xml_files
            else:
                val_files = xml_files
                test_files = []
        else:
            train_files = xml_files
            val_files = []
            test_files = []

        print(f"--- Processando País: {country} ---")
        print(f"Total: {len(xml_files)}, Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

        process_files(images_folder, train_files, "train")
        process_files(images_folder, val_files, "val")
        process_files(images_folder, test_files, "test")

    # -------------------------
    # Cria arquivos YAML
    # -------------------------
    dataset_yaml = {
        'train': os.path.abspath(os.path.join(output_folder, 'images', 'train')),
        'val': os.path.abspath(os.path.join(output_folder, 'images', 'val')),
        'test': os.path.abspath(os.path.join(output_folder, 'images', 'test')),
        'nc': len(original_classes),
        'names': original_classes
    }
    
    yaml_path = os.path.join(output_folder, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f, sort_keys=False)

    print("\nConcluído! Dataset (com classes originais) pronto para YOLO.")
    print(f"Verifique a pasta: {os.path.abspath(output_folder)}")
    print(f"Arquivo YAML criado em: {os.path.abspath(yaml_path)}")