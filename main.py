import cv2
import os
import torch
from ultralytics import YOLO

from config import epochs, batch, output_folder, MODELS_TO_TRAIN

def get_imgsz(images_folder):
    try:
        first_image_name = os.listdir(images_folder)[0]
        first_image_path = os.path.join(images_folder, first_image_name)
        img = cv2.imread(first_image_path)
        h, w = img.shape[:2]
        return max(h, w) 
    except Exception as e:
        print(f"Erro ao ler imagem para definir 'imgsz': {e}")
        print("Usando 'imgsz=640' como padrão.")
        return 640

def train_model(model, data_yaml, base_save_dir, experiment_name, device):
    """
    Função para treinar um modelo YOLO.
    
    Args:
        model (YOLO): O objeto modelo YOLO pré-carregado (ex: YOLO('yolo11n.pt')).
        data_yaml (str): Caminho para o arquivo dataset.yaml.
        base_save_dir (str): A pasta base para todos os 'runs' (ex: './runs').
        experiment_name (str): O nome desta execução (ex: 'yolo11n').
        device (str): O dispositivo para treinar (ex: 'cuda' ou 'cpu').
    """
    
    dataset_path = os.path.dirname(data_yaml)
    train_images_path = os.path.join(dataset_path, "images", "train")
    
    imgsz = get_imgsz(train_images_path)

    print(f"Iniciando treino para: {experiment_name}")
    # print(f"Usando imgsz: {imgsz}")
    print(f"Salvando em: {os.path.join(base_save_dir, experiment_name)}")

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        lr0=0.005,
        lrf=0.01,
        optimizer="auto",
        weight_decay=0.0005,

        mosaic=0.0, 
        mixup=0.0, 
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        erasing=0.0, 
        
        fliplr=0.5,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        
        patience=31,
        plots=True,
        verbose=True,
        # diretprio do projeto e o nome do experimento
        # ex: ./runs/yolov11n, ./runs/yolov11s, etc.
        project=base_save_dir,
        name=experiment_name 
    )
    print(f"--- Treino de {experiment_name} concluído ---")


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")
    
    data_yaml = os.path.join(output_folder, "dataset.yaml")
    base_save_dir = "./runs"
    
    os.makedirs(base_save_dir, exist_ok=True)
    
    if not os.path.exists(data_yaml):
        print(f"Erro: dataset.yaml não encontrado em {data_yaml}")
        print("Rode os scripts voc2yolo.py e merge_classes.py primeiro.")


    else:
        for model_name in MODELS_TO_TRAIN:
            model_pt_file = f'{model_name}.pt'
            

            if not os.path.exists(model_pt_file):
                print(f"Aviso: Arquivo de pesos '{model_pt_file}' não encontrado. Pulando {model_name}.")
                continue


            print(f"\n==================================================")
            print(f"Carregando modelo: {model_pt_file}")
            print(f"==================================================")
            
            model = YOLO(model_pt_file)
            train_model(model, data_yaml, base_save_dir, model_name, device)

    print("\n[FINALIZADO] Benchmark de treinamento concluído.")