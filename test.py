from ultralytics import YOLO
import time
import os

from config import output_folder, MODELS_TO_TRAIN

def validate_model(model_name, data_yaml_path):
    """
    "model.val" para um modelo específico.
    
    """
    model_path = os.path.join("runs", model_name, "weights", "best.pt")

    if not os.path.exists(model_path):
        print(f"Aviso: 'best.pt' não encontrado em {model_path}. Pulando {model_name}.")
        return

    print(f"\n==================================================")
    print(f"Teste: {model_name}")
    print(f"==================================================")
    print(f"Modelo: {model_path}")
    print(f"Dataset: {data_yaml_path}")


    model = YOLO(model_path)

    # Tempo total
    start_val_time = time.time()

    metrics = model.val(data=data_yaml_path, split='test', verbose=True)
    
    end_val_time = time.time()
    total_validation_time = end_val_time - start_val_time

    print(f"\n--- Resultados teste: {model_name} ---")

    print("\n velocidade (médias por imagem):")
    speed_ms = metrics.speed
    print(f"  Pré-processamento: {speed_ms['preprocess']:.2f} ms")
    print(f"  Inferência: {speed_ms['inference']:.2f} ms")
    print(f"  Pós-processamento (NMS): {speed_ms['postprocess']:.2f} ms")
    print(f"  Tempo TOTAL da operação 'model.val()': {total_validation_time:.2f} s")

    # performance
    print("\n metricas (Geral):")
    print(f"  Precisão (P): {metrics.box.p.mean():.4f}")
    print(f"  Recall (R):   {metrics.box.r.mean():.4f}")
    print(f"  F1-Score (média): {metrics.box.f1.mean():.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  mAP50:    {metrics.box.map50:.4f}")

    # F1-classes
    print("\n--- F1-Score por Classe ---")
    class_names = model.names
    for i, f1 in enumerate(metrics.box.f1):
        print(f"  Classe {i} ({class_names[i]}): {f1:.4f}")


if __name__ == "__main__":
    
    data_yaml_path = os.path.join(output_folder, "dataset.yaml")

    if not os.path.exists(data_yaml_path):
        print(f"Erro: dataset.yaml não encontrado em {data_yaml_path}")
        print("Rode os scripts voc2yolo.py e merge_classes.py primeiro.")
    else:
        for model_name in MODELS_TO_TRAIN:
            validate_model(model_name, data_yaml_path)