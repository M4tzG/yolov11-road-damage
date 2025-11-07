# ------------
# Treinamento
# ------------
# listda dos modelos
# usados no main.py e test.py
MODELS_TO_TRAIN = ['yolo11n', 'yolo11s', 'yolo11m']

countries = ["Czech", "India", "Japan", "United_States", "Norway", "China"]
original_classes = ["D00", "D10", "D20", "D30", "D40", "D50" , "D60", "D70", "D80", "D90"]

# classes *apos* o merge (merge_classes.py)
classes = ["D00", "D10", "D20", "D40", "D60", "D70", "other"]

output_folder = f"../yolo"
batch = 16
epochs = 1
val_ratio = 0.15
test_ratio = 0.15
