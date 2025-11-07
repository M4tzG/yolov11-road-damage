# Avaliação Comparativa de Modelos YOLOv11 na Detecção de Danos em Pavimentos Utilizando o Dataset RDD2024

Equipe: Turma U -- Equipe 10 
- Matheus G. P. Nogueira; 
- Nicolas L. de Paulo;
- Nicolas R. S. Zamprogno
- Vítor S. do Vale

## Descrição:

Este projeto avalia o desempenho de arquiteturas de detecção de objetos (YOLOv11n, YOLOv11s, YOLOv11m) para a tarefa de detecção de danos em estradas. O pipeline inclui pré-processamento de dados (conversão de Pascal VOC para YOLO e merge de classes) e um ciclo de treinamento e teste para comparação dos modelos.

## Artefatos:

config.py
    Arquivo de configuração central do projeto.
    Define parâmetros globais, como número de épocas, batch size,
    proporção de validação/teste, classes, e modelos utilizados
    no benchmark (yolov11n, yolov11s, yolov11m).
    Não é executável diretamente.

voc2yolo.py
    Script responsável por converter as anotações originais
    do dataset (formato Pascal VOC) para o formato YOLO.
    Utiliza os parâmetros definidos em config.py.
    Cria o arquivo final `dataset.yaml` e as pastas:
        ../yolo/images/{train,val,test}
        ../yolo/labels/{train,val,test}

merge_classes.py
    Executa o merge das classes originais em categorias
    reduzidas, conforme definido no mapeamento interno.
    Atualiza automaticamente as labels YOLO e o arquivo `dataset.yaml`
    para refletir a nova taxonomia de classes.

main.py
    Script de treinamento principal.
    Treina automaticamente os modelos definidos em `MODELS_TO_TRAIN`
    (yolov11n, yolov11s, yolov11m).
    Os resultados de cada treinamento (pesos, logs, métricas, figuras) são salvos automaticamente pela biblioteca `ultralytics` em diretórios separados por modelo. 
    Ex:
        ./runs/yolov11n/
        ./runs/yolov11s/
        ./runs/yolov11m/
    Parâmetros principais:
        - data: caminho do dataset.yaml (../yolo/dataset.yaml)
        - epochs: definido em config.py
        - batch: definido em config.py
        - device: detectado automaticamente (cuda, mps ou cpu)

test.py
    Responsável pela validação final dos modelos treinados.
    Utiliza os pesos `best.pt` gerados em cada execução de treinamento.
    Principais parâmetros:
        - --weights: caminho do modelo treinado (ex: runs/yolov11m/train/weights/best.pt)
        - --data: caminho do arquivo dataset.yaml
        - split='test': avalia no conjunto de teste
    Retorna um relatório de métricas impresso no console para cada modelo.


## Resultados:

Os resultados experimentais e os dados utilizados estão organizados
em três diretórios principais disponíveis no Google Drive:

[Link geral do Drive] (https://drive.google.com/drive/folders/1UK9znjb-_V_pvVzBIQQvAPRA2cj5vwf-?usp=sharing)

Estrutura do Drive:
`/Dados/`: Contém o dataset original (imagens e anotações em Pascal VOC) utilizado como entrada para o pipeline.

`/Testes_/`: Contém logs, anotações e resultados de experimentos intermediários realizados ao longo do desenvolvimento do projeto.

`/Resultados/`: Contém os resultados finais de cada modelo:
	yolov11n/
	yolov11s/
	yolov11m/
Cada pasta Inclui tabelas comparativas, gráficos de performance, métricas salvas e os pesos finais.
