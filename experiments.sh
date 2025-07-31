#!/bin/bash

# Configuración básica
BASE_CMD="python model.py"
OUTPUT_DIR="experiment_results"
DATASET_SPLITS=("experiment_70_15_15" "experiment_60_20_20" "experiment_80_10_10")
rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/{arquitectura,encoders,loss}/{split1,split2,split3}
source ~/myenv/bin/activate

# Función para ejecutar comandos con log
run_experiment() {
    local cmd=$1
    local log_file=$2
    echo "Comando: $cmd" > "$log_file"
    echo "Inicio: $(date)" >> "$log_file"
    $cmd >> "$log_file" 2>&1
    echo "Fin: $(date)" >> "$log_file"
}

# 1. Experimentos variando arquitecturas (encoder fijo: resnet34)
ENCODER="resnet34"
ARCHITECTURES=("Unet" "FPN" "PSPNet" "DeepLabV3")
LOSS="dice"

echo "=== Ejecutando experimentos variando arquitecturas (encoder: $ENCODER) ==="
for split_idx in "${!DATASET_SPLITS[@]}"; do
    split=${DATASET_SPLITS[$split_idx]}
    split_num=$((split_idx+1))
    echo "--- Usando división de dataset: $split (split$split_num) ---"
    
    for arch in "${ARCHITECTURES[@]}"; do
        echo "Probando arquitectura: $arch"
        CMD="$BASE_CMD -arquitectura $arch -encoder $ENCODER -loss $LOSS -output $OUTPUT_DIR/arquitectura/split${split_num}/${ENCODER}_${arch}_${LOSS} --data_split ../experiments/$split"
        LOG_FILE="$OUTPUT_DIR/arquitectura/split${split_num}/${ENCODER}_${arch}_${LOSS}.log"
        run_experiment "$CMD" "$LOG_FILE"
    done
done

# 2. Experimentos variando encoders (arquitectura fija: Unet)
ARCH="Unet"
ENCODERS=("resnet34" "resnet50" "efficientnet-b0" "mobilenet_v2")

echo "=== Ejecutando experimentos variando encoders (arquitectura: $ARCH) ==="
for split_idx in "${!DATASET_SPLITS[@]}"; do
    split=${DATASET_SPLITS[$split_idx]}
    split_num=$((split_idx+1))
    echo "--- Usando división de dataset: $split (split$split_num) ---"
    
    for encoder in "${ENCODERS[@]}"; do
        echo "Probando encoder: $encoder"
        CMD="$BASE_CMD -arquitectura $ARCH -encoder $encoder -loss $LOSS -output $OUTPUT_DIR/encoders/split${split_num}/${ARCH}_${encoder}_${LOSS} --data_split ../experiments/$split"
        LOG_FILE="$OUTPUT_DIR/encoders/split${split_num}/${ARCH}_${encoder}_${LOSS}.log"
        run_experiment "$CMD" "$LOG_FILE"
    done
done

# 3. Experimentos variando funciones de pérdida (arquitectura: Unet, encoder: resnet34)
ARCH="Unet"
ENCODER="resnet34"
LOSSES=("dice" "bce" "focal")

echo "=== Ejecutando experimentos variando funciones de pérdida ==="
for split_idx in "${!DATASET_SPLITS[@]}"; do
    split=${DATASET_SPLITS[$split_idx]}
    split_num=$((split_idx+1))
    echo "--- Usando división de dataset: $split (split$split_num) ---"
    
    for loss in "${LOSSES[@]}"; do
        echo "Probando función de pérdida: $loss"
        CMD="$BASE_CMD -arquitectura $ARCH -encoder $ENCODER -loss $loss -output $OUTPUT_DIR/loss/split${split_num}/${ARCH}_${ENCODER}_${loss} --data_split ../experiments/$split"
        LOG_FILE="$OUTPUT_DIR/loss/split${split_num}/${ARCH}_${ENCODER}_${loss}.log"
        run_experiment "$CMD" "$LOG_FILE"
    done
done

echo "Todos los experimentos han sido completados."
echo "Resultados organizados en:"
echo "- Arquitecturas: $OUTPUT_DIR/arquitectura/split[1-3]/"
echo "- Encoders: $OUTPUT_DIR/encoders/split[1-3]/"
echo "- Funciones de pérdida: $OUTPUT_DIR/loss/split[1-3]/"

# Generar resumen de resultados
echo "Generando resumen de resultados..."
SUMMARY_FILE="$OUTPUT_DIR/experiment_summary.csv"
echo "experiment_type,split,architecture,encoder,loss,test_loss,iou_score" > "$SUMMARY_FILE"

for experiment_type in "arquitectura" "encoders" "loss"; do
    for split_num in 1 2 3; do
        for log_file in $OUTPUT_DIR/$experiment_type/split$split_num/*.log; do
            if [ -f "$log_file" ]; then
                # Extraer metadatos del nombre del archivo
                filename=$(basename "$log_file" .log)
                IFS='_' read -ra parts <<< "$filename"
                
                if [ "$experiment_type" == "arquitectura" ]; then
                    encoder=${parts[0]}
                    arch=${parts[1]}
                    loss=${parts[2]}
                elif [ "$experiment_type" == "encoders" ]; then
                    arch=${parts[0]}
                    encoder=${parts[1]}
                    loss=${parts[2]}
                else
                    arch=${parts[0]}
                    encoder=${parts[1]}
                    loss=${parts[2]}
                fi
                
                # Extraer métricas del archivo de log
                test_loss=$(grep -oP "Test Loss: \K\d+\.\d+" "$log_file" || echo "NA")
                iou_score=$(grep -oP "IoU: \K\d+\.\d+" "$log_file" || echo "NA")
                
                echo "$experiment_type,split$split_num,$arch,$encoder,$loss,$test_loss,$iou_score" >> "$SUMMARY_FILE"
            fi
        done
    done
done

echo "Resumen de resultados guardado en: $SUMMARY_FILE"
