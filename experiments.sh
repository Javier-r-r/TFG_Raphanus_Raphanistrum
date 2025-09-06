#!/bin/bash

# Configuración básica
BASE_CMD="python model.py"
OUTPUT_DIR="experiment_results"
DATASET_ORIG_IMAGES="../database_petals"
DATASET_ORIG_MASKS="../masks_petals"
SPLIT_ROOT="split_temp"
rm -rf $OUTPUT_DIR $SPLIT_ROOT
mkdir -p $OUTPUT_DIR/{arquitectura,encoders,loss}/{split1,split2,split3}
source ~/myenv/bin/activate

# Generar splits reproducibles 70/15/15 usando database_segmentation.py
echo "Generando splits reproducibles 70/15/15..."
for split_num in 1 2 3; do
    split_dir="$SPLIT_ROOT/split$split_num"
    mkdir -p $split_dir
    python database_segmentation.py --images_dir $DATASET_ORIG_IMAGES --masks_dir $DATASET_ORIG_MASKS --output_dir $split_dir --seed $((42 + split_num))
done

# Función para ejecutar comandos con log
run_experiment() {
    local cmd=$1
    local log_file=$2
    echo "Comando: $cmd" > "$log_file"
    echo "Inicio: $(date)" >> "$log_file"
    $cmd >> "$log_file" 2>&1
    echo "Fin: $(date)" >> "$log_file"
}


# 1. Experimentos variando arquitecturas (encoder fijo: resnet34, loss por defecto: bce_dice)
ENCODER="resnet34"
ARCHITECTURES=("Unet" "FPN" "PSPNet" "DeepLabV3")
LOSS="bce_dice"

echo "=== Ejecutando experimentos variando arquitecturas (encoder: $ENCODER, loss: $LOSS) ==="
for split_num in 1 2 3; do
    split="$SPLIT_ROOT/split$split_num"
    echo "--- Usando división de dataset: $split (split$split_num) ---"
    for arch in "${ARCHITECTURES[@]}"; do
        echo "Probando arquitectura: $arch"
        CMD="$BASE_CMD -arquitectura $arch -encoder $ENCODER -loss $LOSS -output $OUTPUT_DIR/arquitectura/split${split_num}/${ENCODER}_${arch}_${LOSS} --data_split $split"
        LOG_FILE="$OUTPUT_DIR/arquitectura/split${split_num}/${ENCODER}_${arch}_${LOSS}.log"
        run_experiment "$CMD" "$LOG_FILE"
    done
done


# 2. Experimentos variando encoders (arquitectura fija: Unet, loss por defecto: bce_dice)
ARCH="Unet"
ENCODERS=("resnet34" "resnet50" "efficientnet-b0" "mobilenet_v2")
LOSS="bce_dice"

echo "=== Ejecutando experimentos variando encoders (arquitectura: $ARCH, loss: $LOSS) ==="
for split_num in 1 2 3; do
    split="$SPLIT_ROOT/split$split_num"
    echo "--- Usando división de dataset: $split (split$split_num) ---"
    for encoder in "${ENCODERS[@]}"; do
        echo "Probando encoder: $encoder"
        CMD="$BASE_CMD -arquitectura $ARCH -encoder $encoder -loss $LOSS -output $OUTPUT_DIR/encoders/split${split_num}/${ARCH}_${encoder}_${LOSS} --data_split $split"
        LOG_FILE="$OUTPUT_DIR/encoders/split${split_num}/${ARCH}_${encoder}_${LOSS}.log"
        run_experiment "$CMD" "$LOG_FILE"
    done
done


# 3. Experimentos variando funciones de pérdida (arquitectura: Unet, encoder: resnet34)
ARCH="Unet"
ENCODER="resnet34"
LOSSES=("dice" "bce" "focal" "bce_dice")

echo "=== Ejecutando experimentos variando funciones de pérdida ==="
for split_num in 1 2 3; do
    split="$SPLIT_ROOT/split$split_num"
    echo "--- Usando división de dataset: $split (split$split_num) ---"
    for loss in "${LOSSES[@]}"; do
        echo "Probando función de pérdida: $loss"
        CMD="$BASE_CMD -arquitectura $ARCH -encoder $ENCODER -loss $loss -output $OUTPUT_DIR/loss/split${split_num}/${ARCH}_${ENCODER}_${loss} --data_split $split"
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
echo "Generando resumen de resultados completos..."
SUMMARY_FILE="$OUTPUT_DIR/experiment_summary.csv"
DETAILED_FILE="$OUTPUT_DIR/experiment_detailed_summary.csv"

# Encabezados mejorados
echo "experiment_type,split,architecture,encoder,loss,test_loss,iou_score,precision,recall,f1_score,model_path" > "$SUMMARY_FILE"
echo "experiment_type,split,architecture,encoder,loss,test_loss,iou_score,precision,recall,f1_score,epochs,train_time,model_path" > "$DETAILED_FILE"

for experiment_type in "arquitectura" "encoders" "loss"; do
    for split_num in 1 2 3; do
        for exp_dir in $OUTPUT_DIR/$experiment_type/split$split_num/*/; do
            if [ -d "$exp_dir" ]; then
                # Extraer metadatos del nombre del directorio de forma robusta
                dirname=$(basename "$exp_dir")
                # Contar partes
                IFS='_' read -ra parts <<< "$dirname"
                n=${#parts[@]}

                if [ "$experiment_type" == "arquitectura" ]; then
                    # Formato: encoder_arch_loss
                    encoder=${parts[0]}
                    arch=${parts[1]}
                    # loss puede tener guiones bajos
                    loss=$(IFS=_; echo "${parts[@]:2}")
                elif [ "$experiment_type" == "encoders" ]; then
                    # Formato: arch_encoder_loss (encoder puede tener guiones bajos)
                    arch=${parts[0]}
                    # encoder puede ser resnet34, resnet50, efficientnet-b0, mobilenet_v2
                    if [ $n -eq 4 ]; then
                        encoder="${parts[1]}_${parts[2]}"
                        loss=${parts[3]}
                    else
                        encoder=${parts[1]}
                        loss=$(IFS=_; echo "${parts[@]:2}")
                    fi
                else
                    # Formato: arch_encoder_loss (loss puede tener guiones bajos)
                    arch=${parts[0]}
                    encoder=${parts[1]}
                    loss=$(IFS=_; echo "${parts[@]:2}")
                fi

                # Inicializar variables
                test_loss="NA"
                iou_score="NA"
                precision="NA"
                recall="NA"
                f1_score="NA"
                epochs="NA"
                train_time="NA"

                # Leer de metricas_detalladas.csv si existe
                if [ -f "$exp_dir/metricas_detalladas.csv" ]; then
                    metrics=$(tail -n 1 "$exp_dir/metricas_detalladas.csv")
                    test_loss=$(echo "$metrics" | cut -d',' -f1)
                    iou_score=$(echo "$metrics" | cut -d',' -f2)
                    precision=$(echo "$metrics" | cut -d',' -f3)
                    recall=$(echo "$metrics" | cut -d',' -f4)
                    f1_score=$(echo "$metrics" | cut -d',' -f5)
                fi

                # Leer información adicional de config.json
                if [ -f "$exp_dir/config.json" ]; then
                    epochs=$(jq -r '.epochs_max' "$exp_dir/config.json" 2>/dev/null || echo "NA")
                fi

                # Leer tiempo de entrenamiento del log
                if [ -f "$exp_dir/train_history.csv" ]; then
                    train_time=$(tail -n 1 "$exp_dir/train_history.csv" | cut -d',' -f3)
                fi

                # Escribir en el resumen básico (sin repetir el nombre de la arquitectura en f1_score ni en model_path)
                echo "$experiment_type,split$split_num,$arch,$encoder,$loss,$test_loss,$iou_score,$precision,$recall,$f1_score,$exp_dir" >> "$SUMMARY_FILE"

                # Escribir en el resumen detallado
                echo "$experiment_type,split$split_num,$arch,$encoder,$loss,$test_loss,$iou_score,$precision,$recall,$f1_score,$epochs,$train_time,$exp_dir" >> "$DETAILED_FILE"
            fi
        done
    done
done

# Generar también un archivo de resumen por splits
SPLIT_SUMMARY="$OUTPUT_DIR/split_summary.csv"
echo "split,architecture,encoder,loss,avg_iou,avg_precision,avg_recall" > "$SPLIT_SUMMARY"

for split_num in 1 2 3; do
    for arch in "${ARCHITECTURES[@]}"; do
        for encoder in "${ENCODERS[@]}"; do
            for loss in "${LOSSES[@]}"; do
                # Calcular promedios para esta combinación
                metrics=$(grep ",split$split_num,$arch,$encoder,$loss," "$SUMMARY_FILE" | awk -F',' 'BEGIN{OFS=",";count=0;iou=0;prec=0;rec=0}
                {if($6!="NA"){count++; iou+=$7; prec+=$8; rec+=$9}}
                END{if(count>0){print count, iou/count, prec/count, rec/count} else {print "0,NA,NA,NA"}}')

                count=$(echo "$metrics" | cut -d',' -f1)
                if [ "$count" -gt 0 ]; then
                    avg_iou=$(echo "$metrics" | cut -d',' -f2)
                    avg_precision=$(echo "$metrics" | cut -d',' -f3)
                    avg_recall=$(echo "$metrics" | cut -d',' -f4)
                    echo "split$split_num,$arch,$encoder,$loss,$avg_iou,$avg_precision,$avg_recall" >> "$SPLIT_SUMMARY"
                fi
            done
        done
    done
done

echo "Resumen básico guardado en: $SUMMARY_FILE"
echo "Resumen detallado guardado en: $DETAILED_FILE"
echo "Resumen por splits guardado en: $SPLIT_SUMMARY"
touch "$HOME/experiments_done.flag"
