#!/bin/bash
set -e
# Script para evaluar todos los modelos entrenados usando los pesos guardados en experiment_results

BASE_CMD="python model.py"
OUTPUT_DIR="experiment_tests"
WEIGHTS_DIR="experiment_results"
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

# Función para ejecutar test con log (más robusta)
run_test() {
    local log_file=$1
    shift
    echo "Comando: $*" > "$log_file"
    echo "Inicio: $(date)" >> "$log_file"
    "$@" >> "$log_file" 2>&1
    echo "Fin: $(date)" >> "$log_file"
}

# 1. Test de arquitecturas
ENCODER="resnet34"
ARCHITECTURES=("Unet" "FPN" "PSPNet" "DeepLabV3")
LOSS="bce_dice"

for split_num in 1 2 3; do
    split="$SPLIT_ROOT/split$split_num"  # <--- Cambia aquí
    for arch in "${ARCHITECTURES[@]}"; do
        exp_dir="$WEIGHTS_DIR/arquitectura/split${split_num}/${ENCODER}_${arch}_${LOSS}"
        weights_path="$exp_dir/best_model.pth"
        exp_test_dir="$OUTPUT_DIR/arquitectura/split${split_num}/${ENCODER}_${arch}_${LOSS}"
        if [ -f "$weights_path" ]; then
            mkdir -p "$exp_test_dir"
            echo "Evaluando test para $exp_test_dir"
            LOG_FILE="$exp_test_dir/test.log"
            run_test "$LOG_FILE" python model.py --test_only --arch $arch --encoder_name $ENCODER --loss_fn $LOSS --output_dir $exp_test_dir --data_split $split --weights $weights_path
        fi
    done
done

# 2. Test de encoders
ARCH="Unet"
ENCODERS=("resnet34" "resnet50" "efficientnet-b0" "mobilenet_v2")
LOSS="bce_dice"

for split_num in 1 2 3; do
    split="$SPLIT_ROOT/split$split_num"  # <--- Cambia aquí
    for encoder in "${ENCODERS[@]}"; do
        exp_dir="$WEIGHTS_DIR/encoders/split${split_num}/${ARCH}_${encoder}_${LOSS}"
        weights_path="$exp_dir/best_model.pth"
        exp_test_dir="$OUTPUT_DIR/encoders/split${split_num}/${ARCH}_${encoder}_${LOSS}"
        if [ -f "$weights_path" ]; then
            mkdir -p "$exp_test_dir"
            echo "Evaluando test para $exp_test_dir"
            LOG_FILE="$exp_test_dir/test.log"
            run_test "$LOG_FILE" python model.py --test_only --arch $ARCH --encoder_name $encoder --loss_fn $LOSS --output_dir $exp_test_dir --data_split $split --weights $weights_path
        fi
    done
done

# 3. Test de funciones de pérdida
ARCH="Unet"
ENCODER="resnet34"
LOSSES=("dice" "bce" "focal" "bce_dice")

for split_num in 1 2 3; do
    split="$SPLIT_ROOT/split$split_num"  # <--- Cambia aquí
    for loss in "${LOSSES[@]}"; do
        exp_dir="$WEIGHTS_DIR/loss/split${split_num}/${ARCH}_${ENCODER}_${loss}"
        weights_path="$exp_dir/best_model.pth"
        exp_test_dir="$OUTPUT_DIR/loss/split${split_num}/${ARCH}_${ENCODER}_${loss}"
        if [ -f "$weights_path" ]; then
            mkdir -p "$exp_test_dir"
            echo "Evaluando test para $exp_test_dir"
            LOG_FILE="$exp_test_dir/test.log"
            run_test "$LOG_FILE" python model.py --test_only --arch $ARCH --encoder_name $ENCODER --loss_fn $loss --output_dir $exp_test_dir --data_split $split --weights $weights_path
        fi
    done
done

echo "Evaluación de test completada para todos los modelos."

# === Generar resumen de resultados de test ===
SUMMARY_FILE="$OUTPUT_DIR/test_summary.csv"
echo "experiment_type,split,architecture,encoder,loss,test_loss,iou_score,precision,recall,f1_score,model_path" > "$SUMMARY_FILE"

for experiment_type in "arquitectura" "encoders" "loss"; do
    for split_num in 1 2 3; do
        for exp_dir in $OUTPUT_DIR/$experiment_type/split$split_num/*/; do
            if [ -d "$exp_dir" ]; then
                dirname=$(basename "$exp_dir")
                IFS='_' read -ra parts <<< "$dirname"
                n=${#parts[@]}

                if [ "$experiment_type" == "arquitectura" ]; then
                    encoder=${parts[0]}
                    arch=${parts[1]}
                    loss=$(IFS=_; echo "${parts[@]:2}")
                elif [ "$experiment_type" == "encoders" ]; then
                    arch=${parts[0]}
                    if [ $n -eq 4 ]; then
                        encoder="${parts[1]}_${parts[2]}"
                        loss=${parts[3]}
                    else
                        encoder=${parts[1]}
                        loss=$(IFS=_; echo "${parts[@]:2}")
                    fi
                else
                    arch=${parts[0]}
                    encoder=${parts[1]}
                    loss=$(IFS=_; echo "${parts[@]:2}")
                fi

                test_loss="NA"
                iou_score="NA"
                precision="NA"
                recall="NA"
                f1_score="NA"

                if [ -f "$exp_dir/metricas_detalladas.csv" ]; then
                    metrics=$(tail -n 1 "$exp_dir/metricas_detalladas.csv")
                    test_loss=$(echo "$metrics" | cut -d',' -f1)
                    iou_score=$(echo "$metrics" | cut -d',' -f2)
                    precision=$(echo "$metrics" | cut -d',' -f3)
                    recall=$(echo "$metrics" | cut -d',' -f4)
                    f1_score=$(echo "$metrics" | cut -d',' -f5)
                fi

                echo "$experiment_type,split$split_num,$arch,$encoder,$loss,$test_loss,$iou_score,$precision,$recall,$f1_score,$exp_dir" >> "$SUMMARY_FILE"
            fi
        done
    done
done

echo "Resumen de test guardado en: $SUMMARY_FILE"
