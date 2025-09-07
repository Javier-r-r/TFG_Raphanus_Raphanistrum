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
