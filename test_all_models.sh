#!/bin/bash
# Script para evaluar todos los modelos entrenados usando los pesos guardados en experiment_results

BASE_CMD="python model.py"
OUTPUT_DIR="experiment_results"

# 1. Test de arquitecturas
ENCODER="resnet34"
ARCHITECTURES=("Unet" "FPN" "PSPNet" "DeepLabV3")
LOSS="bce_dice"

for split_num in 1 2 3; do
    split="$OUTPUT_DIR/arquitectura/split${split_num}"
    for arch in "${ARCHITECTURES[@]}"; do
        exp_dir="$split/${ENCODER}_${arch}_${LOSS}"
        if [ -d "$exp_dir" ] && [ -f "$exp_dir/best_model.pth" ]; then
            echo "Evaluando test para $exp_dir"
            $BASE_CMD --test_only --arch $arch --encoder_name $ENCODER --loss_fn $LOSS --output_dir $exp_dir --data_split $split --weights $exp_dir/best_model.pth
        fi
    done
done

# 2. Test de encoders
ARCH="Unet"
ENCODERS=("resnet34" "resnet50" "efficientnet-b0" "mobilenet_v2")
LOSS="bce_dice"

for split_num in 1 2 3; do
    split="$OUTPUT_DIR/encoders/split${split_num}"
    for encoder in "${ENCODERS[@]}"; do
        exp_dir="$split/${ARCH}_${encoder}_${LOSS}"
        if [ -d "$exp_dir" ] && [ -f "$exp_dir/best_model.pth" ]; then
            echo "Evaluando test para $exp_dir"
            $BASE_CMD --test_only --arch $ARCH --encoder_name $encoder --loss_fn $LOSS --output_dir $exp_dir --data_split $split --weights $exp_dir/best_model.pth
        fi
    done
done

# 3. Test de funciones de pérdida
ARCH="Unet"
ENCODER="resnet34"
LOSSES=("dice" "bce" "focal" "bce_dice")

for split_num in 1 2 3; do
    split="$OUTPUT_DIR/loss/split${split_num}"
    for loss in "${LOSSES[@]}"; do
        exp_dir="$split/${ARCH}_${ENCODER}_${loss}"
        if [ -d "$exp_dir" ] && [ -f "$exp_dir/best_model.pth" ]; then
            echo "Evaluando test para $exp_dir"
            $BASE_CMD --test_only --arch $ARCH --encoder_name $ENCODER --loss_fn $loss --output_dir $exp_dir --data_split $split --weights $exp_dir/best_model.pth
        fi
    done
done

echo "Evaluación de test completada para todos los modelos."
