#!/bin/bash

# Configuración básica
BASE_CMD="python model.py"
OUTPUT_DIR="experiment_results"
rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR
~/myenv/bin/activate

# 1. Experimentos con encoder fijo (resnet34) y diferentes arquitecturas
ENCODER="resnet34"
ARCHITECTURES=("Unet" "FPN" "PSPNet" "DeepLabV3")
LOSS="dice"

echo "=== Ejecutando experimentos con encoder fijo ($ENCODER) ==="
for arch in "${ARCHITECTURES[@]}"; do
    echo "--- Probando arquitectura: $arch ---"
    CMD="$BASE_CMD -arquitectura $arch -encoder $ENCODER -loss $LOSS -output $OUTPUT_DIR/${ENCODER}_${arch}_${LOSS}"
    OUTPUT_FILE="$OUTPUT_DIR/${ENCODER}_${arch}_${LOSS}.log"
    echo "Comando: $CMD" > $OUTPUT_FILE
    $CMD >> $OUTPUT_FILE 2>&1
done

# 2. Experimentos con arquitectura fija (Unet) y diferentes encoders
ARCH="Unet"
ENCODERS=("resnet34" "resnet50" "efficientnet-b0" "mobilenet_v2")

echo "=== Ejecutando experimentos con arquitectura fija ($ARCH) ==="
for encoder in "${ENCODERS[@]}"; do
    echo "--- Probando encoder: $encoder ---"
    CMD="$BASE_CMD -arquitectura $ARCH -encoder $encoder -loss $LOSS -output $OUTPUT_DIR/${ENCODER}_${arch}_${LOSS}"
    OUTPUT_FILE="$OUTPUT_DIR/${ARCH}_${encoder}_${LOSS}.log"
    echo "Comando: $CMD" > $OUTPUT_FILE
    $CMD >> $OUTPUT_FILE 2>&1
done

# 3. Experimentos con arquitectura y encoder fijos (Unet, resnet34) y diferentes losses
ARCH="Unet"
ENCODER="resnet34"
LOSSES=("dice" "bce" "focal")

echo "=== Ejecutando experimentos con funciones de pérdida diferentes ==="
for loss in "${LOSSES[@]}"; do
    echo "--- Probando función de pérdida: $loss ---"
    CMD="$BASE_CMD -arquitectura $ARCH -encoder $ENCODER -loss $loss -output $OUTPUT_DIR/${ENCODER}_${arch}_${LOSS}"
    OUTPUT_FILE="$OUTPUT_DIR/${ARCH}_${ENCODER}_${loss}.log"
    echo "Comando: $CMD" > $OUTPUT_FILE
    $CMD >> $OUTPUT_FILE 2>&1
done

echo "Todos los experimentos han sido completados. Los resultados se encuentran en $OUTPUT_DIR/"
