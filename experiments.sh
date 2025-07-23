#!/bin/bash

# Configuración básica
BASE_CMD="python model.py"
OUTPUT_DIR="experiment_results"
rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/{arquitectura,encoders,loss}
source ~/myenv/bin/activate


# 1. Experimentos variando arquitecturas (encoder fijo: resnet34)
ENCODER="resnet34"
ARCHITECTURES=("Unet" "FPN" "PSPNet" "DeepLabV3")
LOSS="dice"

echo "=== Ejecutando experimentos variando arquitecturas (encoder: $ENCODER) ==="
for arch in "${ARCHITECTURES[@]}"; do
    echo "--- Probando arquitectura: $arch ---"
    CMD="$BASE_CMD -arquitectura $arch -encoder $ENCODER -loss $LOSS -output $OUTPUT_DIR/arquitectura/${ENCODER}_${arch}_${LOSS}"
    echo "Comando: $CMD" > "$OUTPUT_DIR/arquitectura/${ENCODER}_${arch}_${LOSS}.log"
    $CMD >> "$OUTPUT_DIR/arquitectura/${ENCODER}_${arch}_${LOSS}.log" 2>&1
done

# 2. Experimentos variando encoders (arquitectura fija: Unet)
ARCH="Unet"
ENCODERS=("resnet34" "resnet50" "efficientnet-b0" "mobilenet_v2")

echo "=== Ejecutando experimentos variando encoders (arquitectura: $ARCH) ==="
for encoder in "${ENCODERS[@]}"; do
    echo "--- Probando encoder: $encoder ---"
    CMD="$BASE_CMD -arquitectura $ARCH -encoder $encoder -loss $LOSS -output $OUTPUT_DIR/encoders/${ARCH}_${encoder}_${LOSS}"
    echo "Comando: $CMD" > "$OUTPUT_DIR/encoders/${ARCH}_${encoder}_${LOSS}.log"
    $CMD >> "$OUTPUT_DIR/encoders/${ARCH}_${encoder}_${LOSS}.log" 2>&1
done

# 3. Experimentos variando funciones de pérdida (arquitectura: Unet, encoder: resnet34)
ARCH="Unet"
ENCODER="resnet34"
LOSSES=("dice" "bce" "focal")

echo "=== Ejecutando experimentos variando funciones de pérdida ==="
for loss in "${LOSSES[@]}"; do
    echo "--- Probando función de pérdida: $loss ---"
    CMD="$BASE_CMD -arquitectura $ARCH -encoder $ENCODER -loss $loss -output $OUTPUT_DIR/loss/${ARCH}_${ENCODER}_${loss}"
    echo "Comando: $CMD" > "$OUTPUT_DIR/loss/${ARCH}_${ENCODER}_${loss}.log"
    $CMD >> "$OUTPUT_DIR/loss/${ARCH}_${ENCODER}_${loss}.log" 2>&1
done

echo "Todos los experimentos han sido completados."
echo "Resultados organizados en:"
echo "- Arquitecturas: $OUTPUT_DIR/arquitectura/"
echo "- Encoders: $OUTPUT_DIR/encoders/"
echo "- Funciones de pérdida: $OUTPUT_DIR/loss/"
