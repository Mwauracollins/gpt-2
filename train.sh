#!/bin/bash

TRAIN_DATA="data/tiny-shakespear.txt"
MODEL_OUTPUT_DIR="model"
BATCH_SIZE=32
LEARNING_RATE=5e-5
EPOCHS=3
MAX_LENGTH=1024
DEVICE="cpu"

python train.py \
    --train_data "$TRAIN_DATA" \
    --model_output_dir "$MODEL_OUTPUT_DIR" \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --epochs $EPOCHS \
    --max_length $MAX_LENGTH \
    --device $DEVICE
