#!/bin/bash

MODEL_CHECKPOINT="model/checkpoint_epoch_3.pth"
PROMPT="Once upon a time in a land far, far away..."
OUTPUT_FILE="output/generated_text.txt"
BATCH_SIZE=1
MAX_LENGTH=1024
DEVICE="cuda"

python inference.py \
    --model_checkpoint "$MODEL_CHECKPOINT" \
    --prompt "$PROMPT" \
    --output_file "$OUTPUT_FILE" \
    --batch_size $BATCH_SIZE \
    --max_length $MAX_LENGTH \
    --device $DEVICE
