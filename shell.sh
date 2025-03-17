#!/bin/bash

# Define the models and their directories
MODELS=(
    "RESNET18"
    "RESNET34"
    "RESNET50"
    # "CONVNEXT_TINY"
    # "REGNET"
    # "SWIN_TRANSFORMER_TINY"
    # "VGG11"
    # "VGG13"
    # "VGG16"
    # "VGG19"
    # "EFFICIENT_NET"
    # "MOBILENET"
    # "SHUFFLENET"
    # "INCEPTION_NET"
)

NUM_CLASSES=4
DATA_DIR="/home/Drivehd2tb/garima/datasets/mod_data/d2_classify/sc/complete"
BASE_MODEL_DIR="/home/Drivehd2tb/garima/code/weights_new/d2_classify"
BATCH_SIZE=128
EPOCHS=100
LR=1e-6
WEIGHT_DECAY=1e-3
PATIENCE=20
DELTA=0.01
MULTI_GPU="--multi_gpu"  # Remove if you don't want multi-GPU support
FOLDS=4

# Loop through each model and run training
for MODEL in "${MODELS[@]}"; do
    # conda init
    # conda activate ./env/
    MODEL_DIR="${BASE_MODEL_DIR}/${MODEL}/kfold/bs${BATCH_SIZE}_lr${LR}_wd${WEIGHT_DECAY}"
    LOG_FILE="${MODEL_DIR}/training.log"

    # Create model directory if it doesn't exist
    mkdir -p "$MODEL_DIR"

    echo "Starting training for model: $MODEL" | tee -a "$LOG_FILE"

    python v6/kfold_classification.py \
        --model "$MODEL" \
        --num_classes "$NUM_CLASSES" \
        --patience "$PATIENCE" \
        --delta "$DELTA" \
        --model_dir "$MODEL_DIR" \
        --data_dir "$DATA_DIR" \
        --batch_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --weight_decay "$WEIGHT_DECAY" \
        --device "cuda:1" \
        # $MULTI_GPU \
        2>&1 | tee -a "$LOG_FILE"

    echo "Finished training for model: $MODEL" | tee -a "$LOG_FILE"
    echo "-----------------------------------" | tee -a "$LOG_FILE"
done