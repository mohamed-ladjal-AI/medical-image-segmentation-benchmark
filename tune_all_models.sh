#!/bin/bash

# Required for torch.use_deterministic_algorithms() on CUDA >= 10.2
export CUBLAS_WORKSPACE_CONFIG=:4096:8

MODELS=(
  unet
  unet_plus_plus
  attention_unet
  deeplabv3_plus
  hrnet
  segformer
)

for model in "${MODELS[@]}"
do
  echo "=============================="
  echo "Running model: $model"
  echo "=============================="

  python tune_hyperparameters.py \
    --model $model \
    --n_trials 60 \
    --epochs 5 \
    --batch_size 8

done