#!/bin/bash

source .venv/bin/activate

# Get G2PT_DATA_SHAPENET parameter from command line
if [ -n "$1" ]; then
    G2PT_DATA_SHAPENET="$1"
fi

# If G2PT_DATA_SHAPENET is not set but G2PT_DATA_ROOT is, use default value
if [ -z "$G2PT_DATA_SHAPENET" ] && [ -n "$G2PT_DATA_ROOT" ]; then
    G2PT_DATA_SHAPENET="$G2PT_DATA_ROOT/preprocessed_shapenet_h5"
fi

python exp/pretrain_experimental/train.py \
    datamod=shapenet_h5      \
    datamod.data_dir=$G2PT_DATA_SHAPENET \
    batch_size=80 \
    model=transolver2 \
    datamod.num_workers=6 \
    model_depth=8\
    epochs=500