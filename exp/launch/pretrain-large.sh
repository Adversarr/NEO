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

python exp/pretrain/train.py \
    datamod=shapenet_h5      \
    datamod.data_dir=$G2PT_DATA_SHAPENET \
    batch_size=80 \
    datamod.num_workers=8 \
    run_name=PerceiverModern \
    optimizer.max_lr=5e-4 \
    epochs=400 optimizer.name=muon \
    model=transolver_experimental scheduler.name=onecycle model_depth=8