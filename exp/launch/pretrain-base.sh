#!/bin/bash

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
    batch_size=128 \
    model=transolver_experimental \
    datamod.num_workers=6 \
    model_depth=6 \
    optimizer.max_lr=7e-4

# No RoPE:
 OMP_NUM_THREADS=1 python exp/pretrain/train.py datamod=shapenet_h5 \
    datamod.data_dir=../data/preprocessed_shapenet_h5 batch_size=120 accumulate_grad_batches=2 \
    datamod.num_workers=8 run_name=SmallNoRoPE-Fix optimizer.max_lr=8e-4 epochs=400 \
    optimizer.name=muon model=transolver_experimental model.attn.enable_rope=false \
    model_attn_dim_heads=48 model_depth=4 targ_dim_model=192

OMP_NUM_THREADS=1 python exp/pretrain/train.py datamod=shapenet_h5 \
    datamod.data_dir=../data/preprocessed_shapenet_h5 batch_size=240 \
    datamod.num_workers=8 run_name=SmallNoRoPE-Fix optimizer.max_lr=1e-3 epochs=500 \
    optimizer.name=muon model=transolver_experimental model.attn.enable_rope=false \
    model_attn_dim_heads=48 model_depth=6 targ_dim_model=192

# With RoPE
OMP_NUM_THREADS=1 python exp/pretrain/train.py datamod=shapenet_h5 \
    datamod.data_dir=../data/preprocessed_shapenet_h5 batch_size=120 accumulate_grad_batches=2 \
    datamod.num_workers=8 run_name=PerceiverModern optimizer.max_lr=8e-4 epochs=400 \
    optimizer.name=muon model=transolver_experimental \
    model.attn.enable_rope=true model_attn_dim_heads=48 model_depth=4 targ_dim_model=192

# with RoPE GPUx2
OMP_NUM_THREADS=1 python exp/pretrain/train.py datamod=shapenet_h5 \
    datamod.data_dir=../data/preprocessed_shapenet_h5 batch_size=240 accumulate_grad_batches=2 \
    datamod.num_workers=8 run_name=PerceiverModern optimizer.max_lr=8e-4 epochs=400 \
    optimizer.name=muon model=transolver_experimental \
    model.attn.enable_rope=true model_attn_dim_heads=48 model_depth=4 targ_dim_model=192