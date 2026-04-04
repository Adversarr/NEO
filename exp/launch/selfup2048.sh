python exp/selfup/train_rronly.py batch_size=96 \
    datamod.num_workers=10 datamod.data_dir=../data/preprocessed_shapenet_h5/ \
    datamod.n_points=2048 run_name=rr-2048 epochs=50 \
    n_samples=1024 \
    ckpt_pretrain=checkpoints/PerModern/PerModern-Pretrain.ckpt \
    balancing_delta=3 accumulate_grad_batches=4 \
    scheduler.name=onecycle