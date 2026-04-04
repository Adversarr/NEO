for examples in 30 90 120 300 480
do
    epochs=1000
    check_val=5
    python exp/downstream/shrec16_classification.py freeze_pretrained=true \
        ckpt_pretrain=checkpoints/Finals/Ours-Large.ckpt epochs=${epochs} check_val_every_n_epoch=${check_val} run_name=with-pretrain-${examples} \
        datamod.limit_train=${examples} &

    python exp/downstream/shrec16_classification.py \
        model=only_pos_embed epochs=${epochs} check_val_every_n_epoch=${check_val} run_name=without-pretrain-pointnet-${examples}\
        datamod.limit_train=${examples} &

    python exp/downstream/shrec16_classification.py \
        model=only_pos_embed epochs=${epochs} check_val_every_n_epoch=${check_val} run_name=without-pretrain-pointtransformer-${examples}\
        cls_model=point_transformer     \
        datamod.limit_train=${examples} &
    wait
done