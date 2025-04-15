# dataset options: ['imagenet_c_test', 'imagenet_r', 'imagenet_sketch', 'imagenet_a', 'gaussian_noise', 'imagenet_c_val_mix', 'imagenet_c_val']
# MGTTA
adapt_lr=1e-3
dataset=imagenet_c_test
mgg_path=path_to_mgg_ckpt
CUDA_VISIBLE_DEVICES=0 python main.py \
    --batch_size 64 \
    --workers 8 \
    --data /data/imagenet/ \
    --data_corruption /data/imagenet-c/ \
    --output ./outputs/ \
    --algorithm mgtta \
    --tag /exp_tag/dataset_${dataset}/adapt_lr${adapt_lr} \
    --mgg_path $mgg_path \
    --adapt_lr $adapt_lr \
    --dataset $dataset


# train MGG
train_mgg_epoch=40
dataset=imagenet_c_val_mix #Training MGG on mixed ImageNet-C valiation set
used_data_num=128
bs=2
train_info_path=./train_info.pt
eval_adapt_lr=1e-3
train_mgg_lr=1e-2
train_adapt_lr=1e-4
eval_interval_epochs=1
CUDA_VISIBLE_DEVICES=0 python main.py \
    --batch_size $bs \
    --workers 8 \
    --data /data/imagenet/ \
    --data_corruption /data/imagenet-c/ \
    --output ./outputs/ \
    --algorithm train_mgg \
    --tag /dataset_${dataset}/used_data_num${used_data_num}/train_mgg_lr${train_mgg_lr}_train_adapt_lr${train_adapt_lr}/eval_adapt_lr${eval_adapt_lr}/ \
    --train_mgg_lr $train_mgg_lr --train_adapt_lr $train_adapt_lr --eval_adapt_lr $eval_adapt_lr \
    --train_mgg_epoch $train_mgg_epoch --used_data_num $used_data_num --eval_interval_epochs $eval_interval_epochs \
    --dataset $dataset \
    