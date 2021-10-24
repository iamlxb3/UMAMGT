# bash train_cn_roberta.sh

# top_freq=256
# data_dir=../data/5billion_shuffle_unique_top_$top_freq
# model_save_dir=../model_ckpts/cn-roberta-story-turning-train_5billion_shuffle_unique_top_"$top_freq"_re_init

data_dir=../data/5billion
model_save_dir=../model_ckpts/cn-roberta-story-turning-train_5billion_re_init


python3.6 train_cn_roberta.py --epoch 10 \
                              --per_device_train_batch_size 32 \
                              --gradient_accumulation_steps 4 \
                              --model_save_dir $model_save_dir \
                              --data_dir $data_dir \
                              --re_init_weights 1