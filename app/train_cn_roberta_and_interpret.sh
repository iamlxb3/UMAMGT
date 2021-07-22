bash train_cn_roberta_and_interpret.sh

set -e

top_freqs=(64 128 256 512 1024 2048 4096)
model_name='bert'
debug_N=2000
batch_size=1
max_text_length=512

data_dir=data_process_type
for ((i=0;i<${#top_freqs[@]};++i));do
    top_freq=${top_freqs[i]}
    data_dir=../data/5billion_shuffle_unique_top_$top_freq
    model_save_dir=../model_ckpts/cn-roberta-story-turning-train_5billion_shuffle_unique_top_$top_freq

    python3.6 train_cn_roberta.py --epoch 10 \
                                  --per_device_train_batch_size 32 \
                                  --gradient_accumulation_steps 4 \
                                  --model_save_dir $model_save_dir \
                                  --data_dir $data_dir

    python3.6 run_story_interpret.py --debug_N $debug_N \
                                     --batch_size $batch_size \
                                     --model_dir $model_save_dir \
                                     --max_text_length $max_text_length \
                                     --data_dir $data_dir \
                                     --model_name $model_name
done

