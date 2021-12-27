# debug
# bash save_model_for_interpret.sh cn_novel_5billion cn_roberta 1

dataset_name=${1:-0}
classifier_name=${2:-0}
is_debug=${3:-0}

# model_save_dir
data_dir=../data/$dataset_name
char_freq_txt_path=../data/$dataset_name/sort_char.txt
model_save_dir=../model_ckpts/"$dataset_name"_"$classifier_name"_debug_"$is_debug"

python3.6 train_roberta.py --classifier_name $classifier_name \
                              --dataset_name $dataset_name \
                              --data_dir $data_dir \
                              --save_dir ../result/ \
                              --char_freq_txt_path $char_freq_txt_path \
                              --is_debug $is_debug \
                              --repeat 1 \
                              --char_freq_ranges '0' \
                              --semantic_change 'None' \
                              --is_change_apply_to_test 1 \
                              --re_init_weights 0 \
                              --is_change_apply_to_train 1 \
                              --is_save_record 0 \
                              --model_save_dir $model_save_dir