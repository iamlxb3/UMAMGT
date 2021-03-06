char_freq_ranges=${1:-0}
is_debug=${2:-0}
repeat=${3:-1}
semantic_change=${4:-1}
dataset_name=${5:-1}
classifier_name=${6:-1}

data_dir=../data/$dataset_name
save_dir=../result/
char_freq_txt_path=../data/$dataset_name/sort_char.txt

python3.6 restrict_char_freq_region.py --classifier_name $classifier_name \
                                       --dataset_name $dataset_name \
                                       --data_dir $data_dir \
                                       --save_dir $save_dir \
                                       --char_freq_txt_path $char_freq_txt_path \
                                       --is_debug $is_debug \
                                       --repeat $repeat \
                                       --char_freq_ranges $char_freq_ranges \
                                       --semantic_change $semantic_change