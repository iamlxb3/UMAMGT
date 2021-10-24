# bash train_on_cn_novel_char_deduplicate.sh '256 512 1024 0' 0 0 3

char_freq_ranges=${1:-0}
is_change_apply_to_test=${2:-1}
is_debug=${3:-0}
repeat=${4:-1}

data_dir=../data/5billion
save_dir=../result/
dataset_name=cn_novel_5billion
classifier_name=cn_roberta
char_freq_txt_path=../data/sort_char.txt
semantic_change='char_deduplicate'

python3.6 train_cn_roberta.py --classifier_name $classifier_name \
                              --dataset_name $dataset_name \
                              --data_dir $data_dir \
                              --save_dir $save_dir \
                              --char_freq_txt_path $char_freq_txt_path \
                              --is_debug $is_debug \
                              --repeat $repeat \
                              --char_freq_ranges $char_freq_ranges \
                              --semantic_change $semantic_change \
                              --is_change_apply_to_test $is_change_apply_to_test \
                              --re_init_weights 0