
# Debug
# bash train_on_cn_novel_freq_gaps.sh '128 256 512' 1 1 'rm_chars_in_freq'
# bash train_on_cn_novel_freq_gaps.sh '128 256 512' 1 1 'rm_chars_out_freq'

char_freq_ranges=${1:-0}
is_debug=${2:-0}
repeat=${3:-1}
semantic_change=${4:-1}

data_dir=../data/5billion
save_dir=../result/
dataset_name=cn_novel_5billion
classifier_name=cn_roberta
char_freq_txt_path=../data/sort_char.txt

python3.6 train_cn_roberta_freq_gap.py --classifier_name $classifier_name \
                              --dataset_name $dataset_name \
                              --data_dir $data_dir \
                              --save_dir $save_dir \
                              --char_freq_txt_path $char_freq_txt_path \
                              --is_debug $is_debug \
                              --repeat $repeat \
                              --char_freq_ranges $char_freq_ranges \
                              --semantic_change $semantic_change