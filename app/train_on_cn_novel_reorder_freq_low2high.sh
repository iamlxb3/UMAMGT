# bash train_on_cn_novel.sh

data_dir=../data/5billion
save_dir=../result/
dataset_name=cn_novel_5billion
classifier_name=cn_roberta
char_freq_txt_path=../data/sort_char.txt
repeat=1
is_debug=1
#semantic_change='reorder_shuffle char_deduplicate'
semantic_change='reorder_freq_high2low'
char_freq_ranges='64 256'
is_change_apply_to_test=0

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