# debug-[pretrain]-[repeat1]:
# bash train_on_cn.sh cn_novel_5billion '0' 1 1 0 1 1 'not_use_stopword'
# bash train_on_cn.sh cn_novel_5billion '0' 1 0 0 1 1

dataset_name=${1:-0}
char_freq_ranges=${2:-0}
is_change_apply_to_test=${3:-1}
is_debug=${4:-0}
re_init_weights=${5:-0}
repeat=${6:-1}
is_change_apply_to_train=${7:-1}
semantic_change=${8:-'None'}

data_dir=../data/$dataset_name
save_dir=../result/
classifier_name=cn_roberta
char_freq_txt_path=../data/$dataset_name/sort_char.txt

python3.6 train_roberta.py --classifier_name $classifier_name \
                           --dataset_name $dataset_name \
                           --data_dir $data_dir \
                           --save_dir $save_dir \
                           --char_freq_txt_path $char_freq_txt_path \
                           --is_debug $is_debug \
                           --repeat $repeat \
                           --char_freq_ranges $char_freq_ranges \
                           --semantic_change $semantic_change \
                           --is_change_apply_to_test $is_change_apply_to_test \
                           --re_init_weights $re_init_weights \
                           --is_change_apply_to_train $is_change_apply_to_train