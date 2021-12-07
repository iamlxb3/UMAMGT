# bash run_all_not_apply_to_train.sh

bash train_on_cn_novel_reorder_shuffle.sh             '0' 1 0 0 15 0
bash train_on_cn_novel_char_deduplicate.sh            '0' 1 0 0 15 0
bash train_on_cn_novel_reorder_freq_high2low.sh       '0' 1 0 0 15 0
bash train_on_cn_novel_reorder_freq_low2high.sh       '0' 1 0 0 15 0
bash train_on_cn_novel_reorder_shuffle+deduplicate.sh '0' 1 0 0 15 0