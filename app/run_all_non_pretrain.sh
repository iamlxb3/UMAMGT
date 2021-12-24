# bash run_all_non_pretrain.sh

bash train_on_cn_novel_origin.sh                      '16 256 1024 0' 1 0 1 15 #  non pre-train
bash train_on_cn_novel_reorder_shuffle.sh             '16 256 1024 0' 1 0 1 15
bash train_on_cn_novel_char_deduplicate.sh            '16 256 1024 0' 1 0 1 15
bash train_on_cn_novel_reorder_freq_high2low.sh       '16 256 1024 0' 1 0 1 15
bash train_on_cn_novel_reorder_freq_low2high.sh       '16 256 1024 0' 1 0 1 15
bash train_on_cn_novel_reorder_shuffle+deduplicate.sh '16 256 1024 0' 1 0 1 15
bash train_on_cn_novel_likelihood_rank.sh             '16 256 1024 0' 1 0 1 15