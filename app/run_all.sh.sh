bash train_on_cn_novel_origin.sh                      '256 512 1024 0' 1 0 3
bash train_on_cn_novel_reorder_shuffle.sh             '256 512 1024 0' 1 0 3
bash train_on_cn_novel_char_deduplicate.sh            '256 512 1024 0' 1 0 3
bash train_on_cn_novel_reorder_freq_high2low.sh       '256 512 1024 0' 1 0 3
bash train_on_cn_novel_reorder_freq_low2high.sh       '256 512 1024 0' 1 0 3
bash train_on_cn_novel_reorder_shuffle+deduplicate.sh '256 512 1024 0' 1 0 3