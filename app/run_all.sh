#char_freq_ranges=${1:-0}
#is_change_apply_to_test=${2:-1}
#is_debug=${3:-0}
#re_init_weights=${4:-0}
#repeat=${5:-1}

# apply to test (pretrain)
bash train_on_cn_novel_origin.sh                      '16 256 1024 0' 1 0 0 3
bash train_on_cn_novel_reorder_shuffle.sh             '16 256 1024 0' 1 0 0 3
bash train_on_cn_novel_char_deduplicate.sh            '16 256 1024 0' 1 0 0 3
bash train_on_cn_novel_reorder_freq_high2low.sh       '16 256 1024 0' 1 0 0 3
bash train_on_cn_novel_reorder_freq_low2high.sh       '16 256 1024 0' 1 0 0 3
bash train_on_cn_novel_reorder_shuffle+deduplicate.sh '16 256 1024 0' 1 0 0 3
bash train_on_cn_novel_likelihood_rank.sh             '16 256 1024 0' 1 0 0 3

# apply to test (not pretrain)
bash train_on_cn_novel_origin.sh                      '16 256 1024 0' 1 0 1 3 #  non pre-train
bash train_on_cn_novel_reorder_shuffle.sh             '16 256 1024 0' 1 0 1 3
bash train_on_cn_novel_char_deduplicate.sh            '16 256 1024 0' 1 0 1 3
bash train_on_cn_novel_reorder_freq_high2low.sh       '16 256 1024 0' 1 0 1 3
bash train_on_cn_novel_reorder_freq_low2high.sh       '16 256 1024 0' 1 0 1 3
bash train_on_cn_novel_reorder_shuffle+deduplicate.sh '16 256 1024 0' 1 0 1 3
bash train_on_cn_novel_likelihood_rank.sh             '16 256 1024 0' 1 0 1 3

#char_freq_ranges=${1:-0}
#is_debug=${2:-0}
#repeat=${3:-1}
#semantic_change=${4:-1}

# check freq gap
bash train_on_cn_novel_freq_gaps.sh                   '8 16 32 64' 0 15 'rm_chars_in_freq'  'cn_novel_5billion' 'cn_roberta' # 0
bash train_on_cn_novel_freq_gaps.sh                   '8 16 32 64' 0 15 'rm_chars_out_freq' 'cn_novel_5billion' 'cn_roberta' # 1
bash train_on_cn_novel_freq_gaps.sh                   '8 16 32 64' 0 15 'rm_chars_in_freq'  'en_grover' 'en_roberta' # 2
bash train_on_cn_novel_freq_gaps.sh                   '8 16 32 64' 0 15 'rm_chars_out_freq' 'en_grover' 'en_roberta' # 3
bash train_on_cn_novel_freq_gaps.sh                   '8 16 32 64' 0 15 'rm_chars_in_freq'  'en_writing_prompt' 'en_roberta' # 4
bash train_on_cn_novel_freq_gaps.sh                   '8 16 32 64' 0 15 'rm_chars_out_freq' 'en_writing_prompt' 'en_roberta' # 5
## non pre-train model
#bash train_on_cn_novel_reorder_shuffle.sh             '0' 1 0 1 5
#bash train_on_cn_novel_char_deduplicate.sh            '0' 1 0 1 5
#bash train_on_cn_novel_reorder_freq_high2low.sh       '0' 1 0 1 5
#bash train_on_cn_novel_reorder_freq_low2high.sh       '0' 1 0 1 5
#bash train_on_cn_novel_reorder_shuffle+deduplicate.sh '0' 1 0 1 5

# not apply to test
#bash train_on_cn_novel_reorder_shuffle.sh             '0' 0 0 0 5
#bash train_on_cn_novel_char_deduplicate.sh            '0' 0 0 0 5
#bash train_on_cn_novel_reorder_freq_high2low.sh       '0' 0 0 0 5
#bash train_on_cn_novel_reorder_freq_low2high.sh       '0' 0 0 0 5
#bash train_on_cn_novel_reorder_shuffle+deduplicate.sh '0' 0 0 0 5

# }

