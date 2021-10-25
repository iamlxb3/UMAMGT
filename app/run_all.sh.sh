#char_freq_ranges=${1:-0}
#is_change_apply_to_test=${2:-1}
#is_debug=${3:-0}
#re_init_weights=${4:-0}
#repeat=${5:-1}

# apply to test
bash train_on_cn_novel_origin.sh                      '32 64 128 256 512 1024 0' 1 0 0 10
bash train_on_cn_novel_reorder_shuffle.sh             '32 64 128 256 512 1024 0' 1 0 0 10
bash train_on_cn_novel_char_deduplicate.sh            '32 64 128 256 512 1024 0' 1 0 0 10
bash train_on_cn_novel_reorder_freq_high2low.sh       '32 64 128 256 512 1024 0' 1 0 0 10
bash train_on_cn_novel_reorder_freq_low2high.sh       '32 64 128 256 512 1024 0' 1 0 0 10
bash train_on_cn_novel_reorder_shuffle+deduplicate.sh '32 64 128 256 512 1024 0' 1 0 0 10

# not apply to test
bash train_on_cn_novel_origin.sh                      '32 64 128 256 512 1024 0' 0 0 0 10
bash train_on_cn_novel_reorder_shuffle.sh             '32 64 128 256 512 1024 0' 0 0 0 10
bash train_on_cn_novel_char_deduplicate.sh            '32 64 128 256 512 1024 0' 0 0 0 10
bash train_on_cn_novel_reorder_freq_high2low.sh       '32 64 128 256 512 1024 0' 0 0 0 10
bash train_on_cn_novel_reorder_freq_low2high.sh       '32 64 128 256 512 1024 0' 0 0 0 10
bash train_on_cn_novel_reorder_shuffle+deduplicate.sh '32 64 128 256 512 1024 0' 0 0 0 10