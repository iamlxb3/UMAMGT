

# RUN PRETRAIN EN_GROVER
#
# CUDA_VISIBLE_DEVICES=0 bash run_en_all_no_spacy_no_pretrain.sh 'en_grover'         0 1

# RUN PRETRAIN en_writing_prompt
# 188431
# CUDA_VISIBLE_DEVICES=1 bash run_en_all_no_spacy_no_pretrain.sh 'en_writing_prompt' 0 1

dataset_name=${1:-0}
is_debug=${2:-0}
re_init_weights=${3:-0}


set -e

if [ $is_debug == 1 ]
then
  repeat=1
else
  repeat=15
fi

bash train_on_en.sh $dataset_name '16 256 1024 0' 1 $is_debug $re_init_weights $repeat 1 'char_deduplicate'
bash train_on_en.sh $dataset_name '16 256 1024 0' 1 $is_debug $re_init_weights $repeat 1 'reorder_shuffle'
bash train_on_en.sh $dataset_name '16 256 1024 0' 1 $is_debug $re_init_weights $repeat 1 'reorder_shuffle char_deduplicate'
bash train_on_en.sh $dataset_name '16 256 1024 0' 1 $is_debug $re_init_weights $repeat 1 'reorder_freq_low2high'
bash train_on_en.sh $dataset_name '16 256 1024 0' 1 $is_debug $re_init_weights $repeat 1 'reorder_freq_high2low'

if [ $dataset_name == 'en_grover' ]
then
  bash train_on_en.sh $dataset_name '16 256 1024 0' 1 $is_debug $re_init_weights $repeat 1 'likelihood_rank'
fi