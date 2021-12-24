# RUN DEBUG
# bash run_en_all.sh 'en_grover' 1 0 '0'
# bash run_en_all.sh 'en_writing_prompt' 1 0 '0'

# RUN PRETRAIN EN_GROVER
# bash run_en_all.sh 'en_grover' 0 0 '16 256 1024 0'

# RUN NON PRETRAIN EN_GROVER
# bash run_en_all.sh 'en_grover' 0 1 '16 256 1024 0'

# RUN PRETRAIN en_writing_prompt
# bash run_en_all.sh 'en_writing_prompt' 0 0 '16 256 1024 0'

# RUN NON PRETRAIN en_writing_prompt
# bash run_en_all.sh 'en_writing_prompt' 0 1 '16 256 1024 0'

dataset_name=${1:-0}
is_debug=${2:-0}
re_init_weights=${3:-0}
char_freq_ranges=${4:-'0'}

set -e

if [ $is_debug == 1 ]
then
  repeat=1
else
  repeat=15
fi

bash train_on_en.sh $dataset_name '0' 1 $is_debug $re_init_weights $repeat 1 'dep'
bash train_on_en.sh $dataset_name '0' 1 $is_debug $re_init_weights $repeat 1 'pos'
bash train_on_en.sh $dataset_name '0' 1 $is_debug $re_init_weights $repeat 1 'constit'
bash train_on_en.sh $dataset_name '0' 1 $is_debug $re_init_weights $repeat 1 'ner'

bash train_on_en.sh $dataset_name $char_freq_ranges 1 $is_debug $re_init_weights $repeat 1 'None'
bash train_on_en.sh $dataset_name $char_freq_ranges 1 $is_debug $re_init_weights $repeat 1 'char_deduplicate'
bash train_on_en.sh $dataset_name $char_freq_ranges 1 $is_debug $re_init_weights $repeat 1 'reorder_shuffle'
bash train_on_en.sh $dataset_name $char_freq_ranges 1 $is_debug $re_init_weights $repeat 1 'reorder_shuffle char_deduplicate'
bash train_on_en.sh $dataset_name $char_freq_ranges 1 $is_debug $re_init_weights $repeat 1 'reorder_freq_low2high'
bash train_on_en.sh $dataset_name $char_freq_ranges 1 $is_debug $re_init_weights $repeat 1 'reorder_freq_high2low'

if [ $dataset_name == 'en_grover' ]
then
  bash train_on_en.sh $dataset_name $char_freq_ranges 1 $is_debug $re_init_weights $repeat 1 'likelihood_rank'
fi