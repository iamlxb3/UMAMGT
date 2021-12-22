

# RUN PRETRAIN EN_GROVER
# 12-21: run on a8c02b65
# bash run_en_spacy.sh 'en_grover' 0 0 'dep' # 数据量没有改
# 12-21: run on
# bash run_en_spacy.sh 'en_grover' 0 0 'pos' # 数据量没有改
# 188455
# bash run_en_spacy.sh 'en_grover' 0 0 'constit'
# 188456
# bash run_en_spacy.sh 'en_grover' 0 0 'ner'

# RUN PRETRAIN writing_prompt
# bash run_en_spacy.sh 'en_writing_prompt' 0 0 'dep'
# bash run_en_spacy.sh 'en_writing_prompt' 0 0 'pos'
# bash run_en_spacy.sh 'en_writing_prompt' 0 0 'constit'
# bash run_en_spacy.sh 'en_writing_prompt' 0 0 'ner'

dataset_name=${1:-0}
is_debug=${2:-0}
re_init_weights=${3:-0}
task=${4:-'0'}

set -e

if [ $is_debug == 1 ]
then
  repeat=1
else
  repeat=15
fi

bash train_on_en.sh $dataset_name '0' 1 $is_debug $re_init_weights $repeat 1 $task
