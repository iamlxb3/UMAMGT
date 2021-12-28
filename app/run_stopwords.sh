# RUN PRETRAIN en_writing_prompt
# bash run_stopwords.sh 'cn_novel_5billion' 0 0
# bash run_stopwords.sh 'en_writing_prompt' 0 0
# bash run_stopwords.sh 'en_grover' 0 0

# RUN NON PRETRAIN en_writing_prompt
# bash run_stopwords.sh 'cn_novel_5billion' 0 1
# bash run_stopwords.sh 'en_writing_prompt' 0 1
# bash run_stopwords.sh 'en_grover' 0 1

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

if [ $dataset_name == "cn_novel_5billion" ]
then
  echo "run train on cn"
  bash train_on_cn.sh $dataset_name '0' 1 $is_debug $re_init_weights $repeat 1 'use_stopword'
  bash train_on_cn.sh $dataset_name '0' 1 $is_debug $re_init_weights $repeat 1 'not_use_stopword'
else
  echo "run train on en"
  bash train_on_en.sh $dataset_name '0' 1 $is_debug $re_init_weights $repeat 1 'use_stopword'
  bash train_on_en.sh $dataset_name '0' 1 $is_debug $re_init_weights $repeat 1 'not_use_stopword'
fi


