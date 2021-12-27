# bash run_story_interpret.sh 100
# bash run_story_interpret.sh 100 1 20
# bash run_story_interpret.sh 0 8 20

# bash run_story_interpret.sh 500 1 512
# bash run_story_interpret.sh 0 4 512

debug_N=${1:-0}
batch_size=${2:-1}
max_text_length=${3:-512}

set -e

model_dir='/home/iamlxb3/temp_rsync_dir/story_turing_test/model_ckpts/cn_novel_5billion_cn_roberta_debug_0'
data_dir='../data/cn_novel_5billion'
model_name='bert' # 这里如果要用自己的模型，比如roberta，需要改一下

python3.6 run_story_interpret.py --debug_N $debug_N \
                                 --batch_size $batch_size \
                                 --model_dir $model_dir \
                                 --max_text_length $max_text_length \
                                 --data_dir $data_dir \
                                 --model_name $model_name
