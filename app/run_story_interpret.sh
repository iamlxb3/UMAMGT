# bash run_story_interpret.sh 100
# bash run_story_interpret.sh 0 8 20

debug_N=${1:-0}
batch_size=${2:-1}
max_text_length=${3:-512}

set -e

model_dir='../model_ckpts/cn-roberta-story-turning-train_22789-seq_149'
model_name='bert' # 这里如果要用自己的模型，比如roberta，需要改一下

python3.6 run_story_interpret.py --debug_N $debug_N \
                                 --batch_size $batch_size \
                                 --model_dir $model_dir \
                                 --max_text_length $max_text_length \
                                 --model_name $model_name