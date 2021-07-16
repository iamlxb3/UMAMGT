# bash run_story_interpret.sh 100

debug_N=${1:-0}
set -e

batch_size=1
model_dir='../model_ckpts/cn-roberta-story-turning-train_22789-seq_149'
max_text_length=70

python3.6 run_story_interpret.py --debug_N $debug_N \
                                 --batch_size $batch_size \
                                 --model_dir $model_dir \
                                 --max_text_length $max_text_length