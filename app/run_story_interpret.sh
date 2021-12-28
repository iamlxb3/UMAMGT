# bash run_story_interpret.sh 100
# bash run_story_interpret.sh 100 1 20 cn_novel_5billion cn_novel_5billion_cn_roberta_debug_0 bert
# bash run_story_interpret.sh 100 1 20 en_grover interpret_en_grover_en_roberta_debug_1 roberta
# bash run_story_interpret.sh 50 1 256 en_grover interpret_en_grover_en_roberta_debug_1 roberta 50

# bash run_story_interpret.sh 500 1 512
# bash run_story_interpret.sh 0 4 512

debug_N=${1:-0}
batch_size=${2:-1}
max_text_length=${3:-512}
data_name=${4:-0}
model_name=${5:-0}
model_type=${6:-0}
ig_n_steps=${7:-100}
use_pad_baseline=${8:-1}

set -e

model_dir="../model_ckpts/$model_name"
data_dir="../data/$data_name"

python3.6 run_story_interpret.py --debug_N $debug_N \
                                 --batch_size $batch_size \
                                 --model_dir $model_dir \
                                 --max_text_length $max_text_length \
                                 --data_dir $data_dir \
                                 --model_type $model_type \
                                 --ig_n_steps $ig_n_steps \
                                 --use_pad_baseline $use_pad_baseline
