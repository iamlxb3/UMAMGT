# bash temp_run_12_27.sh

set -e
#bash save_model_for_interpret.sh en_grover en_roberta 0
#bash save_model_for_interpret.sh en_writing_prompt en_roberta 0
# bash run_story_interpret.sh 500 1 128 cn_novel_5billion interpret_cn_novel_5billion_cn_roberta_debug_0 bert 100 1
# bash run_story_interpret.sh 800 1 256 en_grover interpret_en_grover_en_roberta_debug_0 roberta 40 1
# bash run_story_interpret.sh 800 1 128 en_writing_prompt interpret_en_writing_prompt_en_roberta_debug_0 roberta 100 1

bash run_story_interpret.sh 10000 1 128 cn_novel_5billion interpret_cn_novel_5billion_cn_roberta_debug_0 bert 100 0
bash run_story_interpret.sh 10000 1 256 en_grover interpret_en_grover_en_roberta_debug_0 roberta 40 0
bash run_story_interpret.sh 10000 1 128 en_writing_prompt interpret_en_writing_prompt_en_roberta_debug_0 roberta 100 0

bash run_story_interpret.sh 10000 1 128 cn_novel_5billion interpret_cn_novel_5billion_cn_roberta_debug_0 bert 100 1
bash run_story_interpret.sh 10000 1 256 en_grover interpret_en_grover_en_roberta_debug_0 roberta 40 1
bash run_story_interpret.sh 10000 1 128 en_writing_prompt interpret_en_writing_prompt_en_roberta_debug_0 roberta 100 1
