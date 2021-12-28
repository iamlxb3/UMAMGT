# bash compute_idf.sh
set -e

python compute_idf.py --dataset cn_novel_5billion --model cn_roberta
python compute_idf.py --dataset en_grover --model en_roberta
python compute_idf.py --dataset en_writing_prompt --model en_roberta
python compute_idf.py --dataset cn_novel_5billion
python compute_idf.py --dataset en_grover
python compute_idf.py --dataset en_writing_prompt