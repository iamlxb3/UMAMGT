# bash run_pos_dep_constit_ner.sh
set -e

# pre-train model
bash train_on_cn_novel_dep.sh     '0' 1 0 0 15
bash train_on_cn_novel_pos.sh     '0' 1 0 0 15
bash train_on_cn_novel_constit.sh '0' 1 0 0 15
bash train_on_cn_novel_ner.sh     '0' 1 0 0 15

# non-pre-train model
bash train_on_cn_novel_dep.sh     '0' 1 0 1 15
bash train_on_cn_novel_pos.sh     '0' 1 0 1 15
bash train_on_cn_novel_constit.sh '0' 1 0 1 15
bash train_on_cn_novel_ner.sh     '0' 1 0 1 15