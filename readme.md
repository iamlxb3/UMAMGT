老的transformer版本: transformers-3.3.0
新的transformer版本: transformers-4.8.2
torch==1.6.0
标签

label1: 是人写的 label0: 是机器写的

tensorboard --logdir . --bind_all

lab path: http://10.240.137.235:2009/lab?token=cfc9e3fb286739c3ebedf37153e6c4e2f229a1f2dd4d94b5

训练模型:
```python
python3.6 train_roberta.py --epoch 20 \
                              --per_device_train_batch_size 32 \
                              --gradient_accumulation_steps 4
```

可解释性分析:
```bash
args:
debug_N
batch_size
max_text_length

bash run_story_interpret.sh 0 8 512
```

Grover模型：https://github.com/rowanz/grover

从danlu向ubuntu传文件
scp -P 19657 -C -i /home/iamlxb3/.ssh/pujiashu_rsa  root@ai.danlu.netease.com:"/root/story_turing_test/result/*.csv" .

从ubuntu向danlu传文件
rsync -avh --exclude='.git/' -e "ssh -p 19657 -i ~/.ssh/pujiashu_rsa" /home/iamlxb3/temp_rsync_dir/story_turing_test/model_ckpts/interpret_cn_novel_5billion_cn_roberta_debug_0/ root@ai.danlu.netease.com:/root/story_turing_test/model_ckpts/interpret_cn_novel_5billion_cn_roberta_debug_0/
rsync -avh --exclude='.git/' -e "ssh -p 19657 -i ~/.ssh/pujiashu_rsa" /home/iamlxb3/temp_rsync_dir/story_turing_test/model_ckpts/interpret_en_grover_en_roberta_debug_0/ root@ai.danlu.netease.com:/root/story_turing_test/model_ckpts/interpret_en_grover_en_roberta_debug_0/
rsync -avh --exclude='.git/' -e "ssh -p 19657 -i ~/.ssh/pujiashu_rsa" /home/iamlxb3/temp_rsync_dir/story_turing_test/model_ckpts/interpret_en_writing_prompt_en_roberta_debug_0/ root@ai.danlu.netease.com:/root/story_turing_test/model_ckpts/interpret_en_writing_prompt_en_roberta_debug_0/