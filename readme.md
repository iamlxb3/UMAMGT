老的transformer版本: transformers-3.3.0

tensorboard --logdir . --bind_all

lab path: http://10.240.137.235:2009/lab?token=cfc9e3fb286739c3ebedf37153e6c4e2f229a1f2dd4d94b5

训练模型:
```python
python3.6 train_cn_roberta.py --epoch 20 \
                              --per_device_train_batch_size 32 \
                              --gradient_accumulation_steps 4
```

可解释性分析:
```python
python3.6 train_cn_roberta.py --epoch 20 \
                              --per_device_train_batch_size 32 \
                              --gradient_accumulation_steps 4
```