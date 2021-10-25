import os
import sys
import ipdb
import random
import shutil
import ntpath
import numpy as np
import argparse

sys.path.append('..')

from core.task import StoryTuringTest
from core.utils import load_save_json
from core.exp_record import ExpRecorder
from core.semantic_modifier import SemanticModifier
from exp_config import TRAIN_CONFIG, TRAIN_DEBUG_CONFIG, SYSTEM_CONFIG, SEED_OFFSET, TEST_PERCENT, VAL_PERCENT, \
    MAX_SEQ_LENGTH

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from pytorch_lightning import seed_everything


# python3.6 train_cn_roberta.py --epoch 1 --debug_N 100
# python3.6 train_cn_roberta.py --epoch 20 --per_device_train_batch_size 32 --gradient_accumulation_steps 4

# python3.6 train_cn_roberta.py --epoch 10 --per_device_train_batch_size 32 --gradient_accumulation_steps 4 --data_dir ../data/5billion_sort
# python3.6 train_cn_roberta.py --epoch 10 --per_device_train_batch_size 32 --gradient_accumulation_steps 4 --data_dir ../data/5billion_sort_no_reverse
# python3.6 train_cn_roberta.py --epoch 10 --per_device_train_batch_size 32 --gradient_accumulation_steps 4 --data_dir ../data/5billion_sort_unique_no_reverse
# python3.6 train_cn_roberta.py --epoch 10 --per_device_train_batch_size 32 --gradient_accumulation_steps 4 --data_dir ../data/5billion_shuffle_unique_no_reverse

def compute_metrics(eval_predict):
    predict_prob, labels = eval_predict
    predict_label = np.argmax(predict_prob, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=predict_label)
    recall = recall_score(y_true=labels, y_pred=predict_label)
    precision = precision_score(y_true=labels, y_pred=predict_label)
    f1 = f1_score(y_true=labels, y_pred=predict_label)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


def args_parse():
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--char_freq_txt_path', type=str)
    parser.add_argument('--classifier_name', type=str, required=True, choices=['cn_roberta'])
    parser.add_argument('--dataset_name', type=str, required=True, choices=['cn_novel_5billion'])
    parser.add_argument('--is_debug', type=int, default=0)
    parser.add_argument('--char_freq_ranges', nargs='+', type=int, default=[0])
    parser.add_argument('--repeat', type=int, default=1, help='repeat with random seads')
    parser.add_argument('--semantic_change',
                        type=str,
                        required=True)
    args = parser.parse_args()
    return args


def main():
    args = args_parse()

    # config
    classifier_name = args.classifier_name
    dataset_name = args.dataset_name
    repeat = args.repeat
    is_debug = args.is_debug
    is_debug = True if is_debug else False
    is_change_apply_to_test = True
    re_init_weights = False
    data_dir = args.data_dir
    semantic_change = args.semantic_change
    char_freq_ranges = args.char_freq_ranges
    save_dir = args.save_dir
    char_freq_txt_path = args.char_freq_txt_path
    save_dir = os.path.abspath(save_dir)

    # read char frequencies
    char_freq = {}
    with open(char_freq_txt_path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            char_freq[line] = i

    if is_debug:
        train_config = TRAIN_DEBUG_CONFIG
    else:
        train_config = TRAIN_CONFIG

    if classifier_name == 'cn_roberta':
        hugginface_model_id = 'hfl/chinese-roberta-wwm-ext'
    else:
        raise NotImplementedError

    if re_init_weights:
        classifier_name += '_no_pretrain'

    semantic_change_str = semantic_change
    semantic_modifier = SemanticModifier(semantic_change, char_freq=char_freq)

    tokenizer = AutoTokenizer.from_pretrained(hugginface_model_id)

    # 直接用tokenizer的vocab看来也不行，因为前面都是ascii码，后面也不是完全按照词频排序的样子
    # tokenizer_keys = list(tokenizer.vocab.keys())

    # read whole dataset into memory

    # (0.) read dataset
    story_turing_test = StoryTuringTest(tokenizer, dataset_name=dataset_name)
    whole_texts, whole_labels = story_turing_test.read_cn_novel_whole_data(data_dir)
    if is_debug:
        whole_texts, whole_labels = whole_texts[:200], whole_labels[:200]

    origin_all_indices = np.arange(0, len(whole_texts))

    # (1.) init experiment recorder
    exp_recorder = ExpRecorder()

    # (2.) TODO, maybe load model parameters
    for char_freq_range in char_freq_ranges:
        for repeat_i in range(repeat):

            # config tmp output dir
            tmp_ckpts_dir = f'../model_ckpts/tmp/freq_gap_{semantic_change_str}_{repeat_i}'

            repeat_seed = SEED_OFFSET + repeat_i

            # (0.) seed everything
            seed_everything(repeat_seed)

            # (1.) shuffle data and split train/val/test
            shuffle_indices = origin_all_indices.copy()

            random.shuffle(shuffle_indices)
            shuffle_whole_texts = whole_texts.copy()[shuffle_indices]
            shuffle_whole_labels = whole_labels.copy()[shuffle_indices]

            test_size = int(TEST_PERCENT * len(whole_texts))
            val_size = int(VAL_PERCENT * len(whole_texts))

            train_texts = shuffle_whole_texts[:len(whole_texts) - test_size - val_size]
            train_labels = shuffle_whole_labels[:len(whole_texts) - test_size - val_size]
            val_texts = shuffle_whole_texts[len(whole_texts) - test_size - val_size:len(whole_texts) - test_size]
            val_labels = shuffle_whole_labels[len(whole_texts) - test_size - val_size:len(whole_texts) - test_size]
            test_texts = shuffle_whole_texts[len(whole_texts) - test_size:]
            test_labels = shuffle_whole_labels[len(whole_texts) - test_size:]

            # apply semantic change

            if semantic_change_str == 'rm_chars_in_freq':
                train_texts = semantic_modifier.rm_chars_in_freq(train_texts, char_freq_range)
                val_texts = semantic_modifier.rm_chars_in_freq(val_texts, char_freq_range)
                test_texts = semantic_modifier.rm_chars_in_freq(test_texts, char_freq_range)
            elif semantic_change_str == 'rm_chars_out_freq':
                train_texts = semantic_modifier.rm_chars_out_freq(train_texts, char_freq_range)
                val_texts = semantic_modifier.rm_chars_out_freq(val_texts, char_freq_range)
                test_texts = semantic_modifier.rm_chars_out_freq(test_texts, char_freq_range)
            else:
                raise NotImplementedError

            train_dataset = story_turing_test.create_dataset(train_texts, train_labels, max_length=MAX_SEQ_LENGTH)
            val_dataset = story_turing_test.create_dataset(val_texts, val_labels, max_length=MAX_SEQ_LENGTH)
            test_dataset = story_turing_test.create_dataset(test_texts, test_labels, max_length=MAX_SEQ_LENGTH)
            print(f"Train size: {train_dataset.size}, val size: {val_dataset.size}, test size: {test_dataset.size}")
            print(
                f"Train label: {train_dataset.label}, val label: {val_dataset.label}, test label: {test_dataset.label}")

            # (1.) init model
            model = AutoModelForSequenceClassification.from_pretrained(hugginface_model_id, num_labels=2)
            if re_init_weights:
                model.init_weights()
                print("Re init weights for model done!")
            print(model)
            # BertEncoder.layer(ModuleList) -> BertLayer.attention (crossattention) -> BertAttention.self -> BertSelfAttention

            # (2.) train model
            # reference: https://huggingface.co/transformers/v3.0.2/main_classes/trainer.html
            training_args = TrainingArguments(
                output_dir=tmp_ckpts_dir,
                # output directory
                num_train_epochs=train_config['epoch'],  # total number of training epochs
                per_device_train_batch_size=SYSTEM_CONFIG['per_device_train_batch_size'],
                # batch size per device during training
                gradient_accumulation_steps=SYSTEM_CONFIG['gradient_accumulation_steps'],
                per_device_eval_batch_size=SYSTEM_CONFIG['per_device_train_batch_size'] * 2,
                # batch size for evaluation
                warmup_steps=50,  # number of warmup steps for learning rate scheduler
                weight_decay=0.01,  # strength of weight decay
                # directory for storing logs
                logging_steps=train_config['logging_steps'],
                evaluation_strategy='steps',
                eval_steps=train_config['eval_steps'],
                save_total_limit=1,
                load_best_model_at_end=True
            )
            trainer = Trainer(
                model=model,  # the instantiated 🤗 Transformers model to be trained
                compute_metrics=compute_metrics,
                args=training_args,  # training arguments, defined above
                train_dataset=train_dataset,  # training dataset
                eval_dataset=val_dataset  # evaluation dataset
            )
            train_result = trainer.train()
            train_result = dict(train_result._asdict())
            print(f"train_result: {train_result}")
            test_result = trainer.evaluate(test_dataset, metric_key_prefix='test')
            print(f"test_result: {test_result}")
            exp_recorder.add_one_repeat_result(repeat_i,
                                               is_change_apply_to_test,
                                               classifier_name,
                                               dataset_name,
                                               semantic_change,
                                               char_freq_range,
                                               test_result['test_accuracy'],
                                               test_result['test_f1'],
                                               train_result['training_loss'],
                                               test_result['test_loss'],
                                               train_dataset.size,
                                               val_dataset.size,
                                               test_dataset.size)

            # remove temp save dir
            shutil.rmtree(tmp_ckpts_dir)
            print(f"Remove temp dir {tmp_ckpts_dir} SUCCESS!!!!")

    # save path
    save_path = os.path.join(save_dir, f'freq_gap_{dataset_name}_{classifier_name}_{semantic_change_str}.csv')
    exp_recorder.save_to_disk(save_path)

    # # Save model
    # dataset_name = ntpath.basename(data_dir)
    # if model_save_dir is None:
    #     model_save_dir = f"../model_ckpts/cn-roberta-story-turning-train_{train_size}-seq_{seq_len}_{dataset_name}"
    # model_save_dir = os.path.abspath(model_save_dir)
    # model.save_pretrained(model_save_dir)
    # print(f"Save best model ckpt to {model_save_dir}")
    #
    # train_result_save_path = os.path.join(model_save_dir, 'train_result.json')
    # test_result_save_path = os.path.join(model_save_dir, 'test_result.json')
    # load_save_json(train_result_save_path, 'save', data=train_result)
    # load_save_json(test_result_save_path, 'save', data=test_result)


if __name__ == '__main__':
    import time

    t1 = time.time()
    main()
    t2 = time.time()
    print(f'Total minutes: {(t1 - t2) / 60.0}')
