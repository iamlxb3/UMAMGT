import os
import sys
import random
import shutil
import numpy as np
import argparse

sys.path.append('..')

from core.task import StoryTuringTest
from core.exp_record import ExpRecorder
from core.semantic_modifier import SemanticModifier
from exp_config import TRAIN_CONFIG, TRAIN_DEBUG_CONFIG, SYSTEM_CONFIG, SEED_OFFSET, TEST_PERCENT, VAL_PERCENT, \
    MAX_SEQ_LENGTH

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from pytorch_lightning import seed_everything


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
    parser.add_argument('--classifier_name', type=str, required=True, choices=['cn_roberta', 'en_roberta'])
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['cn_novel_5billion', 'en_grover', 'en_writing_prompt'])
    parser.add_argument('--is_debug', type=int, default=0)
    parser.add_argument('--repeat', type=int, default=1, help='repeat with random seads')
    parser.add_argument('--char_freq_ranges', nargs='+', type=int, default=[0])
    parser.add_argument('--semantic_change',
                        nargs='+',
                        type=str,
                        choices=['reorder_shuffle',
                                 'reorder_freq_high2low',
                                 'reorder_freq_low2high',
                                 'char_deduplicate',
                                 'None',
                                 'likelihood_rank',
                                 'pos',
                                 'dep',
                                 'constit',  # phrase structure tree, constituency tree,
                                 'ner',
                                 'use_stopword',
                                 'not_use_stopword'
                                 ],
                        required=True)
    parser.add_argument('--is_change_apply_to_test', type=int, default=0)
    parser.add_argument('--is_change_apply_to_train', type=int, default=1)
    parser.add_argument('--is_save_record', type=int, default=1)
    parser.add_argument('--re_init_weights', type=int, default=0)
    parser.add_argument('--model_save_dir', type=str)
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
    is_change_apply_to_test = args.is_change_apply_to_test
    is_change_apply_to_test = True if is_change_apply_to_test else False
    is_change_apply_to_train = True if args.is_change_apply_to_train else False
    data_dir = args.data_dir
    re_init_weights = args.re_init_weights
    semantic_change = args.semantic_change
    char_freq_ranges = args.char_freq_ranges
    save_dir = args.save_dir
    char_freq_txt_path = args.char_freq_txt_path
    model_save_dir = args.model_save_dir
    is_save_record = args.is_save_record
    save_dir = os.path.abspath(save_dir)

    for x in {'likelihood_rank', 'pos', 'dep'}:
        if x in semantic_change:
            is_change_apply_to_test = True

    if 'pos' in semantic_change or 'dep' in semantic_change:
        assert char_freq_ranges == [0]

    # read char frequencies
    char_freq_rank = {}
    with open(char_freq_txt_path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            char_freq_rank[line] = i

    if is_debug:
        train_config = TRAIN_DEBUG_CONFIG
    else:
        train_config = TRAIN_CONFIG

    if classifier_name == 'cn_roberta':
        hugginface_model_id = 'hfl/chinese-roberta-wwm-ext'
        language = 'cn'
    elif classifier_name == 'en_roberta':
        hugginface_model_id = 'roberta-base'
        language = 'en'
    else:
        raise NotImplementedError

    if re_init_weights:
        classifier_name += '_no_pretrain'

    semantic_change_str = '_'.join(semantic_change)

    # set save path
    # save path
    save_name = f'{dataset_name}_{classifier_name}_{semantic_change_str}' \
                f'_{is_change_apply_to_train}_{is_change_apply_to_test}'
    if is_debug:
        save_name = save_name + '_debug.csv'
    else:
        save_name = save_name + '.csv'
    save_path = os.path.join(save_dir, save_name)

    if os.path.isfile(save_path):
        print(f"=" * 78)
        print(f"{save_path} exist. Skip Training!!!")
        print(f"=" * 78)
        return

    semantic_modifier = SemanticModifier(semantic_change, language, char_freq_rank=char_freq_rank)

    load_complete = False
    try_count = 0
    while not load_complete:
        try:
            tokenizer = AutoTokenizer.from_pretrained(hugginface_model_id)
            try_count += 1
            if try_count >= 5:
                raise Exception(f"Load auto tokenizer failed!")
        except Exception as e:
            print(f"Load tokenizer failed, count: {try_count}, exception: {e}")
            time.sleep(1)
        else:
            load_complete = True

    # (0.) read dataset
    story_turing_test = StoryTuringTest(tokenizer, dataset_name=dataset_name)
    whole_texts, whole_labels = story_turing_test.read_cn_novel_whole_data(data_dir, semantic_change)

    if dataset_name == 'en_grover':
        whole_texts, whole_labels = whole_texts[:28000], whole_labels[:28000]
        print(f"Set en grover dataset size to 28000")

    if is_debug:
        whole_texts, whole_labels = whole_texts[:200], whole_labels[:200]

    origin_all_indices = np.arange(0, len(whole_texts))

    # (1.) init experiment recorder
    exp_recorder = ExpRecorder()

    # (2.) TODO, maybe load model parameters
    for char_freq_range in char_freq_ranges:
        for repeat_i in range(repeat):

            # config tmp output dir
            tmp_ckpts_dir = f'../model_ckpts/tmp/{dataset_name}_{classifier_name}_' \
                            f'{char_freq_range}_{semantic_change_str}_{repeat_i}'

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
            if is_change_apply_to_train:
                if semantic_change != ['None'] or char_freq_range != 0:
                    train_texts = semantic_modifier.change_texts(train_texts, char_freq_range)
                    val_texts = semantic_modifier.change_texts(val_texts, char_freq_range)

            if is_change_apply_to_test:
                if semantic_change != ['None'] or char_freq_range != 0:
                    test_texts = semantic_modifier.change_texts(test_texts, char_freq_range)

            train_dataset = story_turing_test.create_dataset(train_texts, train_labels, max_length=MAX_SEQ_LENGTH)
            val_dataset = story_turing_test.create_dataset(val_texts, val_labels, max_length=MAX_SEQ_LENGTH)
            test_dataset = story_turing_test.create_dataset(test_texts, test_labels, max_length=MAX_SEQ_LENGTH)
            print(f"Train size: {train_dataset.size}, val size: {val_dataset.size}, test size: {test_dataset.size}")
            print(
                f"Train label: {train_dataset.label}, val label: {val_dataset.label}, test label: {test_dataset.label}")

            # (1.) init model

            load_complete = False
            try_count = 0
            while not load_complete:
                try:
                    model = AutoModelForSequenceClassification.from_pretrained(hugginface_model_id, num_labels=2)
                    try_count += 1
                    if try_count >= 5:
                        raise Exception(f"Load auto model failed!")
                except Exception as e:
                    print(f"Load model failed, count: {try_count}, exception: {e}")
                    time.sleep(1)
                else:
                    load_complete = True

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
                                               is_change_apply_to_train,
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

    if is_save_record:
        exp_recorder.save_to_disk(save_path)

    # Save model
    if model_save_dir is not None:
        model_save_dir = os.path.abspath(model_save_dir)
        model.save_pretrained(model_save_dir)
        print(f"Save best model ckpt to {model_save_dir}")


if __name__ == '__main__':
    import time

    t1 = time.time()
    main()
    t2 = time.time()
    print(f'Total {(t2 - t1) / 60.0} minutes')
