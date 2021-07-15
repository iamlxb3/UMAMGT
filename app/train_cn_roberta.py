import os
import sys
import ipdb
import ntpath
import numpy as np
import argparse

sys.path.append('..')

from core.task import StoryTuringTest
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import RobertaForMaskedLM

from transformers import Trainer, TrainingArguments
from transformers import TrainerCallback
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from pytorch_lightning import seed_everything


# python train_cn_roberta.py --epoch 1 --debug_N 100 --model_max_length 30


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
    parser.add_argument('--debug_N', type=int)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--data_dir', type=str, default='../data/5billion')
    parser.add_argument('--model_max_length', type=int, default=512)
    args = parser.parse_args()
    return args


def main():
    args = args_parse()

    # config
    model_max_length = args.model_max_length
    debug_N = args.debug_N
    seed_everything(1)

    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    tokenizer.model_max_length = model_max_length

    # (0.) read dataset
    story_turing_test = StoryTuringTest(tokenizer)
    train_dataset, val_dataset, test_dataset = story_turing_test.read_data(args.data_dir, debug_N=debug_N)
    seq_len = len(val_dataset.encodings['input_ids'][0])
    train_size = len(train_dataset)

    # (1.) init model
    model = AutoModelForSequenceClassification.from_pretrained("hfl/chinese-roberta-wwm-ext", num_labels=2)
    print(model)

    # (2.) train model
    # reference: https://huggingface.co/transformers/v3.0.2/main_classes/trainer.html
    training_args = TrainingArguments(
        output_dir='../result',  # output directory
        num_train_epochs=1,  # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir=f'../logs/cn-roberta-story-turning-train_{train_size}-seq_{seq_len}',  # directory for storing logs
        logging_steps=2,
        report_to='tensorboard',
        evaluation_strategy='steps',
        eval_steps=2,
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
    print(f"train_result: {train_result}")
    test_result = trainer.evaluate(test_dataset, metric_key_prefix='test')
    print(f"test_result: {test_result}")

    # Save model
    model_save_path = f"../model_ckpts/cn-roberta-story-turning-train_{train_size}-seq_{seq_len}"
    model_save_path = os.path.abspath(model_save_path)
    model.save_pretrained(f"../model_ckpts/cn-roberta-story-turning-train_{train_size}-seq_{seq_len}")
    print(f"Save best model ckpt to {model_save_path}")
    ipdb.set_trace()


if __name__ == '__main__':
    main()
