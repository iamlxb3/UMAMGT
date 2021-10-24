import os
import ipdb
import numpy as np
from sklearn.model_selection import train_test_split
from .datasets import StoryTuringTestData


class StoryTuringTest:
    def __init__(self, tokenizer, dataset_name):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name

    def _read_text_label(self, text_file, label_file):
        texts = []
        labels = []

        # read texts
        with open(text_file, 'r') as f:
            for line in f:
                line = line.strip()
                texts.append(line)

        # read labels
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                labels.append(int(line))

        assert len(texts) == len(labels)
        return texts, labels

    def read_cn_novel_whole_data(self, data_dir):
        test_data_path = os.path.join(data_dir, 'valid.tgt')
        test_label_path = os.path.join(data_dir, 'valid.label')
        train_data_path = os.path.join(data_dir, 'train.tgt')
        train_label_path = os.path.join(data_dir, 'train.label')

        train_texts, train_labels = self._read_text_label(train_data_path, train_label_path)
        test_texts, test_labels = self._read_text_label(test_data_path, test_label_path)

        whole_texts = train_texts + test_texts
        whole_labels = train_labels + test_labels

        return np.array(whole_texts), np.array(whole_labels)

    def create_dataset(self, texts, labels, max_length=512):

        text_encodings = self.tokenizer(list(texts), truncation=True, padding=True, max_length=max_length)

        # create dataset
        dataset = StoryTuringTestData(text_encodings, labels)

        return dataset

    def read_test_data(self, data_dir, debug_N):
        test_data_path = os.path.join(data_dir, 'valid.tgt')
        test_label_path = os.path.join(data_dir, 'valid.label')

        test_texts, test_labels = self._read_text_label(test_data_path, test_label_path)
        if debug_N is not None:
            test_texts, test_labels = test_texts[:debug_N], test_labels[:debug_N]
        test_encodings = self.tokenizer(test_texts, truncation=True, padding=True)
        test_dataset = StoryTuringTestData(test_encodings, test_labels)
        return test_texts, test_dataset

    def read_data(self, data_dir, val_ratio=0.1, debug_N=None):
        train_data_path = os.path.join(data_dir, 'train.tgt')
        train_label_path = os.path.join(data_dir, 'train.label')

        train_texts, train_labels = self._read_text_label(train_data_path, train_label_path)
        # train_texts_lens = [len(x.split()) for x in train_texts]

        if debug_N is not None:
            train_texts, train_labels = train_texts[:debug_N], train_labels[:debug_N]

        # split train/val
        train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels,
                                                                            test_size=val_ratio)

        # dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])

        val_encodings = self.tokenizer(val_texts, truncation=True, padding=True)
        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True)

        # create dataset
        train_dataset = StoryTuringTestData(train_encodings, train_labels)
        val_dataset = StoryTuringTestData(val_encodings, val_labels)

        test_texts, test_dataset = self.read_test_data(data_dir, debug_N)

        print(f"Train size: {len(train_texts)}, val size: {len(val_texts)}, test size: {len(test_texts)}")

        return train_dataset, val_dataset, test_dataset
