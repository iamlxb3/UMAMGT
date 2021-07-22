import os
import ipdb
import random
import collections
from shutil import copyfile


def write_dataset(read_data_path, write_data_path, label1_chars_counter, is_unique, is_sort, is_shuffle,
                  is_reverse, topn_chars):
    with open(read_data_path, 'r', encoding='utf-8') as f_read:
        with open(write_data_path, 'w', encoding='utf-8') as f_write:
            for i, line in enumerate(f_read):
                line = line.strip()
                if line:
                    chars = line.split(' ')
                    chars = [x if x in topn_chars else '[MASK]' for x in chars]
                if is_unique:
                    chars = list(set(chars))
                chars_freq = [(x, label1_chars_counter.get(x, 0)) for x in chars]
                if is_sort:
                    chars_freq = sorted(chars_freq, key=lambda x: x[1], reverse=is_reverse)
                if is_shuffle:
                    random.shuffle(chars_freq)
                write_chars = ' '.join([x[0] for x in chars_freq])
                f_write.write(write_chars + '\n')


import argparse


# python3.6 create_sorted_data.py

def args_parse():
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('--top_freq', type=int)
    parser.add_argument('--data_process_type', type=str, choices=['sort',
                                                                  'sort_reverse',
                                                                  'sort_unique_no_reverse',
                                                                  'unique_no_reverse',
                                                                  'shuffle_unique',
                                                                  ],
                        required=True)
    args = parser.parse_args()
    return args


def main():
    #
    args = args_parse()
    top_freq = args.top_freq
    data_process_type = args.data_process_type
    train_label_path = '../data/5billion/train.label'
    train_data_path = '../data/5billion/train.tgt'
    val_data_path = '../data/5billion/valid.tgt'
    random.seed(1)

    train_labels = []
    with open(train_label_path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            train_labels.append(line)

    label0_chars = []
    label1_chars = []

    with open(train_data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            label = int(train_labels[i])
            chars = line.split(' ')

            if label == 0:
                label0_chars.extend(chars)
            elif label == 1:
                label1_chars.extend(chars)

    print(f"label 0 char count: {len(label0_chars)}, label 1 char count: {len(label1_chars)}")
    label0_chars_counter = collections.Counter(label0_chars)
    label1_chars_counter = collections.Counter(label1_chars)
    sorted_label1_counter = sorted(label1_chars_counter.items(), key=lambda x: x[1], reverse=True)
    # maxcount = max([x for _, x in sorted_label1_counter])
    sorted_all_chars = [x for x, _ in sorted_label1_counter]

    if top_freq is not None:
        topn_chars = set(sorted_all_chars[:top_freq])
    else:
        topn_chars = None

    # default config
    is_shuffle = False
    is_unique = False
    is_reverse = False

    if data_process_type == 'sort':
        is_reverse = False
        is_sort = True
    elif data_process_type == 'sort_no_reverse':
        is_reverse = True
        is_sort = True
    elif data_process_type == 'sort_unique_no_reverse':
        is_reverse = True
        is_unique = True
        is_sort = True
    elif data_process_type == 'shuffle_unique':
        is_sort = False
        is_unique = True
        is_shuffle = True
    else:
        raise Exception

    if top_freq is not None:
        data_process_type = data_process_type + f'_top_{top_freq}'

    new_train_data_path = f'../data/5billion_{data_process_type}/train.tgt'
    new_val_data_path = f'../data/5billion_{data_process_type}/valid.tgt'

    save_base_dir = os.path.dirname(new_train_data_path)
    if not os.path.isdir(save_base_dir):
        os.makedirs(save_base_dir)
        print(f"Make new dir {save_base_dir}")

    # copy label
    copyfile(f'../data/5billion/train.label', os.path.join(save_base_dir, 'train.label'))
    copyfile(f'../data/5billion/valid.label', os.path.join(save_base_dir, 'valid.label'))
    print("Copy label done!")

    write_dataset(train_data_path, new_train_data_path, label1_chars_counter, is_unique, is_sort, is_shuffle,
                  is_reverse, topn_chars)
    write_dataset(val_data_path, new_val_data_path, label1_chars_counter, is_unique, is_sort, is_shuffle, is_reverse,
                  topn_chars)
    print("-" * 78)
    print(f"Create Dataset for {data_process_type} done!")


if __name__ == '__main__':
    main()
