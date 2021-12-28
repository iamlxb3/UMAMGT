import os
import ipdb
import math
import argparse
import collections
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def args_parse():
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--model', type=str)
    parser.add_argument('--debug_N', type=int)
    args = parser.parse_args()
    return args


def _read_data(base_dir, dataset):
    file_path = os.path.join(base_dir, dataset, 'train.tgt')
    label_path = os.path.join(base_dir, dataset, 'train.label')

    # grover: 5927,
    texts = []
    invalid_indices = []
    with open(file_path, 'r') as f:
        for i, raw_line in enumerate(f):
            line = raw_line.strip()
            if line:
                texts.append(line)

    texts = np.array(texts)

    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(int(line))
    labels = np.array(labels)
    human_mask = labels == 1
    machine_mask = labels == 0
    try:
        human_texts = texts[human_mask]
    except:
        ipdb.set_trace()
    machine_texts = texts[machine_mask]
    return human_texts, machine_texts


# datasets = ['cn_novel_5billion', 'en_grover', 'en_writing_prompt']

# python compute_idf.py --dataset cn_novel_5billion --model cn_roberta
# python compute_idf.py --dataset en_grover --model en_roberta
# python compute_idf.py --dataset en_writing_prompt --model en_roberta

def main():
    args = args_parse()
    dataset = args.dataset
    model = args.model
    debug_N = args.debug_N
    base_dir = '../data'
    human_texts, machine_texts = _read_data(base_dir, dataset)
    all_texts = list(human_texts) + list(machine_texts)
    if debug_N:
        all_texts = all_texts[:debug_N]
    if model == 'en_roberta':
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        tokenizer_name = tokenizer.name_or_path.replace('-', '_')
    elif model == 'cn_roberta':
        tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        tokenizer_name = tokenizer.name_or_path.replace('/', '_').replace('-', '_')
    else:
        tokenizer = None
        tokenizer_name = 'whitespace'
    text_df = collections.defaultdict(lambda: 0)
    doc_freq = 0
    for text in tqdm(all_texts, total=len(all_texts)):
        doc_freq += 1
        if dataset == 'cn_novel_5billion':
            if tokenizer_name != 'whitespace':
                text = text.replace(' ', '')
                tokenized_text = tokenizer.tokenize(text)
            else:
                tokenized_text = text.split()
        else:
            if tokenizer_name != 'whitespace':
                tokenized_text = tokenizer.tokenize(text)
                tokenized_text = [x.replace('Ġ', '') for x in tokenized_text]
            else:
                tokenized_text = text.split()
        for x in set(tokenized_text):
            text_df[x] += 1

    text_df = {k: math.log(doc_freq / v) for k, v in text_df.items()}
    text_df = {'token': [x for x in text_df.keys()], 'idf': [x for x in text_df.values()]}
    save_path = f'../result/static_data_analysis/{dataset}_{tokenizer_name}_idf.csv'
    text_df = pd.DataFrame(text_df)
    text_df = text_df.sort_values(by='idf', ascending=True)
    text_df.to_csv(save_path, index=False)
    print(f"Save df to {save_path}")


if __name__ == '__main__':
    main()
