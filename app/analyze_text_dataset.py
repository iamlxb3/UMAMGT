import os
import sys
import ipdb
import numpy as np
import pandas as pd

sys.path.append('..')
from core.text_analyzer import TextAnalyzer


# python analyze_text_dataset.py

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


def _save_distinct_result(result, data_analyze_df, dataset, language, src):
    for n_gram, value in result:
        data_analyze_df['value'].append(value)
        data_analyze_df['type'].append(f'distinct-{n_gram}')
        data_analyze_df['dataset'].append(dataset)
        data_analyze_df['src'].append(src)
        data_analyze_df['language'].append(language)
        data_analyze_df['meta'].append('distinct')


def _save_basic_result(data_analyze_df, texts, dataset, src, language):
    unique_tokens = len(set([x for text in texts for x in text.split()]))
    avg_sen_len = np.average([len(text.split()) for text in texts])
    data_analyze_df['value'].append(unique_tokens)
    data_analyze_df['type'].append('unique_token_N')
    data_analyze_df['dataset'].append(dataset)
    data_analyze_df['src'].append(src)
    data_analyze_df['language'].append(language)
    data_analyze_df['meta'].append('basic')
    data_analyze_df['value'].append(avg_sen_len)
    data_analyze_df['type'].append('avg_sen_len')
    data_analyze_df['dataset'].append(dataset)
    data_analyze_df['src'].append(src)
    data_analyze_df['language'].append(language)
    data_analyze_df['meta'].append('basic')


def main():
    base_dir = '../data'
    datasets = ['cn_novel_5billion', 'new_en_grover', 'en_writing_prompt']
    text_analyzer = TextAnalyzer()
    data_analyze_df = {'value': [], 'type': [], 'dataset': [], 'language': [], 'src': [], 'meta': []}

    # ------------------------------------------------------------------------------------------------------------------
    # Basic Data statistics
    # ------------------------------------------------------------------------------------------------------------------

    for dataset in datasets:
        human_texts, machine_texts = _read_data(base_dir, dataset)

    n_grams = (1, 2, 3, 4, 5)
    for dataset in datasets:
        language = dataset.split('_')[0]

        human_texts, machine_texts = _read_data(base_dir, dataset)

        # --------------------------------------------------------------------------------------------------------------
        # read basic
        # --------------------------------------------------------------------------------------------------------------
        _save_basic_result(data_analyze_df, human_texts, dataset, 'human', language)
        _save_basic_result(data_analyze_df, machine_texts, dataset, 'machine', language)
        # --------------------------------------------------------------------------------------------------------------

        # # --------------------------------------------------------------------------------------------------------------
        # # N-gram distinct
        # # --------------------------------------------------------------------------------------------------------------
        # human_result = text_analyzer.compute_ngram_distinct(human_texts, n_grams=n_grams)
        # machine_result = text_analyzer.compute_ngram_distinct(machine_texts, n_grams=n_grams)
        #
        # _save_distinct_result(human_result, data_analyze_df, dataset, language, 'human')
        # _save_distinct_result(machine_result, data_analyze_df, dataset, language, 'machine')
        # # --------------------------------------------------------------------------------------------------------------

    data_analyze_df = pd.DataFrame(data_analyze_df)
    print(data_analyze_df)
    ipdb.set_trace()


if __name__ == '__main__':
    main()
