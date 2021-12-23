import os
import sys
import ipdb
import numpy as np
import pandas as pd

sys.path.append('..')
from core.text_analyzer import TextAnalyzer
from core.utils import load_save_json


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


# python analyze_text_dataset.py

def main():
    base_dir = '../data'
    datasets = ['cn_novel_5billion', 'en_grover', 'en_writing_prompt']
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

        # # --------------------------------------------------------------------------------------------------------------
        # # read basic
        # # --------------------------------------------------------------------------------------------------------------
        # _save_basic_result(data_analyze_df, human_texts, dataset, 'human', language)
        # _save_basic_result(data_analyze_df, machine_texts, dataset, 'machine', language)
        # # --------------------------------------------------------------------------------------------------------------

        # # --------------------------------------------------------------------------------------------------------------
        # # N-gram distinct
        # # --------------------------------------------------------------------------------------------------------------
        # human_result = text_analyzer.compute_ngram_distinct(human_texts, n_grams=n_grams)
        # machine_result = text_analyzer.compute_ngram_distinct(machine_texts, n_grams=n_grams)
        #
        # _save_distinct_result(human_result, data_analyze_df, dataset, language, 'human')
        # _save_distinct_result(machine_result, data_analyze_df, dataset, language, 'machine')
        # # --------------------------------------------------------------------------------------------------------------

        # # --------------------------------------------------------------------------------------------------------------
        # # pos tagging distribution
        # # --------------------------------------------------------------------------------------------------------------
        # text_analyzer = TextAnalyzer(do_lower=True,
        #                              language=language,
        #                              load_spacy_model=True,
        #                              dump_spacy_result=True)
        # # parse_choices = ('pos', 'dep', 'ner')
        # parse_choices = ('pos', 'dep', 'ner')
        # temp_debug_N = 2000
        # for parse_choice in parse_choices:
        #     # human
        #     parse_result = text_analyzer.load_parsed_texts_by_spacy(human_texts[:temp_debug_N], parse_choice)
        #     save_path = os.path.join(f'../result/static_data_analysis/{dataset}_{parse_choice}_human.json')
        #     load_save_json(save_path, 'save', data=parse_result)
        #     # machine
        #     parse_result = text_analyzer.load_parsed_texts_by_spacy(machine_texts[:temp_debug_N], parse_choice)
        #     save_path = os.path.join(f'../result/static_data_analysis/{dataset}_{parse_choice}_machine.json')
        #     load_save_json(save_path, 'save', data=parse_result)
        # # --------------------------------------------------------------------------------------------------------------

        # # --------------------------------------------------------------------------------------------------------------
        # # analyse stopwords
        # # --------------------------------------------------------------------------------------------------------------
        # stopword_sen_ratio_df = {'value': [], 'language': [], 'author': []}
        # text_analyzer = TextAnalyzer(do_lower=True,
        #                              language=language)
        # human_stopword_sen_ratio, human_stopwords_ratio = text_analyzer.analyse_stopwords(human_texts)
        # machine_stopword_sen_ratio, machine_stopwords_ratio = text_analyzer.analyse_stopwords(machine_texts)
        #
        # stopword_sen_ratio_df['value'] = human_stopword_sen_ratio + machine_stopword_sen_ratio
        # stopword_sen_ratio_df['language'] = [language for _ in human_stopword_sen_ratio] + [language for _ in
        #                                                                                     machine_stopword_sen_ratio]
        # stopword_sen_ratio_df['author'] = ['human' for _ in human_stopword_sen_ratio] + ['machine' for _ in
        #                                                                                  machine_stopword_sen_ratio]
        # stopword_sen_ratio_df = pd.DataFrame(stopword_sen_ratio_df)
        # stopword_sen_ratio_df_save_path = f'../result/static_data_analysis/{dataset}_stopword_sen_ratio.csv'
        # stopword_sen_ratio_df.to_csv(stopword_sen_ratio_df_save_path, index=False)
        # print(f"save stopword_sen_ratio_df to {stopword_sen_ratio_df_save_path}")
        #
        # # ---
        # stopword_ratio_df = {'value': [], 'stopword': [], 'language': [], 'author': []}
        # human_stopwords_ratio = sorted(human_stopwords_ratio.items(), key=lambda x: x[1], reverse=True)
        # stopwords = [x[0] for x in human_stopwords_ratio]
        # human_stopword_ratios = [x[1] for x in human_stopwords_ratio]
        # machine_stopword_ratios = [machine_stopwords_ratio.get(x, 0.0) for x in stopwords]
        # stopword_ratio_df['value'] = human_stopword_ratios + machine_stopword_ratios
        # stopword_ratio_df['stopword'] = stopwords + stopwords
        # stopword_ratio_df['index_i'] = list(range(len(stopwords))) + list(range(len(stopwords)))
        # stopword_ratio_df['language'] = [language for _ in human_stopword_ratios] + [language for _ in
        #                                                                              machine_stopword_ratios]
        # stopword_ratio_df['author'] = ['human' for _ in human_stopword_ratios] + ['machine' for _ in
        #                                                                           machine_stopword_ratios]
        # stopword_ratio_df_save_path = f'../result/static_data_analysis/{dataset}_stopword_ratio.csv'
        # stopword_ratio_df = pd.DataFrame(stopword_ratio_df)
        # stopword_ratio_df.to_csv(stopword_ratio_df_save_path, index=False)
        # print(f"Save stopword_ratio_df_save_path to {stopword_ratio_df_save_path}")
        # # --------------------------------------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------------------------------------
        # analyse concreteness
        # --------------------------------------------------------------------------------------------------------------
        if language != 'en':
            continue
        else:
            debug_N = 100
            concreteness_df = {'value': [], 'language': [], 'author': [], 'pos': []}
            text_analyzer = TextAnalyzer(do_lower=True,
                                         language=language,
                                         load_spacy_model=True)
            human_concreteness_dict = text_analyzer.analyse_concreteness(human_texts[:debug_N])
            machine_concreteness_dict = text_analyzer.analyse_concreteness(machine_texts[:debug_N])

            for pos, value in human_concreteness_dict.items():
                concreteness_df['value'].extend(value)
                concreteness_df['language'].extend([language for _ in value])
                concreteness_df['author'].extend(['human' for _ in value])
                concreteness_df['pos'].extend([pos for _ in value])

            for pos, value in machine_concreteness_dict.items():
                concreteness_df['value'].extend(value)
                concreteness_df['language'].extend([language for _ in value])
                concreteness_df['author'].extend(['machine' for _ in value])
                concreteness_df['pos'].extend([pos for _ in value])
            concreteness_df = pd.DataFrame(concreteness_df)
            concreteness_df_save_path = f'../result/static_data_analysis/{dataset}_concreteness.csv'
            concreteness_df.to_csv(concreteness_df_save_path, index=False)
            print(f"Save concreteness_df to {concreteness_df_save_path}")
        # --------------------------------------------------------------------------------------------------------------

    ipdb.set_trace()

    data_analyze_df = pd.DataFrame(data_analyze_df)
    print(data_analyze_df)
    ipdb.set_trace()


if __name__ == '__main__':
    main()
