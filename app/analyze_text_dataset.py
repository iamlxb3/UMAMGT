import os
import sys
import ipdb
import math
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

    # n_grams = (1, 2, 3, 4)
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
        #                              load_spacy_model=True)
        # # parse_choices = ('pos', 'dep', 'ner')
        # parse_choices = ('pos', 'dep', 'ner')
        # temp_debug_N = 999999999999
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
        #
        # --------------------------------------------------------------------------------------------------------------
        # analyse stopwords
        # --------------------------------------------------------------------------------------------------------------
        stopword_sen_ratio_df = {'value': [], 'language': [], 'author': []}
        text_analyzer = TextAnalyzer(do_lower=True,
                                     language=language)
        human_stopword_sen_ratio, human_stopwords_ratio = text_analyzer.analyse_stopwords(human_texts)
        machine_stopword_sen_ratio, machine_stopwords_ratio = text_analyzer.analyse_stopwords(machine_texts)

        stopword_sen_ratio_df['value'] = human_stopword_sen_ratio + machine_stopword_sen_ratio
        stopword_sen_ratio_df['language'] = [language for _ in human_stopword_sen_ratio] + [language for _ in
                                                                                            machine_stopword_sen_ratio]
        stopword_sen_ratio_df['author'] = ['human' for _ in human_stopword_sen_ratio] + ['machine' for _ in
                                                                                         machine_stopword_sen_ratio]
        stopword_sen_ratio_df = pd.DataFrame(stopword_sen_ratio_df)
        stopword_sen_ratio_df_save_path = f'../result/static_data_analysis/{dataset}_stopword_sen_ratio.csv'
        stopword_sen_ratio_df.to_csv(stopword_sen_ratio_df_save_path, index=False)
        print(f"save stopword_sen_ratio_df to {stopword_sen_ratio_df_save_path}")

        # ---
        stopword_ratio_df = {'value': [], 'stopword': [], 'language': [], 'author': []}
        human_stopwords_ratio = sorted(human_stopwords_ratio.items(), key=lambda x: x[1], reverse=True)
        stopwords = [x[0] for x in human_stopwords_ratio]
        human_stopword_ratios = [x[1] for x in human_stopwords_ratio]
        machine_stopword_ratios = [machine_stopwords_ratio.get(x, 0.0) for x in stopwords]
        stopword_ratio_df['value'] = human_stopword_ratios + machine_stopword_ratios
        stopword_ratio_df['stopword'] = stopwords + stopwords
        stopword_ratio_df['index_i'] = list(range(len(stopwords))) + list(range(len(stopwords)))
        stopword_ratio_df['language'] = [language for _ in human_stopword_ratios] + [language for _ in
                                                                                     machine_stopword_ratios]
        stopword_ratio_df['author'] = ['human' for _ in human_stopword_ratios] + ['machine' for _ in
                                                                                  machine_stopword_ratios]
        stopword_ratio_df_save_path = f'../result/static_data_analysis/{dataset}_stopword_ratio.csv'
        stopword_ratio_df = pd.DataFrame(stopword_ratio_df)
        stopword_ratio_df.to_csv(stopword_ratio_df_save_path, index=False)
        print(f"Save stopword_ratio_df_save_path to {stopword_ratio_df_save_path}")
        # --------------------------------------------------------------------------------------------------------------
        #
        # --------------------------------------------------------------------------------------------------------------
        # analyse concreteness
        # --------------------------------------------------------------------------------------------------------------
        if language != 'en':
            continue
        else:
            debug_N = 999999999999
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

        # --------------------------------------------------------------------------------------------------------------
        # analyse N-gram overlap between human and machine generated texts / zipflaw
        # --------------------------------------------------------------------------------------------------------------
        debug_N = 999999999999
        text_analyzer = TextAnalyzer(do_lower=True, language=language)
        ngram_idf_df = {'value': [], 'value_name': [], 'token': [], 'author': [], 'index_i': [], 'n_gram': []}
        n_gram_df = {'human_ngram': [], 'machine_ngram': [], 'overlap_ngram': [], 'total_ngram': [],
                     'human_overlap_ratio': [], 'human_overlap_token_freq_ratio': [],
                     'machine_overlap_ratio': [], 'machine_overlap_token_freq_ratio': [],
                     'ngram': []}
        zipflaw_df = {'rank': [], 'freq': [], 'token': [], 'author': [], 'n_gram': []}

        ngram_idf_df_save_path = f'../result/static_data_analysis/{dataset}_ngram_idf.csv'
        ngram_df_save_path = f'../result/static_data_analysis/{dataset}_ngram.csv'
        zipflaw_df_save_path = f'../result/static_data_analysis/{dataset}_zipflaw.csv'

        n_grams = (1, 2, 3)
        for n_gram in n_grams:
            human_n_gram_freq_dict, human_n_gram_idf_dict = text_analyzer.compute_ngram(human_texts[:debug_N],
                                                                                        n_gram=n_gram)
            machine_n_gram_freq_dict, machine_n_gram_idf_dict = text_analyzer.compute_ngram(machine_texts[:debug_N],
                                                                                            n_gram=n_gram)

            human_unique_tokens = set(human_n_gram_freq_dict.keys())
            machine_unique_tokens = set(machine_n_gram_freq_dict.keys())
            over_lap_ngram = human_unique_tokens.intersection(machine_unique_tokens)

            # ---------------------------
            # compute zipflaw
            # ---------------------------
            human_n_gram_freq = sorted(human_n_gram_freq_dict.items(), key=lambda x: x[1], reverse=True)
            machine_n_gram_freq = sorted(machine_n_gram_freq_dict.items(), key=lambda x: x[1], reverse=True)

            for i, (token, freq) in enumerate(human_n_gram_freq):
                zipflaw_df['rank'].append(i)
                zipflaw_df['freq'].append(math.log(freq))
                zipflaw_df['token'].append(token)
                zipflaw_df['author'].append('human')
                zipflaw_df['n_gram'].append(n_gram)

            for i, (token, freq) in enumerate(machine_n_gram_freq):
                zipflaw_df['rank'].append(i)
                zipflaw_df['freq'].append(math.log(freq))
                zipflaw_df['token'].append(token)
                zipflaw_df['author'].append('machine')
                zipflaw_df['n_gram'].append(n_gram)
            # ---------------------------

            over_lap_ngram_for_idf = [(x, human_n_gram_idf_dict[x]) for x in over_lap_ngram]
            over_lap_ngram_for_idf = sorted(over_lap_ngram_for_idf, key=lambda x: x[1])
            for i, (ngram_token, _) in enumerate(over_lap_ngram_for_idf):
                human_idf = human_n_gram_idf_dict[ngram_token]
                machine_idf = machine_n_gram_idf_dict[ngram_token]
                ngram_idf_df['value'].append(human_idf)
                ngram_idf_df['value_name'].append('idf')
                ngram_idf_df['token'].append(ngram_token)
                ngram_idf_df['author'].append('human')
                ngram_idf_df['index_i'].append(i)
                ngram_idf_df['n_gram'].append(n_gram)

                ngram_idf_df['value'].append(machine_idf)
                ngram_idf_df['value_name'].append('idf')
                ngram_idf_df['token'].append(ngram_token)
                ngram_idf_df['author'].append('machine')
                ngram_idf_df['index_i'].append(i)
                ngram_idf_df['n_gram'].append(n_gram)

            result_dict = {f'human_{n_gram}_gram_total': len(human_n_gram_freq_dict),
                           f'machine_{n_gram}_gram_total': len(machine_n_gram_freq_dict),
                           f'overlap_{n_gram}_gram': len(over_lap_ngram),
                           f'total_{n_gram}_gram': len(human_unique_tokens.union(machine_unique_tokens))
                           }
            print(result_dict)

            human_overlap_freq_count = 0
            for token in human_unique_tokens:
                if token in over_lap_ngram:
                    human_overlap_freq_count += human_n_gram_freq_dict[token]
            total_human_freq = np.sum(list(human_n_gram_freq_dict.values()))

            machine_overlap_freq_count = 0
            for token in machine_unique_tokens:
                if token in over_lap_ngram:
                    machine_overlap_freq_count += machine_n_gram_freq_dict[token]
            total_machine_freq = np.sum(list(machine_n_gram_freq_dict.values()))
            human_overlap_ratio = len(over_lap_ngram) / len(human_n_gram_freq_dict)
            human_overlap_token_freq_ratio = human_overlap_freq_count / total_human_freq
            machine_overlap_ratio = len(over_lap_ngram) / len(machine_n_gram_freq_dict)
            machine_overlap_token_freq_ratio = machine_overlap_freq_count / total_machine_freq

            # n_gram_df = {'human_ngram': [], 'machine_ngram': [], 'overlap_ngram': [], 'total_ngram': [], 'ngram': []}
            #                      'human_overlap_ratio': [],'human_overlap_token_freq_ratio': [],
            #                      'machine_overlap_ratio': [], 'machine_overlap_token_freq_ratio': [],
            n_gram_df['human_ngram'].append(len(human_n_gram_freq_dict))
            n_gram_df['machine_ngram'].append(len(machine_n_gram_freq_dict))
            n_gram_df['overlap_ngram'].append(len(over_lap_ngram))
            n_gram_df['total_ngram'].append(len(human_unique_tokens.union(machine_unique_tokens)))
            n_gram_df['human_overlap_ratio'].append(human_overlap_ratio)
            n_gram_df['human_overlap_token_freq_ratio'].append(human_overlap_token_freq_ratio)
            n_gram_df['machine_overlap_ratio'].append(machine_overlap_ratio)
            n_gram_df['machine_overlap_token_freq_ratio'].append(machine_overlap_token_freq_ratio)
            n_gram_df['ngram'].append(n_gram)

            print(f"[Human] Ngram-{n_gram}, overlap ratio / overlap token freq ratio:"
                  f" {human_overlap_ratio} / {human_overlap_token_freq_ratio:.3f}")
            print(
                f"[Machine] Ngram-{n_gram}, overlap ratio / overlap token freq ratio:"
                f" {machine_overlap_ratio} / {machine_overlap_token_freq_ratio:.3f}")

        ngram_idf_df = pd.DataFrame(ngram_idf_df)
        n_gram_df = pd.DataFrame(n_gram_df)
        zipflaw_df = pd.DataFrame(zipflaw_df)
        ngram_idf_df.to_csv(ngram_idf_df_save_path, index=False)
        n_gram_df.to_csv(ngram_df_save_path, index=False)
        zipflaw_df.to_csv(zipflaw_df_save_path, index=False)
        print(f"Save ngram idf df to {ngram_idf_df}")
        print(f"Save ngram df to {ngram_df_save_path}")
        print(f"Save zipflaw_df to {zipflaw_df_save_path}")

    # --------------------------------------------------------------------------------------------------------------

    # data_analyze_df = pd.DataFrame(data_analyze_df)
    # print(data_analyze_df)


if __name__ == '__main__':
    main()
