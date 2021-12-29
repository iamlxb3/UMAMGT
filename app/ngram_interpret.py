from nltk import ngrams as compute_ngrams
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import collections
import ntpath
import math
import os
import ipdb
from nltk.stem.wordnet import WordNetLemmatizer


def args_parse():
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('--path', type=str)
    args = parser.parse_args()
    return args


def compute_per_sentence_attr_score(target_df, class_pos, ngram, save_path, language, words_concreteness):
    all_sen_i = len(set(target_df['sen_i'].values))
    token_df = collections.defaultdict(lambda: 0)
    token_tf = collections.defaultdict(lambda: [])
    token_per_sen_score = collections.defaultdict(lambda: [])
    token_concreteness_dict = collections.defaultdict(lambda: 0)
    save_df = {'token': [], 'attr_score': [], 'idf': [], 'avg_tf': [], 'total_tf': [], 'concreteness': []}

    for sen_i, sen_df in tqdm(target_df.groupby('sen_i'), total=all_sen_i):
        sen_tokens = sen_df['token'].values
        sen_attr_scores = sen_df['label1_attr_score'].values
        n_gram_token = list(compute_ngrams(sen_tokens, ngram))

        # compute concreteness score
        new_n_gram_token = []
        for token_list in n_gram_token:
            if language == 'cn':
                concreteness = 0.0
                if len(token_list) < 2:
                    pass
                elif len(token_list) == 2:
                    concreteness += words_concreteness.get(''.join(token_list), 0.0)
                else:
                    bigram_token = list(compute_ngrams(token_list, 2))
                    for bigram in bigram_token:
                        concreteness += words_concreteness.get(''.join(bigram), 0.0)
                token = ''.join([str(y) for y in token_list])
                # if concreteness > 0.0:
                #     print(f"token: {token}, concreteness: {concreteness}")
            elif language == 'en':
                token = ' '.join([str(y) for y in token_list])

                concreteness = 0
                for x in token_list:
                    x = str(x)
                    x_lower = x.lower()
                    x_upper = x.upper()
                    x_n_lemma = WordNetLemmatizer().lemmatize(x, 'n')
                    x_v_lemma = WordNetLemmatizer().lemmatize(x, 'v')
                    if x in words_concreteness:
                        concreteness += words_concreteness[x]
                    elif x_lower in words_concreteness:
                        concreteness += words_concreteness[x_lower]
                    elif x_upper in words_concreteness:
                        concreteness += words_concreteness[x_upper]
                    elif x_n_lemma in words_concreteness:
                        concreteness += words_concreteness[x_n_lemma]
                    elif x_v_lemma in words_concreteness:
                        concreteness += words_concreteness[x_v_lemma]
            else:
                raise NotImplementedError
            new_n_gram_token.append(token)
            token_concreteness_dict[token] = concreteness
            # print(f"n_gram_token: {n_gram_token}, concreteness: {concreteness}")
        n_gram_token = new_n_gram_token
        n_gram_attr_scores = list(compute_ngrams(sen_attr_scores, ngram))
        n_gram_attr_scores = [np.sum(x) for x in n_gram_attr_scores]
        assert len(n_gram_token) == len(n_gram_attr_scores)

        sen_token_counter = collections.Counter(n_gram_token)
        for token, count in sen_token_counter.items():
            token_tf[token].append(count)

        # compute per sentence attr score
        token_one_sen_score = collections.defaultdict(lambda: 0)
        for token, attr_score in zip(n_gram_token, n_gram_attr_scores):
            token_one_sen_score[token] += attr_score

        for k, v in token_one_sen_score.items():
            token_per_sen_score[k].append(v)

        # compute idf
        for token in set(n_gram_token):
            token_df[token] += 1

    token_idf = {k: math.log(all_sen_i / v) for k, v in token_df.items()}
    token_total_tf = {k: np.sum(v) for k, v in token_tf.items()}
    token_avg_tf = {k: np.average(v) for k, v in token_tf.items()}
    token_per_sen_score = {k: np.average(v) for k, v in token_per_sen_score.items()}

    for token, attr_score in token_per_sen_score.items():
        save_df['token'].append(token)
        save_df['attr_score'].append(attr_score)
        save_df['idf'].append(token_idf[token])
        save_df['avg_tf'].append(token_avg_tf[token])
        save_df['total_tf'].append(token_total_tf[token])
        save_df['concreteness'].append(token_concreteness_dict[token])

    save_df = pd.DataFrame(save_df)
    save_df = save_df.sort_values(by='attr_score', ascending=not class_pos)
    save_df.to_csv(save_path, index=False)
    print(f"Save df to {save_path}")

    return save_df


# python ngram_interpret.py --path '../result/interpret/interpret_cn_novel_5billion_cn_roberta_debug_0_text_len_128_debug_N_10000_use_all_zero_bs_token_attr.csv'
# python ngram_interpret.py --path '../result/interpret/interpret_en_grover_en_roberta_debug_0_text_len_256_debug_N_800_use_pad_bs_token_attr.csv'
# python ngram_interpret.py --path '../result/interpret/interpret_en_writing_prompt_en_roberta_debug_0_text_len_128_debug_N_800_use_pad_bs_token_attr.csv'
# python ngram_interpret.py --path '../result/interpret/interpret_en_grover_en_roberta_debug_0_text_len_256_debug_N_10000_use_all_zero_bs_token_attr.csv'
# python ngram_interpret.py --path '../result/interpret/interpret_en_writing_prompt_en_roberta_debug_0_text_len_128_debug_N_10000_use_all_zero_bs_token_attr.csv'

def main():
    args = args_parse()
    path = args.path
    basename = ntpath.basename(path).split('.')[0].replace('_token_attr', '')

    language = basename.split('_')[1]

    df = pd.read_csv(path)
    ngrams = [1, 2, 3, 4, 5, 6]
    # ngrams = [1, 2, 3, 4, 5, 6]

    # token_freq_dict = collections.Counter(df['token'].values)
    pos_df = df[df['label'] == 1]
    neg_df = df[df['label'] == 0]

    # load concretness dict
    if language == 'cn':
        concreteness = pd.read_csv('../static/Concreteness_ratings_cn_bigrams.csv')
    else:
        concreteness = pd.read_csv('../static/Concreteness_ratings_Brysbaert_et_al_BRM.csv')
    words_concreteness = dict(zip(concreteness['Word'].values, concreteness['Conc.M'].values))

    for ngram in ngrams:

        dataset_save_dir = os.path.join('../result/interpret/ngram/', basename)
        if not os.path.isdir(dataset_save_dir):
            os.makedirs(dataset_save_dir)

        pos_save_path = os.path.join(dataset_save_dir, f'ngram-{ngram}_pos_attr.csv')
        neg_save_path = os.path.join(dataset_save_dir, f'ngram-{ngram}_neg_attr.csv')

        compute_per_sentence_attr_score(pos_df, True, ngram, pos_save_path, language, words_concreteness)
        compute_per_sentence_attr_score(neg_df, False, ngram, neg_save_path, language, words_concreteness)


if __name__ == '__main__':
    main()
