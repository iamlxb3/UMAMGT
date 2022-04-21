import ipdb
import os
import math
import benepar
import spacy
import hashlib
import ntpath
import collections
import numpy as np
import pandas as pd
import jieba
from tqdm import tqdm
from nltk import ngrams as compute_ngrams
import _pickle as pickle


class TextAnalyzer:
    def __init__(self, do_lower=True, language=None, load_spacy_model=False):
        """
        https://github.com/neural-dialogue-metrics/Distinct-N/blob/master/distinct_n/metrics.py
        """
        self.do_lower = do_lower
        self.language = language
        self.load_spacy_model = load_spacy_model

        if load_spacy_model:
            if language == 'cn':
                self.spacy_parser = spacy.load("zh_core_web_trf")  # zh_core_web_trf
                benepar_model = 'benepar_zh2'
            elif language == 'en':
                benepar.download('benepar_en3')
                self.spacy_parser = spacy.load("en_core_web_trf")  # en_core_web_trf
                benepar_model = 'benepar_en3'
            else:
                raise NotImplementedError
            self.benepar_model = benepar_model
            benepar.download(benepar_model)
            self.spacy_parser.add_pipe("benepar", config={"model": benepar_model})
            print(f'Spacy add benepar pipe done! Model: {benepar_model}')
        else:
            self.spacy_parser = None

    def compute_ngram_distinct_from_file(self, file_path):
        texts = []
        with open(file_path, 'r') as f:
            for line in f:
                texts.append(line.strip())

        self.compute_ngram_distinct(texts)

    def compute_ngram_distinct(self, texts, n_grams=(1, 2, 3)):
        result = []
        for n_gram in n_grams:
            distinct_value = []
            for text in texts:
                if self.do_lower:
                    text = text.lower()
                sen_ngrams = compute_ngrams(text.split(), n_gram)
                if self.language == 'cn':
                    sen_ngrams = [''.join(x) for x in sen_ngrams]
                elif self.language == 'en':
                    sen_ngrams = [' '.join(x) for x in sen_ngrams]
                else:
                    raise NotImplementedError

                try:
                    distinct_value.append(len(set(sen_ngrams)) / len(text.split()))
                except ZeroDivisionError:
                    print('ZeroDivisionError!')
                    continue
            avg_distinct_value = np.average(distinct_value)
            result.append((n_gram, avg_distinct_value))
        return result

    def compute_ngram(self, texts, n_gram):
        n_gram_freq_dict = collections.defaultdict(lambda: 0)
        n_gram_idf_dict = collections.defaultdict(lambda: 0)
        for text in texts:
            if self.do_lower:
                text = text.lower()
                sen_ngrams = compute_ngrams(text.split(), n_gram)
                sen_ngrams = [''.join(x) for x in sen_ngrams]

                for x in set(sen_ngrams):
                    n_gram_idf_dict[x] += 1

                for x in sen_ngrams:
                    n_gram_freq_dict[x] += 1
        n_gram_idf_dict = {k: math.log(len(texts) / v) for k, v in n_gram_idf_dict.items()}
        return n_gram_freq_dict, n_gram_idf_dict

    def analyse_concreteness(self, texts, language):
        if language == 'en':
            concreteness = pd.read_csv('../static/Concreteness_ratings_Brysbaert_et_al_BRM.csv')
        elif language == 'cn':
            concreteness = pd.read_csv('../static/Concreteness_ratings_cn_bigrams.csv')
        else:
            raise NotImplementedError

        words_concreteness = dict(zip(concreteness['Word'].values, concreteness['Conc.M'].values))

        target_pos = ('VERB', 'NOUN', 'ADV', 'ADJ')

        concreteness_dict = collections.defaultdict(lambda: [])
        for text in tqdm(texts, total=len(texts)):

            if self.do_lower:
                text = text.lower()

            text_pickle_path, to_parse_text = self._pickle_path_for_spacy(text, 'pos', 'text_analyzer2')

            token_pickle_path = self._get_token_pickle_path(text_pickle_path)
            if os.path.isfile(text_pickle_path) and os.path.isfile(token_pickle_path):
                pos_tags, tokens = pickle.load(open(text_pickle_path, 'rb')), pickle.load(
                    open(token_pickle_path, 'rb'))
            else:
                spacy_parse_result = self._spacy_parse_text(to_parse_text, 'pos', text_pickle_path,
                                                            return_token=True,
                                                            dump_res=True,
                                                            dump_token=True)
                if spacy_parse_result is not None:
                    pos_tags, tokens = spacy_parse_result
                else:
                    continue

            concreteness_tmp_dict = collections.defaultdict(lambda: [])

            for pos_tag, token in zip(pos_tags, tokens):
                if pos_tag in target_pos:

                    if language == 'en':
                        if token in words_concreteness:
                            concreteness_tmp_dict[pos_tag].append(words_concreteness[token])
                    elif language == 'cn':
                        token_list = list(token)
                        concreteness = 0.0
                        if len(token_list) < 2:
                            pass
                        elif len(token_list) == 2:
                            concreteness += words_concreteness.get(''.join(token_list), 0.0)
                        else:
                            bigram_token = list(compute_ngrams(token_list, 2))
                            for bigram in bigram_token:
                                concreteness += words_concreteness.get(''.join(bigram), 0.0)
                        if concreteness > 0.0:
                            concreteness_tmp_dict[pos_tag].append(concreteness)
                    else:
                        raise NotImplementedError

            concreteness_tmp_dict = {k: np.average(v) for k, v in concreteness_tmp_dict.items()}
            for pos in target_pos:
                if pos not in concreteness_tmp_dict:
                    concreteness_tmp_dict[pos] = 0.0

            for k, v in concreteness_tmp_dict.items():
                concreteness_dict[k].append(v)

        return concreteness_dict

    def analyse_stopwords(self, texts):
        if self.language == 'en':
            stopword_path = '../static/en_nltk_baidu_stopwords'
        elif self.language == 'cn':
            stopword_path = '../static/cn_baidu_stopwords'
        else:
            raise NotImplementedError

        stopwords = []
        with open(stopword_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    stopwords.append(line)
        stopwords = set(stopwords)
        stopword_sen_ratio = []
        stopwords_ratio = collections.defaultdict(lambda: 0)

        for text in texts:
            if self.do_lower:
                text = text.lower()
            if self.language == 'en':
                text_tokenized = text.split(' ')
            elif self.language == 'cn':
                text_tokenized = jieba.lcut(text)
            else:
                raise NotImplementedError

            stopword_count = 0
            for x in text_tokenized:
                if x in stopwords:
                    stopwords_ratio[x] += 1
                    stopword_count += 1
            stopword_ratio = stopword_count / len(text_tokenized)
            stopword_sen_ratio.append(stopword_ratio)

        total_stopword_count = np.sum(list(stopwords_ratio.values()))
        stopwords_ratio = {k: v / total_stopword_count for k, v in stopwords_ratio.items()}
        return stopword_sen_ratio, stopwords_ratio

    def _get_token_pickle_path(self, text_pickle_path):
        pickle_base_dir = os.path.dirname(text_pickle_path)
        pickle_file_name = ntpath.basename(text_pickle_path)
        token_pickle_name = pickle_file_name.split('.')[0] + f'_token.' + pickle_file_name.split('.')[1]
        token_pickle_path = os.path.join(pickle_base_dir, token_pickle_name)
        return token_pickle_path

    def _spacy_parse_text(self, to_parse_text, parse_choice, text_pickle_path,
                          return_token=False,
                          dump_res=True,
                          dump_token=False):
        try:
            parse_res = self.spacy_parser(str(to_parse_text))
        except Exception as e:
            if 'exceeds the maximum supported length' in str(e):
                return None
            else:
                ipdb.set_trace()
                return None

        parse_text = []
        tokens = []
        for token in parse_res:
            if return_token:
                tokens.append(token.text)
            if parse_choice == 'pos':
                parse_text.append(token.pos_)
            elif parse_choice == 'dep':
                parse_text.append(token.dep_)
            elif parse_choice == 'ner':
                for ent in parse_res.ents:
                    parse_text.append(ent.label_)
        if dump_res:
            if parse_text:
                if dump_res:
                    pickle.dump(parse_text, open(text_pickle_path, 'wb'))
                if dump_token:
                    assert return_token
                    assert tokens
                    pickle.dump(tokens, open(self._get_token_pickle_path(text_pickle_path), 'wb'))
            else:
                return None
        if return_token:
            return parse_text, tokens
        else:
            return parse_text

    def _pickle_path_for_spacy(self, text, parse_choice, save_prefix):
        if self.language == 'en':
            to_parse_text = text
        elif self.language == 'cn':
            to_parse_text = text.replace(' ', '')
        else:
            raise NotImplementedError

        model_name = self.spacy_parser.meta['name'] + '_' + self.spacy_parser.meta[
            'lang'] + '_' + self.benepar_model
        # text_analyzer2
        text_md5 = hashlib.md5(f'{save_prefix}_{model_name}_{to_parse_text}_{parse_choice}'.encode()).hexdigest()
        text_pickle_path = f'../spacy_temp/{text_md5}.pkl'
        return text_pickle_path, to_parse_text

    def load_parsed_texts_by_spacy(self, texts, parse_choice):
        assert self.spacy_parser is not None
        assert parse_choice in {'pos', 'ner', 'dep'}

        all_results = []
        for text in tqdm(texts, total=len(texts)):

            text_pickle_path, to_parse_text = self._pickle_path_for_spacy(text, parse_choice, 'text_analyzer2')

            # filter text
            if len(to_parse_text) < 10:
                continue

            if os.path.isfile(text_pickle_path):
                try:
                    parse_text = pickle.load(open(text_pickle_path, 'rb'))
                except Exception as e:
                    print(f'Pickle exception: {e}')
                    parse_text = self._spacy_parse_text(to_parse_text, parse_choice, text_pickle_path)
            else:
                parse_text = self._spacy_parse_text(to_parse_text, parse_choice, text_pickle_path)
            if parse_text is None:
                continue
            else:
                all_results.extend(parse_text)

        all_results = collections.Counter(all_results)
        total_count = np.sum(list(all_results.values()))
        all_results = {k: v / total_count for k, v in all_results.items()}
        return all_results
