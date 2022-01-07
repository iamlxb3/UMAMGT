import os
import ipdb
import copy
import random
import numpy as np
import glob
import jieba
import re
import spacy
from tqdm import tqdm
# import pickle
import _pickle as pickle
import hashlib
import benepar
import string

# import nltk

_PUNCTUATIONS = set(list(string.punctuation) + ['。', '，', '；', '：', '！', '？'])

"""
TAG_LIST = [".",",","-LRB-","-RRB-","``","\"\"","''",",","$","#","AFX","CC","CD","DT","EX","FW","HYPH","IN","JJ","JJR","JJS","LS","MD","NIL","NN","NNP","NNPS","NNS","PDT","POS","PRP","PRP$","RB","RBR","RBS","RP","SP","SYM","TO","UH","VB","VBD","VBG","VBN","VBP","VBZ","WDT","WP","WP$","WRB","ADD","NFP","GW","XX","BES","HVS","_SP"]
POS_LIST = ["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"]
DEP_LIST = ["acl", "acomp", "advcl", "advmod", "agent", "amod", "appos", "attr", "aux", "auxpass", "case", "cc", "ccomp", "compound", "conj", "cop", "csubj", "csubjpass", "dative", "dep", "det", "dobj", "expl", "intj", "mark", "meta", "neg", "nn", "npmod", "nsubj", "nsubjpass", "oprd", "obj", "obl", "pcomp", "pobj", "poss", "preconj", "prep", "prt", "punct",  "quantmod", "relcl", "root", "xcomp"]
NER_LIST = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]
"""


class SemanticModifier:
    def __init__(self, semantic_change, language, char_freq_rank=None):
        # len(self.spacy_parser.get_pipe("benepar")._label_vocab)
        # self.spacy_parser.get_pipe("parser").labels
        #
        self.semantic_change = semantic_change
        self.char_freq_rank = char_freq_rank
        self.language = language
        self.max_freq = max(self.char_freq_rank.values())
        if 'pos' in semantic_change or \
                'dep' in semantic_change or \
                'constit' in semantic_change or \
                'ner' in semantic_change:
            # self.spacy_parser = spacy.load("zh_core_web_sm")
            if self.language == 'cn':
                self.spacy_parser = spacy.load("zh_core_web_sm")  # zh_core_web_trf
                benepar_model = 'benepar_zh2'
            elif self.language == 'en':
                benepar.download('benepar_en3')
                self.spacy_parser = spacy.load("en_core_web_sm")  # en_core_web_trf
                benepar_model = 'benepar_en3'
            else:
                raise NotImplementedError
            self.benepar_model = benepar_model
            benepar.download(benepar_model)
            self.spacy_parser.add_pipe("benepar", config={"model": benepar_model})
            print(f'Spacy add benepar pipe done! Model: {benepar_model}')
            self.spacy_results = {}
        else:
            self.spacy_parser = None

        if 'use_stopword' in self.semantic_change or 'not_use_stopword' in self.semantic_change:
            if language == 'cn':
                stopword_path = '../static/cn_baidu_stopwords'
            elif language == 'en':
                stopword_path = '../static/en_nltk_baidu_stopwords'
            else:
                raise NotImplementedError

            self.stopwords = set()
            with open(stopword_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.stopwords.add(line)
        else:
            self.stopwords = None

        ipdb.set_trace()

    def rm_chars_out_freq(self, texts, char_freq_range):
        processed_texts = []
        for text in texts:
            split_text = text.split(' ')
            split_text_order = [(x, self.char_freq_rank.get(x, self.max_freq)) for x in split_text]
            split_text = [x if freq < char_freq_range else '[MASK]' for (x, freq) in split_text_order]
            processed_texts.append(' '.join(split_text))
        return processed_texts

    def rm_chars_in_freq(self, texts, char_freq_range):
        processed_texts = []
        for text in texts:
            split_text = text.split(' ')
            split_text_order = [(x, self.char_freq_rank.get(x, self.max_freq)) for x in split_text]
            split_text = [x if freq > char_freq_range else '[MASK]' for (x, freq) in split_text_order]
            processed_texts.append(' '.join(split_text))
        return processed_texts

    def _cn_stopword_split(self, split_text):
        new_split_text = []
        for x in split_text:
            if x == '[MASK]':
                new_split_text.append(x)
            else:
                for x_ in list(x):
                    new_split_text.append(x_)
        return new_split_text

    def change_texts(self, texts, char_freq_range):
        processed_texts = []

        # TODO: '</s>' 后面后一个空格, split(' ') 之后会出现一个''，暂时没有处理
        for text in tqdm(texts, total=len(texts)):

            # 这里其实是char在词表里的rank
            if char_freq_range != 0:

                if 'likelihood_rank' in self.semantic_change:
                    origin_text, text_rank = text
                    min_len = min(len(origin_text.split()), len(text_rank.split()))
                    origin_text_split = origin_text.split()[:min_len]
                    text_rank_split = text_rank.split()[:min_len]
                    # TODO: 这两个的长度当时没控制好
                    split_text_order = [(x, self.char_freq_rank.get(x, self.max_freq)) for x in origin_text_split]
                    masked_text = [x if freq < char_freq_range else '[MASK]' for (x, freq) in split_text_order]
                    mask_indices = np.where(np.array(masked_text) == '[MASK]')[0]
                    text_rank = np.array(text_rank_split, dtype='U6')
                    text_rank[mask_indices] = '[MASK]'
                    split_text = text_rank.tolist()
                    # assert len(origin_text.split()) == len(split_text)
                else:
                    split_text_order = [(x, self.char_freq_rank.get(x, self.max_freq)) for x in text.split(' ')]
                    split_text = [x if freq < char_freq_range else '[MASK]' for (x, freq) in split_text_order]
            else:
                if 'likelihood_rank' in self.semantic_change:
                    text = text[1]
                    split_text = text.split(' ')
                else:
                    split_text = text.split(' ')

            if 'reorder_shuffle' in self.semantic_change:
                random.shuffle(split_text)

            # 从高频词到低频词
            elif 'reorder_freq_high2low' in self.semantic_change:
                split_text_order = [(x, self.char_freq_rank.get(x, self.max_freq)) for x in split_text]
                split_text = sorted(split_text_order, key=lambda x: x[1])
                split_text = [x[0] for x in split_text]

            # 从低频词到高频词
            elif 'reorder_freq_low2high' in self.semantic_change:
                split_text_order = [(x, self.char_freq_rank.get(x, self.max_freq)) for x in split_text]
                split_text = sorted(split_text_order, key=lambda x: x[1], reverse=True)
                split_text = [x[0] for x in split_text]
            elif 'use_stopword' in self.semantic_change:

                if self.language == 'cn':
                    jieba_tokens = jieba.lcut(text.replace(' ', ''))
                    split_text = [x if (x in self.stopwords) or (x.lower() in self.stopwords) else '[MASK]' for x in
                                  jieba_tokens]
                    split_text = self._cn_stopword_split(split_text)
                elif self.language == 'en':
                    split_text = [x if (x in self.stopwords) or (x.lower() in self.stopwords) else '[MASK]' for x in
                                  split_text]
                else:
                    raise NotImplementedError

            elif 'not_use_stopword' in self.semantic_change:
                if self.language == 'cn':
                    jieba_tokens = jieba.lcut(text.replace(' ', ''))
                    split_text = [x if (x not in self.stopwords) and (x.lower() not in self.stopwords) else '[MASK]' for
                                  x in jieba_tokens]
                    split_text = self._cn_stopword_split(split_text)
                elif self.language == 'en':
                    split_text = [x if (x not in self.stopwords) and (x.lower() not in self.stopwords) else '[MASK]' for
                                  x in split_text]
                else:
                    raise NotImplementedError
            if 'char_deduplicate' in self.semantic_change:
                appear_set = set()
                new_split_text = []
                for x in split_text:
                    if x in appear_set:
                        new_split_text.append('[MASK]')
                    else:
                        new_split_text.append(x)
                        appear_set.add(x)
                split_text = new_split_text

            assert isinstance(split_text, list)

            if 'pos' in self.semantic_change or \
                    'dep' in self.semantic_change or \
                    'constit' in self.semantic_change or \
                    'ner' in self.semantic_change:
                # from spacy.lang.zh.examples import sentences
                # example_sentence = sentences[0]
                if self.language == 'cn':
                    to_parse_text = str(text.replace(' ', ''))
                else:
                    to_parse_text = str(text)
                model_name = self.spacy_parser.meta['name'] + '_' + self.spacy_parser.meta[
                    'lang'] + '_' + self.benepar_model
                text_md5 = hashlib.md5(
                    f'{model_name}_{to_parse_text}_{"".join(self.semantic_change)}'.encode()).hexdigest()

                if text_md5 in self.spacy_results:
                    split_text = self.spacy_results[text_md5]
                    assert isinstance(split_text, str)
                    # print(split_text)
                    processed_texts.append(split_text)
                    continue
                else:
                    text_pickle_path = f'../spacy_temp/{text_md5}.pkl'
                    if os.path.isfile(text_pickle_path):
                        try:
                            split_text = pickle.load(open(text_pickle_path, 'rb'))
                        except Exception as e:
                            print(f'Pickle exception: {e}')
                            continue
                        else:
                            assert isinstance(split_text, str)
                            processed_texts.append(split_text)
                            self.spacy_results[text_md5] = split_text
                            continue
                    else:
                        try:
                            parse_res = self.spacy_parser(to_parse_text)
                        except Exception as e:
                            try:
                                parse_res = self.spacy_parser(to_parse_text[:int(len(to_parse_text) / 2)])
                            except Exception as e:
                                try:
                                    parse_res = self.spacy_parser(to_parse_text[:int(len(to_parse_text) / 4)])
                                except Exception as e:
                                    try:
                                        parse_res = self.spacy_parser(to_parse_text[:512])
                                    except Exception as e:
                                        print(f'Pickle exception: {e}')
                                        ipdb.set_trace()
                    new_text = []
                    # Reference: https://spacy.io/usage/linguistic-features#dependency-parse

                    if 'pos' in self.semantic_change or 'dep' in self.semantic_change:
                        for token in parse_res:
                            if 'pos' in self.semantic_change:
                                new_text.append(token.pos_)
                            elif 'dep' in self.semantic_change:
                                new_text.append(token.dep_)
                                new_text.append(str(token.idx))
                                new_text.append(str(token.head.idx))
                                # new_text.append(token.head.text)
                            else:
                                raise Exception
                        new_text = ' '.join(new_text)
                    elif 'constit' in self.semantic_change:
                        new_text = []
                        for sen in parse_res.sents:
                            parse_string = sen._.parse_string
                            token_set = set(re.findall(r' ([^(]+?)\)', parse_string))
                            token_set = sorted(token_set, key=lambda x: len(x), reverse=True)
                            token_set = [x for x in token_set if x not in _PUNCTUATIONS]
                            no_word_parse_string = parse_string
                            for token in token_set:
                                no_word_parse_string = no_word_parse_string.replace(str(token), '')
                            new_text.append(no_word_parse_string)
                        new_text = '<s>'.join(new_text)
                        new_text = new_text.replace(' ', '')
                    elif 'ner' in self.semantic_change:
                        new_text = []
                        for ent in parse_res.ents:
                            new_text.append(ent.label_)
                            new_text.append(str(ent.start_char))
                            new_text.append(str(ent.end_char))
                        new_text = ' '.join(new_text)
                    else:
                        raise Exception
                    assert isinstance(new_text, str)
                    split_text = new_text
                    self.spacy_results[text_md5] = split_text
                    pickle.dump(split_text, open(text_pickle_path, 'wb'))
                    # 这个pos/dep tag的数量和原本中文的数量是对不上的，因为会对中文做分词，所以会短一点
                    assert isinstance(split_text, str)
                    # print(split_text)
                    processed_texts.append(split_text)
            else:
                processed_texts.append(' '.join(split_text))

        assert len(processed_texts) == len(texts), ipdb.set_trace()

        return processed_texts
