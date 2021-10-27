import ipdb
import copy
import random
import numpy as np


class SemanticModifier:
    def __init__(self, semantic_change, char_freq_rank=None):
        self.semantic_change = semantic_change
        self.char_freq_rank = char_freq_rank
        self.max_freq = max(self.char_freq_rank.values())

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

    def change_texts(self, texts, char_freq_range):
        processed_texts = []

        # TODO: '</s>' 后面后一个空格, split(' ') 之后会出现一个''，暂时没有处理
        for text in texts:
            # 这里其实是char在词表里的rank
            if char_freq_range != 0:

                if 'likelihood_rank' in self.semantic_change:
                    origin_text, text_rank = text
                    split_text_order = [(x, self.char_freq_rank.get(x, self.max_freq)) for x in origin_text.split()]
                    masked_text = [x if freq < char_freq_range else '[MASK]' for (x, freq) in split_text_order]
                    mask_indices = np.where(np.array(masked_text) == '[MASK]')[0]
                    text_rank = np.array(text_rank.split(' '), dtype='U6')
                    text_rank[mask_indices] = '[MASK]'
                    split_text = text_rank.tolist()
                    assert len(origin_text.split()) == len(split_text)
                else:
                    split_text_order = [(x, self.char_freq_rank.get(x, self.max_freq)) for x in text.split(' ')]
                    split_text = [x if freq < char_freq_range else '[MASK]' for (x, freq) in split_text_order]
            else:
                if 'likelihood' in self.semantic_change:
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

            processed_texts.append(' '.join(split_text))

        assert len(processed_texts) == len(texts)
        return processed_texts
