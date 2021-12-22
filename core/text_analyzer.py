import ipdb
import numpy as np
from nltk import ngrams as compute_ngrams


class TextAnalyzer:
    def __init__(self, do_lower=True):
        """
        https://github.com/neural-dialogue-metrics/Distinct-N/blob/master/distinct_n/metrics.py
        """
        self.do_lower = do_lower

    def compute_ngram_from_file(self, file_path):
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
                sen_ngrams = [x[0] for x in sen_ngrams]
                try:
                    distinct_value.append(len(set(sen_ngrams)) / len(sen_ngrams))
                    # distinct_value.append(len(set(sen_ngrams)) / len(text.split()))
                except ZeroDivisionError:
                    print('ZeroDivisionError!')
                    continue
            avg_distinct_value = np.average(distinct_value)
            result.append((n_gram, avg_distinct_value))
        return result
