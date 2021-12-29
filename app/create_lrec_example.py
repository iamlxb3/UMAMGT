import os
import sys
import ipdb

sys.path.append('..')
from core.semantic_modifier import SemanticModifier


# python create_lrec_example.py
def main():
    dataset_name = 'en_grover'
    char_freq_txt_path = os.path.join('../data', dataset_name, 'sort_char.txt')

    # read char frequencies
    char_freq_rank = {}
    with open(char_freq_txt_path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            char_freq_rank[line] = i

    # semantic_changes = ['reorder_shuffle',
    #                     'reorder_freq_high2low',
    #                     'reorder_freq_low2high',
    #                     'char_deduplicate',
    #                     'None',
    #                     'likelihood_rank',
    #                     'pos',
    #                     'dep',
    #                     'constit',  # phrase structure tree, constituency tree,
    #                     'ner',
    #                     'use_stopword',
    #                     'not_use_stopword'
    #                     ]

    semantic_changes = [['None']]
    # reorder_freq_high2low -> Ascend
    # reorder_freq_low2high -> Descend
    texts = ["It is believed Congress is going to finish its authorization of NASA's budget soon."]

    for semantic_change in semantic_changes:
        semantic_modifier = SemanticModifier(semantic_change, 'en', char_freq_rank=char_freq_rank)

        char_freq_range = 1024
        modif_texts = semantic_modifier.change_texts(texts, char_freq_range)
        print(f"semantic_change: {semantic_change}, text: {modif_texts}")
        ipdb.set_trace()


if __name__ == '__main__':
    import time

    t1 = time.time()
    main()
    t2 = time.time()
    print(f'Total {(t2 - t1) / 60.0} minutes')
