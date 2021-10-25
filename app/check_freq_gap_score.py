import ipdb
import collections
import numpy as np
import pandas as pd


# python3.6 check_freq_gap_score.py

def main():
    train_label_path = '../data/5billion/train.label'
    train_data_path = '../data/5billion/train.tgt'
    val_label_path = '../data/5billion/valid.label'
    val_data_path = '../data/5billion/valid.tgt'

    ai_written_chars = []
    ai_written_texts = []
    human_written_chars = []
    human_written_texts = []

    labels = []
    with open(train_label_path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            labels.append(line)

    with open(val_label_path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            labels.append(line)

    with open(train_data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            label = int(labels[i])

            line_split = line.split(' ')

            if label == 0:
                ai_written_chars.extend(line_split)
                ai_written_texts.append(line_split)
            elif label == 1:
                human_written_chars.extend(line_split)
                human_written_texts.append(line_split)

    with open(val_data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            label = int(labels[i])

            line_split = line.split(' ')

            if label == 0:
                ai_written_chars.extend(line_split)
                ai_written_texts.append(line_split)
            elif label == 1:
                human_written_chars.extend(line_split)
                human_written_texts.append(line_split)

    ai_written_char_counter = collections.Counter(ai_written_chars)
    human_written_char_counter = collections.Counter(human_written_chars)

    sort_char_path = '../data/sort_char.txt'

    sorted_chars = []
    with open(sort_char_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                sorted_chars.append(line)

    # freq_gap_df = {'char': [], 'ai_freq': [], 'human_freq': [], 'freq_ratio_gap': [], 'freq_gap_score': []}
    freq_gap_df = {'char': [], 'value': [], 'value_label': [], 'index': []}
    for i, char in enumerate(sorted_chars):

        if ai_written_char_counter[char] == 0 or human_written_char_counter[char] == 0:
            continue

        freq_gap_df['char'].append(char)
        freq_gap_df['value'].append(ai_written_char_counter[char])
        freq_gap_df['value_label'].append('ai_freq')
        freq_gap_df['index'].append(i)

        freq_gap_df['char'].append(char)
        freq_gap_df['value'].append(human_written_char_counter[char])
        freq_gap_df['value_label'].append('human_freq')
        freq_gap_df['index'].append(i)

        freq_ratio_gap = (ai_written_char_counter[char] - human_written_char_counter[char]) / ai_written_char_counter[
            char]
        freq_gap_score = abs((ai_written_char_counter[char] - human_written_char_counter[char])) / min(
            ai_written_char_counter[
                char], human_written_char_counter[char])

        freq_gap_df['char'].append(char)
        freq_gap_df['value'].append(freq_ratio_gap)
        freq_gap_df['value_label'].append('freq_ratio_gap')
        freq_gap_df['index'].append(i)

        freq_gap_df['char'].append(char)
        freq_gap_df['value'].append(freq_gap_score)
        freq_gap_df['value_label'].append('freq_gap_score')
        freq_gap_df['index'].append(i)

    freq_gap_df = pd.DataFrame(freq_gap_df)
    save_path = '../result/vis/freq_gap.csv'
    freq_gap_df.to_csv(save_path, index=False)
    print(f"Save freq gap to {save_path} done!")

    keep_N = 512
    keep_N_sorted_chars = sorted_chars[:1024]
    # keep_N_sorted_chars = sorted_chars

    char_ranges = np.linspace(0, len(keep_N_sorted_chars), num=80)

    char_range_text_length_df = {'char_index': [], 'value': [], 'value_label': [], 'type': []}

    for i, char_index in enumerate(char_ranges):

        if i == 0:
            continue
        else:
            char_index = int(char_index)

        char_range_top_set = set(sorted_chars[:char_index])
        char_range_tail_set = set(sorted_chars[char_index:])

        # char_range_top_set
        for text in human_written_texts:
            valid_text = [x for x in text if x in char_range_top_set]
            char_range_text_length_df['char_index'].append(char_index)
            char_range_text_length_df['value'].append(len(valid_text))
            char_range_text_length_df['value_label'].append('Human')
            char_range_text_length_df['type'].append('top freq')

        for text in ai_written_texts:
            valid_text = [x for x in text if x in char_range_top_set]
            char_range_text_length_df['char_index'].append(char_index)
            char_range_text_length_df['value'].append(len(valid_text))
            char_range_text_length_df['value_label'].append('Ai')
            char_range_text_length_df['type'].append('top freq')

        for text in human_written_texts:
            valid_text = [x for x in text if x in char_range_tail_set]
            char_range_text_length_df['char_index'].append(char_index)
            char_range_text_length_df['value'].append(len(valid_text))
            char_range_text_length_df['value_label'].append('Human')
            char_range_text_length_df['type'].append('tail freq')

        for text in ai_written_texts:
            valid_text = [x for x in text if x in char_range_tail_set]
            char_range_text_length_df['char_index'].append(char_index)
            char_range_text_length_df['value'].append(len(valid_text))
            char_range_text_length_df['value_label'].append('Ai')
            char_range_text_length_df['type'].append('tail freq')

        # human_char_in_top = [x for x in human_written_chars if x in char_range_top_set]
        # ai_char_in_top = [x for x in ai_written_chars if x in char_range_top_set]
        # human_char_in_tail = [x for x in human_written_chars if x in char_range_tail_set]
        # ai_char_in_tail = [x for x in ai_written_chars if x in char_range_tail_set]

    save_path = '../result/vis/char_range_seq_length.csv'
    char_range_text_length_df = pd.DataFrame(char_range_text_length_df)
    char_range_text_length_df.to_csv(save_path, index=False)
    print(f"Save char range text lengths to {save_path} done!")


if __name__ == '__main__':
    main()
