import collections


# python3.6 create_char_freq_txt_from_cn_novel.py

def main():
    train_label_path = '../data/5billion/train.label'
    train_data_path = '../data/5billion/train.tgt'
    val_label_path = '../data/5billion/valid.label'
    val_data_path = '../data/5billion/valid.tgt'

    train_labels = []
    with open(train_label_path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            train_labels.append(line)

    with open(val_label_path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            train_labels.append(line)

    label0_chars = []
    label1_chars = []

    with open(train_data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            label = int(train_labels[i])
            chars = line.split(' ')

            if label == 0:
                label0_chars.extend(chars)
            elif label == 1:
                label1_chars.extend(chars)

    with open(val_data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            label = int(train_labels[i])
            chars = line.split(' ')

            if label == 0:
                label0_chars.extend(chars)
            elif label == 1:
                label1_chars.extend(chars)

    print(f"label 0 char count: {len(label0_chars)}, label 1 char count: {len(label1_chars)}")
    all_chars = label0_chars + label1_chars
    chars_counter = collections.Counter(all_chars)
    sorted_label_counter = sorted(chars_counter.items(), key=lambda x: x[1], reverse=True)
    # maxcount = max([x for _, x in sorted_label1_counter])
    sorted_all_chars = [x for x, _ in sorted_label_counter]
    with open('../data/sort_char.txt', 'w') as f:
        for x in sorted_all_chars:
            f.write(x + '\n')


if __name__ == '__main__':
    main()
