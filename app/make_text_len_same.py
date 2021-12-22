import os
import ipdb
import string
import collections


# python make_text_len_same.py

def shorten_text(long_text, short_text):
    new_long_text = long_text[:len(short_text)]
    next_index = len(short_text)
    while True:
        if next_index >= len(long_text):
            break
        next_token = long_text[next_index]
        new_long_text.append(next_token)
        if next_token in {'.', '!', '?', '``', ';'}:
            break
        else:
            next_index += 1
        if next_index >= len(short_text) + 10:
            break
    return new_long_text


def main():
    base_dir = '../data'
    dataset_name = 'en_grover'
    train_file_path = os.path.join(base_dir, dataset_name, 'train_backup.tgt')
    title_file_path = os.path.join(base_dir, dataset_name, 'train_title.tgt')
    label_file_path = os.path.join(base_dir, dataset_name, 'train.label')
    rank_file_path = os.path.join(base_dir, dataset_name, 'train_rank.tgt')
    new_save_dir = os.path.join(base_dir, 'new_' + dataset_name)
    if not os.path.isdir(new_save_dir):
        os.makedirs(new_save_dir)

    titles = []
    with open(title_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                titles.append(line)

    labels = []
    with open(label_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)

    ranks = []
    with open(rank_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                ranks.append(line)

    dataset = collections.defaultdict(lambda: [])
    with open(train_file_path, 'r') as fr:
        all_texts = fr.readlines()
        for i, line in enumerate(all_texts):
            line = line.strip()
            title = titles[i]
            dataset[title].append([line, ranks[i], labels[i]])

    dataset = {k: v for k, v in dataset.items() if len(v) > 1}
    all_tiles = dataset.keys()

    new_texts = []
    new_labels = []
    new_ranks = []
    new_titles = []
    for title in all_tiles:
        sample_pair = dataset[title]
        human_text, human_rank, human_label = [x for x in sample_pair if x[-1] == '1'][0]
        machine_text, machine_rank, machine_label = [x for x in sample_pair if x[-1] == '0'][0]

        if len(human_text) > len(machine_text):
            human_text = ' '.join(shorten_text(human_text.split(), machine_text.split()))
        elif len(machine_text) > len(human_text):
            machine_text = ' '.join(shorten_text(machine_text.split(), human_text.split()))
        new_texts.append(machine_text)
        new_labels.append(machine_label)
        new_ranks.append(machine_rank)
        new_titles.append(title)

        new_texts.append(human_text)
        new_labels.append(human_label)
        new_ranks.append(human_rank)
        new_titles.append(title)

    #     train_file_path = os.path.join(base_dir, dataset_name, 'train_backup.tgt')
    #     title_file_path = os.path.join(base_dir, dataset_name, 'train_title.tgt')
    #     label_file_path = os.path.join(base_dir, dataset_name, 'train.label')
    #     rank_file_path = os.path.join(base_dir, dataset_name, 'train_rank.tgt')
    with open(os.path.join(new_save_dir, 'train.label'), 'w') as f:
        for x in new_labels:
            f.write(x + '\n')
    with open(os.path.join(new_save_dir, 'train_title.label'), 'w') as f:
        for x in new_titles:
            f.write(x + '\n')
    with open(os.path.join(new_save_dir, 'train_rank.tgt'), 'w') as f:
        for x in new_ranks:
            f.write(x + '\n')
    with open(os.path.join(new_save_dir, 'train.tgt'), 'w') as f:
        for x in new_texts:
            f.write(x + '\n')
    # # --------------
    # dataset_name = 'en_grover'
    # train_file_path = os.path.join(base_dir, dataset_name, 'train_backup.tgt')
    # new_save_path = os.path.join(base_dir, dataset_name, 'train.tgt')
    #
    # with open(new_save_path, 'w') as fw:
    #     with open(train_file_path, 'r') as fr:
    #         all_texts = fr.readlines()
    #         for i, line in enumerate(all_texts):
    #             line = line.strip()
    #             if line:
    #                 if i % 2 != 0:  # machine label
    #                     machine_text = line.split()
    #                     human_text = all_texts[i - 1].split()
    #
    #                     if len(human_text) > len(machine_text):
    #                         human_text = shorten_text(human_text, machine_text)
    #                     elif len(machine_text) > len(human_text):
    #                         machine_text = shorten_text(machine_text, human_text)
    #
    #                     fw.write(' '.join(human_text) + '\n')
    #                     fw.write(' '.join(machine_text) + '\n')


if __name__ == '__main__':
    main()
