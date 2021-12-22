import os
import ipdb


# python temp_check.py
def main():
    base_dir = '../data'
    dataset_name = 'en_grover_backup'
    label_file_path = os.path.join(base_dir, dataset_name, 'train.label')
    text_file_path = os.path.join(base_dir, dataset_name, 'train.tgt')

    labels = []
    with open(label_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)

    label0_text1 = []
    with open(text_file_path, 'r') as f:
        f = f.readlines()
        f = [x.strip() for x in f if x.strip()]
        assert len(f) == len(labels)
        for i, line in enumerate(f):
            if labels[i] == '0':
                label0_text1.append(line)

    ipdb.set_trace()


if __name__ == '__main__':
    main()
