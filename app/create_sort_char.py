import collections
import argparse
import os
import ipdb


# python create_sort_char.py --target_dir ../data/en_grover
# python create_sort_char.py --target_dir ../data/en_writing_prompt

def args_parse():
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('--target_dir', type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = args_parse()
    target_dir = args.target_dir
    file_path = os.path.join(target_dir, 'train.tgt')
    save_path = os.path.join(target_dir, 'sort_char.txt')
    chars = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            line_split = line.split()
            chars.extend(line_split)
    char_dict = collections.Counter(chars)
    char_dict = sorted(char_dict.items(), key=lambda x: x[1], reverse=True)

    with open(save_path, 'w') as f:
        for x, _ in char_dict:
            f.write(x + '\n')
    print(f"Save to {save_path}")
    ipdb.set_trace()


if __name__ == '__main__':
    main()
