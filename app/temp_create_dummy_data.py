import os
import random


def main():
    data_path = '../data/5billion/valid.tgt'
    save_path = '../data/5billion/valid_rank.tgt'

    with open(save_path, 'w') as f_w:
        with open(data_path, 'r') as f_r:
            for line in f_r:
                line = line.strip()
                if line:
                    line = line.split()
                    number_line = [str(random.randint(0, 99)) for _ in line]
                    f_w.write(' '.join(number_line) + '\n')


if __name__ == '__main__':
    main()
