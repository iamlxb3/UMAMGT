import ipdb
import random
import collections
from tqdm import tqdm


# python create_reddit_data.py

def main():
    source_data_path1 = '../raw/writing_prompt_reddit/test.wp_source'
    target_data_path1 = '../raw/writing_prompt_reddit/test.wp_target'
    source_data_path2 = '../raw/writing_prompt_reddit/valid.wp_source'
    target_data_path2 = '../raw/writing_prompt_reddit/valid.wp_target'
    source_data_path3 = '../raw/writing_prompt_reddit/train.wp_source'
    target_data_path3 = '../raw/writing_prompt_reddit/train.wp_target'
    prompts_data_path = '../raw/writing_prompt_reddit/fusion_prompts.txt'
    fusion_data_path = '../raw/writing_prompt_reddit/fusion_stories.txt'
    seq2seq_data_path = '../raw/writing_prompt_reddit/seq2seq_stories.txt'

    source_dict = collections.defaultdict(lambda: [])
    with open(source_data_path1, 'r') as f_s:
        f_s = f_s.readlines()
        with open(target_data_path1, 'r') as f_t:
            f_t = f_t.readlines()
            for i, line_s in enumerate(f_s):
                line_s = line_s.strip()
                line_t = f_t[i].strip()
                source_dict[line_s].append(line_t)

    with open(source_data_path2, 'r') as f_s:
        f_s = f_s.readlines()
        with open(target_data_path2, 'r') as f_t:
            f_t = f_t.readlines()
            for i, line_s in enumerate(f_s):
                line_s = line_s.strip()
                line_t = f_t[i].strip()
                source_dict[line_s].append(line_t)

    with open(source_data_path3, 'r') as f_s:
        f_s = f_s.readlines()
        with open(target_data_path3, 'r') as f_t:
            f_t = f_t.readlines()
            for i, line_s in enumerate(f_s):
                line_s = line_s.strip()
                line_t = f_t[i].strip()
                source_dict[line_s].append(line_t)

    all_keys = list(source_dict.keys())

    title_save_path = '../data/writing_prompt/train_title.tgt'
    train_save_path = '../data/writing_prompt/train.tgt'
    train_label_save_path = '../data/writing_prompt/train.label'
    train_rank_save_path = '../data/writing_prompt/train_rank.tgt'

    fusion_stories = open(fusion_data_path, 'r').readlines()
    seq2seq_stories = open(seq2seq_data_path, 'r').readlines()
    not_valid_count = 0
    with open(prompts_data_path, 'r') as f:
        f_readline = f.readlines()
        for i, raw_line in tqdm(enumerate(f_readline), total=len(f_readline)):
            line = raw_line.strip()
            if not line in source_dict:
                not_valid_count += 1
                print(f"not_valid_count: {not_valid_count}")
                continue
            human_story = random.choice(source_dict[line])
            fusion_story = fusion_stories[i]
            seq2seq_story = seq2seq_stories[i]
            # --------------------------------------------------------------------------------------------------------------
            # Save for fusion
            # --------------------------------------------------------------------------------------------------------------
            with open(title_save_path, 'a') as f:
                f.write(line + '\n')

            with open(train_save_path, 'a') as f:
                f.write(fusion_story.strip() + '\n')

            with open(train_label_save_path, 'a') as f:
                f.write('0' + '\n')

            # --------------------------------------------------------------------------------------------------------------

            # --------------------------------------------------------------------------------------------------------------
            # Save for origin aug
            # --------------------------------------------------------------------------------------------------------------
            with open(title_save_path, 'a') as f:
                f.write(line + '\n')

            with open(train_save_path, 'a') as f:
                f.write(human_story.strip() + '\n')

            with open(train_label_save_path, 'a') as f:
                f.write('1' + '\n')
            # --------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    main()
