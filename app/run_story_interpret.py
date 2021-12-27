import ipdb
import sys
import random
import argparse
import ntpath

sys.path.append('..')

from core.transformers import BertModelWrapper
from core.interpreter import StoryInterpreter
from core.task import StoryTuringTest
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# python run_story_interpret.py --debug_N 100 --batch_size 16

def args_parse():
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('--debug_N', type=int)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--data_dir', type=str, default='../data/5billion')
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--ig_n_steps', type=int, default=50)
    parser.add_argument('--max_text_length', type=int)
    parser.add_argument('--model_name', type=str, default='bert')
    args = parser.parse_args()
    return args


def main():
    args = args_parse()
    debug_N = args.debug_N
    batch_size = args.batch_size
    model_dir = args.model_dir
    data_dir = args.data_dir
    ig_n_steps = args.ig_n_steps
    max_text_length = args.max_text_length
    model_name = args.model_name
    dataset_name = ntpath.basename(data_dir)
    language = ntpath.basename(dataset_name).split('_')[0]

    seed_everything(1)
    interpret_all_labels = True

    if not debug_N:
        debug_N = None

    save_base_name = f'../result/interpret/{dataset_name}_{ntpath.basename(model_dir)}_text_len_{max_text_length}_{interpret_all_labels}'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    roberta = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2).to(device)

    if language == 'cn':
        tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    elif language == 'en':
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    else:
        raise NotImplementedError
    tokenizer.model_max_length = max_text_length

    roberta = BertModelWrapper(roberta, model_name=model_name).to(device)

    interpreter = StoryInterpreter(roberta,
                                   tokenizer,
                                   device,
                                   n_steps=ig_n_steps,
                                   save_base_name=save_base_name,
                                   correct_label_only=True
                                   )

    # sentences = ['测试1。阿斯顿撒大', '测试2，阿斯顿撒大', '测试3，阿斯顿撒大']
    # labels = [1, 1, 0]
    # interpreter.interpret_sentence(sentences, labels, n_steps=200, correct_label_only=True)

    # load test data
    story_turing_test = StoryTuringTest(tokenizer, dataset_name)
    whole_texts, whole_labels = story_turing_test.read_cn_novel_whole_data(data_dir, ['None'])
    if debug_N:
        whole_texts, whole_labels = whole_texts[:debug_N], whole_labels[:debug_N]
    dataset = story_turing_test.create_dataset(whole_texts, whole_labels, max_length=max_text_length)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=0)
    interpreter.interpret_dataloder(dataloader)
    # ipdb.set_trace()


if __name__ == '__main__':
    main()
