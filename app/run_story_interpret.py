import ipdb
import sys
import argparse
import ntpath

sys.path.append('..')

from core.transformers import BertModelWrapper
from core.interpreter import StoryInterpreter
from core.task import StoryTuringTest
from torch.utils.data import DataLoader

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
    parser.add_argument('--vis_record_save_path', type=str)
    parser.add_argument('--max_text_length', type=int)
    args = parser.parse_args()
    return args


def main():
    args = args_parse()
    debug_N = args.debug_N
    batch_size = args.batch_size
    model_dir = args.model_dir
    ig_n_steps = args.ig_n_steps
    vis_record_save_path = args.vis_record_save_path
    max_text_length = args.max_text_length
    if not debug_N:
        debug_N = None

    if not vis_record_save_path:
        vis_record_save_path = f'../result/vis_record_{ntpath.basename(model_dir)}.pkl'
    print(f"Vis record save path: {vis_record_save_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    roberta = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2).to(device)
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    tokenizer.model_max_length = max_text_length

    # roberta = RobertaForSequenceClassification.from_pretrained('roberta-base').to(device)
    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    roberta = BertModelWrapper(roberta).to(device)

    interpreter = StoryInterpreter(roberta,
                                   tokenizer,
                                   device,
                                   n_steps=ig_n_steps,
                                   vis_record_save_path=vis_record_save_path,
                                   correct_label_only=True
                                   )

    # sentences = ['测试1。阿斯顿撒大', '测试2，阿斯顿撒大', '测试3，阿斯顿撒大']
    # labels = [1, 1, 0]
    # interpreter.interpret_sentence(sentences, labels, n_steps=200, correct_label_only=True)

    # load test data
    story_turing_test = StoryTuringTest(tokenizer)
    _, test_dataset = story_turing_test.read_test_data(args.data_dir, debug_N=debug_N)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 num_workers=0)
    interpreter.interpret_dataloder(test_dataloader)


if __name__ == '__main__':
    main()
