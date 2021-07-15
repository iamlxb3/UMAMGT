import ipdb
import sys

sys.path.append('..')

from core.transformers import BertModelWrapper
from core.interpreter import StoryInterpreter

from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = '/home/iamlxb3/temp_rsync_dir/story_turing_test/model_ckpts/cn-roberta-story-turning-train_90-seq_118'
    roberta = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2).to(device)
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    # roberta = RobertaForSequenceClassification.from_pretrained('roberta-base').to(device)
    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    roberta = BertModelWrapper(roberta)

    interpreter = StoryInterpreter(roberta, tokenizer, device)

    sentences = ['测试1。阿斯顿撒大', '测试2，阿斯顿撒大', '测试3，阿斯顿撒大']
    labels = [1, 1, 0]
    interpreter.interpret_sentence(sentences, labels, n_steps=200)
    ipdb.set_trace()


if __name__ == '__main__':
    main()
