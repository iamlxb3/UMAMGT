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

    config = RobertaConfig.from_pretrained("roberta-base")
    config.output_hidden_states = True

    roberta = AutoModelForSequenceClassification.from_pretrained("aychang/roberta-base-imdb").to(device)
    tokenizer = AutoTokenizer.from_pretrained("aychang/roberta-base-imdb")

    # roberta = RobertaForSequenceClassification.from_pretrained('roberta-base').to(device)
    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    ipdb.set_trace()

    roberta = BertModelWrapper(roberta)

    interpreter = StoryInterpreter(roberta, tokenizer, device)

    sentences = ['This is an awesome movie', 'such a great show!', 'This is a really bad movie, like shit']
    labels = [1, 1, 0]
    interpreter.interpret_sentence(sentences, labels, n_steps=200)
    ipdb.set_trace()

if __name__ == '__main__':
    main()
