import ipdb
import sys

sys.path.append('..')

from core.transformers import BertModelWrapper
from core.interpreter import StoryInterpreter
from core.task import StoryTuringTest
from torch.utils.data import DataLoader
from transformers import RobertaConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


# python test_captum_imdb_roberta.py

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = RobertaConfig.from_pretrained("roberta-base")
    config.output_hidden_states = True

    roberta = AutoModelForSequenceClassification.from_pretrained("aychang/roberta-base-imdb").to(device)
    tokenizer = AutoTokenizer.from_pretrained("aychang/roberta-base-imdb")

    # roberta = RobertaForSequenceClassification.from_pretrained('roberta-base').to(device)
    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    roberta = BertModelWrapper(roberta, model_type='roberta')
    print(f"Init roberta done!")
    save_base_name = f'../result/interpret/imdb_test'
    interpreter = StoryInterpreter(roberta,
                                   tokenizer,
                                   device,
                                   save_base_name=save_base_name,
                                   n_steps=300,
                                   use_pad_baseline=False)
    print(f"Init interpreter done!")

    # sentences = [
    #     'This is an awesome movie.',
    #     'This is a bad movie. Holy Shit!'
    # ]
    # labels = [1, 0]
    #
    # sentences = [
    #     'that is a terrible movie',
    #     'This is an awesome movie',
    #     'This is a really bad movie, like shit'
    # ]
    # labels = [0, 1, 0]
    sentences = [
        'that is a terrible movie',
        'This is an awesome movie',
        'such a great show!',
        'This is a really bad movie, like shit',
        'Loved this movie. I love the little details throughout the movie that add suspense and character.',
        'I rarely stop watching a movie although how crappy it is. Well for this one I made an exception since its beyond boring. I cant leave long reviews so this is it.. SKIP IT',
        'Crap, crap and totally crap. Did I mention this film was totally crap? Well, it"s totally crap',
        "The worst movie I've ever seen."
    ]
    labels = [0, 1, 1, 0, 1, 0, 0, 0]
    story_turing_test = StoryTuringTest(tokenizer, 'imdb')
    test_dataset = story_turing_test.create_dataset(sentences, labels)
    dataloader = DataLoader(test_dataset,
                            batch_size=1,
                            num_workers=0)
    interpreter.interpret_dataloder(dataloader)


if __name__ == '__main__':
    main()
