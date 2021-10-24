import sys

sys.path.append('..')
import random
import copy
import ipdb
import torch
import numpy as np
from transformers import RobertaConfig, RobertaModel
from transformers import RobertaTokenizer
from core.utils import load_save_json
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity


# 训练任务是倒数第二个fxnlp
# python3.6 check_shuffle_embedding.py

def compute_cos_sim(arr1, arr2):
    cos_sim = cosine_similarity(arr1, arr2)[0]
    # cos_sim = np.dot(arr1, arr2.T) / (norm(arr1) * norm(arr2)) # 这个地方是norm arr2计算错了
    # cos_sim = cos_sim[0]
    return cos_sim


def compute_embedding(tokenizer, roberta, sample, device, layer=-2):
    with torch.no_grad():
        model_input = tokenizer(sample, padding=True, truncation=True, return_tensors="pt")
        model_input = {k: v.to(device) for k, v in model_input.items()}
        output = roberta(input_ids=model_input['input_ids'].to(device),
                         attention_mask=model_input['attention_mask'].to(device),
                         output_hidden_states=True)
        batch_layer_hiddens = output[2][layer]
        batch_embeddings = torch.mean(batch_layer_hiddens, dim=1).detach().cpu().numpy()

    return batch_embeddings


def main():
    # data path
    data_path = '../data/classification/DBpedia.json'
    data = load_save_json(data_path, 'load')
    all_labels = list(data.keys())
    sample_N_per_label = 10

    # init model
    roberta = RobertaModel.from_pretrained('roberta-base')
    roberta.eval()
    # configuration = roberta.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    roberta.to(device)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    random.seed(1)

    for label in all_labels:

        sample = data[label][:sample_N_per_label]

        for origin_x in sample:

            origin_x_list = origin_x.split()

            # create shuffle samples
            shuffle_xs = []
            debug_xs = []
            for _ in range(sample_N_per_label - 1):
                shuffle_x = copy.deepcopy(origin_x_list)
                debug_x = copy.deepcopy(origin_x_list)
                del_index = random.randint(0, len(debug_x) - 1)
                debug_x = debug_x[:del_index - 1] + debug_x[del_index:]
                random.shuffle(shuffle_x)
                shuffle_xs.append(' '.join(shuffle_x))
                debug_xs.append(' '.join(debug_x))

            # create same-class samples
            same_class_xs = [x for x in sample if x != origin_x]

            origin_embedding = compute_embedding(tokenizer, roberta, [origin_x], device, layer=-2)
            shuffle_embedding = compute_embedding(tokenizer, roberta, shuffle_xs, device, layer=-2)
            same_class_embedding = compute_embedding(tokenizer, roberta, same_class_xs, device, layer=-2)
            debug_embedding = compute_embedding(tokenizer, roberta, debug_xs, device, layer=-2)

            shuffle_cos_sim = compute_cos_sim(origin_embedding, shuffle_embedding)
            same_class_cos_sim = compute_cos_sim(origin_embedding, same_class_embedding)
            debug_cos_sim = compute_cos_sim(origin_embedding, debug_embedding)

            print(f"Shuffle mean: {np.average(shuffle_cos_sim)}, same class mean: {np.average(same_class_cos_sim)}")
            ipdb.set_trace()


if __name__ == '__main__':
    main()
