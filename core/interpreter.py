import os
import ipdb
import torch
import pickle

from captum.attr import IntegratedGradients
from captum.attr import visualization
from captum.attr import LayerIntegratedGradients, TokenReferenceBase


def add_attributions_to_visualizer(attributions, tokens, pred, pred_ind, label, delta, vis_data_records):
    attributions = attributions.squeeze(0)
    attributions = attributions[1:-1][:len(tokens)]
    attributions = attributions.sum(dim=1)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.detach().numpy()

    # storing couple samples in an array for visualization purposes
    vis_data_records.append(visualization.VisualizationDataRecord(
        attributions,
        pred,
        pred_ind,
        label,
        "label",
        attributions.sum(),
        tokens[:len(attributions)],
        delta))


class StoryInterpreter:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.ig = IntegratedGradients(model)
        self.vis_data_records_ig = []

    def interpret_sentence(self,
                           sentences,
                           labels,
                           n_steps=50,
                           vis_record_save_path=''):
        self.model.eval()
        self.model.zero_grad()

        encoded_inputs = self.tokenizer(sentences, padding=True, return_tensors="pt")
        input_embedding = self.model.model_body.embeddings(input_ids=encoded_inputs['input_ids'])

        all_pad_input_ids = torch.full_like(encoded_inputs['input_ids'], self.tokenizer.pad_token_id)
        all_pad_embedding = self.model.model_body.embeddings(input_ids=all_pad_input_ids)

        # baselines_embeddings =
        # predict
        predict_probs = self.model(input_embedding, attention_mask=encoded_inputs['attention_mask'])
        predict_labels = torch.argmax(predict_probs, dim=1).detach().cpu().tolist()

        # compute attributions and approximation delta using integrated gradients
        # attributions_ig shape: batch x max_seq_len x 768

        attributions_ig, delta = self.ig.attribute(inputs=input_embedding,
                                                   baselines=all_pad_embedding,
                                                   target=[1 for _ in predict_labels],
                                                   n_steps=n_steps,
                                                   return_convergence_delta=True)
        print("-" * 78)
        print(f"Predict prob: {predict_probs}")
        print(f"Predict label: {predict_labels}")
        print(f"Delta: {delta}")

        for i, input_id in enumerate(encoded_inputs['input_ids']):
            tokens = self.tokenizer.convert_ids_to_tokens(input_id, skip_special_tokens=True)
            tokens = [x.replace('Ġ', '') for x in tokens]
            pred_ind = predict_labels[i]
            actual_ind = labels[i]
            predict_prob = predict_probs[i][pred_ind]
            add_attributions_to_visualizer(attributions_ig[i].unsqueeze(0), tokens, predict_prob, pred_ind,
                                           actual_ind, delta[i], self.vis_data_records_ig)

        if not vis_record_save_path:
            vis_record_save_path = '../result/temp.pkl'
        pickle.dump(self.vis_data_records_ig, open(vis_record_save_path, 'wb'))
        print(f"Save vis record to {vis_record_save_path}, size: {len(self.vis_data_records_ig)}")
