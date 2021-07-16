import os
import ipdb
import torch
import pickle

from tqdm import tqdm
from captum.attr import IntegratedGradients
from captum.attr import visualization
from captum.attr import LayerIntegratedGradients, TokenReferenceBase


def add_attributions_to_visualizer(attributions, tokens, pred, pred_ind, label, delta, vis_data_records):
    attributions = attributions.squeeze(0)
    attributions = attributions[1:-1][:len(tokens)]
    attributions = attributions.sum(dim=1)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.numpy()

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
    def __init__(self,
                 model,
                 tokenizer,
                 device,
                 n_steps=50,
                 vis_record_save_path='',
                 correct_label_only=True
                 ):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.ig = IntegratedGradients(model)
        self.vis_data_records_ig = []
        self.model.eval()
        self.model.zero_grad()
        self.n_steps = n_steps
        self.vis_record_save_path = vis_record_save_path
        self.correct_label_only = correct_label_only

    def interpret_dataloder(self,
                            dataloder):
        for encoded_inputs in tqdm(dataloder, total=len(dataloder)):
            labels = encoded_inputs['labels']
            self.interpret_encoded_inputs(encoded_inputs, labels)
        self.save_vis_records()

    def _predict_batch_token_input(self, encoded_inputs):
        encoded_inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}
        input_embedding = self.model.model_body.embeddings(input_ids=encoded_inputs['input_ids'])
        all_pad_input_ids = torch.full_like(encoded_inputs['input_ids'], self.tokenizer.pad_token_id)
        all_pad_embedding = self.model.model_body.embeddings(input_ids=all_pad_input_ids)
        predict_probs = self.model(input_embedding,
                                   encoded_inputs['attention_mask'],
                                   encoded_inputs=encoded_inputs)
        predict_labels = torch.argmax(predict_probs, dim=1).detach().cpu().tolist()
        predict_probs = predict_probs.detach().cpu()
        return predict_probs, predict_labels, input_embedding, all_pad_embedding

    def save_vis_records(self):
        if self.vis_record_save_path:
            pickle.dump(self.vis_data_records_ig, open(self.vis_record_save_path, 'wb'))
            print(
                f"Save vis record to {os.path.abspath(self.vis_record_save_path)}, size: {len(self.vis_data_records_ig)}")
        else:
            print(f"Warning!!!!!!! Vis recorad not saved! Path not provided.")

    def interpret_encoded_inputs(self,
                                 encoded_inputs,
                                 labels):
        predict_probs, predict_labels, input_embedding, all_pad_embedding = \
            self._predict_batch_token_input(encoded_inputs)

        # compute attributions and approximation delta using integrated gradients
        # attributions_ig shape: batch x max_seq_len x 768
        print(f"Start computing attribution...")
        attributions_ig, delta = self.ig.attribute(inputs=input_embedding,
                                                   baselines=all_pad_embedding,
                                                   target=[1 for _ in predict_labels],
                                                   n_steps=self.n_steps,
                                                   return_convergence_delta=True)
        print("-" * 78)
        print(f"Predict prob: {predict_probs}")
        print(f"Predict label: {predict_labels}")
        print(f"Delta: {delta}")
        del input_embedding
        del all_pad_embedding
        attributions_ig = attributions_ig.detach().cpu()
        delta = delta.detach().cpu()

        for i, input_id in enumerate(encoded_inputs['input_ids']):
            tokens = self.tokenizer.convert_ids_to_tokens(input_id, skip_special_tokens=True)
            tokens = [x.replace('Ġ', '') for x in tokens]
            pred_ind = predict_labels[i]
            actual_ind = int(labels[i])

            if self.correct_label_only:
                if pred_ind == actual_ind:
                    predict_prob = predict_probs[i][pred_ind]
                    add_attributions_to_visualizer(attributions_ig[i].unsqueeze(0), tokens, predict_prob, pred_ind,
                                                   actual_ind, delta[i], self.vis_data_records_ig)
        torch.cuda.empty_cache()

    def interpret_sentence(self,
                           sentences,
                           labels):
        encoded_inputs = self.tokenizer(sentences, padding=True, return_tensors="pt")
        self.interpret_encoded_inputs(encoded_inputs, labels)
        self.save_vis_records()
