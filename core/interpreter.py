import os
import ipdb
import torch
import pickle
import numpy as np

from tqdm import tqdm
from captum.attr import IntegratedGradients
from captum.attr import visualization
from captum.attr import LayerIntegratedGradients, TokenReferenceBase


def compute_attributions(attributions, tokens):
    attributions = attributions.squeeze(0)
    attributions = attributions[1:-1][:len(tokens)]
    attributions = attributions.sum(dim=1)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.numpy()
    return attributions


class StoryInterpreter:
    def __init__(self,
                 model,
                 tokenizer,
                 device,
                 n_steps=50,
                 vis_record_save_path='',
                 correct_label_only=True,
                 interpret_all_labels=True
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
        self.interpret_all_labels = interpret_all_labels

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
        self.model.zero_grad()
        predict_probs, predict_labels, input_embedding, all_pad_embedding = \
            self._predict_batch_token_input(encoded_inputs)

        # compute attributions and approximation delta using integrated gradients
        # attributions_ig shape: batch x max_seq_len x 768
        # print(f"Start computing attribution...")
        all_1_labels = torch.tensor([1 for _ in predict_labels]).to(self.device)
        # all_0_labels = torch.tensor([0 for _ in predict_labels]).to(self.device)

        label1_attributions_ig, label1_delta = self.ig.attribute(inputs=input_embedding,
                                                                 baselines=all_pad_embedding,
                                                                 target=all_1_labels,
                                                                 n_steps=self.n_steps,
                                                                 return_convergence_delta=True)
        label1_attributions_ig = label1_attributions_ig.detach().cpu()
        label1_delta = label1_delta.detach().cpu()
        del input_embedding
        del all_pad_embedding

        # if self.interpret_all_labels:
        #     self.model.zero_grad()
        #     predict_probs, predict_labels, input_embedding, all_pad_embedding = \
        #         self._predict_batch_token_input(encoded_inputs)
        #     label0_attributions_ig, label0_delta = self.ig.attribute(inputs=input_embedding,
        #                                                              baselines=all_pad_embedding,
        #                                                              target=all_0_labels,
        #                                                              n_steps=self.n_steps,
        #                                                              return_convergence_delta=True)
        #     label0_attributions_ig = label0_attributions_ig.detach().cpu()
        #     label0_delta = label0_delta.detach().cpu()
        #     del input_embedding
        #     del all_pad_embedding

        # print("-" * 78)
        # print(f"Predict prob: {predict_probs}")
        # print(f"Predict label: {predict_labels}")
        # print(f"Delta: {delta}")

        for i, input_id in enumerate(encoded_inputs['input_ids']):
            tokens = self.tokenizer.convert_ids_to_tokens(input_id, skip_special_tokens=True)
            tokens = [x.replace('Ġ', '') for x in tokens]
            pred_ind = predict_labels[i]
            actual_ind = int(labels[i])

            if self.correct_label_only:
                if pred_ind == actual_ind:
                    predict_prob = predict_probs[i][pred_ind]

                    label1_attributions = compute_attributions(label1_attributions_ig[i].unsqueeze(0), tokens)

                    # if self.interpret_all_labels:
                    #     label0_attributions = compute_attributions(label0_attributions_ig[i].unsqueeze(0), tokens)
                    #     ipdb.set_trace()
                    #     label0_attributions = label0_attributions * -1
                    #     attributions = (label1_attributions + label0_attributions) / 2
                    #     delta = label1_delta

                    attributions = label1_attributions
                    delta = label1_delta

                    # storing couple samples in an array for visualization purposes
                    self.vis_data_records_ig.append(visualization.VisualizationDataRecord(
                        attributions,
                        predict_prob,
                        pred_ind,
                        actual_ind,
                        "label",
                        attributions.sum(),
                        tokens[:len(attributions)],
                        delta))

        torch.cuda.empty_cache()

    def interpret_sentence(self,
                           sentences,
                           labels):
        encoded_inputs = self.tokenizer(sentences, padding=True, return_tensors="pt")
        self.interpret_encoded_inputs(encoded_inputs, labels)
        self.save_vis_records()
