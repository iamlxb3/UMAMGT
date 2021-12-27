import os
import ipdb
import torch
import pickle
import pandas as pd
import ntpath
import numpy as np

from tqdm import tqdm
from captum.attr import IntegratedGradients
from captum.attr import visualization
from captum.attr import LayerIntegratedGradients, TokenReferenceBase


def compute_attributions(attributions, tokens):
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
                 save_base_name='',
                 correct_label_only=True
                 ):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.ig = IntegratedGradients(model)
        self.vis_data_records_ig = []
        self.interpret_df = {'token': [], 'label1_attr_score': [], 'sen_len': [],
                             'predict_score': [], 'label': [], 'delta': []}
        self.model.eval()
        self.model.zero_grad()
        self.n_steps = n_steps
        if save_base_name:
            self.vis_record_save_path = save_base_name + '_vis_record.pkl'
            self.interpret_df_save_path = save_base_name + '_token_attr.csv'
        else:
            self.vis_record_save_path = ''
            self.interpret_df_save_path = ''

        self.correct_label_only = correct_label_only

    def interpret_dataloder(self,
                            dataloder):
        for encoded_inputs in tqdm(dataloder, total=len(dataloder)):
            labels = encoded_inputs['labels']
            self.interpret_encoded_inputs(encoded_inputs, labels)
        self.save_vis_records()
        self.save_interpret_df()

    def save_interpret_df(self):
        if self.interpret_df_save_path:
            interpret_df = pd.DataFrame(self.interpret_df)
            interpret_df.to_csv(self.interpret_df_save_path, index=False)
            print(f"Save Token attr to {self.interpret_df_save_path}")
        else:
            print(f"Warning!!!!!!! Token attr df not saved! Path not provided.")

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

        label1_attributions_ig, label1_delta = self.ig.attribute(inputs=input_embedding,
                                                                 baselines=all_pad_embedding,
                                                                 target=all_1_labels,
                                                                 n_steps=self.n_steps,
                                                                 return_convergence_delta=True)
        label1_attributions_ig = label1_attributions_ig.detach().cpu()
        label1_delta = label1_delta.detach().cpu()
        del input_embedding
        del all_pad_embedding

        for i, input_id in enumerate(encoded_inputs['input_ids']):
            tokens = self.tokenizer.convert_ids_to_tokens(input_id, skip_special_tokens=False)
            tokens = [x.replace('Ġ', '') for x in tokens]
            tokens = np.array(tokens)
            ignore_indices = list(np.where(tokens == '[CLS]')[0]) + list(np.where(tokens == '[SEP]')[0]) \
                             + list(np.where(tokens == '[PAD]')[0])
            keep_mask = np.ones_like(tokens, dtype=bool)
            keep_mask[ignore_indices] = False
            tokens = tokens[keep_mask]
            label1_attribution = label1_attributions_ig[i][keep_mask]

            pred_ind = predict_labels[i]
            actual_ind = int(labels[i])

            if self.correct_label_only:
                if pred_ind == actual_ind:
                    predict_prob = predict_probs[i][pred_ind]
                    # label1_attributions_ig[i]: len(tokens) x 768

                    # 这里的attribution是做过normalize，但是，不是做的softmax，所以sum加起来不是1
                    # TODO: 需要思考下这里的attribution score和长度有没有关系
                    label1_attribution_sum = compute_attributions(label1_attribution, tokens)
                    delta = label1_delta
                    assert len(label1_attribution_sum) == len(tokens)
                    for attr_score, token in zip(label1_attribution_sum, tokens):
                        self.interpret_df['token'].append(token)
                        self.interpret_df['label1_attr_score'].append(float(attr_score))
                        self.interpret_df['sen_len'].append(len(tokens))
                        self.interpret_df['predict_score'].append(float(predict_prob))
                        self.interpret_df['label'].append(actual_ind)
                        self.interpret_df['delta'].append(float(delta))

                    # attributions: seq_len x 1

                    # storing couple samples in an array for visualization purposes
                    self.vis_data_records_ig.append(visualization.VisualizationDataRecord(
                        label1_attribution_sum,
                        predict_prob,
                        pred_ind,
                        actual_ind,
                        "label",
                        label1_attribution_sum.sum(),
                        tokens,
                        delta))

        torch.cuda.empty_cache()

    def interpret_sentence(self,
                           sentences,
                           labels):
        encoded_inputs = self.tokenizer(sentences, padding=True, return_tensors="pt")
        self.interpret_encoded_inputs(encoded_inputs, labels)
        self.save_vis_records()
