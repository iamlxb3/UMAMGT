import ipdb
import torch
import torch.nn as nn


def compute_bert_outputs(model_bert, embedding_output, attention_mask):
    encoder_outputs = model_bert.encoder(embedding_output,
                                         encoder_attention_mask=attention_mask,
                                         output_hidden_states=True)
    sequence_output = encoder_outputs[0]
    # 这里和roberta的classification head是冲突的, classification head里面已经包含pooler了
    # pooled_output = model_bert.pooler(sequence_output)
    outputs = (sequence_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
    return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class BertModelWrapper(nn.Module):

    def __init__(self, model, model_name='roberta_cn_wmm'):
        super(BertModelWrapper, self).__init__()
        self.model = model
        self.model.eval()
        self.model_name = model_name
        self.model_body = list(model.children())[0]

        if model_name == 'roberta_cn_wmm':
            self.classifier_head = nn.Sequential(*list(model.children())[1:])
        else:
            self.classifier_head = list(model.children())[1]

    def forward(self, embeddings, attention_mask=None):
        outputs = compute_bert_outputs(self.model_body, embeddings, attention_mask)
        encoder_output = outputs[0]
        # pooled_output = self.model.dropout(pooled_output)
        if self.model_name == 'roberta_cn_wmm':
            logits = self.classifier_head(encoder_output)[:, 0, :]
        else:
            logits = self.classifier_head(encoder_output)

        return torch.softmax(logits, dim=1)
