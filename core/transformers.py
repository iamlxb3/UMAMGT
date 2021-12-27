import ipdb
import copy
import torch
import torch.nn as nn


def get_extended_attention_mask(attention_mask, input_shape, model_dtype):
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (:obj:`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (:obj:`Tuple[int]`):
            The shape of the input to the model.
        device: (:obj:`torch.device`):
            The device of the input to the model.

    Returns:
        :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
    """
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=model_dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


def compute_bert_outputs(model_bert, embedding_output, attention_mask, model_name):
    input_shape = None

    if attention_mask is not None:
        extended_attention_mask = get_extended_attention_mask(attention_mask, input_shape, model_bert.dtype)
    else:
        extended_attention_mask = None

    encoder_outputs = model_bert.encoder(embedding_output,
                                         attention_mask=extended_attention_mask,
                                         output_hidden_states=True)
    sequence_output = encoder_outputs[0]

    if model_name == 'bert':
        # 这里和roberta的classification head是冲突的, classification head里面已经包含pooler了
        pooled_output = model_bert.pooler(sequence_output)
    else:
        pooled_output = None
    outputs = (sequence_output, pooled_output) + encoder_outputs[
                                                 1:]  # add hidden_states and attentions if they are here
    return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class BertModelWrapper(nn.Module):

    def __init__(self, model, model_type='bert'):
        super(BertModelWrapper, self).__init__()
        self.model = model
        self.model.eval()
        self.model_type = model_type
        self.model_body = list(model.children())[0]

        if model_type == 'bert':
            self.classifier_head = nn.Sequential(*list(model.children())[1:])
        else:
            self.classifier_head = list(model.children())[1]

    def forward(self, embeddings, attention_mask=None, encoded_inputs=None):

        outputs = compute_bert_outputs(self.model_body, embeddings, attention_mask, self.model_type)

        if self.model_type == 'bert':
            encoder_output = outputs[1]
        else:
            encoder_output = outputs[0]

        # encoded_inputs.pop('labels')
        # embedding_input = copy.deepcopy(encoded_inputs)
        # embedding_input.pop('attention_mask')
        # embedding_output = self.model.bert.embeddings(**embedding_input)
        # origin_encoder_output = self.model.bert(**encoded_inputs)[0]
        # origin_pooled_output = self.model.bert(**encoded_inputs)[1]
        #

        logits = self.classifier_head(encoder_output)

        return torch.softmax(logits, dim=1)
