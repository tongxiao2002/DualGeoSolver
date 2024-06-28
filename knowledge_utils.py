import torch
from torch.nn.utils.rnn import pad_sequence


def select_explanations_by_prediction(
    logits: torch.FloatTensor,
    knowledge_explanation_input_ids: torch.LongTensor,
    knowledge_explanation_attention_mask: torch.LongTensor,
    max_length: int = 256,
    threshold: float = 0.5,
):
    """select curr_knowledge_description_input_ids & attention_mask
    according to the prediction of `prediction_head`
    """
    knowledge_indices = _get_predicted_knowledge_index(logits, threshold=threshold)
    curr_knowledge_description_input_ids, curr_knowledge_description_attention_mask = [], []
    curr_indices = []
    for index in knowledge_indices:
        input_ids = knowledge_explanation_input_ids.index_select(0, index)
        attention_mask = knowledge_explanation_attention_mask.index_select(0, index)
        curr_input_ids, curr_attention_mask, indices = _merge_multiple_input_ids_and_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            indices=index,
        )
        curr_knowledge_description_input_ids.append(curr_input_ids)
        curr_knowledge_description_attention_mask.append(curr_attention_mask)
        curr_indices.append(indices)
    curr_knowledge_description_input_ids = pad_sequence(
        sequences=curr_knowledge_description_input_ids,
        batch_first=True,
        padding_value=0,    # pad token of BertTokenizer is 0
    )
    curr_knowledge_description_attention_mask = pad_sequence(
        sequences=curr_knowledge_description_attention_mask,
        batch_first=True,
        padding_value=0,
    )
    # because padded position's attention mask is zero,
    # we can set padding_value to any value between [0, num_labels - 1]
    curr_indices = pad_sequence(
        sequences=curr_indices,
        batch_first=True,
        padding_value=0,
    )
    max_length = torch.max(torch.sum(curr_knowledge_description_attention_mask, dim=-1))
    curr_knowledge_description_input_ids = curr_knowledge_description_input_ids[:, :max_length]
    curr_knowledge_description_attention_mask = curr_knowledge_description_attention_mask[:, :max_length]
    curr_indices = curr_indices[:, :max_length]
    return curr_knowledge_description_input_ids, curr_knowledge_description_attention_mask, curr_indices


def _get_predicted_knowledge_index(logits, threshold: float = 0.5):
    """extract predicted knowledge labels from logits.
    Only return labels whose logits is larger than `threshold`,
    if none of labels whose logits is larger than `threshold`, then select label with maximum logit value.
    """
    knowledge_mask = torch.sigmoid(logits) >= threshold
    maximum = torch.max(logits, dim=-1)[0]
    maximum_mask = logits >= maximum[:, None]
    knowledge_mask = torch.logical_or(knowledge_mask, maximum_mask).nonzero()
    # group by batch
    group_indices = knowledge_mask[:, 0].unique()
    knowledge_indices = [
        knowledge_mask[knowledge_mask[:, 0] == idx][:, 1] for idx in group_indices
    ]
    return knowledge_indices


def _merge_multiple_input_ids_and_mask(input_ids, attention_mask, max_length: int, indices):
    """merge input_ids & attention_mask of multiple knowledge descriptions
    how to merge: remove <eos> token of descriptions except last one,
    and then simply concatenate.
    """
    assert input_ids.shape[0] == attention_mask.shape[0], "shape of input_ids and attention_mask mismatch."
    if input_ids.shape[0] == 1:
        length = attention_mask.shape[-1]
        return input_ids.squeeze(0), attention_mask.squeeze(0), indices.repeat_interleave(length, 0)
    concatted_input_ids, concatted_attention_mask = [], []
    concatted_indices = []
    # process input_ids & mask of all batch except last one in the for loop
    for idx, (ids, mask, index) in enumerate(zip(input_ids[:-1], attention_mask[:-1], indices[:-1])):
        # get rid of [CLS] token, [SEP] token and padding tokens
        start_idx = 0 if idx == 0 else 1
        num_ids = torch.sum(mask).item()
        ids = ids[start_idx:num_ids - 1]
        mask = mask[start_idx:num_ids - 1]
        length = max(num_ids - 1 - start_idx, 0)
        concatted_input_ids.append(ids)
        concatted_attention_mask.append(mask)
        concatted_indices.append(index.unsqueeze(0).repeat_interleave(length, dim=0))
    concatted_input_ids.append(input_ids[-1][1:])
    concatted_attention_mask.append(attention_mask[-1][1:])
    concatted_indices.append(indices[-1].unsqueeze(0).repeat_interleave(input_ids[-1][1:].shape[-1], dim=0))
    concatted_input_ids = torch.cat(concatted_input_ids, dim=0)
    concatted_attention_mask = torch.cat(concatted_attention_mask, dim=0)
    concatted_indices = torch.cat(concatted_indices, dim=0)
    concatted_input_ids = concatted_input_ids[:max_length]
    concatted_attention_mask = concatted_attention_mask[:max_length]
    concatted_indices = concatted_indices[:max_length]
    length = torch.sum(concatted_attention_mask).item()
    # 102 is '[SEP]' in bert tokenizer
    concatted_input_ids[length - 1] = 102
    concatted_input_ids = concatted_input_ids[:length]
    concatted_attention_mask = concatted_attention_mask[:length]
    concatted_indices = concatted_indices[:length]
    assert torch.all(concatted_attention_mask.bool()).item()
    return concatted_input_ids, concatted_attention_mask, concatted_indices
