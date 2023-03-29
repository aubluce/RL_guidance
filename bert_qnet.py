import torch
from tqdm.notebook import tqdm

from transformers import BertTokenizer
from torch.utils.data import TensorDataset

from transformers import BertForSequenceClassification

labels_list = [str(x) for x in range(40)]
label_dict = {}
for index, label in enumerate(labels_list):
    label_dict[label] = index

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                          do_lower_case=True)


def encode_batch(df, tokenizer):
    encoded_data = tokenizer.batch_encode_plus(
                        df['text'],
                        add_special_tokens=True,
                        return_attention_mask=True,
                        pad_to_max_length=True,
                        max_length=256,
                        return_tensors='pt'
                        )
    return encoded_data

encoded_data = encode_batch(df, tokenizer)
input_ids = encoded_data['input_ids']
attention_masks = encoded_data['attention_mask']
labels_