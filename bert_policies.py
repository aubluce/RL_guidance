# a bert model for both the actor and the critic model.
# number of output categories for the actor model is the length of the action space
# number of output categories for the critic network is just 1 (basically a value function)
# state space is just the tokenized text.

from torch import nn
from transformers import BertModel


class BERTPolicy(nn.Module):

    def __init__(self, dropout=0.5, num_classes=40):

        super(BERTPolicy, self).__init__() #originally said BertClassifier

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, num_classes)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer