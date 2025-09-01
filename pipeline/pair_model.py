import torch
import torch.nn as nn
from transformers import BertModel

class PairClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", hidden_size=768, num_classes=2, dropout=0.3):
        super(PairClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        return self.fc(x)
