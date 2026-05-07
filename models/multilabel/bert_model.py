import requests
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel, BertConfig, BertForSequenceClassification
from config import opt

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class BertEmbedding(nn.Module):
    
    def __init__(self, num_labels=None):
        super(BertEmbedding, self).__init__()
        # self.bert = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # mesin pertama untuk input pasien
        self.bert = AutoModel.from_pretrained(opt.MODEL_NAME, output_hidden_states=True, output_attentions=True, ignore_mismatched_sizes=True)
        # mesin kedua untuk memahami intent
        self.bertlabelencoder = AutoModel.from_pretrained(opt.MODEL_NAME, ignore_mismatched_sizes=True)
        # self.dropout = nn.Dropout(0.1)
        # self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, utterance_ids, utterance_mask, label_ids, Label_mask):
        # membaca inten
        output_label = self.bertlabelencoder(
            input_ids=label_ids,
            attention_mask=Label_mask,
            return_dict=True
        )
        clusters = output_label.pooler_output

        # membaca pertanyaan
        output_utterance = self.bert(
            input_ids=utterance_ids,
            attention_mask=utterance_mask,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True
        )

        # ekstrak output yang diperlukan
        pooled_output = output_utterance.pooler_output

        gram = torch.mm(clusters, clusters.permute(1,0)) # (n, n)
        weight = torch.mm(pooled_output, clusters.permute(1,0))

        logits = torch.mm(weight, torch.inverse(gram)) * np.sqrt(768)

        return logits
        # loss = self.bert(input_ids, attention_mask=mask, labels=labels)
        # return loss
