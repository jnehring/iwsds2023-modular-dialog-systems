import torch.nn as nn
import transformers
import torch

from transformers import AutoModel


class BERT_BaseArch_Joint(nn.Module):

  def __init__(self, module_labels):
    super(BERT_BaseArch_Joint, self).__init__()
    
    # initialize BERT model
    self.bert = AutoModel.from_pretrained('bert-base-uncased')
    
    # dropout layer
    self.dropout = nn.Dropout(0.1)
    
    # dense layer 1
    self.fc_bert = nn.Linear(768,len(module_labels))
    
    # output layer
    self.fc_final = nn.Linear(2*len(module_labels),len(module_labels))
  

  # define the forward pass
  def forward(self, sent_id, mask, confidence):
    # pass the inputs to the model
    _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)

    # dropout layer
    x = self.dropout(cls_hs)

    # dense layer 1
    x = self.fc_bert(x)

    # concatenate bert output with normalized confidence
    x = torch.cat((x, confidence), 1)

    # output layer
    x = self.fc_final(x)

    return x