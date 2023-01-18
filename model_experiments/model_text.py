import torch.nn as nn
import transformers

from transformers import AutoModel


class BERT_BaseArch_Text(nn.Module):

  def __init__(self, module_labels):
    super(BERT_BaseArch_Text, self).__init__()
    
    # initialize BERT model
    self.bert = AutoModel.from_pretrained('bert-base-uncased')
    
    # dropout layer
    self.dropout = nn.Dropout(0.1)
     
    # output layer
    self.fc_final = nn.Linear(768, len(module_labels))
  
  
  # define the forward pass
  def forward(self, sent_id, mask):
    # pass the inputs to the model
    _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)

    # dropout layer
    x = self.dropout(cls_hs)
    
    # output layer
    x = self.fc_final(x)

    return x