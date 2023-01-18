import torch.nn as nn


class BERT_BaseArch_Confidence(nn.Module):

  def __init__(self, module_labels):
    super(BERT_BaseArch_Confidence, self).__init__()
    # softmax activation function
    self.softmax = nn.Softmax(dim = 1)

    self.fc_final = nn.Linear(len(module_labels),len(module_labels))

  # define the forward pass
  def forward(self, confidence):
    # sofrmax layer
    x = self.fc_final(confidence)
    
    return x