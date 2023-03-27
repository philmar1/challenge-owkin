import torch
from torch import nn

import logging

logger = logging.getLogger(__name__)

class NN_classifier(torch.nn.Module):

    def __init__(self, input_dim = 512,
                 hidden_layers_size = [1024],
                 dropout = 0.,
                 end_activation = True
                ):
        super(NN_classifier, self).__init__()
        self.end_activation = end_activation
        self.linear_stack = nn.ModuleList()
        hidden_layers_size = [input_dim] + hidden_layers_size
        for k in range(len(hidden_layers_size)-1):
            self.linear_stack.append(nn.Linear(hidden_layers_size[k], hidden_layers_size[k+1]))
            self.linear_stack.append(nn.LeakyReLU())
            if dropout > 0:
                self.linear_stack.append(nn.Dropout(dropout))
        self.linear_stack.append(nn.Linear(hidden_layers_size[-1], 1))
    
    def forward(self, x):   
        # Feedforward
        for layer in self.linear_stack:
            x = layer(x)
        if self.end_activation:
            x = torch.nn.Sigmoid()(x)
        return x                
                
def get_instance_classifier(input_dim: int, 
                            hidden_layers_size: list,
                            dropout: 0.,
                            end_activation: bool = True):
    model = NN_classifier(input_dim, hidden_layers_size, dropout, end_activation)
    logger.info("Creating Instance Classifier: \n {} ".format(model))
    return model