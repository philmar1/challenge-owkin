from sklearn.linear_model import LogisticRegression

import torch
from torch import nn

class LogisticRegression(torch.nn.Module):
    def __init__(self, n=512, n_out=1):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(n, n_out)
        self.scoring = torch.nn.Softmax() if n_out>1 else torch.nn.Sigmoid()

    def forward(self, x):
        z = self.linear(x)
        y_pred = self.scoring(z)
        return y_pred
    
    
class NN_classifier(torch.nn.Module):

    def __init__(self, input_dim = 512,
                 hidden_layers_size = [1024],
                 dropout=0.
                ):
        super(NN_classifier, self).__init__()
        self.linear_stack = nn.ModuleList()
        hidden_layers_size = [input_dim] + hidden_layers_size
        for k in range(len(hidden_layers_size)-1):
            self.linear_stack.append(nn.Linear(hidden_layers_size[k], hidden_layers_size[k+1]))
            self.linear_stack.append(nn.LeakyReLU())
            if dropout > 0:
                self.linear_stack.append(nn.Dropout(dropout))
    
    def forward(self, x):   
        # Feedforward
        for layer in self.linear_stack:
            x = layer(x)
        output = torch.nn.Sigmoid()
        return output
                
                
    
class AttentionSoftMax(torch.nn.Module):
    def __init__(self, in_features = 3, out_features = None):
        """
        given a tensor `x` with dimensions [N * M],
        where M -- dimensionality of the featur vector
                   (number of features per instance)
              N -- number of instances
        initialize with `AggModule(M)`
        returns:
        - weighted result: [M]
        - gate: [N]
        """
        super(AttentionSoftMax, self).__init__()
        if out_features is None:
            out_features = in_features
        self.linear_keys = nn.Linear(in_features, out_features)
        self.linear_values = nn.Linear(in_features, out_features)
        self.activation = nn.LeakyReLU()
        self.layer_linear_query = nn.Linear(out_features, 1)
        
    def forward(self, x):
        keys = self.linear_keys(x)
        keys = self.activation(keys)
        values = self.linear_values(x)
        values = self.activation(values)
        attention_map_raw = self.layer_linear_query(keys)[...,0]
        attention_map = nn.Softmax(dim=-1)(attention_map_raw)
        result = torch.einsum(f'ki,kij->kj', attention_map, values) # torch.einsum(f'ki,kij->kj', attention_map, x)
        return result, attention_map
    
    
class MIL_NN(torch.nn.Module):
    def __init__(self,
                 agg = None,
                 classifier = None,
                ):
        super(MIL_NN, self).__init__()
        self.agg = agg
        self.classifier = classifier
        

    def forward(self, bag_features):
        bag_feature_agg, att_map = self.agg(bag_features)
        y_pred = self.classifier(bag_feature_agg)
        return y_pred
        