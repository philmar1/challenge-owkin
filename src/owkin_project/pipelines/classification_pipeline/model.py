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
        self.linear_stack.append(nn.Linear(hidden_layers_size[-1], 1))
    
    def forward(self, x):   
        # Feedforward
        for layer in self.linear_stack:
            x = layer(x)
        output = torch.nn.Sigmoid()(x)
        return output
                
                
class SimpleAttentionBlock(torch.nn.Module):
    def __init__(self, input_dim = 3, embed_dim = None):
        super(SimpleAttentionBlock, self).__init__()
        self.att_block = nn.ModuleList()
        if embed_dim is None:
            embed_dim = input_dim
        self.linear = nn.Linear(input_dim, embed_dim)
        self.activation = nn.Tanh()
        self.linear_last = nn.Linear(embed_dim, 1)
        
    def forward(self, x):
        scores = self.linear(x)
        scores = self.activation(scores)
        scores = self.linear_last(scores)
        scores = nn.Softmax(dim=-1)
        return self.att_block(x)

class GatedAttentionBlock(torch.nn.Module):
    def __init__(self, input_dim = 3, embed_dim = None):
        super(GatedAttentionBlock, self).__init__()
        if embed_dim is None:
            embed_dim = input_dim
        self.linear_left = nn.Linear(input_dim, embed_dim)
        self.linear_right = nn.Linear(input_dim, embed_dim)
        self.activation_left = nn.Tanh()
        self.activation_right = nn.Sigmoid()
        self.linear_last = nn.Linear(embed_dim, 1)
        
    def forward(self, x):
        right = self.linear_right(x)
        right = self.activation_right(right)
        left = self.linear_left(x)   
        left = self.activation_left(left)
        scores = right * left
        scores = self.linear_last(scores)
        attention_map = nn.Softmax(dim=-1)(scores)
        
        return attention_map   
                              
class AttentionModule(torch.nn.Module):
    def __init__(self, input_dim = 3, 
                 embed_dim = None,
                 att = 'simple'):
        """
        given a tensor `x` with dimensions [N * M],
        where M -- dimensionality of the featur vector
                   (number of features per instance)
              N -- number of instances
        initialize with `AggModule(M)`
        returns:
        - weighted result: [M]
        - attention_map: [N]
        """
        super(AttentionModule, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        if att == 'simple':
            self.att_block = SimpleAttentionBlock(input_dim, embed_dim) 
        if att == 'gated':
            self.att_block = GatedAttentionBlock(input_dim, embed_dim) 
        
    def forward(self, x):
        attention_map = self.att_block(x)
        attention_map = attention_map.reshape(attention_map.shape[0], attention_map.shape[1])
        result = torch.einsum(f'ki,kij->kj', attention_map, x) # torch.einsum(f'ki,kij->kj', attention_map, x)
        return result, attention_map
    

class MIL_NN(torch.nn.Module):
    def __init__(self,
                 aggregator = None,
                 classifier = None,
                 transformers_first = False
                ):
        super(MIL_NN, self).__init__()
        self.transformers_first = transformers_first
        if self.transformers_first:
            self.multihead_att = nn.MultiheadAttention(embed_dim=aggregator.input_dim,
                                                  num_heads=8,
                                                  batch_first=True)
        self.aggregator = aggregator
        self.classifier = classifier
        
    def forward(self, bag_features):
        if self.transformers_first:
            bag_features, _ = self.multihead_att(bag_features, bag_features, bag_features)
        bag_feature_agg, _ = self.aggregator(bag_features)
        y_pred = self.classifier(bag_feature_agg)
        return y_pred
        