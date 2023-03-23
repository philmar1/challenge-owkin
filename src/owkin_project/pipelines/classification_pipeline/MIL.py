import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()
        
        self.feature_extractor = feature_extractor      
        self.fc = nn.Linear(feature_size, output_class)
        
        
    def forward(self, x):
        device = x.device
        feats = self.feature_extractor(x) # N x K
        c = self.fc(feats) # N x C
        return feats, c
    

class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=False): # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()
        
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)  
        
    def forward(self, feats, c): # N x K, N x C
        device = feats.device
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats) # N x Q, unsorted
        
         # handle multiple classes without for loop
        batch_size = feats.shape[0]
        _, m_indices = torch.sort(c, 1, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.torch.index_select(feats[0], dim=0, index=m_indices[0, 1, :]) # select critical instances, m_feats in shape C x K 
        m_feats = m_feats.unsqueeze(0) # add batch_axis
        for batch in range(1, batch_size):
            m_feats_i = torch.torch.index_select(feats[batch], dim=0, index=m_indices[batch, 1, :])
            m_feats_i = m_feats_i.unsqueeze(0)
            m_feats = torch.cat((m_feats, m_feats_i))
            
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.matmul(Q, q_max.transpose(1, 2)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[-1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
        
        B = torch.matmul(A.transpose(1, 2), V) # compute bag representation, B in shape C x V
    
        C = self.fcc(B) # batch x C x V -> batch x C x 1
        C = C.view(batch_size, C.shape[1]) # batch x C
        
        return C, A, B 
    
    
class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier
        
    def forward(self, x):
        feats, classes = self.i_classifier(x)
        prediction_bag, A, B = self.b_classifier(feats, classes)
        
        return classes, prediction_bag, A, B