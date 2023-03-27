import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
    
import logging 

logger = logging.getLogger(__name__)

class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class, instance_classifier = None):
        super(IClassifier, self).__init__()
        
        self.feature_extractor = feature_extractor      
        self.instance_classifier = instance_classifier
        if instance_classifier == None:
            self.instance_classifier = nn.Linear(feature_size, output_class)
        
        
    def forward(self, x):
        feats = self.feature_extractor(x) # N x K
        c = self.instance_classifier(feats) # N x C
        return feats, c
    

class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, embed_dim, dropout_v=0.0, nonlinear=True, passing_v=False): # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, embed_dim), 
                                   nn.LeakyReLU(), 
                                   nn.Linear(embed_dim, embed_dim), 
                                   nn.Tanh())
        else:
            self.q = nn.Linear(input_size, embed_dim)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.LeakyReLU()
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
 
class MultiheadAttentionHook(nn.Module):
    def __init__(self, input_dim, num_heads, batch_first = True) -> None:
        super(MultiheadAttentionHook, self).__init__()  
        self.multi_head_att = nn.MultiheadAttention(embed_dim=input_dim,
                                                  num_heads=num_heads,
                                                  batch_first=batch_first) 
    
    def forward(self, x):
        outputs, attn_weights = self.multi_head_att.forward(x, x, x)
        return outputs
    
class DualStreamNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(DualStreamNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier
        self.name = 'DualStreamNet'
        
    def forward(self, x):
        feats, classes = self.i_classifier(x)
        prediction_bag, A, B = self.b_classifier(feats, classes)
        
        return classes, prediction_bag, A, B
     
def get_model(input_dim = 2048,
              embed_dim = 512,
              dropout = 0.0, 
              passing_v = False,
              transformers_first = False,
              instance_classifier = None
              ):
    
    if transformers_first:
        feature_extractor = MultiheadAttentionHook(input_dim=input_dim,
                                                  num_heads=4,
                                                  batch_first=True)
    else:
        feature_extractor = nn.Identity()
        
    i_classifier = IClassifier(feature_extractor=feature_extractor, feature_size=input_dim, output_class=1, instance_classifier=instance_classifier)
    b_classifier = BClassifier(input_size=input_dim, output_class=1, embed_dim=embed_dim, dropout_v=dropout, passing_v=passing_v)
    model = DualStreamNet(i_classifier, b_classifier)
    logger.info("Creating model: \n {} ".format(model))
    return model