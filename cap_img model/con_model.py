import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from v_classifier import SimpleClassifier
from transformers import *
#from counting import Counter



class BERTVQAModel(nn.Module):
    def __init__(self, dataset, c_model, v_model, classifier):
        super(BERTVQAModel, self).__init__()
        self.dataset = dataset
        self.c_model = c_model
        self.v_model = v_model
                
        self.classifier = classifier
        #self.counter = counter
        #self.drop = nn.Dropout(.5)
        #self.tanh = nn.Tanh()
        
    
    def forward(self, c_input_ids, c_input_mask, c_input_token_type, v_input_ids, v_input_mask, v_input_token_type):
        """Forward

        return: logits, not probs
        """
        
        c_logits, c_attention = self.c_model(c_input_ids, c_input_mask, c_input_token_type)
        v_logits, v_attention = self.v_model(v_input_ids, v_input_mask, v_input_token_type)
        
        logits = c_logits + v_logits
        
        logits = self.classifier(logits)

        return logits, c_attention, v_attention



def build_bertvqa(dataset, c_model, v_model):

    in_dim = dataset.num_ans_candidates  
    hid_dim = in_dim * 2 
    out_dim = dataset.num_ans_candidates

    classifier = SimpleClassifier(
        in_dim, hid_dim, out_dim, .5)
    
    return BERTVQAModel(dataset, c_model, v_model, classifier)
