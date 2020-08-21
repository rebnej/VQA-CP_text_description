import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from v_classifier import SimpleClassifier
from transformers import *
#from counting import Counter



class BERTVQAModel(nn.Module):
    def __init__(self, dataset, bert, classifier, freeze_bert = True):
        super(BERTVQAModel, self).__init__()
        self.dataset = dataset
        self.bert = bert
        
        if freeze_bert:
            for name, param in self.bert.named_parameters():
                param.requires_grad = False
                
                if 'encoder.layer.11' in name or 'pooler.dence' in name:
                    param.requires_grad = True
                
        for name, param in self.bert.named_parameters():
            print(name, param.requires_grad)
                
        self.classifier = classifier
        #self.counter = counter
        #self.drop = nn.Dropout(.5)
        #self.tanh = nn.Tanh()
        
    


    def forward(self, input_ids, input_mask, input_token_type):
        """Forward

        return: logits, not probs
        """
        #print('model forward satart!!!')
        #last_hidden_states = self.bert(input_ids, attention_mask=input_mask, token_type_ids=input_token_type)[0] #(batch_size, 64, 768)
        all_hidden_states, cls_hidden_state, attention = self.bert(input_ids, attention_mask=input_mask, token_type_ids=input_token_type, output_attentions=True)
        #batch_size = last_hidden_states.size(0)
        batch_size = cls_hidden_state.size(0)
        
        #print('last_hidden_states.shape: ', last_hidden_states.shape)
        #print('batch_size: ', batch_size)
        
        #last_hidden_states = last_hidden_states.view(batch_size, -1)
        cls_hidden_state = cls_hidden_state.view(batch_size, -1)
        
        #logits = self.classifier(last_hidden_states)
        logits = self.classifier(cls_hidden_state)

        return logits, attention



def build_bertvqa(dataset):

    bert = BertModel.from_pretrained('bert-base-uncased')

    #in_dim = 64 * 768
    in_dim = 768  #[cls]のみ使うといい感じの次元にはなる
    #hid_dim = 64 #!!!適切な値がわからん
    hid_dim = 768 * 2 #[cls]のみ使うといい感じの次元にはなる
    out_dim = dataset.num_ans_candidates

    classifier = SimpleClassifier(
        in_dim, hid_dim, out_dim, .5)
    #counter = Counter(objects)
    return BERTVQAModel(dataset, bert, classifier)
