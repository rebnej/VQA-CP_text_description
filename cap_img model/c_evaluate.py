"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from c_dataset import VQABERTDataset
import con_model
import model
import v_model
import utils

import _pickle as cPickle
import logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_ans', action='store_false')
    parser.add_argument('--model', type=str, default='bertvqa')
    parser.add_argument('--input', type=str, default='saved_models/con_bertvqa')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    print('Evaluate a given model optimized by training split using validation split.')
    args = parse_args()

    torch.backends.cudnn.benchmark = True
    
    logging.basicConfig(level=logging.ERROR)

    
    from c_train import evaluate
    
    eval_dset = VQABERTDataset('val', args.use_ans)
    

    n_device = torch.cuda.device_count()
    batch_size = args.batch_size * n_device
    
    #v_model
    constructor = 'build_%s' % 'bertvqa'
    v_model = getattr(v_model, constructor)(eval_dset).cuda()

    v_model_path = 'saved_models/v_bertvqa'+'/model%s.pth' % \
        ('' if 0 > 19 else '_epoch%d' % 14)
    
    print('loading %s' % v_model_path)
    v_model_data = torch.load(v_model_path)

    v_model = nn.DataParallel(v_model).cuda()
    v_model.load_state_dict(v_model_data.get('model_state', v_model_data))
    
    
    #c_model (model)
    constructor = 'build_%s' % 'bertvqa'
    model = getattr(model, constructor)(eval_dset).cuda()

    model_path = 'saved_models/bertvqa'+'/model%s.pth' % \
        ('' if 0 > 12 else '_epoch%d' % 12)
    
    print('loading %s' % model_path)
    model_data = torch.load(model_path)

    model = nn.DataParallel(model).cuda()
    model.load_state_dict(model_data.get('model_state', model_data))
    
    #con_model
    constructor = 'build_%s' % args.model
    con_model = getattr(con_model, constructor)(eval_dset, model, v_model).cuda()
    con_model_data = torch.load(args.input+'/model'+('_epoch%d' % 19 if 0 < 19 else '')+'.pth')

    con_model = nn.DataParallel(con_model).cuda()
    con_model.load_state_dict(con_model_data.get('model_state', con_model_data))
    

    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=False, num_workers=1)
    con_model.train(False)

    
    eval_score, bound, entropy = evaluate(con_model, eval_loader)
    
    print('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
    


