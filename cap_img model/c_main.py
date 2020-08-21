"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np

from c_dataset import VQABERTDataset
import con_model
import model
import v_model
import utils

import logging





def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--task', type=str, default='vqa', help='vqa or flickr')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--use_ans', action='store_false')
    parser.add_argument('--model', type=str, default='bertvqa')
    #parser.add_argument('--op', type=str, default='c')

    parser.add_argument('--use_both', action='store_true', help='use both train/val datasets to train?')
    #parser.add_argument('--use_vg', action='store_true', help='use visual genome dataset to train?')

    #parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default='saved_models/con_bertvqa')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=1204, help='random seed')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    
    logging.basicConfig(level=logging.ERROR)

#ここらへんは結果の再現性のためのコード
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True


    from c_train import train

    train_dset = VQABERTDataset('train', args.use_ans)
    val_dset = VQABERTDataset('val', args.use_ans)

    utils.create_dir(args.output) #'saved_models/bertvqa'ディレクトリを作成
    logger = utils.Logger(os.path.join(args.output, 'args.txt')) #'saved_models/bertvqa/args.txt'にloggerを作成
    logger.write(args.__repr__()) #.__repr__():インスタンスをできるかぎり元の状態に戻した文字列を返す

    batch_size = args.batch_size
    
    #v_model
    constructor = 'build_%s' % 'bertvqa'
    v_model = getattr(v_model, constructor)(train_dset).cuda()

    v_model_path = 'saved_models/v_bertvqa'+'/model%s.pth' % \
        ('' if 0 > 19 else '_epoch%d' % 14)
    
    print('loading %s' % v_model_path)
    v_model_data = torch.load(v_model_path)

    v_model = nn.DataParallel(v_model).cuda()
    v_model.load_state_dict(v_model_data.get('model_state', v_model_data))
    
    
    #c_model (model)
    constructor = 'build_%s' % 'bertvqa'
    model = getattr(model, constructor)(train_dset).cuda()

    model_path = 'saved_models/bertvqa'+'/model%s.pth' % \
        ('' if 0 > 19 else '_epoch%d' % 19)
    
    print('loading %s' % model_path)
    model_data = torch.load(model_path)

    model = nn.DataParallel(model).cuda()
    model.load_state_dict(model_data.get('model_state', model_data))


    constructor = 'build_%s' % args.model #'build_bertvqa'
    #getattr(宣言したインスタンス名, インスタンス内の関数名)(引数1, 引数2, ...) : 関数が実行される
    con_model = getattr(con_model, constructor)(train_dset, model, v_model).cuda()
    con_model = nn.DataParallel(con_model).cuda() #Implements data parallelism at the module level
    

    optim = None
    epoch = 0

    # load snapshot
    #if args.input is not None:
        #print('loading %s' % args.input)
        #model_data = torch.load(args.input)
        #model.load_state_dict(model_data.get('model_state', model_data))
        #optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()))
        #optim.load_state_dict(model_data.get('optimizer_state', model_data))
        #epoch = model_data['epoch'] + 1



    if args.use_both:
        # use train & val splits to optimize
        #use train & val splits to optimizeだが、a portion of Visual Genome datasetは使わない時
        trainval_dset = ConcatDataset([train_dset, val_dset]) #複数のdatasets(train/val dataset)をconcatenate
        train_loader = DataLoader(trainval_dset, batch_size, shuffle=True, num_workers=1)
        eval_loader = None
    else:
        train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1)
        eval_loader = DataLoader(val_dset, batch_size, shuffle=False, num_workers=1)


    train(con_model, train_loader, eval_loader, args.epochs, args.output, optim, epoch)
