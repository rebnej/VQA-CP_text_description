"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import os
import time
import itertools
import torch
import torch.nn as nn
import utils
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import numpy as np


def instance_bce_with_logits(logits, labels, reduction='mean'):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)
    if reduction == 'mean':
        loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


#train(model, train_loader, eval_loader, args.epochs, args.output, optim, epoch)
def train(model, train_loader, eval_loader, num_epochs, output, opt=None, s_epoch=0):
    #lr_default = 1e-3 if eval_loader is not None else 7e-4
    #lr_decay_step = 2
    #lr_decay_rate = .25
    #lr_decay_epochs = range(10,20,lr_decay_step) if eval_loader is not None else range(10,20,lr_decay_step)
    #gradual_warmup_steps = [0.5 * lr_default, 1.0 * lr_default, 1.5 * lr_default, 2.0 * lr_default]
    saving_epoch = 3
    grad_clip = .25

    #utils.create_dir(output) #'saved_models/ban'ディレクトリを作成
    #optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_default) \
        #if opt is None else opt
    optim = torch.optim.Adam(model.parameters(), lr=2e-5, eps=1e-8)
    logger = utils.Logger(os.path.join(output, 'log.txt'))#'saved_models/ban/log.txt'にloggerを作成
    best_eval_score = 0

    utils.print_model(model, logger)
    #logger.write('optim: adamax lr=%.4f, decay_step=%d, decay_rate=%.2f, grad_clip=%.2f' % \
        #(lr_default, lr_decay_step, lr_decay_rate, grad_clip))

    for epoch in range(s_epoch, num_epochs):
        total_loss = 0
        train_score = 0
        total_norm = 0
        count_norm = 0
        t = time.time()
        N = len(train_loader.dataset)
        #if epoch < len(gradual_warmup_steps):
            #optim.param_groups[0]['lr'] = gradual_warmup_steps[epoch]
            #logger.write('gradual warmup lr: %.4f' % optim.param_groups[0]['lr'])
        #elif epoch in lr_decay_epochs:
            #optim.param_groups[0]['lr'] *= lr_decay_rate
            #logger.write('decreased lr: %.4f' % optim.param_groups[0]['lr'])
        #else:
            #logger.write('lr: %.4f' % optim.param_groups[0]['lr'])

        for i, (i_ids, mask, token_type, a) in tqdm(enumerate(train_loader)):
            
            batch_size = i_ids.size(0)
            i_ids = i_ids.view(batch_size, -1)
            mask = mask.view(batch_size, -1)
            token_type = token_type.view(batch_size, -1)
            
            if i == 0:
                print('i_ids, mask, token_type : shape : ', i_ids.shape, mask.shape, token_type.shape)
                
            i_ids = i_ids.cuda()
            mask = mask.cuda()
            token_type = token_type.cuda()
            a = a.cuda() #target/question_id

            pred, attention = model(i_ids, mask, token_type)
            loss = instance_bce_with_logits(pred, a)
            loss.backward()
            total_norm += nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            count_norm += 1
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += loss.item() * i_ids.size(0)
            train_score += batch_score.item()

        total_loss /= N
        train_score = 100 * train_score / N
        if None != eval_loader:
            model.train(False)
            eval_score, bound, entropy = evaluate(model, eval_loader)
            model.train(True)

        logger.write('epoch %d, time: %.2f' % (epoch, time.time()-t))
        logger.write('\ttrain_loss: %.2f, norm: %.4f, score: %.2f' % (total_loss, total_norm/count_norm, train_score))
        if eval_loader is not None:
            logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

        #if eval_loader is not None and entropy is not None:
            #info = ''
            #for i in range(entropy.size(0)):
            #    info = info + ' %.2f' % entropy[i]
            #logger.write('\tentropy: ' + info)

        if (eval_loader is not None and eval_score > best_eval_score) or (eval_loader is None and epoch >= saving_epoch):
            model_path = os.path.join(output, 'model_epoch%d.pth' % epoch)
            utils.save_model(model_path, model, epoch, optim)
            if eval_loader is not None:
                best_eval_score = eval_score


@torch.no_grad()
def evaluate(model, dataloader):
    score = 0
    upper_bound = 0
    num_data = 0
    entropy = None
    #if hasattr(model.module, 'glimpse'):
        #entropy = torch.Tensor(model.module.glimpse).zero_().cuda()
    for i, (i_ids, mask, token_type, a) in enumerate(dataloader):
        
        batch_size = i_ids.size(0)
        i_ids = i_ids.view(batch_size, -1)
        mask = mask.view(batch_size, -1)
        token_type = token_type.view(batch_size, -1)
        
        i_ids = i_ids.cuda()
        mask = mask.cuda()
        token_type = token_type.cuda()
        
        pred, attention = model(i_ids, mask, token_type)
        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score.item()
        upper_bound += (a.max(1)[0]).sum().item()
        num_data += pred.size(0)
        #if att is not None and 0 < model.module.glimpse:
            #entropy += calc_entropy(att.data)[:model.module.glimpse]
            
        if i <= 0:
            print(batch_score.shape)#[] (value)
            print(batch_score) # the number of the correct answers
            s = compute_score_with_logits(pred, a.cuda())
            print(s.shape)#[256, 28]
            print(s)
            s = s.cpu().numpy()
            res = np.any(s, axis=1)
            print(res)
            correct = []
            failure = []
            for i, r in enumerate(res):
                if r:
                    correct.append(i)
                else:
                    failure.append(i)
            print(len(correct))
            print(len(failure))
            print('--------------failure')
            print(failure)
            #print('--------------question 1')
            #print(q[1])
            print('--------------answer 1')
            print(a[1])
            print('----------------pred 1')
            print(torch.max(pred, 1)[1].data)
            

    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)

    if entropy is not None:
        entropy = entropy / len(dataloader.dataset)

    return score, upper_bound, entropy

def calc_entropy(att): # size(att) = [b x g x v x q]
    sizes = att.size()
    eps = 1e-8
    p = att.view(-1, sizes[1], sizes[2] * sizes[3])
    return (-p * (p+eps).log()).sum(2).sum(0) # g
