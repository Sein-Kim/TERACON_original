import torch
import torch.nn as nn
import data_loader as data_loader
import generator_hat_other as generator_recsys

import math
import numpy as np
import argparse
import random

import time
from tqdm import tqdm
np.random.seed(10)

from utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=5,
                        help='Sample from top k predictions')
    parser.add_argument('--gpu_num', type=int, default=4)
    parser.add_argument('--epochs',type=int, default=10)
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='hyperpara-Adam')
    parser.add_argument('--L2', default=0.75, type=float)
    parser.add_argument('--savepath',type=str, default='./saved_models/task1')
    parser.add_argument('--rho', default=0.2, type=float)
    parser.add_argument('--seed', type=int, default = 10)
    parser.add_argument('--lr', type = float, default=0.001)
    parser.add_argument('--datapath', type=str, default='./Data/ColdRec/original_desen_pretrain.csv',
                        help='data path')
    parser.add_argument('--datapath_index', type=str, default='Data/Session/index.csv',
                        help='data path')
    parser.add_argument('--split_percentage', type=float, default=0.2,
                        help='0.2 means 80% training 20% testing')
    parser.add_argument('--clipgrad',type=int, default = 1000)
    parser.add_argument('--model_name', type=str,default='NextitNet')
    parser.add_argument('--smax',type=int, default = 50)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    print(args.datapath)
    dl = data_loader.Data_Loader({'model_type': 'generator', 'dir_name': args.datapath,'dir_name_index': args.datapath_index})
    all_samples = dl.item
    items = dl.item_dict
    bigemb= dl.embed_len
    print("len(source)",len(items))
    print("len(allitems)", bigemb)

    shuffle_indices = np.random.permutation(np.arange(len(all_samples)))
    all_samples = all_samples[shuffle_indices]

    dev_sample_index = -1 * int(args.split_percentage * float(len(all_samples)))
    # train_set, valid_set = all_samples[:dev_sample_index], all_samples[dev_sample_index:]
    dev_sample_index_valid = int(dev_sample_index*0.75)
    train_set, valid_set, test_set = all_samples[:dev_sample_index], all_samples[dev_sample_index:dev_sample_index_valid], all_samples[dev_sample_index_valid:]
    print('len valid', len(valid_set))

    model_para = {
        'item_size': len(items),
        'bigemb':bigemb,
        'dilated_channels': 256,
        'dilations': [1,4,1,4,1,4,1,4,],
        'kernel_size': 3,
        'learning_rate':args.lr,
        'batch_size':32,
        'iterations':4,
        'task_embs': [0,2],
        'target_item_size': 10,
    }

    args.gpu_num_ = args.gpu_num
    if args.gpu_num_ == 'cpu':
        args.device = 'cpu'
    else:
        args.device = torch.device("cuda:" + str(args.gpu_num_) if torch.cuda.is_available() else "cpu")
    model = generator_recsys.NextItNet_Decoder(model_para).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=model_para['learning_rate'], weight_decay=0)

    criterion = nn.CrossEntropyLoss()
    len_train = len(train_set)
    count = 0
    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        batch_size = model_para['batch_size']
        batch_num = train_set.shape[0] / batch_size
        start = time.time()
        INFO_LOG("-------------------------------------------------------train")

        for batch_idx, batch_sam in enumerate(getBatch(train_set, batch_size)):
            
            inputs, targets = torch.LongTensor(batch_sam[:, :-1]).to(args.device), torch.LongTensor(batch_sam[:, 1:]).to(
                args.device).view([-1])
            smax = args.smax
            r = batch_num
            s = (smax-1/smax)*batch_idx/r+1/smax

            model.train()
            optimizer.zero_grad()
            masks_list = []
            outputs, masks = model(inputs,s,masks_list, 0, args.gpu_num)
            loss = criterion(outputs, targets)
            clipgrad = args.clipgrad
            reg = 0
            count_reg = 0
            for m in masks:
                reg+=m.sum()
                count_reg+=np.prod(m.size()).item()
            reg/=count_reg

                    
            loss += args.L2 * reg
            loss.backward()
            train_loss += loss.item()

            thres_cosh = 50
            # for n,p in model.named_parameters():
            #         if ('.ec' in n) and (not ('emb' in n)):
            #             num=torch.cosh(torch.clamp(s*p.data,-thres_cosh,thres_cosh))+1
            #             den=torch.cosh(p.data)+1
            #             p.grad.data*=smax/s*num/den
            # torch.nn.utils.clip_grad_norm_(model.parameters(),clipgrad)
            optimizer.step()

            thres_emb = 6
            for n,p in model.named_parameters():
                if ('.ec' in n) and (not ('emb' in n)):
                    p.data=torch.clamp(p.data,-thres_emb,thres_emb)
                    

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % max(10, batch_num//100) == 0:
            # if batch_idx % 1000 == 0:
                INFO_LOG("epoch: {}\t {}/{}".format(epoch, batch_idx, batch_num))
                print('Loss: %.3f | Acc(hit@1): %.3f%% (%d/%d)' % (
                    train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
                print(reg)
        end = time.time()


        model.eval()
        correct = 0
        total = 0
        batch_size_test = 512
        batch_num_test = valid_set.shape[0] / batch_size
        list_ = [[] for i in range(6)]
        INFO_LOG("-------------------------------------------------------valid")
        with torch.no_grad():
            start = time.time()
            for batch_idx, batch_sam in enumerate(getBatch(valid_set, batch_size_test)):
                inputs, targets = torch.LongTensor(batch_sam[:,:-1]).to(args.device), torch.LongTensor(batch_sam[:,-1]).to(args.device).view([-1])
                masks_list = []
                outputs,_ = model(inputs,smax,masks_list,0, args.gpu_num,onecall=True)
                _, sort_idx_20 = torch.topk(outputs, k=args.top_k + 15, sorted=True)  # [batch_size, 20]
                _, sort_idx_5 = torch.topk(outputs, k=args.top_k, sorted=True)  # [batch_size, 5]
                result_ = accuracy_test(sort_idx_5.data.cpu().numpy(), sort_idx_20.data.cpu().numpy(), targets.data.cpu().numpy(),
                        batch_idx, batch_num_test, epoch, args, list_)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            end = time.time()
            print('Acc(hit@1): %.3f%% (%d/%d)' % (100. * correct / total, correct, total))
        
        INFO_LOG("Accuracy mrr_5: {}".format(sum(result_[0]) / float(len(result_[0]))))
        INFO_LOG("Accuracy mrr_20: {}".format(sum(result_[3]) / float(len(result_[3]))))
        INFO_LOG("Accuracy hit_5: {}".format(sum(result_[1]) / float(len(result_[1]))))
        INFO_LOG("Accuracy hit_20: {}".format(sum(result_[4]) / float(len(result_[4]))))
        INFO_LOG("Accuracy ndcg_5: {}".format(sum(result_[2]) / float(len(result_[2]))))
        INFO_LOG("Accuracy ndcg_20: {}".format(sum(result_[5]) / float(len(result_[5]))))
        
        if epoch == 0:
            best_acc = (100.*correct/total)
            count = 0
            list__ = [[] for i in range(6)]
            print('-----testing in best validation-----')
            model_test_(model_para, test_set, model, epoch, args, list__,smax)
        else:
            if best_acc < (100.*correct/total):
                best_acc = (100.*correct/total)
                count = 0
                list__ = [[] for i in range(6)]
                print('-----testing in best validation-----')
                model_test_(model_para, test_set, model, epoch, args, list__,smax)
            else:
                count+=1
        if count == 5:
            break
        print('count', count)
        
        INFO_LOG("TIME FOR EPOCH During Training: {}".format(end - start))
        INFO_LOG("TIME FOR BATCH (mins): {}".format((end - start) / batch_num))


if __name__ == '__main__':
    curr_preds_5 = []
    rec_preds_5 = []
    ndcg_preds_5 = []
    curr_preds_20 = []
    rec_preds_20 = []
    ndcg_preds_20 = []
    main()
