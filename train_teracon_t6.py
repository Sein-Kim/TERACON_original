import torch
import torch.nn as nn
import data_loader as data_loader
import tracon as generator_recsys

import math
import numpy as np
import argparse

import random
import time
from tqdm import tqdm
from tqdm import trange
import collections

from utils import *
from load_model import *
import copy



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=5,
                        help='Sample from top k predictions')
    parser.add_argument('--gpu_num', type=int, default=2)
    parser.add_argument('--epochs',type=int, default=100)
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='hyperpara-Adam')
    parser.add_argument('--alpha', default=0.7, type=float)

    parser.add_argument('--savepath',type=str, default='./saved_models/task6')
    parser.add_argument('--datapath', type=str, default='./Data/ColdRec/original_desen_lifestatus.csv',
                        help='data path')
    parser.add_argument('--datapath_index', type=str, default='Data/Session/index.csv',
                        help='data path')
    parser.add_argument('--eval_iter', type=int, default=20000,
                        help='Sample generator output evry x steps')
    parser.add_argument('--save_para_every', type=int, default=20000,
                        help='save model parameters every')
    parser.add_argument('--split_percentage', type=float, default=0.2,
                        help='0.2 means 80% training 20% testing')
    parser.add_argument('--smax',type=int, default = 50)
    parser.add_argument('--clipgrad',type=int, default = 1000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--paths', type=str, default='./saved_models/task5.t7')
    parser.add_argument('--batch', type=int,default=1024)
    parser.add_argument('--model_name', type=str,default='NextitNet')
    parser.add_argument('--seed', type=int, default = 10)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    dl = data_loader.Data_Loader_Sup({'model_type': 'generator', 'dir_name': args.datapath,'dir_name_index': args.datapath_index})
    items = dl.item_dict
    bigemb= dl.embed_len
    print("len(source)",len(items))

    targets_ = dl.target_dict
    targets_len_ = len(targets_)
    print('len(target)', targets_len_)
    print("len(allitems)", bigemb)

    all_samples = dl.example
    print('len(all_samples)',len(all_samples))

    shuffle_indices = np.random.permutation(np.arange(len(all_samples)))
    all_samples = all_samples[shuffle_indices]

    # Split train/test set
    dev_sample_index = -1 * int(args.split_percentage * float(len(all_samples)))
    dev_sample_index_valid = int(dev_sample_index*0.75)
    train_set, valid_set_, test_set_ = all_samples[:dev_sample_index], all_samples[dev_sample_index:dev_sample_index_valid], all_samples[dev_sample_index_valid:]
    print('len valid', len(valid_set_))

    model_para = {
        'item_size': len(items),
        'bigemb':bigemb,
        'dilated_channels': 256,#256
        'target_item_size': targets_len_,
        'past_target_size':len(items),
        'dilations': [1,4,1,4,1,4,1,4,],
        'kernel_size': 3,
        'learning_rate':args.lr,#0.0001,
        'batch_size':args.batch,
        'iterations':15,
        'task_embs':[0,2],#[0,2]
        'num_task':6,
    }

    args.gpu_num_ = args.gpu_num
    if args.gpu_num_ == 'cpu':
        args.device = 'cpu'
    else:
        args.device = torch.device("cuda:" + str(args.gpu_num_) if torch.cuda.is_available() else "cpu")
    model = generator_recsys.NextItNet_Decoder(model_para).to(args.device)

    model_para['num_task'] = 5
    model_1 = generator_recsys.NextItNet_Decoder(model_para).to(args.device)
    model_1.past_final_layer = nn.Linear(model_para['dilated_channels'],645974).to(args.device)

    model_2 = generator_recsys.NextItNet_Decoder(model_para).to(args.device)
    model_2.past_final_layer =nn.Linear(model_para['dilated_channels'],17880).to(args.device)

    model_3 = generator_recsys.NextItNet_Decoder(model_para).to(args.device)
    model_3.past_final_layer =nn.Linear(model_para['dilated_channels'],7540).to(args.device)

    model_4 = generator_recsys.NextItNet_Decoder(model_para).to(args.device)
    model_4.past_final_layer =nn.Linear(model_para['dilated_channels'],9).to(args.device)

    model_5 = generator_recsys.NextItNet_Decoder(model_para).to(args.device)
    model_5.past_final_layer =nn.Linear(model_para['dilated_channels'],3).to(args.device)

    model.one_layer_task1 = nn.Linear(model_para['dilated_channels'], 645974).to(args.device)
    model.one_layer_task2 = nn.Linear(model_para['dilated_channels'],17880).to(args.device)
    model.one_layer_task3 = nn.Linear(model_para['dilated_channels'],7540).to(args.device)
    model.one_layer_task4 = nn.Linear(model_para['dilated_channels'],9).to(args.device)
    model.one_layer_task5 = nn.Linear(model_para['dilated_channels'],3).to(args.device)

    task1_dict, task2_dict, task3_dict, task4_dict,task5_dict, task6_dict = task6_model(args)
    model.load_state_dict(task6_dict, strict=False)
    model_1.load_state_dict(task1_dict, strict=False)
    model_2.load_state_dict(task2_dict, strict=False)
    model_3.load_state_dict(task3_dict, strict=False)
    model_4.load_state_dict(task4_dict, strict=False)
    model_5.load_state_dict(task5_dict, strict=False)

    # model.embeding.weight.requires_grad = False
    # model.past_final_layer.weight.requires_grad= False
    # model.past_final_layer.bias.requires_grad = False
    for n,p in model_1.named_parameters():
        p.requires_grad = False
    for n,p in model_2.named_parameters():
        p.requires_grad = False
    for n,p in model_3.named_parameters():
        p.requires_grad = False
    for n,p in model_4.named_parameters():
        p.requires_grad = False
    for n,p in model_5.named_parameters():
        p.requires_grad = False
    criterion = nn.CrossEntropyLoss()
    criterion_ = nn.MSELoss()


    optimizer = torch.optim.Adam(model.parameters(), lr=model_para['learning_rate'], weight_decay=0)
    count = 0
    best_acc = 0
    len_train = len(train_set)

    random_batch_size = 0.9
    random_batch_size2 = 0.9
    random_batch_size3 = 0.9
    random_batch_size4 = 0.9
    random_batch_size5 = 0.9

    cos = nn.CosineSimilarity(dim=1,eps=1e-6)
    gate = nn.Sigmoid()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        transfer_loss_t=0
        correct = 0
        total = 0
        batch_size = model_para['batch_size']
        batch_num = train_set.shape[0] / batch_size
        start = time.time()
        shuffle_indices_ = np.random.permutation(np.arange(len(train_set)))
        train_set = train_set[shuffle_indices_]
        print(shuffle_indices_)
        INFO_LOG("-------------------------------------------------------train")
        for batch_idx, batch_sam in enumerate(getBatch(train_set, batch_size)):
            inputs = torch.LongTensor(batch_sam[:, :-2]).to(args.device)
            target = torch.LongTensor(batch_sam[:,-1]).to(args.device)
            random_buffer = []
            random_buffer = [i for i in range(len(inputs))]
            random.shuffle(random_buffer)
            rand_idx = np.array(random_buffer[:int(len(random_buffer)*random_batch_size)])
            rand_idx2 = np.array(random_buffer[:int(len(random_buffer)*random_batch_size2)])
            rand_idx3 = np.array(random_buffer[:int(len(random_buffer)*random_batch_size3)])
            rand_idx4 = np.array(random_buffer[:int(len(random_buffer)*random_batch_size4)])

            smax = args.smax
            r = batch_num
            s = (smax-1/smax)*batch_idx/r+1/smax
            optimizer.zero_grad()

            _,prev_output1,pmasks1 = model_1(inputs[rand_idx,:-1],s,[], 0,args.gpu_num,onecall=True,current_task=5)
            _,prev_output2,pmasks2 = model_2(inputs[rand_idx2,:],s,[], 1,args.gpu_num,onecall=True,current_task=5)
            _,prev_output3,pmasks3 = model_3(inputs[rand_idx3,:],s,[], 2,args.gpu_num,onecall=True,current_task=5)
            _,prev_output4,pmasks4 = model_4(inputs[rand_idx4,:],s,[], 3,args.gpu_num,onecall=True,current_task=5)
            _,prev_output5,pmasks5 = model_5(inputs[rand_idx4,:],s,[], 4,args.gpu_num,onecall=True,current_task=5)


            outputs, _,masks = model(inputs,s,[], 5,args.gpu_num,onecall=True,current_task=5)

            student_out1,_,masks1 = model(inputs[rand_idx,:-1],s,[], 0, args.gpu_num,onecall=True,backward=True,current_task=5)
            student_out2,_,masks2 = model(inputs[rand_idx2,:],s,[], 1, args.gpu_num,onecall=True,backward=True,current_task=5)
            student_out3,_,masks3 = model(inputs[rand_idx3,:],s,[], 2, args.gpu_num,onecall=True,backward=True,current_task=5)
            student_out4,_,masks4 = model(inputs[rand_idx4,:],s,[], 3, args.gpu_num,onecall=True,backward=True,current_task=5)
            student_out5,_,masks5 = model(inputs[rand_idx4,:],s,[], 4, args.gpu_num,onecall=True,backward=True,current_task=5)

            transfer_loss_5 = criterion_(student_out5,prev_output5)
            transfer_loss_4 = criterion_(student_out4,prev_output4)
            transfer_loss_3 = criterion_(student_out3,prev_output3)
            transfer_loss_2 = criterion_(student_out2,prev_output2)
            transfer_loss_1 = criterion_(student_out1,prev_output1)


            loss = criterion(outputs, target)
            clipgrad = args.clipgrad

            random_batch_size =sample_ratio(pmasks1,masks)
            random_batch_size2=sample_ratio(pmasks2,masks)
            random_batch_size3=sample_ratio(pmasks3,masks)
            random_batch_size4=sample_ratio(pmasks4,masks)
            random_batch_size5=sample_ratio(pmasks5,masks)

            coef_sum = random_batch_size+random_batch_size2+random_batch_size3+random_batch_size4+random_batch_size5
            coef1 = random_batch_size/coef_sum
            coef2 = random_batch_size2/coef_sum
            coef3 = random_batch_size3/coef_sum
            coef4 = random_batch_size4/coef_sum
            coef5 = random_batch_size5/coef_sum

            if not torch.isfinite(loss):
                print("Occured Nan", loss)
                loss = 0
                total += 0
            else:
                loss += (args.alpha * (transfer_loss_1*coef1+transfer_loss_2*coef2 + transfer_loss_3*coef3 + transfer_loss_4*coef4 + transfer_loss_5*coef5))
                loss.backward()
                train_loss += loss.item()
                transfer_loss_t +=args.alpha * (transfer_loss_1.item()*coef1+transfer_loss_2.item()*coef2 + transfer_loss_3.item()*coef3 + transfer_loss_4.item()*coef4 + transfer_loss_5.item()*coef5)
                thres_cosh = 50
                # for n,p in model.named_parameters():
                #     if ('.fec' in n):
                #         num=torch.cosh(torch.clamp(s*p.data,-thres_cosh,thres_cosh))+1
                #         den=torch.cosh(p.data)+1
                #         if p.grad != None:
                #             p.grad.data*=smax/s*num/den
                # torch.nn.utils.clip_grad_norm_(model.parameters(),clipgrad)
                optimizer.step()

            thres_emb = 6
            for n,p in model.named_parameters():
                if ('.fec' in n):
                    p.data=torch.clamp(p.data,-thres_emb,thres_emb)
                    
            _,predicted = outputs.max(1)
            total +=target.size(0)
            correct += predicted.eq(target).sum().item()
            if batch_idx % max(10, batch_num//100) == 0:
                INFO_LOG("epoch: {}\t {}/{}".format(epoch, batch_idx, batch_num))
                print('Transfer Loss: %.3f'%(transfer_loss_t/(batch_idx+1)))
                print('Loss: %.3f | Acc(hit@1): %.3f%% (%d/%d)' % (
                    train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        end = time.time()

        model.eval()
        correct = 0
        total = 0
        batch_size_test = model_para['batch_size']
        batch_num_test = valid_set_.shape[0] / batch_size
        INFO_LOG("-------------------------------------------------------valid")
        with torch.no_grad():
            start = time.time()
            for batch_idx, batch_sam in enumerate(getBatch(valid_set_, batch_size_test)):
                inputs = torch.LongTensor(batch_sam[:, :-2]).to(args.device)
                target = torch.LongTensor(batch_sam[:,-1]).to(args.device)
                masks_list = []

                outputs, _,masks1 = model(inputs,smax,masks_list, 5,args.gpu_num,onecall=True,current_task=5)

                output_mean = outputs
                _, predicted = output_mean.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            end = time.time()
            print('Acc(hit@1): %.3f%% (%d/%d)' % (100. * correct / total, correct, total))
        
        
        if epoch == 0:
            best_acc = (100.*correct/total)
            count = 0
            print('-----testing in best validation-----')
            list__ = [[] for i in range(6)]
            model_test_acc(model_para, test_set_, model,epoch,args,list__,smax, backward=False,task_num = 5, current_task=5)
        else:
            if best_acc < (100.*correct/total):
                best_acc = (100.*correct/total)
                count = 0
                print('-----testing in best validation-----')
                list__ = [[] for i in range(6)]
                model_test_acc(model_para, test_set_, model,epoch,args,list__,smax,backward=False,task_num=5,current_task=5)

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

