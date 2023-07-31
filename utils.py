import numpy as np
import time
import math
import torch
import copy
import torch.nn as nn

def getBatch_train(data, index, batch_size):
    shuffle_indices = np.random.permutation(np.arange(len(data)))
    data = data[shuffle_indices]
    index = index[shuffle_indices]

    start_inx = 0
    end_inx = batch_size

    while end_inx < len(data):
        batch = data[start_inx:end_inx]
        index = index[start_inx:end_inx]
        start_inx = end_inx
        end_inx += batch_size
        yield [batch, index]

def getBatch(data, batch_size):
    shuffle_indices = np.random.permutation(np.arange(len(data)))
    data = data[shuffle_indices]

    start_inx = 0
    end_inx = batch_size

    while end_inx < len(data):
        batch = data[start_inx:end_inx]
        start_inx = end_inx
        end_inx += batch_size
        yield batch


def INFO_LOG(info):
    print("[%s]%s"%(time.strftime("%Y-%m-%d %X", time.localtime()), info))

def accuracy_test(pred_items_5, pred_items_20, target, batch_idx, batch_num, epoch,args, list_): # output: [batch_size, 20] target: [batch_size]
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # print(type(pred_items_20[0]))
    # print(type(pred_items_5[0]))
    curr_preds_5_ = list_[0]
    rec_preds_5_ = list_[1]
    ndcg_preds_5_ = list_[2]
    curr_preds_20_ = list_[3]
    rec_preds_20_ = list_[4]
    ndcg_preds_20_ = list_[5]
    for bi in range(pred_items_5.shape[0]):

        true_item=target[bi]
        predictmap_5={ch : i for i, ch in enumerate(pred_items_5[bi])}
        predictmap_20 = {ch: i for i, ch in enumerate(pred_items_20[bi])}

        rank_5 = predictmap_5.get(true_item)
        rank_20 = predictmap_20.get(true_item)
        if rank_5 == None:
            curr_preds_5_.append(0.0)
            rec_preds_5_.append(0.0)
            ndcg_preds_5_.append(0.0)
        else:
            MRR_5 = 1.0/(rank_5+1)
            Rec_5 = 1.0#3
            ndcg_5 = 1.0 / math.log(rank_5 + 2, 2)  # 3
            curr_preds_5_.append(MRR_5)
            rec_preds_5_.append(Rec_5)#4
            ndcg_preds_5_.append(ndcg_5)  # 4
        if rank_20 == None:
            curr_preds_20_.append(0.0)
            rec_preds_20_.append(0.0)#2
            ndcg_preds_20_.append(0.0)#2
        else:
            MRR_20 = 1.0/(rank_20+1)
            Rec_20 = 1.0#3
            ndcg_20 = 1.0 / math.log(rank_20 + 2, 2)  # 3
            curr_preds_20_.append(MRR_20)
            rec_preds_20_.append(Rec_20) # 4
            ndcg_preds_20_.append(ndcg_20)  # 4
    
    return [curr_preds_5_,rec_preds_5_,ndcg_preds_5_,curr_preds_20_,rec_preds_20_,ndcg_preds_20_]


def model_test(model_para, test_set, model, epoch, args, list_,smax,backward,task_num,current_task):
    best_acc = 0
    model.eval()
    correct = 0
    total = 0
    batch_size = model_para['batch_size']

    batch_num = test_set.shape[0] / batch_size
    INFO_LOG("-------------------------------------------------------test")
    with torch.no_grad():
        start = time.time()
        for batch_idx, batch_sam in enumerate(getBatch(test_set, batch_size)):
            if task_num == 0:
                inputs = torch.LongTensor(batch_sam[:,:-1]).to(args.device)
                target = torch.LongTensor(batch_sam[:,-1]).to(args.device)
            else:
                inputs = torch.LongTensor(batch_sam[:, :-2]).to(args.device)
                target = torch.LongTensor(batch_sam[:,-1]).to(args.device)
            masks_list = []
            student_out, _,masks = model(inputs,smax,masks_list, task_num, args.gpu_num,onecall=True,backward=backward,current_task=current_task)
            output_mean = student_out

            list_toy = [[] for i in range(6)]

            _, sort_idx_20 = torch.topk(output_mean, k=args.top_k + 15, sorted=True)  # [batch_size, 20]
            _, sort_idx_5 = torch.topk(output_mean, k=args.top_k, sorted=True)  # [batch_size, 5]
            list__ = accuracy_test(sort_idx_5.data.cpu().numpy(), sort_idx_20.data.cpu().numpy(), target.data.cpu().numpy(),
                     batch_idx, batch_num, epoch, args, list_toy)
            for i in range(len(list_)):
                list_[i] +=list__[i]

            _, predicted = output_mean.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        end = time.time()
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        state = {
            'net': model.state_dict(),
            'acc(hit@1)': acc
        }
        torch.save(state, '%s.t7' % (args.savepath))

    print('epoch:%d    accuracy(hit@1):%.3f    best:%.3f' % (epoch, acc, best_acc))

    INFO_LOG("epoch: {}\t total_epoch:{}\t total_batches:{}".format(
        epoch, args.epochs, batch_num))
    INFO_LOG("Accuracy mrr_5: {}".format(sum(list_[0]) / float(len(list_[0]))))
    INFO_LOG("Accuracy mrr_20: {}".format(sum(list_[3]) / float(len(list_[3]))))
    INFO_LOG("Accuracy hit_5: {}".format(sum(list_[1]) / float(len(list_[1]))))
    INFO_LOG("Accuracy hit_20: {}".format(sum(list_[4]) / float(len(list_[4]))))
    INFO_LOG("Accuracy ndcg_5: {}".format(sum(list_[2]) / float(len(list_[2]))))
    INFO_LOG("Accuracy ndcg_20: {}".format(sum(list_[5]) / float(len(list_[5]))))



def model_test_acc(model_para, test_set, model, epoch, args, list_,smax,backward,task_num,current_task):
    best_acc = 0
    model.eval()
    correct = 0
    total = 0
    iftest=True
    batch_size = model_para['batch_size']
    print(task_num)
    batch_num = test_set.shape[0] / batch_size
    INFO_LOG("-------------------------------------------------------test")
    with torch.no_grad():
        start = time.time()
        for batch_idx, batch_sam in enumerate(getBatch(test_set, batch_size)):
            inputs = torch.LongTensor(batch_sam[:, :-2]).to(args.device)
            target = torch.LongTensor(batch_sam[:,-1]).to(args.device)
            masks_list = []

            student_out, _,masks = model(inputs,smax,masks_list, task_num, args.gpu_num,onecall=True,backward=backward,current_task=current_task)
            output_mean = student_out

            _, predicted = output_mean.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        end = time.time()
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        state = {
            'net': model.state_dict(),
            'acc(hit@1)': acc
        }
        # torch.save(state, '%s/task4_idea_attn.t7' % (args.savedir))
        torch.save(state, '%s.t7' % (args.savepath))

        # torch.save(masked_dict, './saved_mask/' +'_task2_mask.pth')

    print('epoch:%d    accuracy(hit@1):%.3f    best:%.3f' % (epoch, acc, best_acc))

    INFO_LOG("epoch: {}\t total_epoch:{}\t total_batches:{}".format(
        epoch, args.epochs, batch_num))


def sample_ratio(mask1, mask2):
    gate = nn.Sigmoid()
    cos = nn.CosineSimilarity(dim=1,eps=1e-6)
    jaccard = 0
    len_jaccard = 0
    for i in range(len(mask1)):
        jaccard +=gate(6*cos(mask1[i],mask2[i])).detach().cpu().item()
        len_jaccard+=1
    random_batch_size = 1 - jaccard/len_jaccard
    return random_batch_size



def model_test_(model_para, test_set, model, epoch, args, list_,smax):
    best_acc = 0
    model.eval()
    correct = 0
    total = 0
    batch_size = 512
    batch_num = test_set.shape[0] / batch_size
    INFO_LOG("-------------------------------------------------------test")
    with torch.no_grad():
        start = time.time()
        for batch_idx, batch_sam in enumerate(getBatch(test_set, batch_size)):
            inputs, target = torch.LongTensor(batch_sam[:,:-1]).to(args.device), torch.LongTensor(batch_sam[:,-1]).to(args.device).view([-1])

            masks_list = []
            outputs,_ = model(inputs,smax,masks_list,0, args.gpu_num,onecall=True)
            output_mean = outputs


            list_toy = [[] for i in range(6)]
            _, sort_idx_20 = torch.topk(output_mean, k=args.top_k + 15, sorted=True)  # [batch_size, 20]
            _, sort_idx_5 = torch.topk(output_mean, k=args.top_k, sorted=True)  # [batch_size, 5]
            list__ = accuracy_test(sort_idx_5.data.cpu().numpy(), sort_idx_20.data.cpu().numpy(), target.data.cpu().numpy(),
                     batch_idx, batch_num, epoch, args, list_toy)
            for i in range(len(list_)):
                list_[i] +=list__[i]

            _, predicted = output_mean.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        end = time.time()
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        state = {
            'net': model.state_dict(),
            'acc(hit@1)': acc
        }
        torch.save(state, '%s.t7' % (args.savepath))

    print('epoch:%d    accuracy(hit@1):%.3f    best:%.3f' % (epoch, acc, best_acc))

    INFO_LOG("epoch: {}\t total_epoch:{}\t total_batches:{}".format(
        epoch, args.epochs, batch_num))
    INFO_LOG("Accuracy mrr_5: {}".format(sum(list_[0]) / float(len(list_[0]))))
    INFO_LOG("Accuracy mrr_20: {}".format(sum(list_[3]) / float(len(list_[3]))))
    INFO_LOG("Accuracy hit_5: {}".format(sum(list_[1]) / float(len(list_[1]))))
    INFO_LOG("Accuracy hit_20: {}".format(sum(list_[4]) / float(len(list_[4]))))
    INFO_LOG("Accuracy ndcg_5: {}".format(sum(list_[2]) / float(len(list_[2]))))
    INFO_LOG("Accuracy ndcg_20: {}".format(sum(list_[5]) / float(len(list_[5]))))
