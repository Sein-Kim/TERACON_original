from torch import nn
import torch
import torch.functional as F
import torch.nn.functional as F2
import time
import math
from torch.autograd import Variable
import numpy as np


class ResidualBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, dilation=None, task_embs=[0,2],num_task=1):
        super(ResidualBlock, self).__init__()
        self.out_channel = out_channel
        self.num_task = num_task
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation)
        self.ln1 = nn.LayerNorm(out_channel, eps=1e-8)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation*2)
        self.ln2 = nn.LayerNorm(out_channel, eps=1e-8)
        
        self.ec1 = nn.Embedding(6,out_channel)
        self.ec2 = nn.Embedding(6,out_channel)
        
        lo, hi = task_embs
        self.ec1.weight.data.uniform_(lo,hi)
        self.ec2.weight.data.uniform_(lo,hi)
        
        self.gate = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.dilation = dilation
        self.kernel_size = kernel_size

        self.mlp1 = nn.ModuleList(nn.Linear(2*num_task-1,1) for i in range(num_task))
        self.mlp2 = nn.ModuleList(nn.Linear(2*num_task-1,1) for i in range(num_task))

        for i in range(num_task):
            self.mlp1[i].weight.data.normal_(0.0,0.01)
            self.mlp1[i].bias.data.fill_(0.1)
            self.mlp2[i].weight.data.normal_(0.0,0.01)
            self.mlp2[i].bias.data.fill_(0.1)

    def mask(self, t,s):
        gc1 = self.gate(s*self.ec1(t))
        gc2 = self.gate(s*self.ec2(t))
        return [gc1,gc2]
    
    def attn_fmask(self,t,s, gpu):
        k_ = torch.tensor([i for i in range(self.num_task)]).to(gpu)
        new_task_emb_1 = self.ec1(t)
        for i in range(0,self.num_task):
            if not i ==t:
                new_task_emb_1 = torch.cat((new_task_emb_1,self.ec1(torch.tensor([i]).to(gpu))))
                new_task_emb_1 = torch.cat((new_task_emb_1,-1*self.ec1(torch.tensor([i]).to(gpu))))

        new_task_emb_2 = self.ec2(t)
        for i in range(0,self.num_task):
            if not i ==t:
                new_task_emb_2 = torch.cat((new_task_emb_2,self.ec2(torch.tensor([i]).to(gpu))))
                new_task_emb_2 = torch.cat((new_task_emb_2,-1*self.ec2(torch.tensor([i]).to(gpu))))


        attn_output_1 = self.mlp1[t](self.tanh(s*new_task_emb_1).transpose(1,0)).transpose(1,0)
        attn_output_2 = self.mlp2[t](self.tanh(s*new_task_emb_2).transpose(1,0)).transpose(1,0)
        gc1 = self.gate(s*attn_output_1)
        gc2 = self.gate(s*attn_output_2)
        return [gc1,gc2]

    def forward(self, x_): # x: [batch_size, seq_len, embed_size]
        s = x_[1]
        x = x_[0]
        t_ = x_[3]
        gpu_num = x_[4]
        backward = x_[5]
        t = torch.tensor([t_]).to(gpu_num)
        masks_list = x_[2]
        current_task = x_[6]

        if t_ ==current_task:
            masks = self.attn_fmask(t,s,gpu_num)
        else:
            if (not current_task ==1) or backward:
                masks = self.attn_fmask(t,s,gpu_num)
            else:
                masks = self.mask(t,s)

        masks_list+=masks
        gc1, gc2 = masks
        x_pad = self.conv_pad(x, self.dilation)
        out =  self.conv1(x_pad).squeeze(2).permute(0, 2, 1)
        out = F2.relu(self.ln1(out))
        out = out*gc1.expand_as(out)

        out_pad = self.conv_pad(out, self.dilation*2)
        out = self.conv2(out_pad).squeeze(2).permute(0, 2, 1)
        out = F2.relu(self.ln2(out))
        out = out*gc2.expand_as(out)

        out = out + x
        # out_ = self.residual_blocks([inputs,s,masks_list,task_num,gpu_num, backward,current_task])

        return [out,s,masks_list, t_, gpu_num,backward,current_task]

    def conv_pad(self, x, dilation):
        inputs_pad = x.permute(0, 2, 1)  # [batch_size, embed_size, seq_len]
        inputs_pad = inputs_pad.unsqueeze(2)  # [batch_size, embed_size, 1, seq_len]
        pad = nn.ZeroPad2d(((self.kernel_size - 1) * dilation, 0, 0, 0))
        inputs_pad = pad(inputs_pad)  # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*dilations]
        return inputs_pad

    def truncated_normal_(self, tensor, mean=0, std=0.09):
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor

class NextItNet_Decoder(nn.Module):

    def __init__(self, model_para):
        super(NextItNet_Decoder, self).__init__()
        self.model_para = model_para
        self.item_size = model_para['item_size']
        self.embed_size = model_para['dilated_channels']
        self.embeding = nn.Embedding(self.item_size, self.embed_size)
        self.task_embs = model_para['task_embs']
        stdv = np.sqrt(1. / self.item_size)
        self.embeding.weight.data.uniform_(-stdv, stdv) # important initializer

        self.num_task = model_para['num_task']
        self.target_size = model_para['target_item_size']
        self.past_target_size = model_para['past_target_size']

        self.dilations = model_para['dilations']
        self.residual_channels = model_para['dilated_channels']
        self.kernel_size = model_para['kernel_size']
        rb = [ResidualBlock(self.residual_channels, self.residual_channels, kernel_size=self.kernel_size,
                            dilation=dilation, task_embs=self.task_embs,num_task=self.num_task) for dilation in self.dilations]
        self.residual_blocks = nn.Sequential(*rb)

        self.final_layer = nn.Linear(self.residual_channels, self.target_size)

        self.final_layer.weight.data.normal_(0.0, 0.01)  # initializer
        self.final_layer.bias.data.fill_(0.1)

        self.past_final_layer = nn.Linear(self.residual_channels, self.past_target_size)
        self.past_final_layer.weight.data.normal_(0.0, 0.01)  # initializer
        self.past_final_layer.bias.data.fill_(0.1)

        self.one_layer_task1 = nn.Linear(self.residual_channels, self.past_target_size)

        self.one_layer_task1.weight.data.normal_(0.0,0.01)
        self.one_layer_task1.bias.data.fill_(0.1)

        self.one_layer_task2 = nn.Linear(self.residual_channels, self.past_target_size)
        self.one_layer_task3 = nn.Linear(self.residual_channels, self.past_target_size)
        self.one_layer_task4 = nn.Linear(self.residual_channels, self.past_target_size)
        self.one_layer_task5 = nn.Linear(self.residual_channels, self.past_target_size)
        
    def forward(self, x,s,masks_list,task_num, gpu_num,onecall=False,backward=False,current_task=2): # inputs: [batch_size, seq_len]
        inputs = self.embeding(x) # [batch_size, seq_len, embed_size]
        out_ = self.residual_blocks([inputs,s,masks_list,task_num,gpu_num, backward,current_task])
        dilate_outputs = out_[0]
        masks_ = out_[2]

        if onecall:
            hidden = dilate_outputs[:, -1, :].view(-1, self.residual_channels) # [batch_size, embed_size]
        else:
            
            hidden = dilate_outputs.view(-1, self.residual_channels) # [batch_size*seq_len, embed_size] 
        if backward:
            if task_num == 0:
                out = self.one_layer_task1(hidden)
            elif task_num == 1:
                out = self.one_layer_task2(hidden)
            elif task_num == 2:
                out = self.one_layer_task3(hidden)
            elif task_num ==3:
                out = self.one_layer_task4(hidden)
            elif task_num == 4:
                out = self.one_layer_task5(hidden)
            out_2 = 0
        else:
            out = self.final_layer(hidden)
            out_2 = self.past_final_layer(hidden)


        return out,out_2, masks_
