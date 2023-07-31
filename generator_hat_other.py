from torch import nn
import torch
import torch.functional as F
import torch.nn.functional as F2
import time
import math
from torch.autograd import Variable
import numpy as np


class ResidualBlock(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3, dilation=None, task_embs=[0,2]):
        super(ResidualBlock, self).__init__()
        
        
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
        
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.tanh = nn.Tanh()

    # def mask_(self, t,s):
    #     gc1 = self.gate(s*self.tanh(s*self.ec1(t)))
    #     gc2 = self.gate(s*self.tanh(s*self.ec2(t)))
    #     return [gc1,gc2]

    def mask(self, t,s):
        gc1 = self.gate(s*self.ec1(t))
        gc2 = self.gate(s*self.ec2(t))
        return [gc1,gc2]

    def forward(self, x_): # x: [batch_size, seq_len, embed_size]
        s = x_[1]
        x = x_[0]
        t_ = x_[3]
        t = torch.tensor([t_]).to(x_[4])
        masks_list = x_[2]
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

        return [out,s,masks_list, t_, x_[4]]

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
        self.embeding.weight.data.uniform_(-stdv, stdv)
        
        self.target_size = model_para['target_item_size']


        self.dilations = model_para['dilations']
        self.residual_channels = model_para['dilated_channels']
        self.kernel_size = model_para['kernel_size']
        rb = [ResidualBlock(self.residual_channels, self.residual_channels, kernel_size=self.kernel_size,
                            dilation=dilation, task_embs=self.task_embs) for dilation in self.dilations]
        self.residual_blocks = nn.Sequential(*rb)

        self.final_layer = nn.Linear(self.residual_channels, self.item_size)

        self.final_layer.weight.data.normal_(0.0, 0.01)  
        self.final_layer.bias.data.fill_(0.1)

        self.two_layer_1 = nn.Linear(self.residual_channels, self.residual_channels)
        self.two_layer_2 = nn.Linear(self.residual_channels, self.target_size)

        self.two_layer_1.weight.data.normal_(0.0,0.01)
        self.two_layer_1.bias.data.fill_(0.1)
        self.two_layer_2.weight.data.normal_(0.0,0.01)
        self.two_layer_2.bias.data.fill_(0.1)

    def forward(self, x,s,masks_list,task_num, gpu_num,onecall=False,backward=False):
        inputs = self.embeding(x) 
        if backward:
            inputs += torch.normal(0,0.01,inputs.size()).to(gpu_num)
        out_ = self.residual_blocks([inputs,s,masks_list,task_num,gpu_num])
        dilate_outputs = out_[0]
        masks_ = out_[2]

        if onecall:
            hidden = dilate_outputs[:, -1, :].view(-1, self.residual_channels) 
        else:
            
            hidden = dilate_outputs.view(-1, self.residual_channels) 
        if backward:
            out = self.two_layer_1(hidden)
            out = self.two_layer_2(out)
        else:
            out = self.final_layer(hidden)


        return out, masks_
