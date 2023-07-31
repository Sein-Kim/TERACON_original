import os
from os import listdir
from os.path import isfile, join
import numpy as np

import math
from tqdm import trange

import torch

class Data_Loader:
    def __init__(self,options):
        positive_data_file = options['dir_name']
        index_data_file = options['dir_name_index']
        positive_examples = list(open(positive_data_file,'r').readlines())
        positive_examples = [s for s in positive_examples]
        x_list = [x[:-1].split(",") for x in positive_examples]
        unique_ = set()
        for x in x_list:
            unique_.update(x)

        self.item_dict = {'<UNK>':0}
        i = 0
        for k in sorted(unique_,key=int):
            self.item_dict[k] = i+1
            i+=1
        
        self.item = np.array([[self.item_dict[x] for x in xs] for xs in x_list])

        self.embed_len = len(self.item_dict)

        f = open(index_data_file,'w')
        f.write(str(self.item_dict))
        f.close()
        print("The index has been written to {}".format(index_data_file))



class Data_Loader_Sup:
    def __init__(self,options):
        positive_data_file = options['dir_name']
        index_data_file = options['dir_name_index']
        self.item_dict = self.read_dict(index_data_file)

        positive_examples = list(open(positive_data_file, "r").readlines())
        colon = ",,"
        source = [s[:-1].split(colon)[0] for s in positive_examples]
        target = [s[:-1].split(colon)[1] for s in positive_examples]

        self.item = self.map_dict(self.item_dict, source)
        self.item_seq_len = self.item.shape[1]
        x_list = [x.split(",") for x in target]
        unique_ = set()
        for x in x_list:
            unique_.update(x)

        self.target_dict = {'<UNK>':0}
        i = 0
        for k in sorted(unique_,key=int):
            self.target_dict[k] = i+1
            i+=1
        
        self.maxsource = len(self.item_dict)
        self.maxtarget = len(self.target_dict)

        max_target_len = max([len(x) for x in x_list])
        target_list = []
        for lines in x_list:
            zeros = np.zeros(max_target_len)
            i = 0
            for l in lines:
                zeros[i] = self.target_dict[l]
                i+=1
            target_list.append(zeros)
        self.target = np.array(target_list)

        self.separator = 0
        lens = self.item.shape[0]

        self.example = np.array([])
        self.example = []
        
        for line in range(lens):
            source_line = self.item[line]
            target_line = self.target[line]
            target_num = len(target_line)
            for j in range(target_num):
                if target_line[j] != 0:
                    unit = np.append(source_line, np.array(self.separator)) #3
                    unit = np.append(unit, np.array(target_line[j] ))
                    self.example.append(unit)
        self.example = np.array(self.example) 

        self.embed_len = len(self.item_dict)

    def read_dict(self, index_data_file):
        dict_temp = {}
        file = open(index_data_file, 'r')
        for line in file.readlines():
            dict_temp = eval(line)
        return dict_temp

    def map_dict(self, dict_pretrain, source):
        items = []
        for lines in source:
            trueline = [dict_pretrain[x] for x in lines.split(',')]
            # print trueline
            trueline = np.array(trueline)
            items.append(trueline)
        return np.array(items)