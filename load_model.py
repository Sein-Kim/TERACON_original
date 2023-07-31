import torch
import copy

#We will rewrite load_model code with nn.MoudleList at in camera ready

def task2_model(args):
    model_path_ = args.paths
    new_dict_ = torch.load(model_path_, map_location=torch.device(args.device))
    for key in list(new_dict_['net'].keys()):
        if 'two_layer_1' in key:
            del new_dict_['net'][key]

    new_dict_['net']['past_final_layer.weight'] = new_dict_['net']['final_layer.weight']
    new_dict_['net']['past_final_layer.bias'] = new_dict_['net']['final_layer.bias']

    new_dict_['net']['one_layer_task1.weight'] = new_dict_['net']['final_layer.weight']
    new_dict_['net']['one_layer_task1.bias'] = new_dict_['net']['final_layer.bias']
    del new_dict_['net']['final_layer.weight']
    del new_dict_['net']['final_layer.bias']
    return [new_dict_['net'],new_dict_['net']]

def task3_model(args):
    model_path_ = args.paths
    new_dict_ = torch.load(model_path_, map_location=torch.device(args.device))
    for key in list(new_dict_['net'].keys()):
        if 'two_layer_1' in key:
            del new_dict_['net'][key]

    task1_dict = copy.deepcopy(new_dict_['net'])
    task1_dict['past_final_layer.weight'] = task1_dict['one_layer_task1.weight']
    task1_dict['past_final_layer.bias'] = task1_dict['one_layer_task1.bias']


    task2_dict = copy.deepcopy(new_dict_['net'])
    task2_dict['past_final_layer.weight'] = task2_dict['final_layer.weight']
    task2_dict['past_final_layer.bias'] = task2_dict['final_layer.bias']


    new_dict_['net']['one_layer_task2.weight'] = new_dict_['net']['final_layer.weight']
    new_dict_['net']['one_layer_task2.bias'] = new_dict_['net']['final_layer.bias']

    del new_dict_['net']['final_layer.weight']
    del new_dict_['net']['final_layer.bias']
    del task1_dict['final_layer.weight']
    del task1_dict['final_layer.bias']
    del task2_dict['final_layer.weight']
    del task2_dict['final_layer.bias']
    del task2_dict['one_layer_task1.weight']
    del task2_dict['one_layer_task1.bias']

    del new_dict_['net']['past_final_layer.weight']
    del new_dict_['net']['past_final_layer.bias']
    
    for key in list(new_dict_['net'].keys()):
        if 'mlp' in key:
            del new_dict_['net'][key]
        if 'one_layer' in key:
            if (not 'task2' in key) and (not 'task1' in key) :
                del new_dict_['net'][key]
    for key in list(task2_dict.keys()):
        if 'two_layer_2' in key:
            del task2_dict[key]
    for key in list(task1_dict.keys()):
        if 'two_layer_2' in key:
                del task1_dict[key]

    return [task1_dict,task2_dict, new_dict_['net']]

def task4_model(args):
    model_path_ = args.paths
    new_dict_ = torch.load(model_path_, map_location=torch.device(args.device))
    for key in list(new_dict_['net'].keys()):
        if 'two_layer_1' in key:
            del new_dict_['net'][key]

    task1_dict = copy.deepcopy(new_dict_['net'])
    task1_dict['past_final_layer.weight'] = task1_dict['one_layer_task1.weight']
    task1_dict['past_final_layer.bias'] = task1_dict['one_layer_task1.bias']

    task2_dict = copy.deepcopy(new_dict_['net'])
    task2_dict['past_final_layer.weight'] = task2_dict['one_layer_task2.weight']
    task2_dict['past_final_layer.bias'] = task2_dict['one_layer_task2.bias']

    task3_dict = copy.deepcopy(new_dict_['net'])
    task3_dict['past_final_layer.weight'] = task3_dict['final_layer.weight']
    task3_dict['past_final_layer.bias'] = task3_dict['final_layer.bias']

    new_dict_['net']['one_layer_task3.weight'] = new_dict_['net']['final_layer.weight']
    new_dict_['net']['one_layer_task3.bias'] = new_dict_['net']['final_layer.bias']

    del new_dict_['net']['final_layer.weight']
    del new_dict_['net']['final_layer.bias']
    del task1_dict['final_layer.weight']
    del task1_dict['final_layer.bias']
    del task2_dict['final_layer.weight']
    del task2_dict['final_layer.bias']
    del task3_dict['final_layer.weight']
    del task3_dict['final_layer.bias']

    del new_dict_['net']['past_final_layer.weight']
    del new_dict_['net']['past_final_layer.bias']
    
    for key in list(new_dict_['net'].keys()):
        if 'mlp' in key:
            del new_dict_['net'][key]
        if 'one_layer' in key:
            if (not 'task2' in key) and (not 'task3' in key) and (not 'task1' in key):
                del new_dict_['net'][key]

    for key in list(task1_dict.keys()):
        if 'one_layer' in key:
            del task1_dict[key]
    for key in list(task2_dict.keys()):
        if 'one_layer' in key:
            del task2_dict[key]
    for key in list(task3_dict.keys()):
        if 'one_layer' in key:
            del task3_dict[key]
    return [task1_dict, task2_dict, task3_dict, new_dict_['net']]

def task5_model(args):
    model_path_ = args.paths
    new_dict_ = torch.load(model_path_, map_location=torch.device(args.device))

    for key in list(new_dict_['net'].keys()):
        if 'two_layer_1' in key:
            del new_dict_['net'][key]

    task1_dict = copy.deepcopy(new_dict_['net'])
    task1_dict['past_final_layer.weight'] = task1_dict['one_layer_task1.weight']
    task1_dict['past_final_layer.bias'] = task1_dict['one_layer_task1.bias']

    task2_dict = copy.deepcopy(new_dict_['net'])
    task2_dict['past_final_layer.weight'] = task2_dict['one_layer_task2.weight']
    task2_dict['past_final_layer.bias'] = task2_dict['one_layer_task2.bias']

    task3_dict = copy.deepcopy(new_dict_['net'])
    task3_dict['past_final_layer.weight'] = task3_dict['one_layer_task3.weight']
    task3_dict['past_final_layer.bias'] = task3_dict['one_layer_task3.bias']

    task4_dict = copy.deepcopy(new_dict_['net'])
    task4_dict['past_final_layer.weight'] = task4_dict['final_layer.weight']
    task4_dict['past_final_layer.bias'] = task4_dict['final_layer.bias']

    new_dict_['net']['one_layer_task4.weight'] = new_dict_['net']['final_layer.weight']
    new_dict_['net']['one_layer_task4.bias'] = new_dict_['net']['final_layer.bias']

    del new_dict_['net']['final_layer.weight']
    del new_dict_['net']['final_layer.bias']
    del task1_dict['final_layer.weight']
    del task1_dict['final_layer.bias']
    del task2_dict['final_layer.weight']
    del task2_dict['final_layer.bias']
    del task3_dict['final_layer.weight']
    del task3_dict['final_layer.bias']
    del task4_dict['final_layer.weight']
    del task4_dict['final_layer.bias']
    del new_dict_['net']['past_final_layer.weight']
    del new_dict_['net']['past_final_layer.bias']

    for key in list(new_dict_['net'].keys()):
        if 'mlp' in key:
            del new_dict_['net'][key]
        if 'one_layer' in key:
            if (not 'task1' in key) and (not 'task2' in key) and (not 'task3' in key) and (not 'task4' in key):
                del new_dict_['net'][key]

    for key in list(task1_dict.keys()):
        if 'one_layer' in key:
            del task1_dict[key]
    for key in list(task2_dict.keys()):
        if 'one_layer' in key:
            del task2_dict[key]
    for key in list(task3_dict.keys()):
        if 'one_layer' in key:
            del task3_dict[key]
    for key in list(task4_dict.keys()):
        if 'one_layer' in key:
            del task4_dict[key]
    return [task1_dict, task2_dict, task3_dict, task4_dict, new_dict_['net']]


def task6_model(args):
    model_path_ = args.paths
    new_dict_ = torch.load(model_path_, map_location=torch.device(args.device))

    for key in list(new_dict_['net'].keys()):
        if 'two_layer_1' in key:
            del new_dict_['net'][key]
    task1_dict = copy.deepcopy(new_dict_['net'])
    task1_dict['past_final_layer.weight'] = task1_dict['one_layer_task1.weight']
    task1_dict['past_final_layer.bias'] = task1_dict['one_layer_task1.bias']

    task2_dict = copy.deepcopy(new_dict_['net'])
    task2_dict['past_final_layer.weight'] = task2_dict['one_layer_task2.weight']
    task2_dict['past_final_layer.bias'] = task2_dict['one_layer_task2.bias']

    task3_dict = copy.deepcopy(new_dict_['net'])
    task3_dict['past_final_layer.weight'] = task3_dict['one_layer_task3.weight']
    task3_dict['past_final_layer.bias'] = task3_dict['one_layer_task3.bias']

    task4_dict = copy.deepcopy(new_dict_['net'])
    task4_dict['past_final_layer.weight'] = task4_dict['one_layer_task4.weight']
    task4_dict['past_final_layer.bias'] = task4_dict['one_layer_task4.bias']

    task5_dict = copy.deepcopy(new_dict_['net'])
    task5_dict['past_final_layer.weight'] = task5_dict['final_layer.weight']
    task5_dict['past_final_layer.bias'] = task5_dict['final_layer.bias']

    new_dict_['net']['one_layer_task5.weight'] = new_dict_['net']['final_layer.weight']
    new_dict_['net']['one_layer_task5.bias'] = new_dict_['net']['final_layer.bias']

    del new_dict_['net']['final_layer.weight']
    del new_dict_['net']['final_layer.bias']
    del task1_dict['final_layer.weight']
    del task1_dict['final_layer.bias']
    del task2_dict['final_layer.weight']
    del task2_dict['final_layer.bias']
    del task3_dict['final_layer.weight']
    del task3_dict['final_layer.bias']
    del task4_dict['final_layer.weight']
    del task4_dict['final_layer.bias']
    del task5_dict['final_layer.weight']
    del task5_dict['final_layer.bias']

    del new_dict_['net']['past_final_layer.weight']
    del new_dict_['net']['past_final_layer.bias']

    for key in list(new_dict_['net'].keys()):
        if '.mlp' in key:
            del new_dict_['net'][key]
        if 'one_layer' in key:
            if (not 'task1' in key) and (not 'task2' in key) and (not 'task3' in key) and (not 'task4' in key) and (not 'task5' in key):
                del new_dict_['net'][key]

    for key in list(task1_dict.keys()):
        if 'one_layer' in key:
            del task1_dict[key]
    for key in list(task2_dict.keys()):
        if 'one_layer' in key:
            del task2_dict[key]
    for key in list(task3_dict.keys()):
        if 'one_layer' in key:
            del task3_dict[key]
    for key in list(task4_dict.keys()):
        if 'one_layer' in key:
            del task4_dict[key]
    for key in list(task5_dict.keys()):
        if 'one_layer' in key:
            del task5_dict[key]
    return [task1_dict, task2_dict, task3_dict, task4_dict, task5_dict, new_dict_['net']]
