#！/usr/bin/python
# -*- coding: utf-8 -*-#

'''
---------------------------------
 Name:         model_pool.py
 Description:  API for algorithm selection 
 Author:       MASA
---------------------------------
'''

import sys
sys.path.append('..')
from stable_baselines3 import TD3
from RL_controller.TD3_controller import TD3Controller

def model_select(model_name, mode):
    if mode == 'RLonly':
        model_dict = {
            'TD3': TD3,
        }
    elif mode == 'RLcontroller':
        model_dict = {
            'TD3': TD3Controller,
        }
    else:
        raise ValueError('Unexpected mode [{}]..'.format(mode))
    
    try:
        model_cls = model_dict[model_name]
    except:
        raise ValueError("Cannot find the model [{}] in the registration list, please add it to utils.model_pool.py".format(model_name))
    return model_cls
    
def benchmark_algo_select(model_name):
    # Register the CLASS of the benchmark algorithms here
    # Example: model_dict = {'PPN': PPN}
    model_dict = {
    }
    try:
        model_cls = model_dict[model_name]
    except:
        raise ValueError("Cannot find the benchmark model [{}] in the registration list, please add it to utils.model_pool.py".format(model_name))
    return model_cls


