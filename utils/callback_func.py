#！/usr/bin/python
# -*- coding: utf-8 -*-#
'''
---------------------------------
 Name: callback_func.py  
 Author: MASA
--------------------------------
'''
import numpy as np
import os
import pandas as pd
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from .model_pool import model_select
import sys
sys.path.append('..')
from RL_controller.controllers import RL_withoutController, RL_withController

class PoCallback(BaseCallback):

    def __init__(self, config, train_env, valid_env=None, test_env=None, verbose=0):
        super(PoCallback, self).__init__(verbose)
        self.train_env = train_env
        self.valid_env = valid_env
        self.test_env = test_env
        self.config = config
        if self.config.mode == 'RLonly': 
            self.risk_controller = RL_withoutController
        elif self.config.mode == 'RLcontroller':
            self.risk_controller = RL_withController
        else:
            raise ValueError("Unexpected mode [{}]..".format(self.config.mode))
    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        # Save model
        if self.train_env.model_save_flag:
            exclusive_start_cputime = time.process_time()
            exclusive_start_systime = time.perf_counter()
            curmpath = os.path.join(self.config.res_model_dir, 'current_model')
            self.model.save(curmpath)
            # Evaluate model in validation set and test set
            ModelCls = model_select(model_name=self.config.rl_model_name,  mode=self.config.mode)
            trained_model = ModelCls.load(curmpath)
            if self.valid_env is not None:
                obs_valid = self.valid_env.reset()
                while True:
                    a_rlonly, _ = trained_model.predict(obs_valid)
                    a_rlonly = np.reshape(a_rlonly, (-1))
                    a_rl = a_rlonly
                    if np.sum(np.abs(a_rl)) == 0:
                        a_rl = np.array([1/len(a_rl)]*len(a_rl))
                    else:
                        a_rl = a_rl / np.sum(np.abs(a_rl))
                    a_final = self.risk_controller(a_rl=a_rl, env=self.valid_env)
                    a_final = a_final / np.sum(np.abs(a_final))
                    a_final = np.array([a_final])
                    obs_valid, rewards, terminal_flag, _ = self.valid_env.step(a_final) 
                    if terminal_flag:
                        break    
            
                cur_ep = self.valid_env.epoch # self.valid_env.epoch is the epoch number before reset().
                env_type = 'valid'
                fpath = os.path.join(self.config.res_dir, '{}_bestmodel.csv'.format(env_type))
                model_records = pd.DataFrame(pd.read_csv(fpath, header=0))
                if cur_ep == int(model_records['{}_ep'.format(self.config.trained_best_model_type)][0]):
                    mpath = os.path.join(self.config.res_model_dir, '{}_{}'.format(env_type, self.config.trained_best_model_type))
                    trained_model.save(mpath)
                
            if self.test_env is not None:
                obs_test = self.test_env.reset()
                while True:
                    a_rlonly, _ = trained_model.predict(obs_test)
                    a_rlonly = np.reshape(a_rlonly, (-1))
                    a_rl = a_rlonly
                    if np.sum(np.abs(a_rl)) == 0:
                        a_rl = np.array([1/len(a_rl)]*len(a_rl))
                    else:
                        a_rl = a_rl / np.sum(np.abs(a_rl))
                    a_final  = self.risk_controller(a_rl=a_rl, env=self.test_env)
                    a_final = a_final / np.sum(np.abs(a_final))
                    a_final = np.array([a_final])
                    obs_test, rewards, terminal_flag, _ = self.test_env.step(a_final) 
                    if terminal_flag:
                        break

            del trained_model
            # delete the current model file
            os.remove(os.path.join(self.config.res_model_dir, 'current_model.zip'))
            exclusive_end_cputime = time.process_time()
            exclusive_end_systime = time.perf_counter()
            self.train_env.exclusive_cputime = exclusive_end_cputime - exclusive_start_cputime
            self.train_env.exclusive_systime = exclusive_end_systime - exclusive_start_systime
        self.train_env.model_save_flag = False
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
