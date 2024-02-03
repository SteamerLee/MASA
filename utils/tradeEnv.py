#！/usr/bin/python
# -*- coding: utf-8 -*-#
'''
---------------------------------
 Name: tradeEnv.py  
 Description: Define the trading environment for the trading agent.
 Author: MASA
--------------------------------
'''
import numpy as np
import os
import pandas as pd
import time
import copy
from gym.utils import seeding
import gym
from gym import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from scipy.stats import entropy
import scipy.stats as spstats

class StockPortfolioEnv(gym.Env):

    def __init__(self, config, rawdata, mode, stock_num, action_dim, tech_indicator_lst, max_shares,
                 initial_asset=1000000, reward_scaling=1, norm_method='sum', transaction_cost=0.001, slippage=0.001, 
                 seed_num=2022, extra_data=None, mkt_observer=None):
        
        self.config = config
        self.rawdata = rawdata
        self.mode = mode # train, valid, test
        self.stock_num = stock_num # Number of stocks
        self.action_dim = action_dim # Number of assets
        self.tech_indicator_lst = tech_indicator_lst
        self.tech_indicator_lst_wocov = copy.deepcopy(self.tech_indicator_lst) # without cov feature
        if 'cov' in self.tech_indicator_lst_wocov:
            self.tech_indicator_lst_wocov.remove('cov')
        self.max_shares = max_shares # Maximum number of shares
        self.seed_num = seed_num 
        self.seed(seed=self.seed_num)
        self.epoch = 0
        self.curTradeDay = 0
        self.eps = 1e-6

        self.initial_asset = initial_asset # Initial portfolio value
        self.reward_scaling = reward_scaling 
        self.norm_method = norm_method
        self.transaction_cost = transaction_cost # 0.001
        self.slippage = slippage # 0.001 for one-side, 0.002 for two-side
        self.cur_slippage_drift = np.random.random(self.stock_num) * (self.slippage * 2) - self.slippage
        if extra_data is not None:
            self.extra_data = extra_data
        else:
            self.extra_data = None

        if self.norm_method == 'softmax':
            self.weights_normalization = self.softmax_normalization
        elif self.norm_method == 'sum':
            self.weights_normalization = self.sum_normalization
        else:
            raise ValueError("Unexpected normalization method of stock weights: {}".format(self.norm_method))
        if self.config.enable_cov_features:        
            self.state_dim = ((len(self.tech_indicator_lst_wocov)+self.stock_num) * self.stock_num) + 1 # +1: current portfolio value 
        else:
            self.state_dim = (len(self.tech_indicator_lst_wocov) * self.stock_num) + 1 # +1: current portfolio value
        if self.config.enable_market_observer:
            self.state_dim = self.state_dim + self.stock_num
            self.mkt_observer = mkt_observer
        else:
            self.mkt_observer = None
        
        if self.config.benchmark_algo in self.config.only_long_algo_lst:
            # Long only
            self.action_space = spaces.Box(low=0, high=1, shape=(self.action_dim, ))
            self.bound_flag = 1 # 1 for long and long+short, -1 for short
        else:
            if self.config.trade_pattern == 1:
                # Long only
                self.action_space = spaces.Box(low=0, high=1, shape=(self.action_dim, ))
                self.bound_flag = 1 # 1 for long and long+short, -1 for short
            else:
                raise ValueError("Unexpected trade pattern: {}".format(self.config.trade_pattern))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim, ))

        self.rawdata.sort_values(['date', 'stock'], ascending=True, inplace=True)
        self.rawdata.index = self.rawdata.date.factorize()[0]
        self.totalTradeDay = len(self.rawdata['date'].unique())
        self.stock_lst = np.sort(self.rawdata['stock'].unique())

        self.curData = copy.deepcopy(self.rawdata.loc[self.curTradeDay, :])
        self.curData.sort_values(['stock'], ascending=True, inplace=True)
        self.curData.reset_index(drop=True, inplace=True)

        if self.config.enable_cov_features:   
            self.covs = np.array(self.curData['cov'].values[0])
            self.state = np.append(self.covs, np.transpose(self.curData[self.tech_indicator_lst_wocov].values), axis=0)
        else:
            self.state = np.transpose(self.curData[self.tech_indicator_lst_wocov].values)
        self.state = self.state.flatten()
        self.state = np.append(self.state, [0], axis=0)
        self.ctl_state = {k:np.array(list(self.curData[k].values)) for k in self.config.otherRef_indicator_lst}
        self.terminal = False

        self.profit_lst = [0] # percentage of portfolio daily returns
        cur_risk_boundary, stock_ma_price = self.run_mkt_observer(stage='init') # after curData and state, before cur_risk_boundary
        if stock_ma_price is not None:
            self.ctl_state['MA-{}'.format(self.config.otherRef_indicator_ma_window)] = stock_ma_price
        self.cur_capital = self.initial_asset
        
        self.cvar_lst = [0]
        self.cvar_raw_lst = [0]

        self.asset_lst = [self.initial_asset] 
        self.date_memory = [self.curData['date'].unique()[0]]
        self.reward_lst = [0]
        self.action_cbf_memeory = [np.array([0] * self.stock_num)]

        self.actions_memory = [np.array([1/self.stock_num]*self.stock_num) * self.bound_flag] 
        self.action_rl_memory = [np.array([1/self.stock_num]*self.stock_num) * self.bound_flag]

        self.risk_adj_lst = [cur_risk_boundary]
        self.is_last_ctrl_solvable = False
        self.risk_raw_lst = [0] # For performance analysis. Record the risk without using risk controllrt during the validation/test period.
        self.risk_cbf_lst = [0]
        self.return_raw_lst = [self.initial_asset] 
        self.solver_stat = {'solvable': 0, 'insolvable': 0, 'stochastic_solvable': 0, 'stochastic_time': [], 'socp_solvable': 0, 'socp_time': []} 

        self.ctrl_weight_lst = [1.0]
        self.solvable_flag = []
        self.risk_pred_lst = []

        self.rl_reward_risk_lst = []
        self.rl_reward_profit_lst = []
        self.cnt1 = 0
        self.cnt2 = 0
        self.stepcount = 0

        risk_free = self.config.mkt_rf[self.config.market_name] / 100
        self.start_cputime = time.process_time()
        self.start_systime = time.perf_counter()
        if self.mode == 'train':
            self.exclusive_cputime = 0
            self.exclusive_systime = 0
        # For saveing profile
        self.profile_hist_field_lst = [            
            'ep', 'trading_days', 'annualReturn_pct', 'mdd', 'sharpeRatio', 'final_capital', 'volatility', 
            'calmarRatio', 'sterlingRatio',
            'netProfit', 'netProfit_pct', 'winRate',
            'vol_max', 'vol_min', 'vol_avg', 
            'risk_max', 'risk_min', 'risk_avg', 'riskRaw_max', 'riskRaw_min', 'riskRaw_avg',
            'dailySR_max', 'dailySR_min', 'dailySR_avg', 'dailySR_wocbf_max', 'dailySR_wocbf_min', 'dailySR_wocbf_avg',
            'dailyReturn_pct_max', 'dailyReturn_pct_min', 'dailyReturn_pct_avg',
            'sigReturn_max', 'sigReturn_min', 'mdd_high', 'mdd_low', 'mdd_high_date', 'mdd_low_date', 'sharpeRatio_wocbf',
            'reward_sum', 'final_capital_wocbf', 'cbf_contribution',
            'risk_downsideAtVol', 'risk_downsideAtVol_daily_max', 'risk_downsideAtVol_daily_min', 'risk_downsideAtVol_daily_avg',
            'risk_downsideAtValue_daily_max', 'risk_downsideAtValue_daily_min', 'risk_downsideAtValue_daily_avg',
            'cvar_max', 'cvar_min', 'cvar_avg', 'cvar_raw_max', 'cvar_raw_min', 'cvar_raw_avg',
            'solver_solvable', 'solver_insolvable', 'cputime', 'systime', 
        ]
        self.profile_hist_ep = {k: [] for k in self.profile_hist_field_lst}

    def step(self, actions):
        self.terminal = self.curTradeDay >= (self.totalTradeDay - 1)
        if self.terminal:
            self.cur_capital = self.cur_capital * (1 - self.transaction_cost)
            self.asset_lst[-1] = self.cur_capital
            self.profit_lst[-1] = (self.cur_capital - self.asset_lst[-2]) / self.asset_lst[-2]   
            if len(self.action_rl_memory) > 1:
                self.return_raw_lst[-1] = self.return_raw_lst[-1] * (1 - self.transaction_cost)

            if (self.config.enable_market_observer) and (self.mode == 'train'):
                # Training at the end of epoch
                ori_profit_rate = np.append([1], np.array(self.return_raw_lst)[1:] / np.array(self.return_raw_lst)[:-1], axis=0)
                adj_profit_rate = np.array(self.profit_lst) + 1
                label_kwargs = {'mode': self.mode, 'ori_profit': ori_profit_rate, 'adj_profit': adj_profit_rate, 'ori_risk': np.array(self.risk_raw_lst), 'adj_risk': np.array(self.risk_cbf_lst)}
                self.mkt_observer.train(**label_kwargs)       
            
            self.end_cputime = time.process_time()
            self.end_systime = time.perf_counter()
            self.model_save_flag = True
            invest_profile = self.get_results()
            self.save_profile(invest_profile=invest_profile)

            return self.state, self.reward, self.terminal, {}
        else:
            actions = np.reshape(actions, (-1)) # [1, num_of_stocks] or [num_of_stocks, ]
            weights = self.weights_normalization(actions=actions) # Unnormalized weights -> normalized weights 
            self.actions_memory.append(weights)
            if self.curTradeDay == 0:
                self.cur_capital = self.cur_capital * (1 - self.transaction_cost)
            else:
                cur_p = np.array(self.curData['close'].values) * (1 + self.cur_slippage_drift)
                last_p = np.array(self.lastDayData['close'].values) * (1 + self.last_slippage_drift)
                x_p = cur_p / last_p
                last_action = np.array(self.actions_memory[-2])
                x_p_adj = np.where((x_p>=2)&(last_action<0), 2, x_p)
                sgn = np.sign(last_action)
                # Check if loss the whole capital
                adj_w_ay = sgn * (last_action * (x_p_adj - 1) + np.abs(last_action))
                adj_cap = np.sum((x_p_adj - 1) * last_action) + 1
                if (adj_cap <= 0) or np.all(adj_w_ay==0):
                    raise ValueError("Loss the whole capital! [Day: {}, date: {}, adj_cap: {}, adj_w_ay: {}]".format(self.curTradeDay, self.date_memory[-1], adj_cap, adj_w_ay))
                last_w_adj = adj_w_ay / adj_cap
                self.cur_capital = self.cur_capital * (1 - (np.sum(np.abs(self.actions_memory[-1] - last_w_adj)) * self.transaction_cost))
                self.asset_lst[-1] = self.cur_capital
                self.profit_lst[-1] = (self.cur_capital - self.asset_lst[-2]) / self.asset_lst[-2]
                if len(self.action_rl_memory) > 1:
                    last_rl_action = np.array(self.action_rl_memory[-2])
                    sgn_rl = np.sign(last_rl_action)
                    prev_rl_cap = self.return_raw_lst[-1]
                    x_p_adjrl = np.where((x_p>=2)&(last_rl_action<0), 2, x_p)
                    adj_w_ay = sgn_rl * (last_rl_action * (x_p_adjrl - 1) + np.abs(last_rl_action))
                    adj_cap = np.sum((x_p_adjrl - 1) * last_rl_action) + 1
                    if (adj_cap <= 0) or np.all(adj_w_ay==0):
                        print("Loss the whole capital if using RL actions only! [Day: {}, date: {}, adj_cap: {}, adj_w_ay: {}]".format(self.curTradeDay, self.date_memory[-1], adj_cap, adj_w_ay))
                        adj_w_ay = np.array([1/self.stock_num]*self.stock_num) * self.bound_flag
                        adj_cap = 1
                    last_rlw_adj =  adj_w_ay / adj_cap 
                    return_raw = prev_rl_cap * (1 - (np.sum(np.abs(self.action_rl_memory[-1] - last_rlw_adj)) * self.transaction_cost))
                    self.return_raw_lst[-1] = return_raw

            # Jump to the next day
            self.curTradeDay = self.curTradeDay + 1
            self.lastDayData = self.curData
            self.last_slippage_drift = self.cur_slippage_drift
            self.curData = copy.deepcopy(self.rawdata.loc[self.curTradeDay, :])
            self.curData.sort_values(['stock'], ascending=True, inplace=True)
            self.curData.reset_index(drop=True, inplace=True)
            if self.config.enable_cov_features:   
                self.covs = np.array(self.curData['cov'].values[0])
                self.state = np.append(self.covs, np.transpose(self.curData[self.tech_indicator_lst_wocov].values), axis=0)
            else:
                self.state = np.transpose(self.curData[self.tech_indicator_lst_wocov].values)
            self.state = self.state.flatten()
            self.ctl_state = {k:np.array(list(self.curData[k].values)) for k in self.config.otherRef_indicator_lst} # State data for the controller
            cur_date = self.curData['date'].unique()[0]
            self.date_memory.append(cur_date)

            self.cur_slippage_drift = np.random.random(self.stock_num) * (self.slippage * 2) - self.slippage
            curDay_ClosePrice_withSlippage = np.array(self.curData['close'].values) * (1 + self.cur_slippage_drift)
            lastDay_ClosePrice_withSlippage = np.array(self.lastDayData['close'].values) * (1 + self.last_slippage_drift)
            rate_of_price_change = curDay_ClosePrice_withSlippage / lastDay_ClosePrice_withSlippage
            rate_of_price_change_adj = np.where((rate_of_price_change>=2)&(weights<0), 2, rate_of_price_change)
            sigDayReturn = (rate_of_price_change_adj - 1) * weights # [s1_pct, s2_pct, .., px_pct_returns]
            poDayReturn = np.sum(sigDayReturn)
            if poDayReturn <= (-1):
                raise ValueError("Loss the whole capital! [Day: {}, date: {}, poDayReturn: {}]".format(self.curTradeDay, self.date_memory[-1], poDayReturn))

            updatePoValue = self.cur_capital * (poDayReturn + 1) 
            poDayReturn_withcost = (updatePoValue - self.cur_capital) / self.cur_capital # Include the cost in the last timestamp

            self.cur_capital = updatePoValue
            self.state = np.append(self.state, [np.log(self.cur_capital/self.initial_asset)], axis=0) # current portfolio value observation
            
            self.profit_lst.append(poDayReturn_withcost) # Daily return
            self.asset_lst.append(self.cur_capital)

            # Receive info from the market observer
            cur_risk_boundary, stock_ma_price = self.run_mkt_observer(stage='run', rate_of_price_change=np.array([rate_of_price_change]))
            if stock_ma_price is not None:
                self.ctl_state['MA-{}'.format(self.config.otherRef_indicator_ma_window)] = stock_ma_price
            self.risk_adj_lst.append(cur_risk_boundary)
            self.ctrl_weight_lst.append(1.0)       

            daily_return_ay = np.array(list(self.curData['DAILYRETURNS-{}'.format(self.config.dailyRetun_lookback)].values))
            cur_cov = np.cov(daily_return_ay) 
            self.risk_cbf_lst.append(np.sqrt(np.matmul(np.matmul(weights, cur_cov), weights.T))) # Daily risk
            w_rl = self.action_rl_memory[-1] # weights - self.action_cbf_memeory[-1]
            w_rl = w_rl / np.sum(np.abs(w_rl))
            self.risk_raw_lst.append(np.sqrt(np.matmul(np.matmul(w_rl, cur_cov), w_rl.T)))

            if self.curTradeDay == 1:
                prev_rl_cap = self.return_raw_lst[-1] * (1 - self.transaction_cost)
            else:
                prev_rl_cap = self.return_raw_lst[-1]
            
            rate_of_price_change_adj_rawrl = np.where((rate_of_price_change>=2)&(w_rl<0), 2, rate_of_price_change)
            po_r_rl = np.sum((rate_of_price_change_adj_rawrl - 1) * w_rl)
            if po_r_rl <= (-1):
                raise ValueError("Loss the whole capital if using RL actions only! [Day: {}, date: {}, po_r_rl: {}]".format(self.curTradeDay, self.date_memory[-1], po_r_rl))
            return_raw = prev_rl_cap * (po_r_rl + 1) 
            self.return_raw_lst.append(return_raw)

            # CVaR
            expected_r_series = daily_return_ay[:, -21:]
            expected_r_prev = np.mean(expected_r_series[:, -1:], axis=1)
            expected_r_prev = np.where((expected_r_prev>=1)&(weights<0), 1, expected_r_prev)
            expected_r = np.sum(np.reshape(expected_r_prev, (1, -1)) @ np.reshape(weights, (-1, 1)))
            expected_cov = np.cov(expected_r_series)
            expected_std = np.sum(np.sqrt(np.reshape(weights, (1, -1)) @ expected_cov @ np.reshape(weights, (-1, 1))))
            cvar_lz = spstats.norm.ppf(1-0.05) # positive 1.65 for 95%(=1-alpha) confidence level.
            cvar_Z = np.exp(-0.5*np.power(cvar_lz, 2)) / 0.05 / np.sqrt(2*np.pi)
            cvar_expected = -expected_r + expected_std * cvar_Z
            self.cvar_lst.append(cvar_expected)

            # CVaR without risk controller
            expected_r_prevrl = np.mean(expected_r_series[:, -1:], axis=1)
            expected_r_prevrl = np.where((expected_r_prevrl>=1)&(w_rl<0), 1, expected_r_prevrl)
            expected_r_raw = np.sum(np.reshape(expected_r_prevrl, (1, -1)) @ np.reshape(w_rl, (-1, 1)))
            expected_std_raw = np.sum(np.sqrt(np.reshape(w_rl, (1, -1)) @ expected_cov @ np.reshape(w_rl, (-1, 1))))
            cvar_expected_raw = -expected_r_raw + expected_std_raw * cvar_Z
            self.cvar_raw_lst.append(cvar_expected_raw)

            profit_part = np.log(poDayReturn_withcost+1)
            if (self.config.trained_best_model_type == 'js_loss') and (self.config.enable_controller):
                # Action reward guiding mechanism
                if self.config.trade_pattern == 1:
                    weights_norm = weights
                    w_rl_norm = w_rl
                elif self.config.trade_pattern == 2:
                    # [-1, 1] -> [0, 1]
                    weights_norm = (weights + 1) / 2
                    w_rl_norm = (w_rl + 1) / 2
                elif self.config.trade_pattern == 3:
                    # [-1, 0] -> [0, 1]
                    weights_norm  = -weights
                    w_rl_norm = -w_rl
                else:
                    raise ValueError("Unexpected trade pattern: {}".format(self.config.trade_pattern))
                
                js_m = 0.5 * (w_rl_norm + weights_norm)
                js_divergence = (0.5 * entropy(pk=w_rl_norm, qk=js_m, base=2)) + (0.5 * entropy(pk=weights_norm, qk=js_m, base=2))
                js_divergence = np.clip(js_divergence, 0, 1)
                risk_part = (-1) * js_divergence

                scaled_profit_part = self.config.lambda_1 * profit_part               
                scaled_risk_part = self.config.lambda_2 * risk_part
                cur_reward = scaled_profit_part + scaled_risk_part

            elif (self.config.mode == 'RLonly') and (self.config.trained_best_model_type == 'pr_loss'):
                # overall return maximisation + risk minimisation
                cov_r_t0 = np.cov(self.ctl_state['DAILYRETURNS-{}'.format(self.config.dailyRetun_lookback)])
                risk_part = np.sqrt(np.matmul(np.matmul(np.array([weights]), cov_r_t0), np.array([weights]).T)[0][0])
                scaled_risk_part = (-1) * risk_part * 50
                scaled_profit_part = profit_part * self.config.lambda_1
                cur_reward = scaled_profit_part + scaled_risk_part

            elif (self.config.mode == 'RLonly') and (self.config.trained_best_model_type == 'sr_loss'):
                # Sharpe ratio maximisation
                cov_r_t0 = np.cov(self.ctl_state['DAILYRETURNS-{}'.format(self.config.dailyRetun_lookback)])
                risk_part = np.sqrt(np.matmul(np.matmul(np.array([weights]), cov_r_t0), np.array([weights]).T)[0][0])
                profit_part = poDayReturn_withcost
                scaled_profit_part = profit_part
                scaled_risk_part = risk_part
                cur_reward = (scaled_profit_part - (self.config.mkt_rf[self.config.market_name] * 0.01)) / scaled_risk_part

            else:
                risk_part = 0
                scaled_risk_part = 0
                scaled_profit_part = profit_part * self.config.lambda_1
                cur_reward = scaled_profit_part + scaled_risk_part

            self.rl_reward_risk_lst.append(scaled_risk_part)
            self.rl_reward_profit_lst.append(scaled_profit_part)
            self.reward = cur_reward
            self.reward_lst.append(self.reward)
            self.model_save_flag = False
            return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.epoch = self.epoch + 1
        self.curTradeDay = 0

        self.curData = copy.deepcopy(self.rawdata.loc[self.curTradeDay, :])
        self.curData.sort_values(['stock'], ascending=True, inplace=True)
        self.curData.reset_index(drop=True, inplace=True)
        if self.config.enable_cov_features:   
            self.covs = np.array(self.curData['cov'].values[0])
            self.state = np.append(self.covs, np.transpose(self.curData[self.tech_indicator_lst_wocov].values), axis=0)
        else:
            self.state = np.transpose(self.curData[self.tech_indicator_lst_wocov].values)
        self.state = self.state.flatten()
        self.state = np.append(self.state, [0], axis=0)
        self.ctl_state = {k:np.array(list(self.curData[k].values)) for k in self.config.otherRef_indicator_lst} 
        self.terminal = False

        self.profit_lst = [0] 
        cur_risk_boundary, stock_ma_price = self.run_mkt_observer(stage='reset')  
        if stock_ma_price is not None:
            self.ctl_state['MA-{}'.format(self.config.otherRef_indicator_ma_window)] = stock_ma_price

        self.cur_capital = self.initial_asset


        self.cvar_lst = [0]
        self.cvar_raw_lst = [0]

        self.asset_lst = [self.initial_asset] 

        self.actions_memory = [np.array([1/self.stock_num]*self.stock_num) * self.bound_flag]
        self.date_memory = [self.curData['date'].unique()[0]]
        self.reward_lst = [0]
        self.action_cbf_memeory = [np.array([0] * self.stock_num)]
        self.action_rl_memory = [np.array([1/self.stock_num]*self.stock_num) * self.bound_flag]

        self.risk_adj_lst = [cur_risk_boundary]
        self.is_last_ctrl_solvable = False
        self.risk_raw_lst = [0]
        self.risk_cbf_lst = [0]
        self.return_raw_lst = [self.initial_asset]
        self.solver_stat = {'solvable': 0, 'insolvable': 0, 'stochastic_solvable': 0, 'stochastic_time': [], 'socp_solvable': 0, 'socp_time': []} 

        self.ctrl_weight_lst = [1.0]
        self.solvable_flag = []
        self.risk_pred_lst = []

        self.rl_reward_risk_lst = []
        self.rl_reward_profit_lst = []
        self.cnt1 = 0
        self.cnt2 = 0
        self.stepcount = 0

        self.start_cputime = time.process_time()
        self.start_systime = time.perf_counter()
        return self.state

    def render(self, mode='human'):
        return self.state
    

    def softmax_normalization(self, actions):
        if np.sum(np.abs(actions)) == 0:  
            norm_weights = np.array([1/len(actions)]*len(actions)) * self.bound_flag
        else:
            norm_weights = np.exp(actions)/np.sum(np.abs(np.exp(actions)))
        return norm_weights
    
    def sum_normalization(self, actions):
        if np.sum(np.abs(actions)) == 0:
            norm_weights = np.array([1/len(actions)]*len(actions)) * self.bound_flag
        else:
            norm_weights = actions / np.sum(np.abs(actions))
        return norm_weights

    def save_action_memory(self):

        action_pd = pd.DataFrame(np.array(self.actions_memory), columns=self.stock_lst)
        action_pd['date'] = self.date_memory
        return action_pd

    def seed(self, seed=2022):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs


    def get_results(self):
        self.profit_lst = np.array(self.profit_lst)
        self.asset_lst = np.array(self.asset_lst)

        netProfit = self.cur_capital - self.initial_asset # Profits
        netProfit_pct = netProfit / self.initial_asset # Rate of overall returns

        diffPeriodAsset = np.diff(self.asset_lst)
        sigReturn_max = np.max(diffPeriodAsset) # Maximal returns in a single transaction.
        sigReturn_min = np.min(diffPeriodAsset) # Minimal returns in a single transaction

        # Annual Returns
        annualReturn_pct = np.power((1 + netProfit_pct), (self.config.tradeDays_per_year/len(self.asset_lst))) - 1

        dailyReturn_pct_max = np.max(self.profit_lst)
        dailyReturn_pct_min = np.min(self.profit_lst)
        avg_dailyReturn_pct = np.mean(self.profit_lst)
        # strategy volatility
        volatility = np.sqrt(np.sum(np.power((self.profit_lst - avg_dailyReturn_pct), 2)) * self.config.tradeDays_per_year / (len(self.profit_lst) - 1))

        # SR_Vol, Long-term risk
        sharpeRatio = ((annualReturn_pct * 100) - self.config.mkt_rf[self.config.market_name])/ (volatility * 100)
        # sharpeRatio = np.max([sharpeRatio, 0])

        dailyAnnualReturn_lst = np.power((1+np.array(self.profit_lst)), self.config.tradeDays_per_year) - 1
        dailyRisk_lst = np.array(self.risk_cbf_lst) * np.sqrt(self.config.tradeDays_per_year) # Daily Risk to Anuual Risk
        dailySR = ((dailyAnnualReturn_lst[1:] * 100) - self.config.mkt_rf[self.config.market_name]) / (dailyRisk_lst[1:] * 100)
        dailySR = np.append(0, dailySR)
        # dailySR = np.where(dailySR < 0, 0, dailySR)
        dailySR_max = np.max(dailySR)
        dailySR_min = np.min(dailySR[dailySR!=0])
        dailySR_avg = np.mean(dailySR)

        # For performance analysis
        dailyReturnRate_wocbf = np.diff(self.return_raw_lst)/np.array(self.return_raw_lst)[:-1]
        dailyReturnRate_wocbf = np.append(0, dailyReturnRate_wocbf)
        dailyAnnualReturn_wocbf_lst = np.power((1+dailyReturnRate_wocbf), self.config.tradeDays_per_year) - 1
        dailyRisk_wocbf_lst = np.array(self.risk_raw_lst) * np.sqrt(self.config.tradeDays_per_year)  
        dailySR_wocbf = ((dailyAnnualReturn_wocbf_lst[1:] * 100) - self.config.mkt_rf[self.config.market_name]) / (dailyRisk_wocbf_lst[1:] * 100)
        dailySR_wocbf = np.append(0, dailySR_wocbf)
        # dailySR_wocbf = np.where(dailySR_wocbf < 0, 0, dailySR_wocbf)
        dailySR_wocbf_max = np.max(dailySR_wocbf)
        dailySR_wocbf_min = np.min(dailySR_wocbf[dailySR_wocbf!=0])
        dailySR_wocbf_avg = np.mean(dailySR_wocbf)

        annualReturn_wocbf_pct = np.power((1 + ((self.return_raw_lst[-1] - self.initial_asset) / self.initial_asset)), (self.config.tradeDays_per_year/len(self.return_raw_lst))) - 1
        volatility_wocbf = np.sqrt((np.sum(np.power((dailyReturnRate_wocbf - np.mean(dailyReturnRate_wocbf)), 2)) * self.config.tradeDays_per_year / (len(self.return_raw_lst) - 1)))
        sharpeRatio_woCBF = ((annualReturn_wocbf_pct * 100) - self.config.mkt_rf[self.config.market_name])/ (volatility_wocbf * 100)
        sharpeRatio_woCBF = np.max([sharpeRatio_woCBF, 0])

        winRate = len(np.argwhere(diffPeriodAsset>0))/(len(diffPeriodAsset) + 1)

        # MDD
        repeat_asset_lst = np.tile(self.asset_lst, (len(self.asset_lst), 1))
        mdd_mtix = np.triu(1 - repeat_asset_lst / np.reshape(self.asset_lst, (-1, 1)), k=1)
        mddmaxidx = np.argmax(mdd_mtix)
        mdd_highidx = mddmaxidx // len(self.asset_lst)
        mdd_lowidx = mddmaxidx % len(self.asset_lst)
        self.mdd = np.max(mdd_mtix)
        self.mdd_high = self.asset_lst[mdd_highidx]
        self.mdd_low = self.asset_lst[mdd_lowidx]
        self.mdd_highTimepoint = self.date_memory[mdd_highidx]
        self.mdd_lowTimepoint = self.date_memory[mdd_lowidx]

        # Strategy volatility during trading
        cumsum_r = np.cumsum(self.profit_lst)/np.arange(1, self.totalTradeDay+1) # average cumulative returns rate
        repeat_profit_lst = np.tile(self.profit_lst, (len(self.profit_lst), 1))
        stg_vol_lst = np.sqrt(np.sum(np.power(np.tril(repeat_profit_lst - np.reshape(cumsum_r, (-1,1)), k=0), 2), axis=1)[1:] / np.arange(1, len(repeat_profit_lst)) * self.config.tradeDays_per_year)
        stg_vol_lst = np.append([0], stg_vol_lst, axis=0)
        # stg_vol_lst  = np.sqrt((np.cumsum(np.power((self.profit_lst - cumsum_r), 2))/np.arange(1, self.totalTradeDay+1)) * self.config.tradeDays_per_year)

        vol_max = np.max(stg_vol_lst)
        vol_min = np.min(np.array(stg_vol_lst)[np.array(stg_vol_lst)!=0])
        vol_avg = np.mean(stg_vol_lst)

        # short-term risk
        risk_max = np.max(self.risk_cbf_lst)
        risk_min = np.min(np.array(self.risk_cbf_lst)[np.array(self.risk_cbf_lst)!=0])
        risk_avg = np.mean(self.risk_cbf_lst)

        risk_raw_max = np.max(self.risk_raw_lst)
        risk_raw_min = np.min(np.array(self.risk_raw_lst)[np.array(self.risk_raw_lst)!=0])
        risk_raw_avg = np.mean(self.risk_raw_lst)

        # Downside risk at volatility        
        risk_downsideAtVol_daily = np.sqrt(np.sum(np.power(np.tril((repeat_profit_lst - np.reshape(cumsum_r, (-1,1))) * (repeat_profit_lst<np.reshape(cumsum_r, (-1,1))), k=0), 2), axis=1)[1:] / np.arange(1, len(repeat_profit_lst)) * self.config.tradeDays_per_year)
        risk_downsideAtVol_daily = np.append([0], risk_downsideAtVol_daily, axis=0)
        risk_downsideAtVol = risk_downsideAtVol_daily[-1]
        risk_downsideAtVol_daily_max = np.max(risk_downsideAtVol_daily)
        risk_downsideAtVol_daily_min = np.min(risk_downsideAtVol_daily)
        risk_downsideAtVol_daily_avg = np.mean(risk_downsideAtVol_daily)

        # Downside risk at value against initial capital
        risk_downsideAtValue_daily = (self.asset_lst / self.initial_asset) - 1
        risk_downsideAtValue_daily_max = np.max(risk_downsideAtValue_daily)
        risk_downsideAtValue_daily_min = np.min(risk_downsideAtValue_daily)
        risk_downsideAtValue_daily_avg = np.mean(risk_downsideAtValue_daily)

        # CVaR curve
        cvar_max = np.max(self.cvar_lst)
        cvar_min = np.min(np.array(self.cvar_lst)[np.array(self.cvar_lst)!=0])
        cvar_avg = np.mean(self.cvar_lst)

        cvar_raw_max = np.max(self.cvar_raw_lst)
        cvar_raw_min = np.min(np.array(self.cvar_raw_lst)[np.array(self.cvar_raw_lst)!=0])
        cvar_raw_avg = np.mean(self.cvar_raw_lst)

        # Calmar ratio
        time_T = len(self.profit_lst)
        avg_return = netProfit_pct / time_T
        variance_r = np.sum(np.power((self.profit_lst - avg_dailyReturn_pct), 2)) / (len(self.profit_lst) - 1)
        volatility_daily = np.sqrt(variance_r)
 
        if netProfit_pct > 0:
            shrp = avg_return / volatility_daily
            calmarRatio = (time_T * np.power(shrp, 2)) / (0.63519 + 0.5 * np.log(time_T) + np.log(shrp))
        elif netProfit_pct == 0:
            calmarRatio = (netProfit_pct) / (1.2533 * volatility_daily * np.sqrt(time_T))
        else:
            # netProfit_pct < 0
            calmarRatio = (netProfit_pct) / (-(avg_return * time_T) - (variance_r / avg_return))

        # Sterling ratio
        move_mdd_mask = np.where(np.array(self.profit_lst)<0, 1, 0)
        moving_mdd = np.sqrt(np.sum(np.power(self.profit_lst * move_mdd_mask, 2))  * self.config.tradeDays_per_year / (len(self.profit_lst) - 1))
        sterlingRatio =  ((annualReturn_pct * 100) - self.config.mkt_rf[self.config.market_name]) / (moving_mdd * 100)

        if self.mode == 'train':
            cputime_use = self.end_cputime - self.start_cputime - self.exclusive_cputime
            systime_use = self.end_systime - self.start_systime - self.exclusive_systime
        else:
            cputime_use = self.end_cputime - self.start_cputime
            systime_use = self.end_systime - self.start_systime

        if np.shape(np.array(self.actions_memory)) != (self.totalTradeDay, self.stock_num):
            if (self.config.mode =='RLcontroller') and (self.config.enable_controller):
                raise ValueError('actions_memory shape error in the RLcontroller mode')
            else:
                self.actions_memory = np.ones((self.totalTradeDay, self.stock_num)) * (1/self.stock_num) * self.bound_flag
        if np.shape(np.array(self.action_rl_memory)) != (self.totalTradeDay+1, self.stock_num):
            if (self.config.mode =='RLcontroller') and (self.config.enable_controller):
                raise ValueError('action_rl_memory shape error in the RLcontroller mode')
            else:
                self.action_rl_memory = np.ones((self.totalTradeDay+1, self.stock_num)) * (1/self.stock_num) * self.bound_flag
        if np.shape(np.array(self.action_cbf_memeory)) != (self.totalTradeDay+1, self.stock_num):
            if (self.config.mode =='RLcontroller') and (self.config.enable_controller):
                raise ValueError('action_cbf_memeory shape error in the RLcontroller mode')
            else:
                self.action_cbf_memeory = np.zeros((self.totalTradeDay+1, self.stock_num))
        if len(self.solvable_flag) == 0:
            self.solvable_flag = np.zeros(len(self.asset_lst))
        if len(self.risk_pred_lst) == 0:
            self.risk_pred_lst = np.zeros(len(self.asset_lst))
  
        cbf_abssum_contribution = np.sum(np.abs(self.action_cbf_memeory[:-1]))

        info_dict = {
            'ep': self.epoch, 'trading_days': self.totalTradeDay, 'annualReturn_pct': annualReturn_pct, 'volatility': volatility, 'sharpeRatio': sharpeRatio, 'sharpeRatio_wocbf': sharpeRatio_woCBF,
            'mdd': self.mdd, 'calmarRatio': calmarRatio, 'sterlingRatio': sterlingRatio, 'netProfit': netProfit, 'netProfit_pct': netProfit_pct, 'winRate': winRate,
            'vol_max': vol_max, 'vol_min': vol_min, 'vol_avg': vol_avg,
            'risk_max': risk_max, 'risk_min': risk_min, 'risk_avg': risk_avg,
            'riskRaw_max': risk_raw_max, 'riskRaw_min': risk_raw_min, 'riskRaw_avg': risk_raw_avg,
            'dailySR_max': dailySR_max, 'dailySR_min': dailySR_min, 'dailySR_avg': dailySR_avg, 'dailySR_wocbf_max': dailySR_wocbf_max, 'dailySR_wocbf_min': dailySR_wocbf_min, 'dailySR_wocbf_avg': dailySR_wocbf_avg,
            'dailyReturn_pct_max': dailyReturn_pct_max, 'dailyReturn_pct_min': dailyReturn_pct_min, 'dailyReturn_pct_avg': avg_dailyReturn_pct,
            'sigReturn_max': sigReturn_max, 'sigReturn_min': sigReturn_min, 
            'mdd_high': self.mdd_high, 'mdd_low': self.mdd_low, 'mdd_high_date': self.mdd_highTimepoint, 'mdd_low_date': self.mdd_lowTimepoint, 
            'final_capital': self.cur_capital, 'reward_sum': np.sum(self.reward_lst),
            'final_capital_wocbf': self.return_raw_lst[-1], 
            'cbf_contribution': cbf_abssum_contribution,
            'risk_downsideAtVol': risk_downsideAtVol, 'risk_downsideAtVol_daily_max': risk_downsideAtVol_daily_max, 'risk_downsideAtVol_daily_min': risk_downsideAtVol_daily_min, 'risk_downsideAtVol_daily_avg': risk_downsideAtVol_daily_avg,
            'risk_downsideAtValue_daily_max': risk_downsideAtValue_daily_max, 'risk_downsideAtValue_daily_min': risk_downsideAtValue_daily_min, 'risk_downsideAtValue_daily_avg': risk_downsideAtValue_daily_avg,
            'cvar_max': cvar_max, 'cvar_min': cvar_min, 'cvar_avg': cvar_avg, 'cvar_raw_max': cvar_raw_max, 'cvar_raw_min': cvar_raw_min, 'cvar_raw_avg': cvar_raw_avg,
            'solver_solvable': self.solver_stat['solvable'], 'solver_insolvable': self.solver_stat['insolvable'], 'cputime': cputime_use, 'systime': systime_use,
            'asset_lst': copy.deepcopy(self.asset_lst), 'daily_return_lst': copy.deepcopy(self.profit_lst), 'reward_lst': copy.deepcopy(self.reward_lst), 
            'stg_vol_lst': copy.deepcopy(stg_vol_lst), 'risk_lst': copy.deepcopy(self.risk_cbf_lst), 'risk_wocbf_lst': copy.deepcopy(self.risk_raw_lst),
            'capital_wocbf_lst': copy.deepcopy(self.return_raw_lst), 'daily_sr_lst': copy.deepcopy(dailySR), 'daily_sr_wocbf_lst': copy.deepcopy(dailySR_wocbf),
            'risk_adj_lst': copy.deepcopy(self.risk_adj_lst), 'ctrl_weight_lst': copy.deepcopy(self.ctrl_weight_lst), 
            'solvable_flag': copy.deepcopy(self.solvable_flag), 'risk_pred_lst': copy.deepcopy(self.risk_pred_lst),
            'final_action_abssum_lst': copy.deepcopy(np.sum(np.abs(np.array(self.actions_memory)), axis=1)), 
            'rl_action_abssum_lst': copy.deepcopy(np.sum(np.abs(np.array(self.action_rl_memory)), axis=1)[:-1]), 
            'cbf_action_abssum_lst': copy.deepcopy(np.sum(np.abs(np.array(self.action_cbf_memeory)), axis=1)[:-1]), 
            'daily_downsideAtVol_risk_lst': copy.deepcopy(risk_downsideAtVol_daily), 'daily_downsideAtValue_risk_lst': copy.deepcopy(risk_downsideAtValue_daily),
            'cvar_lst': copy.deepcopy(self.cvar_lst), 'cvar_raw_lst': copy.deepcopy(self.cvar_raw_lst),
        }

        return info_dict

    def save_profile(self, invest_profile):
        # basic data
        for fname in self.profile_hist_field_lst:
            if fname in list(invest_profile.keys()):
                self.profile_hist_ep[fname].append(invest_profile[fname])
            else:
                raise ValueError('Cannot find the field [{}] in invest profile..'.format(fname))
        phist_df = pd.DataFrame(self.profile_hist_ep, columns=self.profile_hist_field_lst)
        phist_df.to_csv(os.path.join(self.config.res_dir, '{}_profile.csv'.format(self.mode)), index=False)

        cputime_avg = np.mean(phist_df['cputime'])
        systime_avg = np.mean(phist_df['systime'])

        bestmodel_dict = {}
        if self.config.trained_best_model_type == 'max_capital':
            field_name = 'final_capital'
            v = np.max(phist_df[field_name]) # Please noted that the maximum value will be recorded.
        elif 'loss' in self.config.trained_best_model_type:
            field_name = 'reward_sum'
            v = np.max(phist_df[field_name]) # Please noted that the maximum value will be recorded.
        elif self.config.trained_best_model_type == 'sharpeRatio':
            field_name = 'sharpeRatio'
            v = np.max(phist_df[field_name]) # Please noted that the maximum value will be recorded.
        elif self.config.trained_best_model_type == 'volatility':
            field_name = 'volatility'
            v = np.min(phist_df[field_name]) # Please noted that the minimum value will be recorded.
        elif self.config.trained_best_model_type == 'mdd':
            field_name = 'mdd'
            v = np.min(phist_df[field_name]) # Please noted that the minimum value will be recorded.
        else:
            raise ValueError('Unknown implementation with the best model type [{}]..'.format(self.config.trained_best_model_type))
        v_ep = list(phist_df[phist_df[field_name]==v]['ep'])[0]
        bestmodel_dict['{}_ep'.format(self.config.trained_best_model_type)] = v_ep
        bestmodel_dict[self.config.trained_best_model_type] = v
        
        if True:
            print("-"*30)
            # log_str = "Mode: {}, Ep: {}, Current epoch capital: {}, historical best captial ({} ep): {}, cputime cur: {} s, avg: {} s, system time cur: {} s/ep, avg: {} s/ep..".format(self.mode, self.epoch, self.cur_capital, v_ep, v, np.round(np.array(phist_df['cputime'])[-1], 2), np.round(cputime_avg, 2), np.round(np.array(phist_df['systime'])[-1], 2), np.round(systime_avg, 2))
            log_str = "Mode: {}, Ep: {}, Current epoch capital: {}, historical best captial ({} ep): {} | solvable: {}, insolvable: {} | step count: {} | cputime cur: {} s, avg: {} s, system time cur: {} s/ep, avg: {} s/ep..".format(self.mode, self.epoch, self.cur_capital, v_ep, v, np.array(phist_df['solver_solvable'])[-1], np.array(phist_df['solver_insolvable'])[-1], self.stepcount, np.round(np.array(phist_df['cputime'])[-1], 2), np.round(cputime_avg, 2), np.round(np.array(phist_df['systime'])[-1], 2), np.round(systime_avg, 2))
            print(log_str)
        bestmodel_df = pd.DataFrame([bestmodel_dict])
        bestmodel_df.to_csv(os.path.join(self.config.res_dir, '{}_bestmodel.csv'.format(self.mode)), index=False)

        # save data of each step in 1st/best/last model
        fpath = os.path.join(self.config.res_dir, '{}_stepdata.csv'.format(self.mode))
        if not os.path.exists(fpath):
            step_data = {'capital_policy_1': invest_profile['asset_lst'], 'dailyReturn_policy_1': invest_profile['daily_return_lst'],
                        'reward_policy_1': invest_profile['reward_lst'], 'strategyVolatility_policy_1': invest_profile['stg_vol_lst'],
                        'risk_policy_1': invest_profile['risk_lst'], 'risk_wocbf_policy_1': invest_profile['risk_wocbf_lst'], 'capital_wocbf_policy_1': invest_profile['capital_wocbf_lst'],
                        'dailySR_policy_1': invest_profile['daily_sr_lst'], 'dailySR_wocbf_policy_1': invest_profile['daily_sr_wocbf_lst'], 
                        'riskAccepted_policy_1': invest_profile['risk_adj_lst'],
                        'ctrlWeight_policy_1': invest_profile['ctrl_weight_lst'],
                        'solvable_flag_policy_1': invest_profile['solvable_flag'],
                        'risk_pred_policy_1': invest_profile['risk_pred_lst'],
                        'final_action_abssum_policy_1': invest_profile['final_action_abssum_lst'],
                        'rl_action_abssum_policy_1': invest_profile['rl_action_abssum_lst'],
                        'cbf_action_abssum_policy_1': invest_profile['cbf_action_abssum_lst'],
                        'downsideAtVol_risk_policy_1': invest_profile['daily_downsideAtVol_risk_lst'],
                        'downsideAtValue_risk_policy_1': invest_profile['daily_downsideAtValue_risk_lst'],
                        'cvar_policy_1': invest_profile['cvar_lst'], 'cvar_raw_policy_1': invest_profile['cvar_raw_lst'],
                        }
            step_data = pd.DataFrame(step_data)
        else:
            step_data = pd.DataFrame(pd.read_csv(fpath, header=0))
            
        if bestmodel_dict['{}_ep'.format(self.config.trained_best_model_type)] == invest_profile['ep']:
            step_data['capital_policy_best'] = invest_profile['asset_lst']
            step_data['dailyReturn_policy_best'] = invest_profile['daily_return_lst']
            step_data['reward_policy_best'] = invest_profile['reward_lst']
            step_data['strategyVolatility_policy_best'] = invest_profile['stg_vol_lst']  
            step_data['risk_policy_best'] = invest_profile['risk_lst']
            step_data['risk_wocbf_policy_best'] = invest_profile['risk_wocbf_lst']
            step_data['capital_wocbf_policy_best'] = invest_profile['capital_wocbf_lst']
            step_data['dailySR_policy_best'] = invest_profile['daily_sr_lst']
            step_data['dailySR_wocbf_policy_best'] = invest_profile['daily_sr_wocbf_lst']
            step_data['riskAccepted_policy_best'] = invest_profile['risk_adj_lst']
            step_data['ctrlWeight_policy_best'] = invest_profile['ctrl_weight_lst']
            step_data['solvable_flag_policy_best'] = invest_profile['solvable_flag']
            step_data['risk_pred_policy_best'] = invest_profile['risk_pred_lst']
            step_data['final_action_abssum_policy_best'] = invest_profile['final_action_abssum_lst']
            step_data['rl_action_abssum_policy_best'] = invest_profile['rl_action_abssum_lst']
            step_data['cbf_action_abssum_policy_best'] = invest_profile['cbf_action_abssum_lst']
            step_data['downsideAtVol_risk_policy_best'] = invest_profile['daily_downsideAtVol_risk_lst']
            step_data['downsideAtValue_risk_policy_best'] = invest_profile['daily_downsideAtValue_risk_lst']
            step_data['cvar_policy_best'] = invest_profile['cvar_lst']
            step_data['cvar_raw_policy_best'] = invest_profile['cvar_raw_lst']
        # Record the test set performance on valid_best_policy
        if self.mode == 'test':
            valid_fpath = os.path.join(self.config.res_dir, 'valid_bestmodel.csv')
            if os.path.exists(valid_fpath):
                valid_records = pd.DataFrame(pd.read_csv(valid_fpath, header=0))
                if int(valid_records['{}_ep'.format(self.config.trained_best_model_type)][0]) == invest_profile['ep']:
                    step_data['capital_policy_validbest'] = invest_profile['asset_lst']
                    step_data['dailyReturn_policy_validbest'] = invest_profile['daily_return_lst'] # Plot the scatter plot of efficient frontier.
                    step_data['reward_policy_validbest'] = invest_profile['reward_lst']
                    step_data['strategyVolatility_policy_validbest'] = invest_profile['stg_vol_lst']
                    step_data['risk_policy_validbest'] = invest_profile['risk_lst'] # Plot the scatter plot of efficient frontier.
                    step_data['risk_wocbf_policy_validbest'] = invest_profile['risk_wocbf_lst']
                    step_data['capital_wocbf_policy_validbest'] = invest_profile['capital_wocbf_lst']
                    step_data['dailySR_policy_validbest'] = invest_profile['daily_sr_lst']
                    step_data['dailySR_wocbf_policy_validbest'] = invest_profile['daily_sr_wocbf_lst']
                    step_data['riskAccepted_policy_validbest'] = invest_profile['risk_adj_lst']
                    step_data['ctrlWeight_policy_validbest'] = invest_profile['ctrl_weight_lst']
                    step_data['solvable_flag_policy_validbest'] = invest_profile['solvable_flag']
                    step_data['risk_pred_policy_validbest'] = invest_profile['risk_pred_lst']
                    step_data['final_action_abssum_policy_validbest'] = invest_profile['final_action_abssum_lst']
                    step_data['rl_action_abssum_policy_validbest'] = invest_profile['rl_action_abssum_lst']
                    step_data['cbf_action_abssum_policy_validbest'] = invest_profile['cbf_action_abssum_lst']           
                    step_data['downsideAtVol_risk_policy_validbest'] = invest_profile['daily_downsideAtVol_risk_lst']
                    step_data['downsideAtValue_risk_policy_validbest'] = invest_profile['daily_downsideAtValue_risk_lst']
                    step_data['cvar_policy_validbest'] = invest_profile['cvar_lst']
                    step_data['cvar_raw_policy_validbest'] = invest_profile['cvar_raw_lst']
                print("-"*30)
                log_str = "Mode: Best-{}, Ep: {}, Capital (test set, by using the best validation model, {} ep): {} ".format(self.mode, self.epoch, int(valid_records['{}_ep'.format(self.config.trained_best_model_type)][0]), np.array(step_data['capital_policy_validbest'])[-1])
                print(log_str)

        if invest_profile['ep'] == self.config.num_epochs:
            step_data['capital_policy_last'] = invest_profile['asset_lst']
            step_data['dailyReturn_policy_last'] = invest_profile['daily_return_lst']
            step_data['reward_policy_last'] = invest_profile['reward_lst']
            step_data['strategyVolatility_policy_last'] = invest_profile['stg_vol_lst']
            step_data['risk_policy_last'] = invest_profile['risk_lst']
            step_data['risk_wocbf_policy_last'] = invest_profile['risk_wocbf_lst']
            step_data['capital_wocbf_policy_last'] = invest_profile['capital_wocbf_lst']
            step_data['dailySR_policy_last'] = invest_profile['daily_sr_lst']
            step_data['dailySR_wocbf_policy_last'] = invest_profile['daily_sr_wocbf_lst']
            step_data['riskAccepted_policy_last'] = invest_profile['risk_adj_lst']  
            step_data['ctrlWeight_policy_last'] = invest_profile['ctrl_weight_lst']   
            step_data['solvable_flag_policy_last'] = invest_profile['solvable_flag'] 
            step_data['risk_pred_policy_last'] = invest_profile['risk_pred_lst']
            step_data['final_action_abssum_policy_last'] = invest_profile['final_action_abssum_lst']
            step_data['rl_action_abssum_policy_last'] = invest_profile['rl_action_abssum_lst']
            step_data['cbf_action_abssum_policy_last'] = invest_profile['cbf_action_abssum_lst']   
            step_data['downsideAtVol_risk_policy_last'] = invest_profile['daily_downsideAtVol_risk_lst']
            step_data['downsideAtValue_risk_policy_last'] = invest_profile['daily_downsideAtValue_risk_lst']
            step_data['cvar_policy_last'] = invest_profile['cvar_lst']
            step_data['cvar_raw_policy_last'] = invest_profile['cvar_raw_lst']
        step_data.to_csv(fpath, index=False)


    def run_mkt_observer(self, stage=None, rate_of_price_change=None):
        cur_date = self.curData['date'].unique()[0]
        if self.config.enable_market_observer:
            if stage in ['reset', 'init'] and (self.mode == 'train'):
                self.mkt_observer.reset()

            finemkt_feat = self.extra_data['fine_market']
            ma_close = finemkt_feat[finemkt_feat['date']==cur_date][['mkt_{}_close'.format(self.config.finefreq), 'mkt_{}_ma'.format(self.config.finefreq)]].values[-1]
            mkt_cur_close_price = ma_close[0]
            mkt_ma_price = ma_close[1]
            finemkt_feat = finemkt_feat[finemkt_feat['date']==cur_date][self.config.finemkt_feat_cols_lst].values
            finemkt_feat = np.reshape(finemkt_feat, (len(self.config.use_features), self.config.fine_window_size)) # -> (features, window_size)
            finemkt_feat = np.expand_dims(finemkt_feat, axis=0) # -> (batch=1, features, window_size)
            if (rate_of_price_change is not None) and (self.mode == 'train'):
                if mkt_cur_close_price > self.mkt_last_close_price:
                    mkt_direction = 0
                elif mkt_cur_close_price < self.mkt_last_close_price:
                    mkt_direction = 2
                else:
                    mkt_direction = 1
                mkt_direction = np.array([mkt_direction])
                self.mkt_observer.update_hidden_vec_reward(mode=self.mode, rate_of_price_change=rate_of_price_change, mkt_direction=mkt_direction)

            finestock_feat = self.extra_data['fine_stock']
            stock_cur_close_price = finestock_feat[finestock_feat['date']==cur_date]['stock_{}_close'.format(self.config.finefreq)].values # (num_of_stock, )
            stock_ma_price = finestock_feat[finestock_feat['date']==cur_date]['stock_{}_ma'.format(self.config.finefreq)].values # (num_of_stock, )
            if self.config.is_gen_dc_feat:
                dc_events = finestock_feat[finestock_feat['date']==cur_date]['stock_{}_dc'.format(self.config.finefreq)].values # (num_of_stocks, )
            else:
                dc_events = None
            
            finestock_feat = finestock_feat[finestock_feat['date']==cur_date][self.config.finestock_feat_cols_lst].values
            finestock_feat = np.reshape(finestock_feat, (self.config.topK, len(self.config.use_features), self.config.fine_window_size)) # -> (num_of_stocks, features, window_size)
            finestock_feat = np.transpose(finestock_feat, (1, 0, 2)) # -> (features, num_of_stocks, window_size)
            finestock_feat = np.expand_dims(finestock_feat, axis=0) # -> (batch=1, features, num_of_stocks, window_size)
            input_kwargs = {'mode': self.mode, 'stock_ma_price': np.array([stock_ma_price]), 'stock_cur_close_price': np.array([stock_cur_close_price]), 'dc_events': np.array([dc_events])}

            cur_hidden_vector_ay, lambda_val, sigma_val = self.mkt_observer.predict(finemkt_feat=finemkt_feat, finestock_feat=finestock_feat, **input_kwargs) # lambda_val: not applicable
            if self.config.is_enable_dynamic_risk_bound:
                if int(sigma_val[-1]) == 0:
                    # up
                    cur_risk_boundary = self.config.risk_up_bound
                elif int(sigma_val[-1]) == 1:
                    # hold
                    cur_risk_boundary = self.config.risk_hold_bound
                elif int(sigma_val[-1]) == 2:
                    # down
                    cur_risk_boundary = self.config.risk_down_bound
                else:
                    raise ValueError('Unknown sigma value [{}]..'.format(sigma_val[-1]))

            else:
                cur_risk_boundary = self.config.risk_default
            
            self.state = np.append(self.state, cur_hidden_vector_ay[-1], axis=0)
            self.mkt_last_close_price = mkt_cur_close_price
        else:
            cur_risk_boundary = self.config.risk_default
            if self.config.mode == 'RLcontroller':
                finestock_feat = self.extra_data['fine_stock']
                stock_cur_close_price = finestock_feat[finestock_feat['date']==cur_date]['stock_{}_close'.format(self.config.finefreq)].values # (num_of_stock, )
                stock_ma_price = finestock_feat[finestock_feat['date']==cur_date]['stock_{}_ma'.format(self.config.finefreq)].values # (num_of_stock, )
            else:
                stock_ma_price = None

        return cur_risk_boundary, stock_ma_price

class StockPortfolioEnv_cash(StockPortfolioEnv):
    # Considering cash item
    def step(self, actions):
        self.terminal = self.curTradeDay >= (self.totalTradeDay - 1)
        if self.terminal:
            self.cur_capital = self.cur_capital * (1 - (np.sum(np.abs(self.actions_memory[-1])) * self.transaction_cost))
            self.asset_lst[-1] = self.cur_capital
            self.profit_lst[-1] = (self.cur_capital - self.asset_lst[-2]) / self.asset_lst[-2]     
            if len(self.action_rl_memory) > 1:
                self.return_raw_lst[-1] = self.return_raw_lst[-1] * (1 - (np.sum(np.abs(self.action_rl_memory[-1])) * self.transaction_cost))

            if (self.config.enable_market_observer) and (self.mode == 'train'):
                # Training at the end of epoch
                ori_profit_rate = np.append([1], np.array(self.return_raw_lst)[1:] / np.array(self.return_raw_lst)[:-1], axis=0)
                adj_profit_rate = np.array(self.profit_lst) + 1
                label_kwargs = {'ori_profit': ori_profit_rate, 'adj_profit': adj_profit_rate, 'ori_risk': np.array(elf.risk_raw_lst), 'adj_risk': np.array(self.risk_cbf_lst)}
                self.mkt_observer.train(**label_kwargs)

            self.end_cputime = time.process_time()
            self.end_systime = time.perf_counter()
            self.model_save_flag = True
            invest_profile = self.get_results()
            self.save_profile(invest_profile=invest_profile)

            return self.state, self.reward, self.terminal, {}
        else:
            actions = np.reshape(actions, (-1)) # [1, num_of_stocks] or [num_of_stocks, ]
            weights = self.weights_normalization(actions=actions) # Unnormalized weights -> normalized weights 
            self.actions_memory.append(weights[1:]) 
            if self.curTradeDay == 0:
                self.cur_capital = self.cur_capital * (1 - (1-1/len(weights)) * self.transaction_cost)
            else:
                cur_p = np.array(self.curData['close'].values) * (1 + self.cur_slippage_drift)
                last_p = np.array(self.lastDayData['close'].values) * (1 + self.last_slippage_drift)
                x_p = cur_p / last_p
                last_action = np.array(self.actions_memory[-2])
                last_action = np.append([1.0 - np.sum(np.abs(last_action))], last_action, axis=0) # cash
                x_p_adj = np.where((x_p>=2)&(last_action[1:]<0), 2, x_p)
                x_p_adj = np.append([1.0], x_p_adj, axis=0) # cash
                sgn = np.sign(last_action)
                sgn[0] = 1.0 # cash sign
                adj_w_ay = sgn * (last_action * (x_p_adj - 1) + np.abs(last_action))
                adj_cap = np.sum((x_p_adj - 1) * last_action) + 1
                if (adj_cap <= 0) or np.all(adj_w_ay==0):
                    raise ValueError("Loss the whole capital! [Day: {}, date: {}, adj_cap: {}, adj_w_ay: {}]".format(self.curTradeDay, self.date_memory[-1], adj_cap, adj_w_ay))
                last_w_adj = adj_w_ay / adj_cap
                self.cur_capital = self.cur_capital * (1 - (np.sum(np.abs(self.actions_memory[-1] - last_w_adj[1:])) * self.transaction_cost))
                self.asset_lst[-1] = self.cur_capital
                self.profit_lst[-1] = (self.cur_capital - self.asset_lst[-2]) / self.asset_lst[-2]
                if len(self.action_rl_memory) > 1:
                    last_rl_action = np.array(self.action_rl_memory[-2])
                    last_rl_action = np.append([1.0 - np.sum(np.abs(last_rl_action))], last_rl_action, axis=0) # cash
                    x_p_adjrl = np.where((x_p>=2)&(last_rl_action[1:]<0), 2, x_p)
                    sgn_rl = np.sign(last_rl_action)
                    sgn_rl[0] = 1.0 # cash sign
                    prev_rl_cap = self.return_raw_lst[-1]
                    adj_w_ay = sgn_rl * (last_rl_action * (x_p_adjrl - 1) + np.abs(last_rl_action))
                    adj_cap = np.sum((x_p_adjrl - 1) * last_rl_action) + 1
                    if (adj_cap <= 0) or np.all(adj_w_ay==0):
                        print("Loss the whole capital if using RL actions only! [Day: {}, date: {}, adj_cap: {}, adj_w_ay: {}]".format(self.curTradeDay, self.date_memory[-1], adj_cap, adj_w_ay))
                        adj_w_ay = np.array([1/(self.stock_num+1)]*(self.stock_num+1)) * self.bound_flag
                        adj_cap = 1
                    last_rlw_adj =  adj_w_ay / adj_cap
                    return_raw = prev_rl_cap * (1 - (np.sum(np.abs(self.action_rl_memory[-1] - last_rlw_adj[1:])) * self.transaction_cost))
                    self.return_raw_lst[-1] = return_raw       
            
            # Jump to the next day
            self.curTradeDay = self.curTradeDay + 1
            self.lastDayData = self.curData       
            self.last_slippage_drift = self.cur_slippage_drift     
            self.curData = copy.deepcopy(self.rawdata.loc[self.curTradeDay, :])
            self.curData.sort_values(['stock'], ascending=True, inplace=True)
            self.curData.reset_index(drop=True, inplace=True)
            if self.config.enable_cov_features:   
                self.covs = np.array(self.curData['cov'].values[0])
                self.state = np.append(self.covs, np.transpose(self.curData[self.tech_indicator_lst_wocov].values), axis=0)
            else:
                self.state = np.transpose(self.curData[self.tech_indicator_lst_wocov].values)
            self.state = self.state.flatten()
            self.ctl_state = {k:np.array(list(self.curData[k].values)) for k in self.config.otherRef_indicator_lst} # State data for the controller
            cur_date = self.curData['date'].unique()[0]
            self.date_memory.append(cur_date)

            self.cur_slippage_drift = np.random.random(self.stock_num) * (self.slippage * 2) - self.slippage
            curDay_ClosePrice_withSlippage = np.array(self.curData['close'].values) * (1 + self.cur_slippage_drift)
            lastDay_ClosePrice_withSlippage = np.array(self.lastDayData['close'].values) * (1 + self.last_slippage_drift)
            rate_of_price_change = curDay_ClosePrice_withSlippage / lastDay_ClosePrice_withSlippage
            rate_of_price_change_adj = np.where((rate_of_price_change>=2)&(weights[1:]<0), 2, rate_of_price_change)
            sigDayReturn = (rate_of_price_change_adj - 1) * weights[1:] # [s1_pct, s2_pct, .., px_pct_returns]
            poDayReturn = np.sum(sigDayReturn)
            if poDayReturn <= (-1):
                raise ValueError("Loss the whole capital! [Day: {}, date: {}, poDayReturn: {}]".format(self.curTradeDay, self.date_memory[-1], poDayReturn))
            
            updatePoValue = self.cur_capital * ((poDayReturn + 1 - np.abs(weights[0])) + np.abs(weights[0]))
            poDayReturn_withcost = (updatePoValue - self.cur_capital) / self.cur_capital  
            
            self.cur_capital = updatePoValue
            self.state = np.append(self.state, [np.log((self.cur_capital/self.initial_asset))], axis=0) # current portfolio value observation
            
            self.profit_lst.append(poDayReturn_withcost) 
            self.asset_lst.append(self.cur_capital)

            # Receive info from the market observer
            rate_of_price_change_withcash = np.append([1.0], rate_of_price_change_adj, axis=0) # cash
            cur_risk_boundary, stock_ma_price = self.run_mkt_observer(stage='run', rate_of_price_change=np.array([rate_of_price_change_withcash]))  
            if stock_ma_price is not None:
                self.ctl_state['MA-{}'.format(self.config.otherRef_indicator_ma_window)] = stock_ma_price
            self.risk_adj_lst.append(cur_risk_boundary)
            self.ctrl_weight_lst.append(1.0) 

            # For debugging
            daily_return_ay = np.array(list(self.curData['DAILYRETURNS-{}'.format(self.config.dailyRetun_lookback)].values))
            cur_cov = np.cov(daily_return_ay) 
            self.risk_cbf_lst.append(np.sqrt(np.matmul(np.matmul(weights[1:], cur_cov), weights[1:].T)))
            w_rl = self.action_rl_memory[-1] # Not applicable # weights[1:] - self.action_cbf_memeory[-1]
            w_rl = w_rl / np.sum(np.abs(w_rl))
            self.risk_raw_lst.append(np.sqrt(np.matmul(np.matmul(w_rl, cur_cov), w_rl.T)))

            if self.curTradeDay == 1:
                prev_rl_cap = self.return_raw_lst[-1] * (1 - (1-1/len(weights)) * self.transaction_cost)
            else:
                prev_rl_cap = self.return_raw_lst[-1]

            rate_of_price_change_adj_rawrl = np.where((rate_of_price_change>=2)&(w_rl<0), 2, rate_of_price_change)
            return_raw = prev_rl_cap * ((np.sum((rate_of_price_change_adj_rawrl - 1) * w_rl) + 1 - np.abs(weights[0])) + np.abs(weights[0]))
            if return_raw <= 0:
                raise ValueError("Loss the whole capital if using RL actions only! [Day: {}, date: {}, return_raw: {}]".format(self.curTradeDay, self.date_memory[-1], return_raw))
            self.return_raw_lst.append(return_raw)

            # CVaR
            expected_r_series = daily_return_ay[:, -21:]
            expected_r_prev = np.mean(expected_r_series[:, -1:], axis=1)
            expected_r_prev = np.where((expected_r_prev>=1)&(weights[1:]<0), 1, expected_r_prev)
            expected_r = np.sum(np.reshape(expected_r_prev, (1, -1)) @ np.reshape(weights[1:], (-1, 1)))
            expected_cov = np.cov(expected_r_series)
            expected_std = np.sum(np.sqrt(np.reshape(weights[1:], (1, -1)) @ expected_cov @ np.reshape(weights[1:], (-1, 1))))
            cvar_lz = spstats.norm.ppf(1-0.05) # positive 1.65 for 95%(=1-alpha) confidence level.
            cvar_Z = np.exp(-0.5*np.power(cvar_lz, 2)) / 0.05 / np.sqrt(2*np.pi)
            cvar_expected = -expected_r + expected_std * cvar_Z
            self.cvar_lst.append(cvar_expected)

            # CVaR without risk controller
            expected_r_prevrl = np.mean(expected_r_series[:, -1:], axis=1)
            expected_r_prevrl = np.where((expected_r_prevrl>=1)&(w_rl<0), 1, expected_r_prevrl)
            expected_r_raw = np.sum(np.reshape(expected_r_prevrl, (1, -1)) @ np.reshape(w_rl, (-1, 1)))
            expected_std_raw = np.sum(np.sqrt(np.reshape(w_rl, (1, -1)) @ expected_cov @ np.reshape(w_rl, (-1, 1))))
            cvar_expected_raw = -expected_r_raw + expected_std_raw * cvar_Z
            self.cvar_raw_lst.append(cvar_expected_raw)

            profit_part = np.log(poDayReturn_withcost+1)
            if (self.config.trained_best_model_type == 'js_loss') and (self.config.enable_controller):
                # Action reward guiding mechanism
                if self.config.trade_pattern == 1:
                    weights_norm = weights
                    w_rl_norm = w_rl
                elif self.config.trade_pattern == 2:
                    # [-1, 1] -> [0, 1]
                    weights_norm = (weights + 1) / 2
                    w_rl_norm = (w_rl + 1) / 2
                elif self.config.trade_pattern == 3:
                    # [-1, 0] -> [0, 1]
                    weights_norm  = -weights
                    w_rl_norm = -w_rl
                else:
                    raise ValueError("Unexpected trade pattern: {}".format(self.config.trade_pattern))
                
                js_m = 0.5 * (w_rl_norm + weights_norm[1:])
                js_divergence = (0.5 * entropy(pk=w_rl_norm, qk=js_m, base=2)) + (0.5 * entropy(pk=weights_norm[1:], qk=js_m, base=2))
                js_divergence = np.clip(js_divergence, 0, 1)
                risk_part = (-1) * js_divergence

                scaled_risk_part = self.config.lambda_2 * risk_part
                scaled_profit_part = self.config.lambda_1 * profit_part         
                cur_reward = scaled_profit_part + scaled_risk_part      

            elif (self.config.mode == 'RLonly') and (self.config.trained_best_model_type == 'pr_loss'):
                # overall return maximisation + risk minimisation
                cov_r_t0 = np.cov(self.ctl_state['DAILYRETURNS-{}'.format(self.config.dailyRetun_lookback)])
                risk_part = np.sqrt(np.matmul(np.matmul(np.array([weights[1:]]), cov_r_t0), np.array([weights[1:]]).T)[0][0])
                scaled_risk_part = (-1) * risk_part * 50
                scaled_profit_part = profit_part * self.config.lambda_1
                cur_reward = scaled_profit_part + scaled_risk_part

            elif (self.config.mode == 'RLonly') and (self.config.trained_best_model_type == 'sr_loss'):
                # Sharpe ratio maximisation                
                cov_r_t0 = np.cov(self.ctl_state['DAILYRETURNS-{}'.format(self.config.dailyRetun_lookback)])
                risk_part = np.sqrt(np.matmul(np.matmul(np.array([weights[1:]]), cov_r_t0), np.array([weights[1:]]).T)[0][0])
                profit_part = poDayReturn_withcost
                scaled_profit_part = profit_part
                scaled_risk_part = risk_part
                cur_reward = (scaled_profit_part - (self.config.mkt_rf[self.config.market_name] * 0.01)) / scaled_risk_part

            else:
                risk_part = 0
                scaled_risk_part = 0
                scaled_profit_part = profit_part * self.config.lambda_1
                cur_reward = scaled_profit_part + scaled_risk_part

            self.rl_reward_risk_lst.append(scaled_risk_part)
            self.rl_reward_profit_lst.append(scaled_profit_part)
            self.reward = cur_reward
            self.reward_lst.append(self.reward)
            self.model_save_flag = False

            return self.state, self.reward, self.terminal, {}
