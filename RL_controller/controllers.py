#！/usr/bin/python
# -*- coding: utf-8 -*-#

'''
---------------------------------
 Name:         controllers.py
 Description:  Implement the solver-based agent of the proposed MASA framework.
 Author:       MASA
---------------------------------
'''

import numpy as np
import pandas as pd
import time
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
import cvxpy as cp
from scipy.linalg import sqrtm
import scipy.stats as spstats
def RL_withoutController(a_rl, env=None):
    a_cbf = np.array([0]*env.stock_num)
    a_rl = np.array(a_rl)
    env.action_cbf_memeory.append(a_cbf)
    env.action_rl_memory.append(a_rl)
    a_final = a_rl + a_cbf
    return a_final

def RL_withController(a_rl, env=None):
    a_rl = np.array(a_rl)
    env.action_rl_memory.append(a_rl)
    if env.config.pricePredModel == 'MA':
        pred_prices_change = get_pred_price_change(env=env)
        pred_dict = {'shortterm': pred_prices_change}
    else:
        raise ValueError("Cannot find the price prediction model [{}]..".format(env.config.pricePredModel))
    a_cbf, is_solvable_status = cbf_opt(env=env, a_rl=a_rl, pred_dict=pred_dict) 
    cur_dcm_weight= 1.0 
    cur_rl_weight = 1.0
    if is_solvable_status:   
        a_cbf_weighted = a_cbf * cur_dcm_weight
        env.action_cbf_memeory.append(a_cbf_weighted)
        a_rl_weighted = a_rl * cur_rl_weight 
        a_final = a_rl_weighted + a_cbf_weighted
    else:
        env.action_cbf_memeory.append(np.array([0]*env.stock_num))
        a_final = a_rl
    return a_final

def get_pred_price_change(env):
    ma_lst = env.ctl_state['MA-{}'.format(env.config.otherRef_indicator_ma_window)]
    pred_prices = ma_lst
    cur_close_price = np.array(env.curData['close'].values)
    pred_prices_change = (pred_prices - cur_close_price) / cur_close_price 
    return pred_prices_change

def cbf_opt(env, a_rl, pred_dict):
    """
    The risk constraint is based on controller barrier function (CBF) method. Not just considering the satisfaction of the current risk constraint, but also considering the trends/gradients of the future risk. 
    """
    pred_prices_change = pred_dict['shortterm']

    a_rl = np.array(a_rl)
    assert np.sum(np.abs(a_rl)) - 1 < 0.0001, "The sum of a_rl is not equal to 1, the value is {}, a_rl list: {}".format(np.sum(a_rl), a_rl)
    N = env.stock_num
    # Past N days daily return rate of each stock, (num_of_stocks, lookback_days), [[t-N+1, t-N+2, .., t-1, t]]
    daily_return_ay = env.ctl_state['DAILYRETURNS-{}'.format(env.config.dailyRetun_lookback)]
    
    cov_r_t0 = np.cov(daily_return_ay)
    w_t0 = np.array([env.actions_memory[-1]])
    try:
        risk_stg_t0 = np.sqrt(np.matmul(np.matmul(w_t0, cov_r_t0), w_t0.T)[0][0])
    except Exception as error:
        print("Risk-(MV model variance): {}".format(error))
        risk_stg_t0 = 0
    risk_market_t0 = env.config.risk_market
    if len(env.risk_adj_lst) <= 1:
        risk_safe_t0 = env.risk_adj_lst[-1]
    else:
        if env.is_last_ctrl_solvable:
            risk_safe_t0 = env.risk_adj_lst[-2]
        else:
            risk_safe_t0 = risk_stg_t0 + risk_market_t0

    gamma = env.config.cbf_gamma
    risk_market_t1 = env.config.risk_market  
    risk_safe_t1 = env.risk_adj_lst[-1] 

    pred_prices_change_reshape = np.reshape(pred_prices_change, (-1, 1))
    r_t1 = np.append(daily_return_ay[:, 1:], pred_prices_change_reshape, axis=1)

    cov_r_t1 = np.cov(r_t1)
    cov_sqrt_t1 = sqrtm(cov_r_t1)
    cov_sqrt_t1 = cov_sqrt_t1.real
    G_ay = np.array([]).reshape(-1, N)

    h_0 = np.array([])
    
    use_cvxopt_threshold = 10 # using cvxopt tool will be faster when the size of portfolio is less than or equal to 10. Otherwise, the cvxpy tool will be faster.
    w_lb = 0
    w_ub = 1

    if env.config.topK <= use_cvxopt_threshold:
        # Implemented by cvxopt
        A_eq = np.array([]).reshape(-1, N)
        linear_g1 = np.array([[1.0] * N]) # (1, N)
        A_eq = np.append(A_eq, linear_g1, axis=0)
        A_eq = matrix(A_eq)
        b_eq = np.array([0.0])
        b_eq = matrix(b_eq)

        h_0 = np.append(h_0, a_rl, axis=0) # linear_h3, 0 <= (a_RL + a_cbf)
        h_0 = np.append(h_0, 1-a_rl, axis=0) # linear_h4 (a_RL + a_cbf) <= 1

        linear_g3 = np.diag([-1.0] * N)
        G_ay = np.append(G_ay, linear_g3, axis=0) # 0 <= (a_RL + a_cbf)
        linear_g4 = np.diag([1.0] * N) 
        G_ay = np.append(G_ay, linear_g4, axis=0) # (a_RL + a_cbf) <= 1
    
    else:
        a_rl_re_sign = np.reshape(a_rl, (-1, 1))
        sign_mul = np.ones((1, N))
        w_lb_sign = w_lb
        w_ub_sign = w_ub
        
    last_h_risk = (-risk_market_t0 - risk_stg_t0 + risk_safe_t0)
    last_h_risk = np.max([last_h_risk, 0.0])
    socp_d = -risk_market_t1 + risk_safe_t1 + (gamma - 1) * last_h_risk

    step_add_lst = [0.002, 0.002, 0.002, 0.002, 0.002, 0.005, 0.005, 0.005, 0.005, 0.005]
    cnt = 1
    if env.config.is_enable_dynamic_risk_bound:
        cnt_th = env.config.ars_trial # Iterative risk relaxation
    else:
        cnt_th = 1 

    if env.config.topK <= use_cvxopt_threshold:
        # Implemented by cvxopt
        socp_b = np.matmul(cov_sqrt_t1, a_rl)
        h = np.append(h_0, [socp_d], axis=0) # socp_d
        h = np.append(h, socp_b, axis=0) # socp_b
        h = matrix(h)
        socp_cx = np.array([[0.0] * N])
        G_ay = np.append(G_ay, -socp_cx, axis=0)
        G_ay = np.append(G_ay, -cov_sqrt_t1, axis=0) # socp_ax
        G = matrix(G_ay) # G = matrix(np.transpose(np.transpose(G_ay)))

        linear_eq_num = 2*N
        dims = {'l': linear_eq_num, 'q': [N+1], 's': []}
        QP_P = matrix(np.eye(N)) * 2 # (1/2) xP'x
        QP_Q = matrix(np.zeros((N, 1))) # q'x
        while cnt <= cnt_th:
            try:
                sol = solvers.coneqp(QP_P, QP_Q, G, h, dims, A_eq, b_eq)
                if sol['status'] == 'optimal':
                    solver_flag = True
                    break
                else:
                    raise
            except:
                solver_flag = False
                cnt += 1
                risk_safe_t1 = risk_safe_t1 + step_add_lst[cnt-2]
                socp_d = -risk_market_t1 + risk_safe_t1 + (gamma - 1) * (-risk_market_t0 - risk_stg_t0 + risk_safe_t0)
                h = np.append(h_0, [socp_d], axis=0) # socp_d
                h = np.append(h, socp_b, axis=0) # socp_b 
                h = matrix(h)

        if solver_flag:
            if sol['status'] == 'optimal':
                a_cbf = np.reshape(np.array(sol['x']), -1)
                env.solver_stat['solvable'] = env.solver_stat['solvable'] + 1
                is_solvable_status = True
                env.risk_adj_lst[-1] = risk_safe_t1
                # Check the solution whether satisfy the risk constraint.
                cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl+a_cbf), cov_r_t1), (a_rl+a_cbf).T))
                assert (cur_alpha_risk - socp_d) <= 0.00001, 'cur risk: {}, socp_d {}'.format(cur_alpha_risk, socp_d)
                assert np.abs(np.sum(np.abs((a_rl+a_cbf))) - 1) <= 0.00001, 'sum of actions: {} \n{} \n{}'.format(np.sum(np.abs((a_rl+a_cbf))), a_rl, a_cbf)
                env.solvable_flag.append(0)
            else:
                a_cbf = np.zeros(N)
                env.solver_stat['insolvable'] = env.solver_stat['insolvable'] + 1
                is_solvable_status = False
                cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl), cov_r_t1), (a_rl).T)) 
                env.solvable_flag.append(1)           
        else:
            a_cbf = np.zeros(N)
            env.solver_stat['insolvable'] = env.solver_stat['insolvable'] + 1
            is_solvable_status = False
            cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl), cov_r_t1), (a_rl).T))
            env.solvable_flag.append(1)
            # print("Failed to solve the problem.")

    else:
        # Complete solver
        # ++ Implemented by cvxpy
        cp_x = cp.Variable((N, 1))
        a_rl_re = np.reshape(a_rl, (-1, 1))
        cp_constraint = []
        cp_constraint.append(cp.sum(sign_mul@cp_x) + cp.sum(a_rl_re_sign) == 1)
        cp_constraint.append(a_rl_re + cp_x >= w_lb_sign) 
        cp_constraint.append(a_rl_re + cp_x <= w_ub_sign) 
        cp_constraint.append(cp.SOC(socp_d, cov_sqrt_t1 @ (a_rl_re + cp_x))) 
        # cvxpy
        while cnt <= cnt_th:
            try:
                obj_f2 = cp.sum_squares(cp_x)
                cp_obj = cp.Minimize(obj_f2)
                cp_prob = cp.Problem(cp_obj, cp_constraint)
                cp_prob.solve(solver=cp.ECOS, verbose=False) 

                if cp_prob.status == 'optimal':
                    solver_flag = True
                    break
                else:
                    raise
            except:
                solver_flag = False
                cnt += 1
                risk_safe_t1 = risk_safe_t1 + step_add_lst[cnt-2]
                socp_d = -risk_market_t1 + risk_safe_t1 + (gamma - 1) * last_h_risk
                cp_constraint[-1] = cp.SOC(socp_d, cov_sqrt_t1 @ (a_rl_re + cp_x))

        if (cp_prob.status == 'optimal') and solver_flag:
            is_solvable_status = True
            a_cbf = np.reshape(np.array(cp_x.value), -1)
            env.solver_stat['solvable'] = env.solver_stat['solvable'] + 1
            # Check the solution whether satisfy the risk constraint.
            env.risk_adj_lst[-1] = risk_safe_t1
            cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl+a_cbf), cov_r_t1), (a_rl+a_cbf).T))
            assert (cur_alpha_risk - socp_d) <= 0.00001, 'cur risk: {}, socp_d {}'.format(cur_alpha_risk, socp_d) 
            assert np.abs(np.sum(np.abs(a_rl+a_cbf)) - 1) <= 0.00001, 'sum of actions: {} \n{} \n{}'.format(np.sum(np.abs((a_rl+a_cbf))), a_rl, a_cbf)
            env.solvable_flag.append(0)
        else:
            a_cbf = np.zeros(N)
            env.solver_stat['insolvable'] = env.solver_stat['insolvable'] + 1
            is_solvable_status = False
            env.solvable_flag.append(1)
            cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl), cov_r_t1), (a_rl).T))
            env.risk_adj_lst[-1] = risk_safe_t1
    env.risk_pred_lst.append(cur_alpha_risk)    
    env.is_last_ctrl_solvable = is_solvable_status
    if cnt > 1:
        env.stepcount = env.stepcount + 1

    return a_cbf, is_solvable_status