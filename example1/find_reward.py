import numpy as np
import pandas
import time

import pandas as pd

np.random.seed(2)  #计算机产生伪随机数列

N_STATES = 6   #多少种状态
ACTIONS = ['left', 'right']  #多少种动作
EPSILON = 0.9               #greedy police  随机策略参数
ALPHA = 0.1                 #learing rate
LAMBDA = 0.9                #discount factor
MAX_EPISODES = 13           #maximum episodes
FRESH_TIME = 0.3            #fresh time for one move

def bulid_q_table(n_states,actions):
    #pandas 库是对数据表格化的工具
    table = pd.DataFrame(
        np.zeros((n_states,len(actions))),  #全零初始化Q表
        columns=actions,  #动作名字x
    )
    return table

def choose_action(state,q_table):
    #选动作
    state_actions = q_table.iloc[state, :]   #根据当前的环境，选择状态
    if(np.random.uniform() > EPSILON) or ((state_actions == 0).all()):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_actions.idxmax()    ##原版是argmax()
    return action_name

def get_env_feedback(S, A):
    if A == 'right':
        if S == N_STATES - 2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S - 1
    return S_,R

def update_env(S, episode, step_counter):

    env_list = ['-']*(N_STATES-1) + ['T']
    if S == 'terminal':
        interaction = 'Episode %s: total_step = %d' %(episode+1,step_counter)
        print('\r{}'.format(interaction),end='')
        time.sleep(2)
        print('\r                           ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)

def rl():
    q_table = bulid_q_table(N_STATES,ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S,episode,step_counter)
        while not is_terminated:

            A = choose_action(S,q_table)

            S_, R = get_env_feedback(S, A)  # take action & get next state and reward
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                q_target = R + LAMBDA * q_table.iloc[S_,:].max()
            else:
                q_target = R
                is_terminated = True

            q_table.loc[S,A] += ALPHA * (q_target - q_predict)
            S = S_
            step_counter += 1
            global EPSILON
         #   EPSILON += 0.01 * step_counter
            update_env(S,episode,step_counter)
    return q_table

if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
