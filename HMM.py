# 导入包
from enum import Flag
import random
import argparse
import codecs
import os
import re
from tkinter.constants import NO
from typing import KeysView
import numpy as np
from numpy.core.fromnumeric import argmax, size
from numpy.lib.function_base import select
from numpy.random.mtrand import gamma
import pandas as pd
import copy
# observations
np.printoptions(precision=5)


class Observation:
    def __init__(self, stateseq, outputseq):
        self.stateseq = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs

    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.outputseq)


def load_observations(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [line.split() for line in lines]
    if len(lines) % 2 == 1:  # remove extra lines
        lines = lines[:len(lines)-1]
    return [Observation(lines[i], lines[i+1]) for i in range(0, len(lines), 2)]


class HMM:

    @property
    def A(self):
        # 处理pi和矩阵a，a就是状态转换矩阵
        A = pd.DataFrame(index=[x for x in self.states], columns=[
                         x for x in self.states])  # 创建一个空的dataframe
        for state_i in self.states:
            for state_j in self.states:
                A.loc[state_i, state_j ] = float(self.transitions[state_i][state_j])
        return A

    @property
    def PI(self):
        trans_dic = copy.deepcopy(self.transitions)
        _pi = trans_dic['#']
        _pi = pd.DataFrame(_pi, index=['p']).T
        return _pi

    @property
    def B(self):
        states = self.states
        # 获取所有的观测值
        outputs = self.outputs
        # 获取状态下所有观测值的概率
        p = []
        for state in self.states:
            temp = []
            for output in self.outputs:
                temp.append(self.emissions[state].get(output,0))
            p.append(temp)
        B = np.array(p)
        B = pd.DataFrame(B,index=states, columns=list(outputs))
        return B

    def __init__(self, transitions=None, emissions=None):
        """creates a model from transition and emission probabilities"""
        self.transitions = transitions
        self.emissions = emissions
        if self.emissions:
            self.states = self.emissions.keys()

    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities if given.
        Initializes probabilities randomly if unspecified."""
        # TODO: fill in for section a
        trans_file_name = str(basename) + '.trans'
        emit_file_name = str(basename) + '.emit'
        # 下面处理状态转移矩阵
        trans_dic = {}
        states = set()
        trans_dic_need_init = False
        with open(trans_file_name, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip() 
                line_after = re.sub(r' +', ' ', line)
                line_after = line_after.split(' ')  # 按空格分割后的数组
                tran_01 = line_after

                if len(tran_01) == 2: # 如果处理后长度没有3，则认为没有概率,并且需要初始化概率，注意是整个初始化
                    trans_dic_need_init = True
                    tran_01.append(0)
                if tran_01[0] not in trans_dic.keys():
                    trans_dic[tran_01[0]] = {}
                trans_dic[tran_01[0]][tran_01[1]] = float(tran_01[2])
                # 将状态添加到集合中
                states.add(tran_01[0])
                states.add(tran_01[1])
        states.remove('#') # 这个是不要的，因为它下面就是π
        # 需要初始化概率
        if trans_dic_need_init:
            for state_i in trans_dic.keys():
                num = len(trans_dic[state_i]) # 统计从该状态转移到其他状态的所有情况有多少个
                p = np.random.randint(1000,size=(num))
                p = p/p.sum()
                p_gen  = (x for x in p) # 创建一个生成器，用于赋值
                for state_j in trans_dic[state_i].keys():
                    trans_dic[state_i][state_j] = next(p_gen)
        # 将缺失的，补足为0
        for state in states:
            trans_dic['#'][state] = trans_dic['#'].get(state,0) #若没有，则赋值为0
            # 补足state作为i状态的键
            if state not in trans_dic.keys(): #
                trans_dic[state] = {}
        # 补足缺失的aij
        for state_i in states:
            for state_j in states:
                trans_dic[state_i][state_j] = trans_dic[state_i].get(state_j,0)
        
        self.transitions = trans_dic
        self.states = trans_dic['#'].keys() # 后面我们遍历transitions的时候，都要用它作为迭代的顺序
        print('状态转移矩阵处理完毕!')
        # 下面处理发射矩阵
        emit_dic = {}
        outputs = set() #A set object is an unordered collection of distinct hashable objects. 
        emit_dic_need_init = False
        with open(emit_file_name, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                line_after = re.sub(r' +', ' ', line)
                line_after = line_after.split(' ')  # 按空格分割后的数组
                p_01 = line_after
                if len(p_01) == 2: # 如果处理后长度没有3，则认为没有概率,并且需要初始化概率，注意是整个初始化
                    emit_dic_need_init = True
                    p_01.append(0)
                if p_01[0] not in emit_dic.keys():
                    emit_dic[p_01[0]] = {}
                emit_dic[p_01[0]][p_01[1]] = float(p_01[2])
                outputs.add(p_01[1])
        # 需要初始化概率
        if emit_dic_need_init:
            for state_j in emit_dic.keys():
                num = len(emit_dic[state_j])
                p = np.random.randint(1000,size=(num))
                p = p/p.sum()
                p_gen  = (x for x in p)
                for output in emit_dic[state_j].keys():
                    emit_dic[state_j][output] = next(p_gen)
        
        # 下面补足
        for state_j in states:
            if state_j not in emit_dic.keys():
                emit_dic[state_j] = {} #这里会出现若是某个状态到观测值的概率全部没有的情况下，都被初始化为0了
            for output in outputs:
                emit_dic[state_j][output] = emit_dic[state_j].get(output,0)
        
        outputs = list(outputs)
        self.emissions = emit_dic
        self.outputs = outputs # 我们后面迭代也用这个output
        print('load success!')

    def dump(self, basename):
        """store HMM model parameters in basename.trans and basename.emit"""
        # TODO: fill in for section a
        trans_file_name = str(basename)+'.trans'
        emit_file_name = str(basename)+'.emit'
        with open(trans_file_name, 'w', encoding='utf-8') as f:
            for keys, items in self.transitions.items():  # {a:{b:f,c:f}}
                for key, item in items.items():
                    if item != 0:
                        f.writelines(keys+' '+key+' '+str(item)+'\n')

        with open(emit_file_name, 'w', encoding='utf-8') as f:
            for keys, items in self.emissions.items():  # {a:{b:f,c:f}}
                for key, item in items.items():
                    if item != 0:
                        f.writelines(keys+' '+key+' '+str(item)+'\n')
        print('dump success!')

    def generate(self, n):
        """return an n-length observation by randomly sampling from this HMM.
        """
        # TODO: fill in for section c
        state_seq = []
        a_i_state = '#'
        for i in range(1, n+1):
            aj_state = [] # 存储aj的状态
            aj_p = [] # 存储aj的概率
            for state, p in self.transitions[a_i_state].items():# 获取i状态下所有尽可能的j状态
                #概率为0的不参与，否则会很慢....
                if p == 0:
                    print('p is 0')
                    continue
                aj_state.append(state) # 获取状态
                aj_p.append(float(p))  # 获取概率
            a_i_state = np.random.choice(
                a=aj_state, size=1, replace=True, p=aj_p)[0]
            state_seq.append(a_i_state)

        # 通过状态链，去随机观测值
        out_seq = []
        for state in state_seq:
            # 获取该状态下的观测值
            out_result = []
            out_p = []
            for out, p in self.emissions[state].items(): # 获取j状态下，所有可能的output
                # print(out,p)
                if p == 0:
                    continue
                out_result.append(out)
                out_p.append(float(p))
            # print('-----------')
            out_result = np.random.choice(
                a=out_result, size=1, replace=True, p=out_p)[0]
            # print(out_result)
            out_seq.append(out_result)
        # print(state_seq)
        return Observation(state_seq, out_seq)

    def viterbi(self, observation):
        """given an observation,
        set its state sequence to be the most likely state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """
        # TODO: fill in for section d
        # 下面开始初始化delta_1 下，各个状态的概率
        obs = observation.outputseq
        delta_1 = []

        for state_i in self.states: # 所有状态的初始概率
            delta_1.append(self.transitions['#'][state_i]*self.emissions[state_i][obs[0]])

        delta_1 = np.array(delta_1)
        # 下面开始初始化psai_1 下，概率最大的路径节点（全部为0）
        psai_1 = np.zeros(len(self.states))

        delta = []
        psai = []
        delta.append(delta_1)
        psai.append(psai_1)

        # 下面开始迭代
        delta_t_1 = delta_1
        for output in obs[1:]:
            temp_delta_t_1 = [] #用来临时存储，每轮迭代后，delta_i的计算结果,不能直接用delta_t_1存储，否则会改变它的值
            temp_psai_t = [] # psai的值不参与迭代，直接取的结果，也就是上一个节点的参数的位置
            for state_i in self.states: # 遍历所有的状态，这些状态作为第i个状态
                a_j_i = [] # 用来存储aji,状态为j的情况下，转移到i的概率
                for state_j in self.states: # 遍历所有状态，做为第j个状态
                    a_j_i.append(self.transitions[state_j][state_i])
                b_i_t = self.emissions[state_i][output] # 用来存储状态为i时，观测值为ot的概率
                a_j_i = np.array(a_j_i)
                delta_t_1_i = (delta_t_1*a_j_i).max()*b_i_t
                # 这里是找寻 (delta_t_1*a_j_i).max()中，
                # 先获取索引,索引应该不是唯一的，但是np的函数返回的是唯一的，该问题待优化
                index = (delta_t_1*a_j_i).argmax() # 返回最大的参数的位置，也就是概率最大的路径当中，第t-i个节点，在所有隐状态中的索引
                # pasai_t = [x for x in self.transitions['、#'].keys()][index] # 返回所有状态，因为字典没有改变过，所有顺序不会变，转成列表，按索引取值即可
                pasai_t = index  # + 1 # 因为ndarray中的索引是从0开始的，但是我们定义“设定”的是从1开始的
                temp_delta_t_1.append(delta_t_1_i)
                temp_psai_t.append(pasai_t)

            delta_t_1 = np.array(temp_delta_t_1)
            temp_psai_t = np.array(temp_psai_t)

            psai.append(temp_psai_t) # 加入到psai中，便于后续核对结果
            delta.append(delta_t_1) # 加入到delta中，便于后续核对结果

        #迭代完成后，最大概率
        max_p = delta[-1].max()
        
        # 将状态赋值给stateseq
        observation.stateseq = []
        
        # 选择终点 
        # 创建状态索引列表
        i_s_l = [] # index state list
        index = delta[-1].argmax()
        i_s_l.append(index) # 最有路径重点的状态索引
        # 下面开始前向迭代
        # 因为psai添加的顺序是从前往后的，,这样保证是从后往前迭代的，第0个元素是不要的
        rever_psai = psai[:0:-1] 
        for k in rever_psai:
            index = k[index]
            i_s_l.append(index)
        i_s_l = i_s_l[::-1] #再反转，因为迭代是从后往前的，所以反转了就是从前往后的状态
        states = [x for x in self.states] # 因为字典的key的顺序不会改变，所以返回的结果的顺序也依然不变
        for i in i_s_l:
            observation.stateseq.append(states[i]) # 字典的key不能当列表使用来索引
        return delta,psai,max_p,i_s_l

    def forward(self, observation):
        """given an observation as a list of T symbols,
        compute and return the forward algorithm parameters alpha_i(t)
        for all 0<=t<T and i HMM states.
        """
        # TODO: fill in for section e
        all_alpha = []
        # 第一步，计算初值
        obs = observation.outputseq
        alpha_1 = []
        for state in self.states:
            t = self.transitions['#'][state]*self.emissions[state].get(obs[0], 0)
            alpha_1.append(t)
        alpha_1 = np.array(alpha_1)  # 转为ndarray，做乘法
        all_alpha.append(alpha_1)

        # 第二步，迭代
        alpha_i = alpha_1  # 将初始值赋值给alpha_i，参见统计学习方法
        for t in range(2, len(obs)+1):  # 从第二部迭代到最后一个观测值
            temp_alpha_i = []
            for i_state in self.states:  # 这个循环标志a_ji当中的i状态，参见统计学习方法. HMM
                a_ji = []  # 用它来标志aji，表示j状态，转移到i状态的概率
                for j_state in self.states:  # 这个循环标志a_ji当中的j状态，参见统计学习方法. HMM
                    a_ji.append(self.transitions[j_state][i_state])
                # 将b_ji 转为ndarray属性，便于计算
                a_ji = np.array(a_ji)
                # alpha_i = alpha_i*b_ji
                # 下面找到状态t到观测值的概率
                b_i_t = self.emissions[i_state].get(obs[t-1], 0)
                temp_alpha_i.append((alpha_i*a_ji).sum()*b_i_t)

            alpha_i = np.array(temp_alpha_i)
            all_alpha.append(alpha_i)
        # 最后一轮alpha的值

        return np.array(all_alpha)

    def forward_probability(self, observation):
        """return probability of observation, computed with forward algorithm.
        """
        all_alpha = self.forward(observation)
        alpha = all_alpha[-1]
        return alpha.sum()

    def backward(self, observation):
        """given an observation as a list of T symbols,
        compute and return the backward algorithm parameters beta_i(t)
        for all 0<=t<T and i HMM states.
        """
        all_beta = []
        # 第一步，计算初值,也就是beta_t的初始值
        obs = observation.outputseq
        
        beta_t = []
        for __ in self.states:
            t = 1            # 全部初始化为1
            beta_t.append(t)
        beta_t = np.array(beta_t)  # 转为ndarray，方便做乘法
        all_beta.append(beta_t)  # 便于观察

        # 第二步，迭代
        beta_t_plus_1 = beta_t  # 将初始值赋值给βt+1,便于计算βt，参见统计学习方法
        for t in range(2, len(obs)+1):  # 从倒数第二个观测值，迭代到第一个观测值
            temp_beta_t_plus_1 = []
            for i_state in self.states:  # 这个循环标志β_t_i当中的i状态，参见统计学习方法. HMM
                a_ij = []  # 用来标志a_ij表示从i状态，转移到j状态的概率
                b_j_t_plus_1 = []  # 用它来标志b_j_t，表示j状态下，观测值为t+1的概率
                for j_state in self.states:  # 这个循环标志b_ji当中的j状态，参见统计学习方法. HMM
                    #print('i state : ',i_state,' ',self.transitions[i_state][j_state],' j :state ',j_state,' ',self.emissions[j_state][obs[len(obs)-t]],' t :',obs[len(obs)-t], ' t+1: ',obs[len(obs)-t+1])
                    a_ij.append(
                        self.transitions[i_state][j_state]) # 逐个添加概率
                    b_j_t_plus_1.append(
                        self.emissions[j_state].get(obs[len(obs)-t+1], 0)) # 逐个添加概率
                # 将a_ij 转为ndarray属性，便于计算
                a_ij = np.array(a_ij)
                b_j_t_plus_1 = np.array(b_j_t_plus_1)
                beta_t_i = (a_ij*b_j_t_plus_1*beta_t_plus_1).sum()
                temp_beta_t_plus_1.append(beta_t_i)
                #print('i_state:',i_state,' ,β is :',(a_ij*b_j_t_plus_1*beta_t_plus_1).sum())
            beta_t_plus_1 = np.array(temp_beta_t_plus_1)
            all_beta.append(beta_t_plus_1)
        
        all_beta = all_beta[::-1]
        return np.array(all_beta)

    def backward_probability(self, observation):
        """return probability of observation, computed with backward algorithm.
        """
        # TODO: fill in for section e
        all_beta = self.backward(observation)
        _pi = []
        _b_i_1 = []
        for state, pro in self.transitions['#'].items():
            # print(state)
            _pi.append(pro)
            _b_i_1.append(
                self.emissions[state].get(observation.outputseq[0], 0))
        _pi = np.array(_pi)
        _b_i_1 = np.array(_b_i_1)

        return (_pi*_b_i_1*all_beta[0]).sum()

    def learn_supervised(self, corpus, emitlock=False, translock=False):
        """Given a corpus, which is a list of observations
        with known state sequences,
        set the HMM parameters that maximize the corpus likelihood.
        Do not update the transitions if translock is True,
        or the emissions if emitlock is True.
        """
        # TODO: fill in for section b
        # 下面开始训练，训练的参数结果将根据设定，来确定是否更新，算法的公式推导参见《统计学习方法第二版》P203 and 204
        # 深拷贝
        num_transitions_dic = copy.deepcopy(self.transitions)
        num_emittionns_dic = copy.deepcopy(self.emissions)
        # 将字典中的值初始化为0，便于后续累加，计算频数
        for state_i in self.states:
            num_transitions_dic['#'][state_i] = 0
            for state_j in self.states:
                num_transitions_dic[state_i][state_j] = 0
        
        for state_j in self.states:
            for output in self.outputs:
                num_emittionns_dic[state_j][output] = 0
        # 下面开始计算频数
        for obs in corpus:
            s_s = obs.stateseq
            o_t = obs.outputseq
            for i in range(len(s_s)-1):
                if i == 0:
                    num_transitions_dic['#'][s_s[i]] += 1 # 序列的首位，初始状态的频数
                num_transitions_dic[s_s[i]][s_s[i+1]] += 1 # aij的情况，直接 +1
                num_emittionns_dic[s_s[i]][o_t[i]] += 1 # bi_t 的情况，直接加 1
            num_emittionns_dic[s_s[-1]][o_t[-1]] += 1 # 因为序列的最后一个我们没有处理，在这里处理

        # print(num_transitions_dic)
        # print(num_emittionns_dic)
        # 下面处理状态转移矩阵
        # 将 π打印出来
        print(num_transitions_dic['#'])
        pi_total = sum(num_transitions_dic['#'].values())
        for state_j in self.states:
            num_transitions_dic['#'][state_j] = num_transitions_dic['#'][state_j]/pi_total
        for state_i in self.states:
            total = sum(num_transitions_dic[state_i].values())
            # print('状态-状态：{} to {}'.format(state_i,total))
            for state_j in self.states:
                p = num_transitions_dic[state_i][state_j]/total
                num_transitions_dic[state_i][state_j] = num_transitions_dic[state_i][state_j]/total
        #         if state_i == 'ADJ':
        #             print('i is {},j is {}. feq is . {}, total is {} '.format(state_i,state_j,num_transitions_dic[state_i][state_j],total))
        #             print(num_transitions_dic[state_i][state_j],p)
        # print('--------------这里再打 ',num_transitions_dic['ADJ']['ADJ'])
        # 下面开始计算发射矩阵的概率
        for state_j in self.states:
            total = sum(num_emittionns_dic[state_j].values()) # 对状态j下面，所有的观测值出现的次数求和
            for output in self.outputs:
                num_emittionns_dic[state_j][output] = num_emittionns_dic[state_j][output]/total
                # if output == 'fawn':
                #     print(total)
                #     print(num_emittionns_dic[state_j][output])
                


        
        #是否改变
        if not emitlock:
            self.emissions = num_emittionns_dic
        if not translock:
            self.transitions = num_transitions_dic


    def learn_unsupervised(self, corpus, convergence=0.001, emitlock=False, translock=False, restarts=0):
        """Given a corpus,
        which is a list of observations with the state sequences unknown,
        apply the Baum Welch EM algorithm
        to learn the HMM parameters that maximize the corpus likelihood.
        Do not update the transitions if translock is True,
        or the emissions if emitlock is True.
        Stop when the log likelihood changes less than the convergence threhshold,
        and return the final log likelihood.
        If restarts>0, re-run EM with random initializations.
        """
        # TODO: fill in for section f

        train_model = []

        for i in corpus:
            obs = i
            if len(obs.outputseq) < 2 :
                print(obs.outputseq)
                print(obs)
                print('序列长度小于2，不参与训练')
                continue
            # 获取α
            alpha = self.forward(obs)
            # 获取β
            beta = self.backward(obs)
            #计算γ
            gama = alpha*beta
            gama = gama/gama.sum(axis=1).reshape(-1,1) # γ[t,i]
            # 下面开始计算A矩阵
            # 下面 transition这个字典，转换为A矩阵
            A = self.A.values
            B = self.B
            # 下面开始计算不同时刻t(从1到T-1)下的epsilon
            epsilon = [] # ε[t,i,j]
            for i in range(len(obs.outputseq)-1):
                t = alpha[i,].reshape(-1,1)*A*B[obs.outputseq[i+1]].values.reshape(1,-1)*beta[i+1,].reshape(1,-1)
                epsilon.append(t)
            epsilon = np.array(epsilon)
            # epsilon还要除以对应的时刻下面的所有的加和
            episilon_t_sum = epsilon.sum(axis=(1,2)).reshape(-1,1,1)
            epsilon = epsilon/episilon_t_sum
            # ε[t,i,j]在t方向上求和，作为aij的分子
            epsilon_sum = epsilon.sum(axis=0)
            # γ[t-1,i]在t方向上求和,作为aji的分母 两者做除法可求出矩阵A
            gama_sum_t_1 = gama[:-1].sum(axis=0)

            # 找到重复值,找到在观测序列中，每个观测值的索引
            output_index_dic = {}
            for k,index in zip(obs.outputseq,range(len(obs.outputseq))):
                if k not in output_index_dic:
                    output_index_dic[k] = []
                if k in obs.outputseq:
                    output_index_dic[k].append(index)
            
            emit_dic = copy.deepcopy(self.emissions)
            # 计算b矩阵当中的分子
            newB_num = []
            for state_j,state_index in zip(self.states,range(len(self.states))):
                j_k_p = []
                for output in self.outputs:
                    if output in output_index_dic.keys():
                        index = output_index_dic[output] # 返回每个观测值的索引
                        p = 0
                        for t in index:
                            p += gama[t,state_index]
                        emit_dic[state_j][output] = p 
                    else:
                        emit_dic[state_j][output] = 0
                    j_k_p.append(emit_dic[state_j][output])
                newB_num.append(j_k_p)
            newB_num = np.array(newB_num)
            # γ在t方向上求和，两者做除法可求出B
            gama_sum_t = gama.sum(axis=0)
            train_model.append([gama,[epsilon_sum,gama_sum_t_1],[newB_num,gama_sum_t]])
        
        # 下面计算tran_model中的值
        gama_sum_0  = np.zeros((1,len(self.states)))

        epsilon_sum = np.zeros((len(self.states),len(self.states)))
        gama_sum_t_1 = np.zeros((1,len(self.states)))

        newB_num_sum = np.zeros((len(self.states),len(self.outputs)))
        gama_sum_t = np.zeros((1,len(self.states)))
        for i in range(len(train_model)):
            gama_sum_0 = gama_sum_0 +train_model[i][0][0,]
            epsilon_sum = epsilon_sum+train_model[i][1][0]
            gama_sum_t_1 = gama_sum_t_1 +train_model[i][1][1]
            newB_num_sum = newB_num_sum +train_model[i][2][0]
            gama_sum_t = gama_sum_t + train_model[i][2][1]
        
        new_PI = gama_sum_0[0,]/len(corpus) #  gama_sum_0是二维的，但是只有一行，所以直接取第一行即可
        new_A = epsilon_sum/gama_sum_t_1
        new_B = newB_num_sum/gama_sum_t.reshape(-1,1)

        return new_PI,new_A,new_B,train_model