#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 18:01:28 2022

@author: seifu
"""

import random
import numpy as np
import math
from  CEF_Propagation import Propagation
from sklearn import preprocessing
import CEF_Latency
huge = 100

class Environment:
    def __init__(self):
      
# topology settings
        self.NumCloudNetwork = 1
        self.NumCoreNetwork = 2
        self.NumAcceNetwork = 8
        self.DownLinkBandWd = 1000000
        self.UpLinkBandWd = 1000000
        
# Arrival traffic rate at jth AN-CEF site of ith CN-CEF site
        # np.random.seed(1)
        # self.lamda = [np.random.randint(low = 2000, high = 8000, size = self.NumAcceNetwork) \
        #               for i in range(self.NumCoreNetwork)]
        # self.lam = (70000,70000,70000,70000,70000,70000,70000,70000)
        self.lam = (50000,50000,50000,50000,50000,50000,50000,50000)       
        # self.lam = (40000,40000,40000,40000,40000,40000,40000,40000)
        # self.lam = (30000,30000,30000,30000,30000,30000,30000,30000)        
        # self.lam = (20000,20000,20000,20000,20000,20000,20000,20000)
        # self.lam = (15000,15000,15000,15000,15000,15000,15000,15000)        
        # self.lam = (10000,10000,10000,10000,10000,10000,10000,10000)
        # self.lam = (5000,5000,5000,5000,5000,5000,5000,5000)
        np.random.seed(1)
        self.lamda = [[np.random.poisson(lam=self.lam, size=self.NumAcceNetwork) \
                      for i in range(self.NumCoreNetwork)] for j in range(self.NumCloudNetwork)]
        # print("Lambda", self.lamda)
# Array to collect latency totall latency(initial will 0) updated during calculation
        self.latency = [[np.zeros((self.NumAcceNetwork, self.NumAcceNetwork+self.NumCoreNetwork+ \
                                  self.NumCloudNetwork)) for i in range(self.NumCoreNetwork)] for k in range(self.NumCoreNetwork)]
        #print("Latency", self.latency[0][0][0][0])
# Inttail Traffic that is served at Access, core and cloud network
        self.AccesNetLamda = [[[0 for i in range(self.NumAcceNetwork)] for j in range(self.NumCoreNetwork)] for j in range(self.NumCloudNetwork)]
        #print("AccessLambda", self.AccesNetLamda)
        self.CoreNetLamda = [[0 for i in range(self.NumCoreNetwork)] for j in range(self.NumCloudNetwork)]
        #print("CoreLambda", self.CoreNetLamda)
        self.CloudNetLamda = [0 for i in range(self.NumCloudNetwork)]
        #print("CloudLambda", self.CloudNetLamda)
# Initial capacity for each tiers and sites       
        self.miuCloudNetwork = [3000000 for i in range(self.NumCloudNetwork)]
        #print("miuCloudNetwork", self.miuCloudNetwork)
        self.miuCoreNetwork = [[270000 for i in range(self.NumCoreNetwork)] for j in range(self.NumCloudNetwork)]
        #print("miuCoreNetwork", self.miuCoreNetwork)         
        self.miuAcceNetwork = [[[40000 for i in range(self.NumAcceNetwork)] for j in range(self.NumCoreNetwork)] for k in range(self.NumCloudNetwork)] 
        #print("miuAcceNetwork", self.miuAcceNetwork)
#method calling for fining distance      
        # self.propagationCloudNetwork = Propagation("clcef.csv").propagation_delay
        # self.propagationCoreNetwork = Propagation("cncef.csv").propagation_delay
        # self.propagationAccesNetwork = Propagation("ancef.csv").propagation_delay

        self.propagationCloudNetwork = Propagation("CLCEF.csv").propagation_delay
        self.propagationCoreNetwork = Propagation("CNCEF.csv").propagation_delay
        self.propagationAccesNetwork = Propagation("ANCEF.csv").propagation_delay
        
        self.steps_left = 10
        self.total_reward = 0.0
        self.done = False 
    
    def get_observation(self):
        latency = self.latency
        lamda = self.lamda
        UpLinkBandWd = self.UpLinkBandWd
        DownLinkBandWd = self.DownLinkBandWd
        
        self.miuCloudNetwork = [3000000 for i in range(self.NumCloudNetwork)]
        #print("miuCloudNetwork", self.miuCloudNetwork)
        self.miuCoreNetwork = [[270000 for i in range(self.NumCoreNetwork)] for j in range(self.NumCloudNetwork)]
        #print("miuCoreNetwork", self.miuCoreNetwork)         
        self.miuAcceNetwork = [[[40000 for i in range(self.NumAcceNetwork)] for j in range(self.NumCoreNetwork)] for k in range(self.NumCloudNetwork)] 
        #print("miuAcceNetwork", self.miuAcceNetwork) 
       
        miuCloudNet = self.miuCloudNetwork       
        # for i in range(len(miuCloudNet)):
        #     miuCloudNet[i] -= self.CloudNetLamda[i]
            
        miuCoreNet = self.miuCoreNetwork
        # for j in range(len(miuCoreNet)):
        #     miuCoreNet[j] -= self.CoreNetLamda[j]
                
        miuAccesNet = self.miuAcceNetwork
        # for l in range(len(miuAccesNet)):
        #     for m in range(len(miuAccesNet[l])):
        #         miuAccesNet[l][m]-= self.AccesNetLamda[l][m]
               
        scalr = preprocessing.RobustScaler()   # standardized data set
       
        # intial_Observation = [latency, lamda, UpLinkBandWd, DownLinkBandWd, miuAccesNet, miuCoreNet, miuCloudNet] # initial observation information
        intial_Observation = [latency, lamda, UpLinkBandWd, DownLinkBandWd, miuAccesNet, miuCoreNet, miuCloudNet] # initial observation information        
        
        lat = scalr.fit_transform(np.array(latency).flatten().reshape(-1, 1))
        lmd = scalr.fit_transform(np.array(lamda).flatten().reshape(-1, 1))
        bw = scalr.fit_transform(np.array([UpLinkBandWd,DownLinkBandWd]).reshape(-1, 1))
        mAcces = scalr.fit_transform(np.array(miuAccesNet).flatten().reshape(-1, 1))
        mCore = scalr.fit_transform(np.array(miuCoreNet).flatten().reshape(-1, 1))
        mCloud = scalr.transform(np.array(miuCloudNet).flatten().reshape(-1, 1))
        
        observe = np.concatenate((lat,lmd,bw,mAcces,mCore,mCloud)).flatten()      # oservation after step started
        # print("Observe", observe)
        return intial_Observation,lat,observe 

# At the first time randomly generate the actions           
    def generate_actions(self):
        # self.miuCloudNetwork = [3000000 for i in range(self.NumCloudNetwork)]
        # #print("miuCloudNetwork", self.miuCloudNetwork)
        # self.miuCoreNetwork = [[270000 for i in range(self.NumCoreNetwork)] for j in range(self.NumCloudNetwork)]
        # #print("miuCoreNetwork", self.miuCoreNetwork)         
        # self.miuAcceNetwork = [[[40000 for i in range(self.NumAcceNetwork)] for j in range(self.NumCoreNetwork)] for k in range(self.NumCloudNetwork)] 
        # #print("miuAcceNetwork", self.miuAcceNetwork) 
       
        # miuCloudNet = self.miuCloudNetwork       
        # for i in range(len(miuCloudNet)):
        #     miuCloudNet[i] -= self.CloudNetLamda[i]
            
        # miuCoreNet = self.miuCoreNetwork
        # for j in range(len(miuCoreNet)):
        #     miuCoreNet[j] -= self.CoreNetLamda[j]
                
        # miuAccesNet = self.miuAcceNetwork
        # for l in range(len(miuAccesNet)):
        #     for m in range(len(miuAccesNet[l])):
        #         miuAccesNet[l][m]-= self.AccesNetLamda[l][m]
        

        # act = [np.zeros((self.NumAcceNetwork, self.NumCoreNetwork+self.NumAcceNetwork+self.NumCloudNetwork)) for i in range(self.NumCoreNetwork)]
        
        # for i in range(self.NumCoreNetwork):
        #     for j in range(self.NumAcceNetwork):
        #         tmp = np.concatenate((np.array(miuAccesNet[i]).flatten(), np.array(miuCoreNet).flatten(), np.array(miuCloudNet).flatten()))
        #         for k in range(len(tmp)):
        #             act[i][j][k] = tmp[k]/sum(tmp)                    

        # return act, np.array(act).flatten()
    
    #comment
        # b=[]
        # # lamdax=[]
        # for n in range(self.NumCoreNetwork):
        #     matrix = np.random.rand(self.NumAcceNetwork,self.NumAcceNetwork+self.NumCoreNetwork+self.NumCloudNetwork)
        #     b.append(matrix/matrix.sum(axis=1)[:,None])
            
        # array = np.array(b)
        
        # return b, array.flatten()   
        Action = []               
     
        for n in range(self.NumCloudNetwork):
            for m in range(self.NumCoreNetwork):
                randm_action = np.random.rand(self.NumAcceNetwork, self.NumAcceNetwork + \
                                              self.NumCoreNetwork + self.NumCloudNetwork)
                Action.append(randm_action/randm_action.sum(axis=1)[:,None])
                
        action_array = np.array(Action)
        print("Action", action_array[0][0][0])
        return Action, action_array.flatten()

# Load (Allocate) trffic    
    def Load_Traffic(self, action):
        
        # mus = [action[x:x+28] for x in range(0, len(action), 28)]
        # index =0
        # for i in mus:
        #     mus[index] = [i[x:x+7] for x in range(0, len(action), 7)] 
        #     index+=1
        # action = mus        
        
        
        task=[]
        for n in range(self.NumCloudNetwork):
            for m in range(self.NumCoreNetwork):
                task.append(np.random.rand(self.NumAcceNetwork, self.NumAcceNetwork + \
                                             self.NumCoreNetwork + self.NumCloudNetwork))
                print("task", task)
        for i in range(len(task)):
            for j in range(len(task[i])):
                for k in range(len(task[i][j])):
                    task[i][j][k] = math.ceil(self.lamda[i][j][k]*np.round(action[i][j],2))
                    print("task[i][j][k][l]", task[i][j][k])
        trafic=[]    
        for n in range(len(task)):
            for m in range(len(task[n])):
                for l in range(len(task[n][m])):
                    trafic.append(task[n][m].sum(axis=1))
                    print("Traffic", trafic)
        for i in range(len(trafic)):
            for j in range(len(trafic[i])):
                for k in range(len(trafic[i][j])):
                    if trafic[i][j] < self.lamda[i][j][k]:
                        task[i][j][k][random.randint(0,len(task[i][j][k])-1)] += self.lamda[i][j][k] - trafic[i][j]
                
                    elif trafic[i][j] > self.lamda[i][j][k]:
                        task[i][j][k][random.randint(0,len(task[i][j][k])-1)] -= trafic[i][j] - self.lamda[i][j][k]
                      
                    else:
                        continue
        return task
    
    def is_done(self, step):
        if step == 0:
            self.done = True
        else:
            self.done = False
        return self.done
    
# Calculate reward for given actions
    def calculate_reward(self, obs, actions):
        latency = obs[0]
        lamda = obs[1]
        UpLinkBandWd = obs[2]
        DownLinkBandWd = obs[3]
        miuAccesNet = obs[4]
        miuCoreNet = obs[5]
        miuCloudNet = obs[6]
        
        Ac = actions
        actions = self.Load_Traffic(Ac)
        
        for i in range(len(latency)):
            print("I",i)
            #lamdaAccesNet =0
            for j in range(len(latency[i])):                
                for k in range(len(latency[i][j])):
                    lamdaAccesNet = CEF_Latency.get_AccesNetwork_lamda(i,k, actions)
                    lamdaCoreNet = CEF_Latency.get_CoreNetwork_lamda(k, actions)
                    lamdaCloudNet = CEF_Latency.get_CloudNetwork_lamda(actions)
                    
    ## Latency that is served at local Access Network
                    if j==k and k<=self.NumAcceNetwork:
                        latency[i][j][k] = CEF_Latency.get_latency_local_AccesNetwork(UpLinkBandWd, DownLinkBandWd, \
                                                                                      miuAccesNet[i][j],lamdaAccesNet, \
                                                                                          lamda[i][j])                                                                                                                                                         
     ## Latency that is served at AN Neighbor                    
                    elif k < self.NumAcceNetwork and k!=j:
                        Dist_Acc_Acc = self.propagationAccesNetwork()[j][k]                       
                        latency[i][j][k] = CEF_Latency.get_latency_neighbour_AccesNetwork(UpLinkBandWd, DownLinkBandWd, \
                                                                                          miuAccesNet[i][j], lamdaAccesNet, \
                                                                                              lamda[i][j],  Dist_Acc_Acc) 
                                                                                  
       ## Latency that is served at local CN
                    elif self.NumAcceNetwork-1 < k < (self.NumAcceNetwork + self.NumCoreNetwork) and k == self.NumAcceNetwork+i:
                        latency[i][j][k] = CEF_Latency.get_latency_CoreNetwork(UpLinkBandWd,DownLinkBandWd, \
                                                                               miuCoreNet[i], lamdaCoreNet,lamda[i][j], \
                                                                                   3.3333333333333333e-6)
                                                                 
        ## Latency that is served at CN neighbour    
                    elif self.NumAcceNetwork-1 < k < (self.NumAcceNetwork + self.NumCoreNetwork) and k != self.NumAcceNetwork+i:
                        Dist_Cor_Cor = self.propagationCoreNetwork()[j][k]                       
                        latency[i][j][k] = CEF_Latency.get_latency_neighbour_CorNetwork(UpLinkBandWd,DownLinkBandWd, \
                                                                                        miuCoreNet[i],lamdaCoreNet, \
                                                                                            lamda[i][j], 3.3333333333333333e-6, Dist_Cor_Cor)
                                                                                                                                                               
      ## Latency that is served at Cloud
                    elif self.NumCoreNetwork+self.NumAcceNetwork-1 < k < (self.NumAcceNetwork+self.NumCoreNetwork+ \
                                                                          self.NumCloudNetwork) and k == self.NumCoreNetwork+self.NumAcceNetwork+i:
                        # Dist_Cor_Cloud = self.propagationCloudNetwork()[i][k]
                        latency[i][j][k]= CEF_Latency.get_latency_CloudNetwork(UpLinkBandWd,DownLinkBandWd, \
                                                                               miuCloudNet[i], lamdaCloudNet, lamda[i][j], \
                                                                                   3.3333333333333333e-6, 3.3333333333333333e-6)
                        
        total_latency = 0
# Jth AN Ith CN of Cloud Latency
        for i in range(len(latency)):
            for j in range(len(latency[i])):
                for k in range(len(latency[i][j])):
                    total_latency += (actions[i][j][k]*latency[i][j][k])

# system average delay 𝑙(𝑖,𝑗)                                    
        total_traffic_latency = sum(sum(map(sum, actions)))
        
# Average Latency (Average system latency)
        avg_syst_latency = total_latency/total_traffic_latency
        
# Reward = 1/l       
        reward = 1/avg_syst_latency
        if self.done == True:
            self.reset()
        self.AccesNetLamda = [[[0 for i in range(self.NumAcceNetwork)] for j in range(self.NumCoreNetwork)] for j in range(self.NumCloudNetwork)]
        #print("AccessLambda", self.AccesNetLamda)
        self.CoreNetLamda = [[0 for i in range(self.NumCoreNetwork)] for j in range(self.NumCloudNetwork)]
        #print("CoreLambda", self.CoreNetLamda)
        self.CloudNetLamda = [0 for i in range(self.NumCloudNetwork)]
        #print("CloudLambda", self.CloudNetLamda)
        
        return reward
    
    def step(self, actions):
        intial_Observation, obs2, observe = self.get_observation()
        reward = self.calculate_reward(intial_Observation, actions)
        self.total_reward += reward
        actions = self.Load_Traffic(actions)
    
        # Acn = [] 
        # for i in actions:
        #     Acn.append([sum(x) for x in zip(*i)])
        # Bcn = [sum(x) for x in zip(*Acn)]
        # Ccn = [sum(Bcn)]
        # for i in range(len(self.CloudNetLamda)):
        #     self.CloudNetLamda[i] += Ccn[i]
        # for j in range(len(self.CoreNetLamda)):
        #     self.CoreNetLamda[j] += Bcn[j]
        # for k in range(len(self.CoreNetLamda)):
        #     for l in range(len(self.AccesNetLamda[k])):
        #         self.AccesNetLamda[k][l] += Acn[k][l]
                
        intial_Observation, obs2, observe = self.get_observation()
        
        self.steps_left -= 1
        done = self.is_done(self.steps_left)
        
        return observe, reward, done
    
    def reset(self):
        self.steps_left = 10
        self.AccesNetLamda = [[[0 for i in range(self.NumAcceNetwork)] for j in range(self.NumCoreNetwork)] for j in range(self.NumCloudNetwork)]
        #print("AccessLambda", self.AccesNetLamda)
        self.CoreNetLamda = [[0 for i in range(self.NumCoreNetwork)] for j in range(self.NumCloudNetwork)]
        #print("CoreLambda", self.CoreNetLamda)
        self.CloudNetLamda = [0 for i in range(self.NumCloudNetwork)]
        #print("CloudLambda", self.CloudNetLamda)
        
        self.latency = [[np.zeros((self.NumAcceNetwork, self.NumAcceNetwork+self.NumCoreNetwork+ \
                                  self.NumCloudNetwork)) for i in range(self.NumCoreNetwork)] for k in range(self.NumCoreNetwork)]
        # self.lamda = [np.random.randint(low = 2000, high = 8000, size = self.NumAcceNetwork) for i in range(self.NumCoreNetwork)]
        self.lamda = [[np.random.poisson(lam=self.lam, size=self.NumAcceNetwork) \
                      for i in range(self.NumCoreNetwork)] for j in range(self.NumCloudNetwork)]
        self.UpLinkBandWd = 100000
        self.DownLinkBandWd = 100000
        
        self.miuCloudNetwork = [3000000 for i in range(self.NumCloudNetwork)]
        #print("miuCloudNetwork", self.miuCloudNetwork)
        self.miuCoreNetwork = [[270000 for i in range(self.NumCoreNetwork)] for j in range(self.NumCloudNetwork)]
        #print("miuCoreNetwork", self.miuCoreNetwork)         
        self.miuAcceNetwork = [[[40000 for i in range(self.NumAcceNetwork)] for j in range(self.NumCoreNetwork)] for k in range(self.NumCloudNetwork)] 
        #print("miuAcceNetwork", self.miuAcceNetwork)
        # self.total_reward = 0.0

if __name__ == "__main__":
    

    env = Environment()
    
    import timeit  
    done = False
    
    while not done:
        start = timeit.default_timer()
        action, array = env.generate_actions()
   
        stop = timeit.default_timer()
        execution_time = stop - start
        print("Executed Time "+str(execution_time))
        obs, obs2, obs3= env.get_observation()
        
        obs, reward, done = env.step(action)
        # print('reeeeward', reward)
        # score += reward

        if done == True:
            env.reset()
       
            obs, obs2, obs3 = env.get_observation()
            # print("Random reward  : ", reward)
            print("Total reward got : ", env.total_reward)


