#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 10:45:11 2022

@author: seifu
"""

import random
import numpy as np
import math
from  CEF_Propagation import Propagation
from sklearn import preprocessing
import CEF_MAdelayCore80k

huge = 100

class Environment: 
    def __init__(self):
      
# topology settings
        self.NumCloudNetwork = 1
        self.NumCoreNetwork = 4
        self.NumAcceNetwork = 8
        
        self.agent = self.NumCoreNetwork
        self.steps_left = 10 *self.agent
        self.total_latency = 0.0
        self.total_reward = 0.0
        self.done = False        

# topology settings
        self.DownLinkBandWd = 1000000
        self.UpLinkBandWd = 1000000    
    
# Arrival traffic rate at jth AN-CEF site of ith CN-CEF site
        # np.random.seed(1)
        # self.lamda = [np.random.randint(low = 2000, high = 8000, size = self.NumAcceNetwork) \
        #               for i in range(self.NumCoreNetwork)]
        
        self.lam = (80000,80000,80000,80000,80000,80000,80000,80000)
        
        np.random.seed(1)
        self.lamda = [np.random.poisson(lam=self.lam, size=self.NumAcceNetwork) \
                      for i in range(self.NumCoreNetwork)]

# Array to collect latency totall latency(initial will 0) updated during calculation
        self.latency = [np.zeros((self.NumAcceNetwork, self.NumAcceNetwork+self.NumCoreNetwork+self.NumCloudNetwork)) for i in range(self.NumCoreNetwork)]
       
# Inttail Traffic that is served at Access, core and cloud network
        self.AccesNetLamda = [[0 for i in range(self.NumAcceNetwork)] for j in range(self.NumCoreNetwork)] 
        self.CoreNetLamda = [0 for i in range(self.NumCoreNetwork)]       
        self.CloudNetLamda = [0 for i in range(self.NumCloudNetwork)]

# Initial capacity for each tiers and sites       
        self.miuCloudNetwork = [3000000 for i in range(self.NumCloudNetwork)]     
        self.miuCoreNetwork = [270000 for i in range(self.NumCoreNetwork)]   
        self.miuAcceNetwork = [[30000 for i in range(self.NumAcceNetwork)] for j in range(self.NumCoreNetwork)] 

#method calling for fining distance      
        # self.propagationCloudNetwork = Propagation("clcef.csv").propagation_delay
        # self.propagationCoreNetwork = Propagation("cncef.csv").propagation_delay
        # self.propagationAccesNetwork = Propagation("ancef.csv").propagation_delay
        
        # self.propagationCoreToCloudNetwork = Propagation("CORTOCLONCEF5site.csv").propagation_delay
        # self.propagationAccToCoreNetwork = Propagation("ACTOCORNCEF2.csv").propagation_delay
        # self.propagationAccToAccesNetwork = Propagation("ACTOACNCEF.csv").propagation_delay
        # self.propagationCoreToCoreNetwork = Propagation("CORTOCORNCEF2site.csv").propagation_delay

        self.propagationCoreToCloudNetwork = Propagation("CoreNetwork_to_CloudNetwork.csv").propagation_delay
        self.propagationAccToCoreNetwork = Propagation("AccessNetwork_to_CoreNetwork.csv").propagation_delay
        self.propagationAccToAccesNetwork = Propagation("AccessNetwork_to_AccessNetwork.csv").propagation_delay
        self.propagationCoreToCoreNetwork = Propagation("CoreNetwork_to_CoreNetwor.csv").propagation_delay

        
        self.actions = [[[0 for i in range(self.NumAcceNetwork+self.NumCoreNetwork+self.NumCloudNetwork)] for j in range(self.NumAcceNetwork)] for k in range(self.NumCoreNetwork)]

        for i in range(len(self.actions)):
            for j in range(len(self.actions[i])):
                for k in range(len(self.actions[i][j])):
                    if j == k:
                        self.actions[i][j][k] = 1
                        
        # print("self.actions", self.actions)
        
    def get_observation(self, agent_id):
        latency = self.latency
        lamda = self.lamda
        
        self.miuCloudNetwork = [3000000 for i in range(self.NumCloudNetwork)]       
        self.miuCoreNetwork = [270000 for i in range(self.NumCoreNetwork)]   
        self.miuAcceNetwork = [[30000 for i in range(self.NumAcceNetwork)] for j in range(self.NumCoreNetwork)] 
       
        miuAccesNet = self.miuAcceNetwork
       
        miuCoreNet = self.miuCoreNetwork
        
        miuCloudNet = self.miuCloudNetwork
        
        scalr = preprocessing.RobustScaler()   # standardized data set
       
        intial_Observation = [latency, lamda, miuAccesNet, miuCoreNet, miuCloudNet] # initial observation information
        # intial_Observation = [latency, lamda, UpLinkBandWd, DownLinkBandWd, miuAccesNet, miuCoreNet, miuCloudNet] # initial observation information
        
        lat = scalr.fit_transform(np.array(latency[agent_id]).flatten().reshape(-1, 1))
        lmd = scalr.fit_transform(np.array(lamda[agent_id]).flatten().reshape(-1, 1))
        # bw = scalr.fit_transform(np.array([UpLinkBandWd,DownLinkBandWd]).reshape(-1, 1))
        mAcces = scalr.fit_transform(np.array(miuAccesNet[agent_id]).flatten().reshape(-1, 1))
        mCore = scalr.fit_transform(np.array(miuCoreNet).flatten().reshape(-1, 1))
        mCloud = scalr.fit_transform(np.array(miuCloudNet).flatten().reshape(-1, 1))
        
        observe = np.concatenate((lat,lmd,mAcces,mCore,mCloud)).flatten()      # oservation after step started
        
        return intial_Observation,lat,observe 

# At the first time randomly generate the actions           
    def generate_actions(self, agent_id):
        
        Action = []
        for n in range(self.NumCloudNetwork):
            for m in range(self.NumCoreNetwork):
                randm_action = np.random.rand(self.NumAcceNetwork, self.NumAcceNetwork + \
                                              self.NumCoreNetwork + self.NumCloudNetwork)
                Action.append(randm_action/randm_action.sum(axis=1)[:,None])

        action_array = np.array(Action)

        return Action[agent_id], action_array[agent_id].flatten()
        
        # self.miuCloudNetwork = [3000000 for i in range(self.NumCloudNetwork)]       
        # self.miuCoreNetwork = [270000 for i in range(self.NumCoreNetwork)]   
        # self.miuAcceNetwork = [[30000 for i in range(self.NumAcceNetwork)] for j in range(self.NumCoreNetwork)]
        
        # miuAccesNet = self.miuAcceNetwork
        # for i in range(len(miuAccesNet)):
        #     for j in range(len(miuAccesNet[i])):
        #         miuAccesNet[i][j]-= self.AccesNetLamda[i][j] 
        #         # print("miuAccesNet[i][j]", miuAccesNet[i][j])
        # miuCoreNet = self.miuCoreNetwork
        # for m in range(len(miuCoreNet)):
        #     miuCoreNet[m] -= self.CoreNetLamda[m]
        # #     # print("miuCoreNet[m]", miuCoreNet[m])
        # miuCloudNet = self.miuCloudNetwork
        # for m in range(len(miuCloudNet)):
        #     miuCloudNet[m] -= self.CloudNetLamda[m]
        # #     # print("miuCloudNet[m]", miuCloudNet[m])
        # act = [np.zeros((self.NumAcceNetwork, self.NumAcceNetwork+self.NumCoreNetwork+self.NumCloudNetwork)) for i in range(self.NumCoreNetwork)]
        # # print("act=1 ", act)
        # # act2 = miuAN
        # for i in range(self.NumCoreNetwork):
        #     for j in range(self.NumAcceNetwork):
        #         tmp = np.concatenate((np.array(miuAccesNet[i]).flatten(), np.array(miuCoreNet).flatten(), np.array(miuCloudNet).flatten()))
        #         for k in range(len(tmp)):
        #             act[i][j][k] = tmp[k]/sum(tmp)                    
        #             # print("act[i][j][k]", act[i][j][k])

        # return act[agent_id], np.array(act[agent_id]).flatten()  
        

        # actions = [[[0 for i in range(self.NumCloudNetwork+self.NumCoreNetwork+self.NumAcceNetwork)] for j in range(self.NumAcceNetwork)] for k in range(self.NumCoreNetwork)]
        # # act = [np.zeros((self.NumAcceNetwork, self.NumCoreNetwork+self.NumAcceNetwork+self.NumCloudNetwork)) for i in range(self.NumCoreNetwork) for j in range(self.NumCloudNetwork)]
        
        # for i in range(self.NumCloudNetwork):
        #     for j in range(self.NumCoreNetwork):
        #         # for k in range(self.NumAcceNetwork):
        #         tmp = np.concatenate((np.array(miuAccesNet[i]).flatten(), np.array(miuCoreNet).flatten(), np.array(miuCloudNet).flatten()))
        #         for k in range(len(tmp)):
        #             actions[i][j][k] = tmp[k]/sum(tmp)
        #                 # print("act[i][j][k]", act[i][j][k])                    
        # # print("act[agent_id]", act[agent_id])
        # print("act[agent_id]", actions[agent_id])
        # return actions[agent_id], np.array(actions[agent_id]).flatten()
    
# Load (Allocate) trffic    
    def Load_Traffic(self, action):
                    
        task=[]
        for n in range(self.NumCoreNetwork):
            # for m in range(self.NumCoreNetwork):
            task.append(np.random.rand(self.NumAcceNetwork, self.NumAcceNetwork +self.NumCoreNetwork + self.NumCloudNetwork))
                
        for i in range(len(task)):
            # print("I", i)
            for j in range(len(task[i])):
                for k in range(len(task[i][j])):
                    # print("action[i][j][k]",action[i][j][k])
                    task[i][j][k] = math.ceil(self.lamda[i][j]*np.round(action[i][j][k],2))
                # print("task[i][j][k]", task[0][2][0])
        # trafic=[]
        # # print('trafic=[]', trafic)
        # for n in range(len(task)):
        #     trafic.append(task[n].sum(axis=1))
        #     # print('trafic=[]', trafic)
        #     # print(" ")
        # for i in range(len(trafic)):
        #     for j in range(len(trafic[i])):
        #         if trafic[i][j] < self.lamda[i][j]:
        #             task[i][j][random.randint(0,len(task[i][j])-1)] += self.lamda[i][j] - trafic[i][j]
        #             # print("if",task[i][j][random.randint(0,len(task[i][j])-1)])
        #             # print("  ")
        #         elif trafic[i][j] > self.lamda[i][j]:
        #               task[i][j][random.randint(0,len(task[i][j])-1)] -= trafic[i][j] - self.lamda[i][j]
        #               # print("else",task[i][j][random.randint(0,len(task[i][j])-1)])
        #         else:
        #             continue
                
        # print('tasksssss', task)
        return task
    
# Calculate latence for given actions
    def calculate_latency(self, obs, action, agent_id):
        latency = obs[0]
        lamda = obs[1]
        
        miuAccesNet = obs[2]
        miuCoreNet = obs[3]
        miuCloudNet = obs[4]
        
        trafic = self.Load_Traffic(action)
        # print("Trafic", trafic)
        for i in range(len(latency)):
            # print("I", i) #[Number of core networks]           
            # AccesNetLamda =0
            for j in range(len(latency[i])):
                # AccesNetLamda = CEF_MAdelayCore80k.get_AccesNetwork_lamda(i,j, trafic)
                # # print("AccesNetLamda", AccesNetLamda)
                # if miuAccesNet<AccesNetLamda:
                #     AccesNetLamda=CEF_MAdelayCore80k.get_AccesNetwork_lamda(i,j, trafic)*0.35
                for k in range(len(latency[i][j])):
                    
                    AccesLamda = CEF_MAdelayCore80k.get_AccesNetwork_lamda(i,j, trafic)
                    # print("AccesNetLamda", AccesNetLamda)
                    if (self.miuAcceNetwork[i][j] < AccesLamda):
                        # AccesNetLamda = self.miuAcceNetwork[i][j] * 0.8
                        AccesNetLamda = ((AccesLamda-self.miuAcceNetwork[i][j])*0.08) + self.miuAcceNetwork[i][j] *0.65
                        # print("If_AccesNetLamda", AccesNetLamda)
                    else:
                        AccesNetLamda = CEF_MAdelayCore80k.get_AccesNetwork_lamda(i,j, trafic) * 0.80
                        # print("Else_AccesNetLamda", AccesNetLamda)
                    CoreLamda = CEF_MAdelayCore80k.get_CoreNetwork_lamda(j,trafic)
                    # print("CoreNetLamda", CoreNetLamda)
                    if (self.miuCoreNetwork[i] < CoreLamda):
                        CoreNetLamda = (CoreLamda *0.65) + (AccesLamda - AccesNetLamda) *0.3 
                        # print("If_CoreNetLamda", CoreNetLamda)
                    else:
                        CoreNetLamda = CoreLamda + (AccesLamda - AccesNetLamda) *0.3
                        # print("Else_CoreNetLamda", CoreNetLamda)    
                   
                    CloudNetLamda = CEF_MAdelayCore80k.get_CloudNetwork_lamda(trafic) + (AccesLamda - (AccesNetLamda +((AccesLamda - AccesNetLamda) *0.3)))
                        # print("If_CoreNetLamda", CoreNetLamda) 
                    
    ## Latency that is served at local Access Network
                    if j==k and k<self.NumAcceNetwork:
                        # print("miuAccesNet[i][j]", miuAccesNet[i][j])
                        # print("AccesLamda", self.AccesNetLamda[agent_id][n])
                        latency[i][j][k] = CEF_MAdelayCore80k.get_latency_serve_ByAN_CEF(self.UpLinkBandWd,self.DownLinkBandWd,miuAccesNet[i][j],AccesNetLamda,lamda[i][j])
                        # print("latency[i][j][k]LocalAn",latency[i][j][k])                                                               
     ## Latency that is served at AN Neighbor
                        # print("AccesLamdaNNN", self.AccesNetLamda[i][j])
                    elif k!=j and k < self.NumAcceNetwork:
                        Dist_Acc_Acc = self.propagationAccToAccesNetwork()[j][k]
                        # print("self.AccesNetLamda[i][j]", self.AccesNetLamda[i][j])
                        latency[i][j][k] = CEF_MAdelayCore80k.get_latency_servesd_Byneighbour_AN_CEF(self.UpLinkBandWd,self.DownLinkBandWd,miuAccesNet[i][j],AccesNetLamda,lamda[i][j],Dist_Acc_Acc) 
                        # print("latency[i][j][k]NeighbourAN",latency[i][j][k])
                        
                        # CoreNetLamda = CEF_MAdelayCore80kNew.get_CoreNetwork_lamda(j,tasks)
                        # print("CoreNetLamda",CoreNetLamda)
                                                                              
       ## Latency that is served at local CN   3.789639064123363e-05
                        # print("self.CoreNetLamda[i]", self.CoreNetLamda[i])
                    elif self.NumAcceNetwork-1 < k < (self.NumAcceNetwork + self.NumCoreNetwork) and k == self.NumAcceNetwork+i:
                        Dist_Acc_Cor = self.propagationAccToCoreNetwork()[j][k-self.NumAcceNetwork]
                        
                        latency[i][j][k] = CEF_MAdelayCore80k.get_latency_served_ByCN_CEF(self.UpLinkBandWd,self.DownLinkBandWd,miuCoreNet[i],CoreNetLamda,lamda[i][j],Dist_Acc_Cor)#1.66666666666666666e-6
                        # print("latency[i][j][k]localCN",latency[i][j][k])
                        # print("self.CoreNetLamda[i]", self.CoreNetLamda[i])                  
        ## Latency that is served at CN neighbour    
                    elif self.NumAcceNetwork-1 < k < (self.NumAcceNetwork + self.NumCoreNetwork) and k != self.NumAcceNetwork+i:
                        Dist_Acc_Cor = self.propagationAccToCoreNetwork()[j][k-self.NumAcceNetwork]
                        Dist_Cor_Cor = self.propagationCoreToCoreNetwork()[i][k-self.NumAcceNetwork]
                                             
                        latency[i][j][k] = CEF_MAdelayCore80k.get_latency_served_Byneighbour_CN_CEF(self.UpLinkBandWd,self.DownLinkBandWd,miuCoreNet[i],CoreNetLamda,lamda[i][j],Dist_Acc_Cor,Dist_Cor_Cor)
                        # print("latency[i][j][k]Neighbour",latency[i][j][k])                                                                                                                                      
      ## Latency that is served at Cloud
      #Dist_Acc_Cor =3.3333333333333333e-6
      # Dist_Cor_Cloud=3.3333333333333333e-4
                        
                    elif (self.NumCoreNetwork+self.NumAcceNetwork)-1 < k < (self.NumAcceNetwork+self.NumCoreNetwork+ \
                                                                          self.NumCloudNetwork) and k == self.NumCoreNetwork+self.NumAcceNetwork+i:
                        
                        Dist_Acc_Cor = self.propagationAccToCoreNetwork()[j][k-self.NumAcceNetwork]
                        Dist_Cor_Cloud = self.propagationCoreToCloudNetwork()[i][k-self.NumAcceNetwork+self.NumCloudNetwork]
                        # print("Dist_Acc_Cor", Dist_Acc_Cor)
                        # print("CloudNetLamda[i]", CloudNetLamda[i])
                        latency[i][j][k]= CEF_MAdelayCore80k.get_latency_served_ByClN_CEF(self.UpLinkBandWd,self.DownLinkBandWd,miuCloudNet[i],CloudNetLamda,lamda[i][j],Dist_Acc_Cor,  Dist_Cor_Cloud)#1.6666666666666666e-4
                        # print("latency[i][j][k]",latency[i][j][k])
                        
        total_latency = 0.0
        # print("Before total_latency", total_latency)
# Jth AN Ith CN of Cloud Latency
        # for i in range(len(latency[agent_id])):
        for j in range(len(latency[agent_id])):
            for k in range(len(latency[agent_id][j])):
                total_latency += (action[agent_id][j][k]*latency[agent_id][j][k])
        # avg_lat=total_latency/self.NumAcceNetwork       
        # print("After total_latency", total_latency)
# system average delay 𝑙(𝑖,𝑗)                                    
        # total_traffic_task = sum(map(sum, trafic[agent_id]))
        
# Average Latency (Average system latency)
        # print("total_traffic_Tasks",total_traffic_task)
        # print(" " )
        # avg_syst_latency = avg_lat/total_traffic_task
        # avg_syst_latency = total_latency/total_traffic_task
        # avg_syst_latency = avg_lat
        # print("avg_syst_latency",avg_syst_latency)
        # print(" " )       
        # latence = avg_syst_latency
        latence = total_latency
        self.AccesNetLamda = [[0 for i in range(self.NumAcceNetwork)] for j in range(self.NumCoreNetwork)] 
        self.CoreNetLamda = [0 for i in range(self.NumCoreNetwork)]
        self.CloudNetLamda = [0 for i in range(self.NumCloudNetwork)]
        if self.done == True:
            self.total_latency = 0.0
            self.total_reward = 0.0
            self.reset()
            self.AccesNetLamda = [[0 for i in range(self.NumAcceNetwork)] for j in range(self.NumCoreNetwork)] 
            self.CoreNetLamda = [0 for i in range(self.NumCoreNetwork)]
            self.CloudNetLamda = [0 for i in range(self.NumCloudNetwork)]
        # print("Latency", latence)
        reward = 1/latence
        return latence, reward
    
    def is_done(self, step):
        if step == 0:
            self.done = True
        else:
            self.done = False
        return self.done
    
    def step(self, actions, agent_id):
        self.actions[agent_id] = actions
        intial_Observation, obs2, observe = self.get_observation(agent_id)
        latence, reward = self.calculate_latency(intial_Observation, self.actions, agent_id)
        # print("latence",  intial_Observation)
        self.total_latency += latence
        self.total_latency += reward
        
        trafic = self.Load_Traffic(self.actions)

        # Actions is the distributed for each ANMEC, this total of the column to get total traffic in each AN site
        A = [sum(x) for x in zip(*trafic[agent_id])]
        
        # This update the AN and CN total traffic
        for l in range(len(self.CloudNetLamda)):
            self.CloudNetLamda[l] += A[self.NumAcceNetwork+self.NumCoreNetwork+l]
            # print("self.CloudNetLamda[l]", self.CloudNetLamda[l])
            
        for i in range(len(self.CoreNetLamda)):
            self.CoreNetLamda[i] += A[self.NumAcceNetwork+i]
            # print("self.CoreNetLamda[i]", self.CoreNetLamda[i])
            
        # print("self.AccesNetLamda[agent_id]", self.AccesNetLamda[agent_id])
        for k in range(len(self.AccesNetLamda[agent_id])):
            self.AccesNetLamda[agent_id][k] += A[k]
            # print("self.AccesNetLamda[agent_id][k]", self.AccesNetLamda[agent_id][k])
            
              
        intial_Observation, obs2, observe = self.get_observation(agent_id)
        
        self.steps_left -= 1
        done = self.is_done(self.steps_left)
        # print("latence",latence)
        return observe, latence, reward, done
    
    def reset(self):
        # self.total_latency = 0.0
        self.steps_left = 10 *self.agent
        self.AccesNetLamda = [[0 for i in range(self.NumAcceNetwork)] for j in range(self.NumCoreNetwork)] 
        self.CoreNetLamda = [0 for i in range(self.NumCoreNetwork)]
        self.CloudNetLamda = [0 for i in range(self.NumCloudNetwork)]
        
        self.latency = [np.zeros((self.NumAcceNetwork, self.NumAcceNetwork+self.NumCoreNetwork+self.NumCloudNetwork)) \
                        for i in range(self.NumCoreNetwork)]
        # self.lamda = [np.random.randint(low = 2000, high = 8000, size = self.NumAcceNetwork) for i in range(self.NumCoreNetwork)]
        self.lamda = [np.random.poisson(lam=self.lam, size=self.NumAcceNetwork) \
                       for i in range(self.NumCoreNetwork)]
        self.UpLinkBandWd = 1000000
        self.DownLinkBandWd = 1000000
        
        self.miuCloudNetwork = [3000000 for i in range(self.NumCloudNetwork)]
        
        self.miuCoreNetwork = [270000 for i in range(self.NumCoreNetwork)]   
        self.miuAcceNetwork = [[30000 for i in range(self.NumAcceNetwork)] for j in range(self.NumCoreNetwork)]
        self.done = False

if __name__ == "__main__":
    

    env = Environment()
    # print("Actions", env.self.actions)
    import timeit  
    done = False
    
    # def getSizeOfNestedList(listElemnet):
    #     count = 0
    #     #Iterte over the list
    #     for elem in listElemnet:
    #         # check if type of element is list
    #         if type(elem) == list:
    #             count +=getSizeOfNestedList(elem)
    #         else:
    #             count +=1
    #     return count
            
    while not done:
        start = timeit.default_timer()
        cum_latence = 0
        cum_reward = 0
        # print("Env>agent", env.agent)
        for agent_id in range(env.agent):
            # print("agent_id", agent_id)
            obs, obs2, obs3= env.get_observation(agent_id)
            action, array = env.generate_actions(agent_id)
            # action= [[0.09293392, 0.09176855, 0.0952002 , 0.08609746, 0.09068488,
            #        0.09113586, 0.09383239, 0.08710034, 0.08821332, 0.08960839,
            #        0.09342471],[0.08713596, 0.0898193 , 0.08993352, 0.09434062, 0.09415349,
            #        0.08711883, 0.0916327 , 0.09181905, 0.09440103, 0.08972584,
            #        0.08991961], [0.08808156, 0.09504043, 0.08765328, 0.09453638, 0.09019633,
            #        0.09646421, 0.09145199, 0.08994867, 0.08774105, 0.09055617,
            #        0.08832989], [0.09289624, 0.09532161, 0.09042565, 0.09317867, 0.08925668,
            #        0.09317803, 0.08870202, 0.08835751, 0.08635185, 0.09549768,
            #        0.08683414], [0.09138437, 0.08985548, 0.09462575, 0.090605  , 0.09586018,
            #        0.08698676, 0.0889536 , 0.09025422, 0.08839212, 0.09597522,
            #        0.08710741], [0.09512476, 0.0907176 , 0.09007521, 0.08723105, 0.09268889,
            #        0.08768371, 0.09323131, 0.09208861, 0.08508853, 0.09144267,
            #        0.09462768], [0.09227143, 0.08911742, 0.09260148, 0.08934538, 0.09346   ,
            #        0.0919222 , 0.09251067, 0.08854633, 0.08913276, 0.08861964,
            #        0.09247261], [0.0925478 , 0.08865507, 0.0954776 , 0.08818795, 0.08920262,
            #        0.09031913, 0.09574074, 0.09078348, 0.08950942, 0.08974963, 0.08982656]]
            # obs, latence, done = env.step(action, agent_id)
            # cum_latence += latence
   
        stop = timeit.default_timer()
        execution_time = stop - start
        print("Executed Time "+str(execution_time))
        obs, latence, reward, done = env.step(action, agent_id)
        cum_latence += latence
        cum_reward += reward
        # print("cummulative_latence=", cum_latence)
        # print("AN====>", env.AccesNetLamda)
        # print("CN====>", sum(env.CoreNetLamda))
        # print("CN====>", sum(env.CloudNetLamda))        
        # print("Total: ", sum(map(sum, env.AccesNetLamda))+sum(env.CoreNetLamda)+sum(env.CloudNetLamda))
        # print("##################################################")
        # print("done=", done)
        if done == True:
            env.reset()
            # print("AN afyer : ",env.AccesNetLamda)
            # print("CN afyer : ",env.CoreNetLamda)
            # print("CL afyer : ",env.CloudNetLamda)            
            # print("self.total_latency = 0.0 : ",env.total_latency)
            obs, obs2, obs3 = env.get_observation(agent_id)
            print(" ")
        
            print("Total Latency got : ", cum_latence)
            print(" ")
            print("Total Reward got : ", cum_reward) 
            print(" ")
        # print('reeeeward', latence)
        # score += latence

        # if done == True:
        #     env.reset()
       
        #     obs, obs2, obs3 = env.get_observation(agent_id)
        #     # print("Random latence  : ", latence)
        #     print("Total Latency got : ", env.total_latency)
        
    # stop = timeit.default_timer()
    # execution_time = stop - start
    # print("Executed Time "+str(execution_time))   
    state_dim = env.get_observation(1)[2].shape[0]
    print("state_dim  =============>", state_dim)
    Number_actions = env.NumAcceNetwork+env.NumCoreNetwork+env.NumCloudNetwork
    print("actor_action_dim =======>", Number_actions)
    critic_action_dim = (env.NumCoreNetwork+env.NumAcceNetwork+env.NumCloudNetwork)*env.NumAcceNetwork
    # critic_action_dim = (env.NumCoreNetwork+env.NumAcceNetwork+env.NumCloudNetwork)*(env.NumAcceNetwork*env.NumCoreNetwork)
    print("critic_action_dim =======>", critic_action_dim)