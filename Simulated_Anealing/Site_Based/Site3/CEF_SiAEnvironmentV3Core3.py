#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 10:54:37 2022

@author: seifu
"""

import numpy as np
import math
from  CEFSA_Propagation import Propagation
from sklearn import preprocessing
# from Plots import plot_learning_curve
# from matplotlib import pyplot
import CEF_SiAdelayCore3
# huge = 100

class Environment:
    def __init__(self):
      
# topology settings
        self.NumCloudNetwork = 1
        self.NumCoreNetwork =3
        self.NumAcceNetwork =8
        
        self.steps_left = 1
        self.total_latency = 0.0
        self.total_reward= 0.0
        self.done = False 
        
        self.DownLinkBandWd = 1000000
        self.UpLinkBandWd = 1000000
        
# Arrival traffic rate at jth AN-CEF site of ith CN-CEF site
        # self.lam = lam
        # np.random.seed(0)
        # self.lamda = np.random.poisson(lam=self.lam, size=(self.NumCoreNetwork, self.NumAcceNetwork))
        
        self.lam = (40000,80000,40000,80000,40000,80000,40000,80000)
        
        # self.lam = (80000,80000,80000,80000,80000,80000)
        
        # self.lam = (80000,80000,80000,80000,80000,80000)
        
        # self.lam = (80000,80000,80000,80000,80000,80000)
        
        # self.lam = (80000,80000,80000,80000,80000,80000)
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

        self.propagationCoreToCloudNetwork = Propagation("CoreNetwork_to_CloudNetwork.csv").propagation_delay
        self.propagationAccToCoreNetwork = Propagation("AccessNetwork_to_CoreNetwork.csv").propagation_delay
        self.propagationAccToAccesNetwork = Propagation("AccessNetwork_to_AccessNetwork.csv").propagation_delay
        self.propagationCoreToCoreNetwork = Propagation("CoreNetwork_to_CoreNetwor.csv").propagation_delay

        # self.propagationCoreToCloudNetwork = Propagation("CORTOCLONCEF5site.csv").propagation_delay
        # self.propagationAccToCoreNetwork = Propagation("ACTOCORNCEF2.csv").propagation_delay
        # self.propagationAccToAccesNetwork = Propagation("ACTOACNCEF.csv").propagation_delay
        # self.propagationCoreToCoreNetwork = Propagation("CORTOCORNCEF2site.csv").propagation_delay

        # self.neighbourCoreNetwork = Propagation("CNCEF.csv").neighbour
        # self.neighbourAccesNetwork = Propagation("ANCEF.csv").neighbour
        
        # self.actions = [[[0 for i in range(self.NumCloudNetwork+self.NumCoreNetwork+self.NumAcceNetwork)] for j in range(self.NumAcceNetwork)] \
        #                   for k in range(self.NumCoreNetwork)] 

        # for i in range(len(self.actions)):
        #     for j in range(len(self.actions[i])):
        #         for k in range(len(self.actions[i][j])):
        #             if j == k:
        #                 self.actions[i][j][k] = 1
        
    def get_observation(self):
        
        latency = self.latency
        lamda = self.lamda
        
        self.miuCloudNetwork = [3000000 for i in range(self.NumCloudNetwork)]       
        self.miuCoreNetwork = [270000 for i in range(self.NumCoreNetwork)]   
        self.miuAcceNetwork = [[30000 for i in range(self.NumAcceNetwork)] for j in range(self.NumCoreNetwork)] 
       
        miuAcce = self.miuAcceNetwork
        # for l in range(len(miuAcce)):
        #     for m in range(len(miuAcce[l])):
        #         miuAcce[l][m]-= self.AccesNetLamda[l][m]
        # print("miuAcce", miuAcce) 
        
        miuCore = self.miuCoreNetwork
        # for j in range(len(miuCore)):
        #     miuCore[j] -= self.CoreNetLamda[j]
        # print("miuCore", miuCore) 
        
        miuCloud = self.miuCloudNetwork       
        # for i in range(len(miuCloud)):
        #     miuCloud[i] -= self.CloudNetLamda[i]       
        
        scalr = preprocessing.RobustScaler()   # standardized data set
       
         # initial observation information        
        intial_Obs = [latency,lamda, miuAcce, miuCore, miuCloud]# initial observation information
        
        lat = scalr.fit_transform(np.array(latency).flatten().reshape(-1, 1))
        
        lmd = scalr.fit_transform(np.array(lamda).flatten().reshape(-1, 1))
        
        # bw = scalr.fit_transform(np.array([UpLinkBandWd,DownLinkBandWd]).reshape(-1, 1))
       
        mAcces = scalr.fit_transform(np.array(miuAcce).flatten().reshape(-1, 1))
        
        mCore = scalr.fit_transform(np.array(miuCore).flatten().reshape(-1, 1))
        
        mCloud = scalr.fit_transform(np.array(miuCloud).flatten().reshape(-1, 1))
        
        observe = np.concatenate((lat,lmd,mAcces,mCore,mCloud)).flatten()
        
        return intial_Obs,lat,observe 

# At the first time randomly generate the actions           
    def generate_actions(self):
        
        
        self.miuCloudNetwork = [3000000 for i in range(self.NumCloudNetwork)]       
        self.miuCoreNetwork = [270000 for i in range(self.NumCoreNetwork)]   
        self.miuAcceNetwork = [[30000 for i in range(self.NumAcceNetwork)] for j in range(self.NumCoreNetwork)] 
     
        miuAcce = self.miuAcceNetwork
        for l in range(len(miuAcce)):
            for m in range(len(miuAcce[l])):
                miuAcce[l][m]-= self.AccesNetLamda[l][m]
        # print("miuAcce", miuAcce) 
        
        miuCore = self.miuCoreNetwork
        for j in range(len(miuCore)):
            miuCore[j] -= self.CoreNetLamda[j]
        # print("miuCore", miuCore) 
        
        miuCloud = self.miuCloudNetwork       
        for i in range(len(miuCloud)):
            miuCloud[i] -= self.CloudNetLamda[i]
        
        # actions = []
        # for n in range(self.NumCloudNetwork):
        #     for m in range(self.NumCoreNetwork):
        #         randm_action = np.random.rand(self.NumAcceNetwork, self.NumAcceNetwork + \
        #                                       self.NumCoreNetwork + self.NumCloudNetwork)
        actions = [np.zeros((self.NumAcceNetwork, self.NumCoreNetwork+self.NumAcceNetwork+self.NumCloudNetwork)) for i in range(self.NumCoreNetwork)]        # randm_action= np.random.normal(scale=0.1,
                #                                 size=(self.NumAcceNetwork, self.NumAcceNetwork + self.NumCoreNetwork + self.NumCloudNetwork))                    
                # actions.append(randm_action)

        for i in range(self.NumCoreNetwork):
            for j in range(self.NumAcceNetwork):
                tmp = np.concatenate((np.array(miuAcce[i]).flatten(), np.array(miuCore).flatten(), np.array(miuCloud).flatten()))
                for k in range(len(tmp)):
                    actions[i][j][k] = tmp[k]/sum(tmp) 
        action_array =np.array(actions)
        # print("actions",actions)
        return actions, action_array
        
        # action = []

        # for i in range(self.NumCloudNetwork):
        #     for j in range(self.NumCoreNetwork):
        #         randm_action = np.random.rand(self.NumAcceNetwork,self.NumAcceNetwork+self.NumCoreNetwork+self.NumCloudNetwork)
        #         action.append(randm_action/randm_action.sum(axis=1)[:,None])

        # action_array = np.array(action)
        # # print("Action", action)
        # # print(" ")
        # # print("action_array",action_array)
        # return action, action_array.flatten()
        
        
# Load (Allocate) trffic    
    def Load_Traffic(self, actions):
        # actn = sorted(actions)
        task=[]
        # for n in range(self.NumCloudNetwork):
        for m in range(self.NumCoreNetwork):
            task.append(np.random.rand(self.NumAcceNetwork, self.NumAcceNetwork+self.NumCoreNetwork+self.NumCloudNetwork))
        # print("Tasks", task)
        # task = [np.zeros((self.NumAcceNetwork, self.NumCoreNetwork+self.NumAcceNetwork+self.NumCloudNetwork)) for i in range(self.NumCoreNetwork)] 
        for i in range(len(task)):
            
            for j in range(len(task[i])):
                
                for k in range(len(task[i][j])):
                    # print("AAAAA",actions)
                    task[i][j][k] = math.ceil(self.lamda[i][j]*np.round(actions[i][j][k],2))
        # print("Task", task)       
        return task
    
        
# Calculate latency for given actions
    def calculate_latency(self, obs, action):
        latency = obs[0]
        lamda = obs[1]
        miuAccesNet = obs[2]
        miuCoreNet = obs[3]
        miuCloudNet = obs[4]
        
        trafic = self.Load_Traffic(action)
        
        for i in range(len(latency)):
            for j in range(len(latency[i])):
                # AccesNetLamda = CEF_SiAdelayCore3.get_AccesNetwork_lamda(i,j, trafic)
                # # print('AccesNetLamdadddddddddd', AccesNetLamda)
                
                for k in range(len(latency[i][j])):
                
                    AccesNetLamda = CEF_SiAdelayCore3.get_AccesNetwork_lamda(i,k, trafic)
                    # print('AccesNetLamdadddddddddd', AccesNetLamda)
                    CoreNetLamda = CEF_SiAdelayCore3.get_CoreNetwork_lamda(k, trafic)
                    # print('lamdaCoreNet', CoreNetLamda)
                    CloudNetLamda = CEF_SiAdelayCore3.get_CloudNetwork_lamda(trafic)
                    # print('lamdaAccesNet', CloudNetLamda)
                    
                    # AccesLamda = CEF_SiAdelayCore3.get_AccesNetwork_lamda(i,j, trafic)
                    # # print("AccesNetLamda", AccesNetLamda)
                    # if (self.miuAcceNetwork[i][j] <= AccesLamda):
                    #     AccesNetLamda = ((AccesLamda-self.miuAcceNetwork[i][j])*0.08) + self.miuAcceNetwork[i][j] *0.65
                    #     # print("AccesNetLamda", AccesNetLamda)
                    # else:
                    #     AccesNetLamda = CEF_SiAdelayCore3.get_AccesNetwork_lamda(i,j, trafic) * 0.80
                    #     # print("Else_AccesNetLamda", AccesNetLamda)
                    # CoreLamda = CEF_SiAdelayCore3.get_CoreNetwork_lamda(j,trafic) + ((AccesLamda-AccesNetLamda) * 0.40)
                    # # print("CoreNetLamda", CoreNetLamda)
                    # if (self.miuCoreNetwork[i] <= CoreLamda):
                    #     CoreNetLamda = CEF_SiAdelayCore3.get_CoreNetwork_lamda(j,trafic) + ((AccesLamda-AccesNetLamda) * 0.40) * 0.90
                    #     # print("If_CoreNetLamda", CoreNetLamda)
                    # else:
                    #     CoreNetLamda = CEF_SiAdelayCore3.get_CoreNetwork_lamda(j,trafic) + ((AccesLamda-AccesNetLamda) * 0.40)
                    #     # print("Else_CoreNetLamda", CoreNetLamda)    
                    
                    # CloudNetLamda = CEF_SiAdelayCore3.get_CloudNetwork_lamda(trafic) + ((AccesLamda-AccesNetLamda) * 0.60)
                    #     # print("Else_CoreNetLamda", CoreNetLamda)  
                    
    ## Latency that is served at local Access Network
                    if j==k and k<self.NumAcceNetwork:
                        # print("miuAccesNet[i][j]", miuAccesNet[i][j])
                        # print("AccesLamda", self.AccesNetLamda[agent_id][n])
                        latency[i][j][k] = CEF_SiAdelayCore3.get_latency_serve_ByAN_CEF(self.UpLinkBandWd,self.DownLinkBandWd,miuAccesNet[i][j],AccesNetLamda,lamda[i][j])
                        # print("latency[i][j][k]LocalAn",latency[i][j][k])                                                               
     ## Latency that is served at AN Neighbor
                        # print("AccesLamdaNNN", self.AccesNetLamda[i][j])
                    elif k!=j and k < self.NumAcceNetwork:
                        Dist_Acc_Acc = self.propagationAccToAccesNetwork()[j][k]
                        # print("self.AccesNetLamda[i][j]", self.AccesNetLamda[i][j])
                        latency[i][j][k] = CEF_SiAdelayCore3.get_latency_servesd_Byneighbour_AN_CEF(self.UpLinkBandWd,self.DownLinkBandWd,miuAccesNet[i][j],AccesNetLamda,lamda[i][j],Dist_Acc_Acc) 
                        # print("latency[i][j][k]NeighbourAN",latency[i][j][k])
                        
                        # CoreNetLamda = CEF_SiAdelayCore3New.get_CoreNetwork_lamda(j,tasks)
                        # print("CoreNetLamda",CoreNetLamda)
                                                                              
       ## Latency that is served at local CN   3.789639064123363e-05
                        # print("self.CoreNetLamda[i]", self.CoreNetLamda[i])
                    elif self.NumAcceNetwork-1 < k < (self.NumAcceNetwork + self.NumCoreNetwork) and k == self.NumAcceNetwork+i:
                        Dist_Acc_Cor = self.propagationAccToCoreNetwork()[j][k-self.NumAcceNetwork]
                        
                        latency[i][j][k] = CEF_SiAdelayCore3.get_latency_served_ByCN_CEF(self.UpLinkBandWd,self.DownLinkBandWd,miuCoreNet[i],CoreNetLamda,lamda[i][j],Dist_Acc_Cor)#1.66666666666666666e-6
                        # print("latency[i][j][k]localCN",latency[i][j][k])
                        # print("self.CoreNetLamda[i]", self.CoreNetLamda[i])                  
        ## Latency that is served at CN neighbour    
                    elif self.NumAcceNetwork-1 < k < (self.NumAcceNetwork + self.NumCoreNetwork) and k != self.NumAcceNetwork+i:
                        Dist_Acc_Cor = self.propagationAccToCoreNetwork()[j][k-self.NumAcceNetwork]
                        Dist_Cor_Cor = self.propagationCoreToCoreNetwork()[i][k-self.NumAcceNetwork]
                                             
                        latency[i][j][k] = CEF_SiAdelayCore3.get_latency_served_Byneighbour_CN_CEF(self.UpLinkBandWd,self.DownLinkBandWd,miuCoreNet[i],CoreNetLamda,lamda[i][j],Dist_Acc_Cor,Dist_Cor_Cor)
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
                        latency[i][j][k]= CEF_SiAdelayCore3.get_latency_served_ByClN_CEF(self.UpLinkBandWd,self.DownLinkBandWd,miuCloudNet[i],CloudNetLamda,lamda[i][j],Dist_Acc_Cor,  Dist_Cor_Cloud)#1.6666666666666666e-4
                        # print("latency[i][j][k]",latency[i][j][k])
                        
        total_latency = 0.0
        
# Jth AN Ith CN of Cloud Latency
        for i in range(len(latency)):
            for j in range(len(latency[i])):
                for k in range(len(latency[i][j])):
                    # print("action[i][j][k]",action[i][j][k])
                    total_latency += (action[i][j][k]*latency[i][j][k])
        # print("total_latency",total_latency)
        # avg_lat=total_latency/self.NumAcceNetwork
# system average delay 𝑙(𝑖,𝑗)
                              
        # total_traffic = sum(sum(map(sum, trafic)))
        # avg_syst_latency = avg_lat/ total_traffic
# Average Latency (Average system latency)
        # avg_syst_latency = total_latency/total_traffic
        # avg_syst_latency = total_latency
# latency = 1/l 
        # latency = avg_syst_latency
        latency = total_latency
        # print("latency", latency)
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
        reward = 1/latency
        return latency, reward   
    
    
    def is_done(self, step):
        if step == 0:
            self.done = True
        else:
            self.done = False
        return self.done
    
    def step(self, actions):
        # nr_actions =[]
        # # for j in range(len(actions)):
        # totalAcc = sum(map(sum,actions))
        # # print("totalAcc", totalAcc)
        # for i in range(len(actions)):
        #     # print("acn",acn[i])
        #     nr_actions.append(actions[i]/totalAcc)
        # print("nr_actions",nr_actions) 
        # action = []

        # for i in range(len(actions)):
        #     for j in range(len(actions[i])):
        #         print("actions.sum(axis=1)[:,None])",actions.sum(axis=1)[:,None])
        #         action.append(actions[i]/actions.sum(axis=1)[:,None])

        # action_array = np.array(action)
        # self.actions =actions
        intial_Observation, obs,observe = self.get_observation()
        latency,reward = self.calculate_latency(intial_Observation, actions)
        self.total_latency += latency
        self.total_reward += reward
        
        trafic = self.Load_Traffic(actions)
        # print("trafic", trafic)
        # print("  ")
        Acn = [] 
        for r in trafic:
            
            Acn.append([sum(x) for x in zip(*r)])
        # print("Acn", Acn)
        # print(" ")
        Bcn = [sum(x) for x in zip(*Acn)]
        # print("Bcn", Bcn)
        # Ccn = [sum(Bcn)]
        # print("Ccn", Ccn)
        # This update the AN and CN total traffic
        for l in range(len(self.CloudNetLamda)):
            self.CloudNetLamda[l] += Bcn[self.NumAcceNetwork+self.NumCoreNetwork+l]
            # print(" ")
            # print('self.CloudNetLamda[l]', self.CloudNetLamda[l])
        for i in range(len(self.CoreNetLamda)):
            # print('I', l)
            self.CoreNetLamda[i] += Bcn[self.NumAcceNetwork+i]
            # print('self.CoreNetLamda[i]', self.CoreNetLamda[i])       
        for j in range(len(self.AccesNetLamda)):
            
            for k in range(len(self.AccesNetLamda[j])):
                
                self.AccesNetLamda[j][k] += Acn[j][k]
                # print("self.AccesNetLamda[j][k]", self.AccesNetLamda[j][k])       
        
        intial_Observation,obs ,observe = self.get_observation()
        
        self.steps_left -= 1
        done = self.is_done(self.steps_left)
        if self.done == True:
            self.reset()
        # # return observe, latency, done
        return observe, latency, reward, done
    
    def reset(self):
        self.steps_left =1
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
    
    import timeit  
    done = False
    
    while not done:
        start = timeit.default_timer()
        obs,obs2, obs3= env.get_observation()
        actions, array = env.generate_actions()
        
        stop = timeit.default_timer()
        execution_time = stop - start
        print("Executed Time "+str(execution_time))
        # obs,obs2, obs3= env.get_observation()
        obs, latency, reward, done = env.step(actions)
        # print("action", actions)
        # print("latency: ", latency)
        # print("  ")
        # print("Lambda : ", env.lamda)
        # print("  ")
        # print("lam : ", env.lam)
        # print("  ")
        print("AN trafic distribution  : ", env.AccesNetLamda)
        print("Sum : ", sum(map(sum,env.AccesNetLamda)))
        print("  ")
        print("CON trafic distribution: ", env.CoreNetLamda)
        print("Sum : ", sum(env.CoreNetLamda))
        print("  ")
        print("CLN trafic distribution : ", env.CloudNetLamda)
        print("Sum : ", sum(env.CloudNetLamda))
        print("  ")
        print("Total Trafic: ", sum(map(sum,env.AccesNetLamda))+sum(env.CoreNetLamda)+sum(env.CloudNetLamda))
        print("  ")
        if done == True:
            env.reset()
       
            obs, obs2,obs3 = env.get_observation()
            print(" ")
            print("Total latency got : ", env.total_latency)
            print("Total reward got : ", env.total_reward)
            print(" ")
    state_dim = env.get_observation()[2].shape[0]
    print("state_dim  =============>", state_dim)
    actor_action_dim = env.NumAcceNetwork+env.NumCoreNetwork+env.NumCloudNetwork
    print("actor_action_dim =======>", actor_action_dim)
    critic_action_dim = env.NumCoreNetwork*env.NumAcceNetwork*actor_action_dim
    print("critic_action_dim =======>", critic_action_dim)  
            
