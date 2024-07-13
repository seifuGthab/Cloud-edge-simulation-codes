#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 12:20:10 2022

@author: seifu
"""
import numpy as np
import math
from  CEF_Propagation import Propagation
from sklearn import preprocessing
import CEF_delayCore3


class Environment: 
    def __init__(self):
      
# topology settings
        self.NumCloudNetwork = 1
        self.NumCoreNetwork = 3
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
        
        self.lam = (40000,80000,40000,80000,40000,80000,40000,80000)
        
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
                        
        
    def get_observation(self, agent_id):
        latency = self.latency
        lamda = self.lamda
        
        self.miuCloudNetwork = [3000000 for i in range(self.NumCloudNetwork)]       
        self.miuCoreNetwork = [270000 for i in range(self.NumCoreNetwork)]   
        self.miuAcceNetwork = [[30000 for i in range(self.NumAcceNetwork)] for j in range(self.NumCoreNetwork)] 
       
        miuAccesNet = self.miuAcceNetwork
        # for i in range(len(miuAccesNet)):
        #     for j in range(len(miuAccesNet[i])):
        #         miuAccesNet[i][j]-= self.AccesNetLamda[i][j] 
        # #         # print("miuAccesNet[i][j]", miuAccesNet[i][j])
        miuCoreNet = self.miuCoreNetwork
        # for m in range(len(miuCoreNet)):
        #     miuCoreNet[m] -= self.CoreNetLamda[m]
        # # #     # print("miuCoreNet[m]", miuCoreNet[m])
        miuCloudNet = self.miuCloudNetwork
        # for n in range(len(miuCloudNet)):
        #     miuCloudNet[n] -= self.CloudNetLamda[n]
        # # #     # print("miuCloudNet[m]", miuCloudNet[m])
        
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
        
        # Action = []
        # for n in range(self.NumCloudNetwork):
        #     for m in range(self.NumCoreNetwork):
        #         randm_action = np.random.rand(self.NumAcceNetwork, self.NumAcceNetwork + \
        #                                       self.NumCoreNetwork + self.NumCloudNetwork)
                    
        #         Action.append(randm_action/randm_action.sum(axis=1)[:,None])
        #         # print("Action",np.shape(Action))
        # # Ac = []
        # # for n in range(self.NumCloudNetwork):
        # #     for m in range(self.NumCoreNetwork):
        # #         Ac.append(Action[n][m].sort())
        # #         print("Action",Ac)
        # action_array = np.array(Action)
    
        # return Action[agent_id], action_array[agent_id].flatten()
        
        self.miuCloudNetwork = [3000000 for i in range(self.NumCloudNetwork)]       
        self.miuCoreNetwork = [270000 for i in range(self.NumCoreNetwork)]   
        self.miuAcceNetwork = [[30000 for i in range(self.NumAcceNetwork)] for j in range(self.NumCoreNetwork)]
        
        miuAccesNet = self.miuAcceNetwork
        for i in range(len(miuAccesNet)):
            for j in range(len(miuAccesNet[i])):
                miuAccesNet[i][j]-= self.AccesNetLamda[i][j] 
        #         # print("miuAccesNet[i][j]", miuAccesNet[i][j])
        miuCoreNet = self.miuCoreNetwork
        for m in range(len(miuCoreNet)):
            miuCoreNet[m] -= self.CoreNetLamda[m]
        # #     # print("miuCoreNet[m]", miuCoreNet[m])
        miuCloudNet = self.miuCloudNetwork
        for m in range(len(miuCloudNet)):
            miuCloudNet[m] -= self.CloudNetLamda[m]
        # #     # print("miuCloudNet[m]", miuCloudNet[m])
        act = [np.zeros((self.NumAcceNetwork, self.NumAcceNetwork+self.NumCoreNetwork+self.NumCloudNetwork)) for i in range(self.NumCoreNetwork)]
        # print("act=1 ", act)
        # act2 = miuAN
        for i in range(self.NumCoreNetwork):
            for j in range(self.NumAcceNetwork):
                tmp = np.concatenate((np.array(miuAccesNet[i]).flatten(), np.array(miuCoreNet).flatten(), np.array(miuCloudNet).flatten()))
                # print("tmp", tmp)
                for k in range(len(tmp)):
                    act[i][j][k] = tmp[k]/sum(tmp)                    
                    # print("act[i][j][k]", act[i][j][k])
        # print("act[agent_id]", act[agent_id])
        return act[agent_id], np.array(act[agent_id]).flatten()  
        

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
                    
        # task=[]
        # for n in range(self.NumCoreNetwork):
        # #     # for m in range(self.NumCoreNetwork):
        #     task.append(np.random.rand(self.NumAcceNetwork, self.NumAcceNetwork +self.NumCoreNetwork + self.NumCloudNetwork))
        task = [np.zeros((self.NumAcceNetwork, self.NumAcceNetwork+self.NumCoreNetwork+self.NumCloudNetwork)) for i in range(self.NumCoreNetwork)]   
        for i in range(len(task)):
            for j in range(len(task[i])):
                for k in range(len(task[i][j])):
                    task[i][j][k] = math.ceil(self.lamda[i][j]*np.round(action[i][j][k],2))
        # print("task", task)
        return task
        
    
# Calculate latence for given actions
    def calculate_latency(self, obs, action, agent_id):
        latency = obs[0]
        lamda = obs[1]
        
        miuAccesNet = obs[2]
        miuCoreNet = obs[3]
        miuCloudNet = obs[4]
        Ac=action
        trafc = self.Load_Traffic(Ac)
        
        for i in range(len(latency)):
            
            for j in range(len(latency[i])):
                
                for k in range(len(latency[i][j])):
                    AccesNetLamda = CEF_delayCore3.get_AccesNetwork_lamda(i,j, trafc)
                    # print("AccesLamda",AccesNetLamda)
                    CoreNetLamda = CEF_delayCore3.get_CoreNetwork_lamda(j,trafc)
                    # print("CoreLamda",CoreLamda)
                    CloudNetLamda = CEF_delayCore3.get_CloudNetwork_lamda(trafc)
                    # print("CloudLamda",CloudLamda)
                    # AccesLamda = CEF_delayCore3.get_AccesNetwork_lamda(i,j, trafic)
                    # # print("AccesNetLamda", AccesNetLamda)
                    # if (self.miuAcceNetwork[i][j] <= AccesLamda):
                    #     AccesNetLamda = ((AccesLamda-self.miuAcceNetwork[i][j])*0.08) + self.miuAcceNetwork[i][j] *0.65
                    #     # print("AccesNetLamda", AccesNetLamda)
                    # else:
                    #     AccesNetLamda = CEF_delayCore3.get_AccesNetwork_lamda(i,j, trafic) * 0.80
                    #     # print("Else_AccesNetLamda", AccesNetLamda)
                    # CoreLamda = CEF_delayCore3.get_CoreNetwork_lamda(j,trafic) 
                    # # print("CoreNetLamda", CoreLamda)
                    # if (self.miuCoreNetwork[i] <= CoreLamda):
                    #     CoreNetLamda = ((CoreLamda-self.miuCoreNetwork[i])*0.08) + self.miuCoreNetwork[i] *0.65
                    #     # print("If_CoreNetLamda", CoreNetLamda)
                    # else:
                    #     CoreNetLamda = CEF_delayCore3.get_CoreNetwork_lamda(j,trafic) + ((AccesLamda-AccesNetLamda) * 0.20)
                    #     print("Else_CoreNetLamda", CoreNetLamda)    
                    
                    # CloudNetLamda = CEF_delayCore3.get_CloudNetwork_lamda(trafic) + ((AccesLamda-AccesNetLamda) * 0.60)
                        # print("Else_CoreNetLamda", CoreNetLamda)
    ## Latency that is served at local Access Network
                    if j==k and k<self.NumAcceNetwork:
                        
                        latency[i][j][k] = CEF_delayCore3.get_latency_serve_ByAN_CEF(self.UpLinkBandWd,self.DownLinkBandWd,miuAccesNet[i][j],AccesNetLamda,lamda[i][j])
                        # print("latency[i][j][k]LocalAn",latency[i][j][k])                                                               
     ## Latency that is served at AN Neighbor
                        # print("AccesLamdaNNN", self.AccesNetLamda[i][j])
                    elif k!=j and k < self.NumAcceNetwork:
                        Dist_Acc_Acc = self.propagationAccToAccesNetwork()[j][k]
                        # print("self.AccesNetLamda[i][j]", self.AccesNetLamda[i][j])
                        latency[i][j][k] = CEF_delayCore3.get_latency_servesd_Byneighbour_AN_CEF(self.UpLinkBandWd,self.DownLinkBandWd,miuAccesNet[i][j],AccesNetLamda,lamda[i][j],Dist_Acc_Acc) 
                        # print("latency[i][j][k]NeighbourAN",latency[i][j][k])
                                       
       ## Latency that is served at local CN   3.789639064123363e-05
                        # print("self.CoreNetLamda[i]", self.CoreNetLamda[i])
                    elif self.NumAcceNetwork-1 < k < (self.NumAcceNetwork + self.NumCoreNetwork) and k == self.NumAcceNetwork+i:
                        Dist_Acc_Cor = self.propagationAccToCoreNetwork()[j][k-self.NumAcceNetwork]
                        
                        latency[i][j][k] = CEF_delayCore3.get_latency_served_ByCN_CEF(self.UpLinkBandWd,self.DownLinkBandWd,miuCoreNet[i],CoreNetLamda,lamda[i][j],Dist_Acc_Cor)#1.66666666666666666e-6
                        # print("latency[i][j][k]localCN",latency[i][j][k])
                        # print("self.CoreNetLamda[i]", self.CoreNetLamda[i])                  
        ## Latency that is served at CN neighbour    
                    elif self.NumAcceNetwork-1 < k < (self.NumAcceNetwork + self.NumCoreNetwork) and k != self.NumAcceNetwork+i:
                        Dist_Acc_Cor = self.propagationAccToCoreNetwork()[j][k-self.NumAcceNetwork]
                        Dist_Cor_Cor = self.propagationCoreToCoreNetwork()[i][k-self.NumAcceNetwork]
                                             
                        latency[i][j][k] = CEF_delayCore3.get_latency_served_Byneighbour_CN_CEF(self.UpLinkBandWd,self.DownLinkBandWd,miuCoreNet[i],CoreNetLamda,lamda[i][j],Dist_Acc_Cor,Dist_Cor_Cor)
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
                        latency[i][j][k]= CEF_delayCore3.get_latency_served_ByClN_CEF(self.UpLinkBandWd,self.DownLinkBandWd,miuCloudNet[i],CloudNetLamda,lamda[i][j],Dist_Acc_Cor,  Dist_Cor_Cloud)#1.6666666666666666e-4
                        # print("latency[i][j][k]",latency[i][j][k])
                        
        total_latency = 0.0
        # print("Before total_latency", total_latency)
# Jth AN Ith CN of Cloud Latency
        # for i in range(len(latency[agent_id])):
        for j in range(len(latency[agent_id])):
            for k in range(len(latency[agent_id][j])):
                # print("actions",action)
                total_latency += (action[agent_id][j][k]*latency[agent_id][j][k])
        # avg_lat=total_latency/self.NumAcceNetwork       
        # print("After total_latency", total_latency)
# system average delay 𝑙(𝑖,𝑗)                                    
        # total_traffic_task = sum(map(sum, action[agent_id]))
        
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
        # print("latence", latence)
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
        reward = (1/latence)
        return latence, reward
    
    def is_done(self, step):
        if step == 0:
            self.done = True
        else:
            self.done = False
        return self.done
    
    def step(self, actions, agent_id):
        self.actions[agent_id] = actions
        # Action = []
        # for n in range(self.NumCloudNetwork):
        #     for m in range(self.NumCoreNetwork):
        #         Action=(actions/sum(actions))
                # print("Action",Action)
        intial_Observation, obs2, observe = self.get_observation(agent_id)
        latence, reward = self.calculate_latency(intial_Observation, self.actions, agent_id)
        # print("latence",  intial_Observation)
        self.total_latency += latence
        self.total_reward += reward
        
        trafic = self.Load_Traffic(self.actions)
        # print("trafic",sum(sum(map(sum,trafic))))
        
        # # Actions is the distributed for each ANMEC, this total of the column to get total traffic in each AN site
        
        # A = [sum(x) for x in zip(*trafic[agent_id])]
        # print("A", A)
        # # This update the AN and CN total traffic
        # for l in range(len(self.CloudNetLamda)):
        #     self.CloudNetLamda[l] += A[self.NumAcceNetwork+self.NumCoreNetwork]
        #     # print("self.CloudNetLamda[l]", self.CloudNetLamda[l])
            
        # for i in range(len(self.CoreNetLamda)):
        #     self.CoreNetLamda[i] += A[self.NumAcceNetwork+i]
        #     # print("self.CoreNetLamda[i]", self.CoreNetLamda[i])
            
        # # print("self.AccesNetLamda[agent_id]", self.AccesNetLamda[agent_id])
        # for k in range(len(self.AccesNetLamda[agent_id])):
        #     self.AccesNetLamda[agent_id][k] += A[k]
            # print("self.AccesNetLamda[agent_id][k]", self.AccesNetLamda[agent_id][k])
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
        # print("CL====>", sum(env.CloudNetLamda)) 
        for i in range(len(self.CoreNetLamda)):
            # print('I', l)
            self.CoreNetLamda[i] += Bcn[self.NumAcceNetwork+i]
        # print("CN====>", sum(env.CoreNetLamda))      
        for j in range(len(self.AccesNetLamda)):
            
            for k in range(len(self.AccesNetLamda[j])):
                
                self.AccesNetLamda[j][k] += Acn[j][k]
        # print("AN====>", sum(map(sum, env.AccesNetLamda)))        
              
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
            
            # obs, latence, done = env.step(action, agent_id)
            # cum_latence += latence
   
        stop = timeit.default_timer()
        execution_time = stop - start
        print("Executed Time "+str(execution_time))
        obs, latence, reward, done = env.step(action, agent_id)
        cum_latence += latence
        cum_reward += reward
        # print(" ")
        print("AN afyer : ",env.AccesNetLamda)
        print("AN====>", sum(map(sum, env.AccesNetLamda)))
        print(" ")
        print("CN afyer : ",env.CoreNetLamda)
        print("CN====>", sum(env.CoreNetLamda))
        print(" ")
        print("CL afyer : ",env.CloudNetLamda) 
        print("CL====>", sum(env.CloudNetLamda))
        print(" ")
        print("Total: ", sum(map(sum, env.AccesNetLamda))+sum(env.CoreNetLamda)+sum(env.CloudNetLamda))   
        print("##################################################")
        # print("done=", done)
        if done == True:
            env.reset()
            # print("AN afyer : ",env.AccesNetLamda)
            # print("AN====>", sum(map(sum, env.AccesNetLamda)))
            # print(" ")
            # print("CN afyer : ",env.CoreNetLamda)
            # print("CN====>", sum(env.CoreNetLamda))
            # print(" ")
            # print("CL afyer : ",env.CloudNetLamda) 
            # print("CL====>", sum(env.CloudNetLamda))
            # print(" ")
            # print("Total: ", sum(map(sum, env.AccesNetLamda))+sum(env.CoreNetLamda)+sum(env.CloudNetLamda))
            # print("self.total_latency = 0.0 : ",env.total_latency)
            obs, obs2, obs3 = env.get_observation(agent_id)
            # print(" ")
        
            # print("Total Latency got : ", cum_latence)
            # print(" ")
            # print("Total Reward got : ", cum_reward) 
            # print(" ")
         
        # print("Executed Time "+str(execution_time)) 
        # score += latence

        if done == True:
            env.reset()
       
            obs, obs2, obs3 = env.get_observation(agent_id)
            # print("Random latence  : ", latence)
            print("Total Latency got : ", env.total_latency)
    
     
    # print("Executed Time "+str(execution_time))    
    stop = timeit.default_timer()
    execution_time = stop - start
    # print("Executed Time "+str(execution_time))   
    state_dim = env.get_observation(1)[2].shape[0]
    print("state_dim  =============>", state_dim)
    Number_actions = env.NumAcceNetwork+env.NumCoreNetwork+env.NumCloudNetwork
    print("actor_action_dim =======>", Number_actions)
    critic_action_dim = (env.NumCoreNetwork+env.NumAcceNetwork+env.NumCloudNetwork)*env.NumAcceNetwork
    # critic_action_dim = (env.NumCoreNetwork+env.NumAcceNetwork+env.NumCloudNetwork)*(env.NumAcceNetwork*env.NumCoreNetwork)
    print("critic_action_dim =======>", critic_action_dim)