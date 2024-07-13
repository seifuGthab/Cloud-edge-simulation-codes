#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 10:56:39 2022

@author: seifu
"""

import numpy as np
import matplotlib.pyplot as plt
# from CEF_SiAEnvironmentV2Core2 import Environment
from CEF_SiAEnvironmentV3Core4T40k import Environment
import random
import timeit 

#%% Environtment Setup
NumCloudNetwork = 1
NumCoreNetwork = 4
NumAcceNetwork = 8

env = Environment()

#%% SA Initiation
np.seterr(divide="ignore")

# randomly generate initial solution
# intial_actions, arry = env.generate_actions()
intial_actions = [[[0 for i in range(NumCloudNetwork+NumCoreNetwork+NumAcceNetwork)] for j in range(NumAcceNetwork)] \
                  for k in range(NumCoreNetwork)] 

for i in range(len(intial_actions)):
    for j in range(len(intial_actions[i])):
        for k in range(len(intial_actions[i][j])):
            if j == k:
                intial_actions[i][j][k] = 1
                # print("intial_actions[i][j][k]",intial_actions[i][j][k])
curret_obs, obs2, obs = env.get_observation()
current_latency,current_reward = env.calculate_latency(curret_obs, intial_actions)

new_actions = intial_actions
latency = 0
reward = 0
index = 0                   #index for record reward every 100 steps
temp =[]
record_best_value=[]
start = timeit.default_timer()
  
for m in range(len(intial_actions)):   
    # print("MMMMMMM", m)
    for n in range(len(intial_actions[m])):    
        # print("NNNN",n)
        intial_temp = 1000 #5000 #1000
        final_temp = 100
        num_decrease_temp = 300 #1000 #300   # How many times decrease the temperature
        # num_decrease_temp = NumCoreNetwork*NumAcceNetwork*(NumAcceNetwork+NumCloudNetwork+NumCoreNetwork)
        
        Num_attampts = 20 # How many times search the neighbour before decrease the temprature (num actions)
        cooling_pram = 0.99                  # how much you want to decrease the temp(Cooling parameter)
        k = 0.1                              # boltzman constant
        # step_start = timeit.default_timer()
        for i in range (num_decrease_temp):
            # print("IIIII",i)
            # step_start = timeit.default_timer()
            for j in range (Num_attampts):
                # print("Num_attampts",j)
                step_start = timeit.default_timer()
                
        # Get total arrival traffic in site
                
                arival_trafic = env.Load_Traffic(intial_actions)
                trafic = [] 
                for i in arival_trafic:
                    trafic.append([sum(x) for x in zip(*i)])
                    # print(trafic)
                # Get capacity
                capacity = env.miuAcceNetwork[m]
                # print("capacity", capacity[n])
                capacity.extend(env.miuCoreNetwork)
                # print("capacity", capacity)
                capacity.extend(env.miuCloudNetwork)
                # print("capacity", capacity)
                residual_value =[]
                
                #subtract capacity with arrival trafic
                result = zip(capacity,trafic[m])
                # print("trafic[m] ----",trafic[m])
                for x,y in result:
                    residual_value.append(x-y)
                # print("Arivl Traffic -------------",trafic[m])
                # print("Capacity ------------", capacity)
                # print("Difference value ----",residual_value)
                
                rand_values = random.uniform(0,0.01)
                neighbor1 = n                                       
                # print("neighbor1",neighbor1)
                Lists = []
                for i in range(len(residual_value)):
                    if residual_value[i] > 2300:  #(A[m][rand_num_x_1]*rand_int)*10:
                            Lists.append(i)
                # print("List",Lists)
                while True:
                    
                    neighbor2 = random.choice(Lists)
                                   
                    if neighbor2!=neighbor1:
                        break
                
                # rand_values = random.uniform(0,0.01)
                # neighbor1 = n  # n source states
                # neighbor2 = np.argmax(residual_value)                                       
        
                # Lists = []
                # for i in range(len(residual_value)):
                #     if residual_value[i] > 2300:      #(A[m][neighbor1]*offloading_ratio)*20%:
                #         Lists.append(i)
                        
                # while True:
                #     # neighbor2 = random.choice(Lists)
                #     if len(Lists)==0:
                #         neighbor2 = random.randint(0, NumAcceNetwork)
                #     else:
                #         #neighbor2 = random.randint(0,NumAcceNetwork)
                #         neighbor2 = np.argmax(residual_value)
                        
                #     if neighbor2!=neighbor1:
                #         break
                # print("residual_value[n]",n,residual_value[n])
                # if env.AccesNetLamda[n][m] > env:.miuAcceNetwork[n][m]: 
                if residual_value[n] < 0 and intial_actions[m][n][n] > rand_values:
                    new_actions[m][n][neighbor1] = intial_actions[m][n][neighbor1] - rand_values
                    # print("new_actions",new_actions)
                    new_actions[m][n][neighbor2] = intial_actions[m][n][neighbor2] + rand_values
                    # print("residual_value[n]",n,residual_value[n])
                elif residual_value[n] < 0 and intial_actions[m][n][n] < rand_values:            
                    pass
                    
                elif residual_value[n] > 0 and intial_actions[m][n][n] > rand_values:
                    rand_num = np.random.rand()
                    # print("rand_num", rand_num)
                    if rand_num > 0.5:     # to accept worse solution
                        new_actions[m][n][neighbor1] = intial_actions[m][n][neighbor1] - rand_values
                        # print("new_actions",new_actions)
                        new_actions[m][n][neighbor2] = intial_actions[m][n][neighbor2] + rand_values
                        # print("new_actions",new_actions)
                    elif rand_num < 0.5:
                        if intial_actions[m][n][neighbor2] < rand_values:
                            new_actions[m][n][neighbor1] = intial_actions[m][n][neighbor1] - rand_values
                            # print("new_actions",new_actions)
                            new_actions[m][n][neighbor2] = intial_actions[m][n][neighbor2] + rand_values
                            # print("new_actions",new_actions)
                        else:
                            new_actions[m][n][neighbor1] = intial_actions[m][n][neighbor1] + rand_values
                            new_actions[m][n][neighbor2] = intial_actions[m][n][neighbor2] - rand_values
                            
                elif residual_value[n] > 0 and intial_actions[m][n][n] < rand_values:
                    new_actions[m][n][neighbor1] = intial_actions[m][n][neighbor1] + rand_values
                    new_actions[m][n][neighbor2] = intial_actions[m][n][neighbor2] - rand_values                 
                
                curret_obs, obs2, obs = env.get_observation()
                # the possible new move with temporary values
                # new_latency, new_reward = env.calculate_latency(curret_obs, new_actions)
                # print("New Latency ======> : %0.10f" % new_latency, "New Reward ======> : %0.10f" % new_reward)
                
                # # where we are currently
                # current_latency, current_reward = env.calculate_latency(curret_obs, intial_actions)
                # print("Current Latency ======> : %0.10f" % current_latency,"Current Reward ======> : %0.10f" % current_reward)
                # rand_num = np.random.rand()
                # # asd
                # formula = 1/(np.exp((current_reward-new_reward)/intial_temp))
                # # print("formula================ %f: " % formula)
                # if current_reward > new_reward :
                #     intial_actions[m][n][neighbor1] = new_actions[m][n][neighbor1]
                #     intial_actions[m][n][neighbor2] = new_actions[m][n][neighbor2]
                # elif rand_num <= formula:
                #     intial_actions[m][n][neighbor1] = new_actions[m][n][neighbor1]
                #     intial_actions[m][n][neighbor2] = new_actions[m][n][neighbor2]
                # else:
                #     intial_actions[m][n][neighbor1] = new_actions[m][n][neighbor1]
                #     intial_actions[m][n][neighbor2] = new_actions[m][n][neighbor2]
            #     # if i == 2:
            #         # xc
            #     new_actions = intial_actions
            # # print(new_actions)
            # temp.append(intial_temp)
            # record_best_value.append(current_reward)
            
            # intial_temp = cooling_pram*intial_temp                    
                
                #the possible new move with temporary values
                new_latency, new_reward = env.calculate_latency(curret_obs, new_actions)
                
                print("New Latency ======> : %0.10f" % new_latency, "New Reward ======> : %0.10f" % new_reward)
                # print(" ")
                # where we are currently
                current_latency, current_reward = env.calculate_latency(curret_obs, intial_actions)
                print("Current Latency ======> : %0.10f" % current_latency,"Current Reward ======> : %0.10f" % current_reward)
                # print(" ")
                # print("new_actions",new_actions)
                # Generate randome number to accept or reject the worse solution
                acceptance_retio = np.random.rand()
               
                # make decioin to accept the worse solution
                if  current_reward < new_reward :
                    prob = 1/(np.exp((current_reward-new_reward)/intial_temp))
                    # print("Prob=======>", prob)
                    # make decioin to accept the worse solution
                    if acceptance_retio < prob:
                        accept = True
                    else:
                        accept = False
                        
                else:
                    accept = True    # Accept better solutions
                    
                if accept == True:
                    intial_actions[m][n][neighbor1] = new_actions[m][n][neighbor1]
                    intial_actions[m][n][neighbor2] = new_actions[m][n][neighbor2]
                
                new_actions = intial_actions
                current_latency, current_reward = env.calculate_latency(curret_obs, intial_actions)
                if index % 100 == 0:
                    temp.append(intial_temp)
                    print("Latency=====> : %0.10f" % current_latency, "Reward=====> : %0.10f" % current_reward)
                    record_best_value.append(current_reward)
                index += 1   
            
            # temp.append(intial_temp)
            # record_best_value.append(current_latency)
            # step_stop = timeit.default_timer()
            # Step_decision_time = step_stop - step_start
            intial_temp = cooling_pram*intial_temp
            step_stop = timeit.default_timer()
        Step_decision_time = step_stop - step_start
            # print(" ")
            # print("intial_temp",intial_temp)
# print(" ")
# print("Actions:=============", intial_actions)
# print("Get observations:========== ", env.get_observation()[0])
obs, latency_final,reward_final, done = env.step(new_actions)
print(" ")
print("new_actions",new_actions)
latency = latency_final
reward = reward_final
# print("Total latency = ", total_latency)
# print("Total reward = ", reward)
# print("AccNet traffic:  ", env.AccesNetLamda)
# print("CoreNet traffic: ", env.CoreNetLamda)
# print("CloudNet Trafic: ", env.CloudNetLamda)
# print("miuAccNet====>", env.miuAcceNetwork)
# print("miuCoreNet====>", env.miuCoreNetwork)
# print("miuCloudNet====>", env.miuCloudNetwork)
  
    
stop = timeit.default_timer()
print(" ")
print(" ")
print("======= Simulated Annealing Result ==========")
print(" ")
print("Latency = ", latency)
print(" ")
print("Reward = ", reward)
print(" ")
execution_time = stop - start
print("Program Executed in "+str(execution_time))
print(" ")
print("Traffic : ", env.lam)
print(" ")
print("Trafic Distribution in AccNetwork = : ", env.AccesNetLamda)
print(" ")
print("Trafic Distribution in CoreNetwork = : ", env.CoreNetLamda)
print(" ")
print("Trafic Distribution in Cloud = : ", env.CloudNetLamda)
print("=============================================")
from datetime import datetime
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
with open("resultLatency.txt", "a+") as file_object:
    # Move read cursor to the start of file.
    file_object.seek(0)
    # If file is not empty then append '\n'
    data = file_object.read(100)
    if len(data) > 0 :
        file_object.write("\n")
    # Append text at the end of file
    file_object.write("\n")
    file_object.write("############")
    file_object.write(dt_string)
    file_object.write("#############")
    file_object.write("\n")
    file_object.write("Total Latency= ")
    file_object.write(str(latency))
    file_object.write("Total Reward= ")
    file_object.write(str(reward))
    file_object.write("\n")
    file_object.write("Step_decision_time ")
    file_object.write(str(Step_decision_time))
    file_object.write("\n")
    file_object.write("Max reward ")
    file_object.write(str(max(record_best_value)))
    file_object.write("\n")
    file_object.write("Index_Values ")
    file_object.write(str(record_best_value.index(max(record_best_value))))
    file_object.write("\n")
    file_object.write("Program Executed in ")
    file_object.write(str(execution_time))
    file_object.write("\n")
    file_object.write("\n")
    file_object.write("Cloud Num = ")
    file_object.write(str(env.NumCloudNetwork))
    file_object.write("\n")
    file_object.write("Core Num = ")
    file_object.write(str(env.NumCoreNetwork))
    file_object.write("\n")
    file_object.write("Access Num = ")
    file_object.write(str(env.NumAcceNetwork))
    file_object.write("\n")
    file_object.write("Traffic = ")
    file_object.write(str(env.lam))
    file_object.write("\n")
    file_object.write("Trafic Distribution in AccNetwork = ")
    file_object.write(str(sum(map(sum, env.AccesNetLamda))))
    file_object.write("\n")
    file_object.write("Trafic Distribution in CoreNetwork = ")
    file_object.write(str(sum(env.CoreNetLamda)))
    file_object.write("\n")
    file_object.write("Trafic Distribution in Cloud = ")
    file_object.write(str(sum(env.CloudNetLamda)))
    file_object.write("\n")
        
    # with open("Latency2.txt", "a+") as file_object:
    #     # Move read cursor to the start of file.
    #     file_object.seek(0)
    #     # If file is not empty then append '\n'
    #     data = file_object.read(100)
    #     if len(data) > 0 :
    #         file_object.write("\n")
    #     # Append text at the end of file
    #     file_object.write("\n")
    #     file_object.write("Traffic = ")
    #     file_object.write(str(env.lam))
    #     file_object.write("\n")
    #     file_object.write("latency= ")
    #     file_object.write(str(record_best_value))
    #     file_object.write("\n")
    #     file_object.write("len latency = ")
    #     file_object.write(str(len(record_best_value)))
    #     file_object.write("\n")
    #     file_object.write("############")
print("Step_decision_time",Step_decision_time)
print(" ")
print("Decision Time:"+str(execution_time))
print("Max", max(record_best_value), "index", record_best_value.index(max(record_best_value)))
# print("Each step Time:"+str(step_time))

print(" ")
print("Trafic Distribution in AccNetwork = : ", sum(map(sum, env.AccesNetLamda)))
print(" ")
print("Trafic Distribution in CoreNetwork = : ", sum(env.CoreNetLamda))
print(" ")
print("Trafic Distribution in Cloud = : ", sum(env.CloudNetLamda))
print()
print("Total trafic: ", sum(map(sum, env.AccesNetLamda))+sum(env.CoreNetLamda)+sum(env.CloudNetLamda))        
# plt.plot(temp,obj_val)
# plt.title("Z at temperature values", fontsize=20, fontweight='bold')
# plt.xlabel("Temperature")
# plt.ylabel("Z")

# plt.xlim(temp_for_plot,0)
# plt.xticks(np.arange(min(temp),max(temp),100), fontweight="bold")
# plt.yticks(fontweight="bold")
plt.plot(record_best_value)
plt.show()