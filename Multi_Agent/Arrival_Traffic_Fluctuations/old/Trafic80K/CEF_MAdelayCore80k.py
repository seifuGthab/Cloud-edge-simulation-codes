#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 16:51:53 2021

@author: seifu

"""
# from CEF_EnvironmentV4 import Environment
NumAcceNetwork = 8
NumCoreNetwork = 4
NumCloudNetwork = 1


# def get_latency(miu, lamda):
#     latence = 1/(miu-lamda)
#     # print("latence", latence)
#     return latence
    
def get_latency_serve_ByAN_CEF(UpLinkBandWd, DownLinkBandWd, miuAcce, lamdaAcce, lamda):
    # latency = (1/(UpLinkBandWd-lamda)) + (1/(miuAcce-lamdaAcce)) + (1/(DownLinkBandWd - lamda))
    
    if (miuAcce - lamdaAcce) <= 0:
        latency = -(miuAcce - lamdaAcce)
        # latency = 0
    
    else:
        latency = (1/(UpLinkBandWd-lamda)) + (1/(miuAcce-lamdaAcce)) + (1/(DownLinkBandWd - lamda))
       
    return latency
    
def get_latency_servesd_Byneighbour_AN_CEF(UpLinkBandWdAcc, DownLinkBandWdAcc, miuAcce, lamdaAcce, lamda, Dist_Acc_Acc):
    # latency = (1/(UpLinkBandWdAcc-lamda)) + (1/(miuAcce-lamdaAcce)) + (1/(DownLinkBandWdAcc - lamda)) + (2*Dist_Acc_Acc)
    if (miuAcce - lamdaAcce) <= 0:
        latency = -(miuAcce - lamdaAcce)
        # latency = 0
        
    else:    
        latency = (1/(UpLinkBandWdAcc-lamda)) + (1/(miuAcce-lamdaAcce)) + (1/(DownLinkBandWdAcc - lamda)) + (2*Dist_Acc_Acc)
        
    return latency
    
def get_latency_served_ByCN_CEF(UpLinkBandWdC, DownLinkBandWdC, miuCore, lamdaCore, lamda, Dist_Acc_Cor):
    # latency = (1/(UpLinkBandWdC-lamda)) + (1/(miuCore-lamdaCore)) + (1/(DownLinkBandWdC - lamda)) + (2*Dist_Acc_Cor)
    if (miuCore-lamdaCore) <=0:
        
        latency = -(miuCore-lamdaCore)
        # latency = 0
        
    else:
        latency = (1/(UpLinkBandWdC-lamda)) + (1/(miuCore-lamdaCore)) + (1/(DownLinkBandWdC - lamda)) + (2*Dist_Acc_Cor)
        
    return latency
    
def get_latency_served_Byneighbour_CN_CEF(UpLinkBandWdCC, DownLinkBandWdCC, miuCore, lamdaCore, lamda, Dist_Acc_Cor, Dist_Cor_Cor):
    # latency = (1/(UpLinkBandWdCC-lamda)) + (1/(miuCore-lamdaCore)) + (1/(DownLinkBandWdCC - lamda)) + (2*Dist_Acc_Cor) + (2*Dist_Cor_Cor)
    if (miuCore-lamdaCore) <= 0:
        latency = -(miuCore-lamdaCore)
        # latency = 0
        
    else:
        latency = (1/(UpLinkBandWdCC-lamda)) + (1/(miuCore-lamdaCore)) + (1/(DownLinkBandWdCC - lamda)) + (2*Dist_Acc_Cor) + (2*Dist_Cor_Cor)
        
    return latency
    
def get_latency_served_ByClN_CEF(UpLinkBandWdCL, DownLinkBandWdCL, miuCloud, lamdaCloud, lamda, Dist_Acc_Cor, Dist_Cor_Cloud):
    # latency = (1/(UpLinkBandWdCL - lamda)) + (1/(miuCloud-lamdaCloud)) + (1/(DownLinkBandWdCL - lamda)) + (2*Dist_Acc_Cor) + (2*Dist_Cor_Cloud)
    if (miuCloud-lamdaCloud) <= 0:
        latency = -(miuCloud-lamdaCloud)
        # latency = 0
        
    else:
        latency = (1/(UpLinkBandWdCL - lamda)) + (1/(miuCloud-lamdaCloud)) + (1/(DownLinkBandWdCL - lamda)) + (2*Dist_Acc_Cor) + (2*Dist_Cor_Cloud)
        
    return latency

def get_AccesNetwork_lamda(i,k, task):
        lamdaAccesNet = 0
        for j in range(NumAcceNetwork):
            lamdaAccesNet += task[i][j][k]
            # print('task[0][0][0]', task[0][1][10])
            # print('lamdaAccesNet', lamdaAccesNet)
            
        return lamdaAccesNet    
def get_CoreNetwork_lamda(k,task):
    lamdaCoreNet = 0
    # for i in range(NumCloudNetwork):
    for i in range(NumCoreNetwork):
        for j in range(NumAcceNetwork):
            lamdaCoreNet += task[i][j][k]
            # print('lamdaCoreNet', lamdaCoreNet)
            # print('actions[i][j][k]', actions[i][j][k])
    return lamdaCoreNet
     
def get_CloudNetwork_lamda(task):
    lamdaCloudNet = 0
    
    for i in range(NumCloudNetwork):
        for j in range(NumCoreNetwork):
            for k in range(NumAcceNetwork):
                lamdaCloudNet += task[i][j][k]
            # print('lamdaCloudNet', lamdaCloudNet)
            # print('actions[i][j][k]', task[i][j][k])
    return lamdaCloudNet
