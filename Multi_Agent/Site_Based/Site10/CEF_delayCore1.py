#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 16:51:53 2021

@author: seifu

"""
# from CEF_EnvironmentV4 import Environment
NumAcceNetwork = 8
NumCoreNetwork = 1
NumCloudNetwork = 1


# def get_latency(miu, lamda):
#     latence = 1/(miu-lamda)
#     # print("latence", latence)
#     return latence
    
def get_latency_serve_ByAN_CEF(UpLinkBandWd, DownLinkBandWd, miuAcce, lamdaAcce, lamda):
    # print("============================")
    # print("UpLinkBandWd",UpLinkBandWd)
    # print("DownLinkBandWd", DownLinkBandWd)
    # print("miuAccesNet[i][j]", miuAcce)
    # print("AccesNetLamda", lamdaAcce)
    # print("lamda[i][j]", lamda)
    # print("Dist_Acc_Acc", Dist_Acc_Acc)
    if (miuAcce - lamdaAcce) <= 0:
        latency = -(miuAcce - lamdaAcce)
        # latency = 0
        # print("miuAcce",miuAcce)
        # print("lamdaAcce",lamdaAcce)
        # print("Latency", latency)
    else:
        latency = (1/(UpLinkBandWd-lamda)) + (1/(miuAcce-lamdaAcce)) + (1/(DownLinkBandWd - lamda))
        # print("miuAcce",miuAcce)
        # print("lamdaAcce",lamdaAcce)
        # print("Latency", latency)
    return latency
    
def get_latency_servesd_Byneighbour_AN_CEF(UpLinkBandWdAcc, DownLinkBandWdAcc, miuAcce, lamdaAcce, lamda, Dist_Acc_Acc):
    # print("============================")
    # print("============================")
    # print("UpLinkBandWd",UpLinkBandWdAcc)
    # print("DownLinkBandWd", DownLinkBandWdAcc)
    # print("miuAccesNet[i][j]", miuAcce)
    # print("AccesNetLamda", lamdaAcce)
    # print("lamda[i][j]", lamda)
    # print("Dist_Acc_Acc", Dist_Acc_Acc)
    if (miuAcce - lamdaAcce) <= 0:
        latency = -(miuAcce - lamdaAcce)
        # latency = 0
        # print("miuAcce",miuAcce)
        # print("lamdaAcce",lamdaAcce)
        # print("Latency", latency)
    else:    
        latency = (1/(UpLinkBandWdAcc-lamda)) + (1/(miuAcce-lamdaAcce)) + (1/(DownLinkBandWdAcc - lamda)) + (2*Dist_Acc_Acc)
        # print("miuAcce",miuAcce)
        # print("lamdaAcce",lamdaAcce)
        # print("Latency", latency)
    return latency
    
def get_latency_served_ByCN_CEF(UpLinkBandWdC, DownLinkBandWdC, miuCore, lamdaCore, lamda, Dist_Acc_Cor):
    # print("============================")
    # print("UpLinkBandWd",UpLinkBandWdC)
    # print("DownLinkBandWd", DownLinkBandWdC)
    # print("miuAccesNet[i][j]", miuCore)
    # print("AccesNetLamda", lamdaCore)
    # print("lamda[i][j]", lamda)
    # print("Dist_Acc_Acc", Dist_Acc_Cor)
    if (miuCore-lamdaCore) <=0:
        
        latency = -(miuCore-lamdaCore)
        # print("miuAcce",miuCore)
        # print("lamdaAcce",lamdaCore)
        # print("Latency", latency)
    else:
        latency = (1/(UpLinkBandWdC-lamda)) + (1/(miuCore-lamdaCore)) + (1/(DownLinkBandWdC - lamda)) + (2*Dist_Acc_Cor)
        # print("miuCore",miuCore)
        # print("lamdaCore",lamdaCore)
        # print("LatencyAccneghbrEls",latency)
    return latency
    
def get_latency_served_Byneighbour_CN_CEF(UpLinkBandWdCC, DownLinkBandWdCC, miuCore, lamdaCore, lamda, Dist_Acc_Cor, Dist_Cor_Cor):
    # print("============================")
    # print("UpLinkBandWd",UpLinkBandWdCC)
    # print("DownLinkBandWd", DownLinkBandWdCC)
    # print("miuAccesNet[i][j]", miuCore)
    # print("AccesNetLamda", lamdaCore)
    # print("lamda[i][j]", lamda)
    # print("Dist_Acc_Acc", Dist_Acc_Cor)
    if (miuCore-lamdaCore) <= 0:
        latency = -(miuCore-lamdaCore)
        # print("miuCoreN",miuCore)
        # print("lamdaCoreN",lamdaCore)
        # print("LatencyAccneghbrEls",1/(miuCore-lamdaCore))
    else:
        latency = (1/(UpLinkBandWdCC-lamda)) + (1/(miuCore-lamdaCore)) + (1/(DownLinkBandWdCC - lamda)) + (2*Dist_Acc_Cor) + (2*Dist_Cor_Cor)
        # print("miuCoreN",miuCore)
        # print("lamdaCoreN",lamdaCore)
        # print("LatencyAccneghbrEls",1/(miuCore-lamdaCore))
    return latency
    
def get_latency_served_ByClN_CEF(UpLinkBandWdCL, DownLinkBandWdCL, miuCloud, lamdaCloud, lamda, Dist_Acc_Cor, Dist_Cor_Cloud):
    # print("============================")
    # print("UpLinkBandWd",UpLinkBandWdCL)
    # print("DownLinkBandWd", DownLinkBandWdCL)
    # print("miuAccesNet[i][j]", miuCloud)
    # print("AccesNetLamda", lamdaCloud)
    # print("lamda[i][j]", lamda)
    # print("Dist_Acc_Acc", Dist_Acc_Cor)
    if (miuCloud-lamdaCloud) <= 0:
        latency = -(miuCloud-lamdaCloud)
        # print("miuCloud",miuCloud)
        # print("lamdaCloud",lamdaCloud)
        # print("LatencyAccneghbrEls",1/(miuCloud-lamdaCloud))
    else:
        latency = (1/(UpLinkBandWdCL - lamda)) + (1/(miuCloud-lamdaCloud)) + (1/(DownLinkBandWdCL - lamda)) + (2*Dist_Acc_Cor) + (2*Dist_Cor_Cloud)
        # print("miuCloud",miuCloud)
        # print("lamdaCloud",lamdaCloud)
        # print("LatencyAccneghbrEls",1/(miuCloud-lamdaCloud))
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
