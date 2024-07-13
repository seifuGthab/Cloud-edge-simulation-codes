#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 21:20:11 2021

@author: seifu
"""
import numpy as np
from numpy import genfromtxt
import math
import pandas as pd


class Propagation:
    def __init__(self, file):
        self.my_data = genfromtxt(file, delimiter=',')
        self.long = []
        self.lat =[]
        self.i = 1
        self.ANNum = 8
        while self.i < len(self.my_data):
            self.long.append(self.my_data[self.i][0])
            self.lat.append(self.my_data[self.i][1])
            self.i +=1
        self.column = [i for i in range(len(self.long))]
        self.index = self.column
        self.distnce_diference = np.zeros((self.ANNum,self.ANNum))
        self.propagation_difference = np.zeros((self.ANNum,self.ANNum))
        self.neighbour = [[] for self.i in range(len(self.long))]

    def distance(self, long1, long2, lati1, lati2):
        lon1 = math.radians(long1)
        lon2 = math.radians(long2)
        lat1 = math.radians(lati1)
        lat2 = math.radians(lati2)
        R = 6373.0
        dist_lon = lon2 - lon1
        dist_lat = lat2 - lat1
        a = math.sin(dist_lat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dist_lon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        # print("distance", distance)
        return distance
    
    def propagation_delay(self):
    #Measure propagation delay
        for i in self.index:
            for j in self.column:
                self.distnce_diference[i][j] = self.distance(self.long[i], self.long[j], self.lat[i], self.lat[j])
                self.propagation_difference[i][j] = self.distnce_diference[i][j]/(3*10**5)
        return self.propagation_difference

    # Find neighbour
    def neighbour(self):
        for m in range(len(self,self.neighbour)):
            neightmp = np.where(self.distnce_diference < 1, 0, self.distnce_diference)
            indeks = 0
            for n in neightmp[m]:
                print(n)
                if n <= 2:
                    self.neighbour[m].append(indeks)
                indeks += 1
                
        return self.neighbour

# propa = Propagation('CoreNetwork_to_CloudNetwork.csv')
# print("propa.propagation_delay()",propa.propagation_delay())
# # print("propa.propagation_delay()",propa.neighbour())
# dist = propa.distance(112.05,112.06,30.61,30.28)
# print(dist)