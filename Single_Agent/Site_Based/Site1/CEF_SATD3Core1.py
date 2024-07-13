#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 19:45:13 2021

@author: seifu
"""
import os
import time
import numpy as np
from matplotlib import pyplot
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from CEF_SAEnvironmentCore1 import Environment
# from CEF_SAEnvironmentCore2V1 import Environment
# We initialize the Experience Replay memory
class ReplayBuffer(object):

  def __init__(self, max_size=1000000):
    self.storage = []
    self.max_size = max_size
    self.mem_counter = 0

  def Transitions(self, transition):
      
# if the memory has been populated so it placed in the beginning (replace the old) 
    if len(self.storage) == self.max_size:
      self.storage[int(self.mem_counter)] = transition
      self.mem_counter = (self.mem_counter + 1) % self.max_size
    else:
      self.storage.append(transition)
      
  def SampleBuffer(self, batch_size):
    cntr = np.random.randint(0, len(self.storage), size=batch_size)
    states, new_states, actions, rewards, dones = [],[],[],[],[]
    
    for i in cntr:
        state, new_state, action, reward, done = self.storage[i]
        states.append(np.array(state, copy=False))
        new_states.append(np.array(new_state, copy=False))
        actions.append(np.array(action, copy=False))
        rewards.append(np.array(reward, copy=False))
        # print('Reward', rewards)
        dones.append(np.array(done, copy=False))
      
    return np.array(states), np.array(new_states), np.array(actions), np.array(rewards).reshape(-1, 1), np.array(dones).reshape(-1, 1)
# We build one neural network for the Actor model and 
# one neural network for the Actor target

class Actor_Network(nn.Module):
  
  def __init__(self, state_dim, actor_action_dim,layer1_size,layer2_size,min_actions,max_action):
    super(Actor_Network, self).__init__()
    self.state_dim = state_dim
    self.actor_action_dim = actor_action_dim
    self.layer1_size = layer1_size
    self.layer2_size = layer2_size
    
    self.actor_1 = nn.Linear(self.state_dim, self.layer1_size)
    self.actor_2 = nn.Linear(self.layer1_size,self.layer2_size)
    self.actor_3 = nn.Linear(self.layer2_size,self.layer2_size)
    
    self.actor_11 = nn.Linear(self.layer2_size,self.layer1_size)
    self.actor_12 = nn.Linear(self.layer1_size,self.layer2_size)
    self.actor_13 = nn.Linear(self.layer2_size, self.actor_action_dim)
    
    self.actor_21 = nn.Linear(self.layer2_size,self.layer1_size)
    self.actor_22 = nn.Linear(self.layer1_size,self.layer2_size)
    self.actor_23 = nn.Linear(self.layer2_size, self.actor_action_dim)
    
    self.actor_31 = nn.Linear(self.layer2_size,self.layer1_size)
    self.actor_32 = nn.Linear(self.layer1_size,self.layer2_size)
    self.actor_33 = nn.Linear(self.layer2_size, self.actor_action_dim)
    
    self.actor_41 = nn.Linear(self.layer2_size,self.layer1_size)
    self.actor_42 = nn.Linear(self.layer1_size,self.layer2_size)
    self.actor_43 = nn.Linear(self.layer2_size, self.actor_action_dim)
    
    self.actor_51 = nn.Linear(self.layer2_size,self.layer1_size)
    self.actor_52 = nn.Linear(self.layer1_size,self.layer2_size)
    self.actor_53 = nn.Linear(self.layer2_size, self.actor_action_dim)
    
    self.actor_61 = nn.Linear(self.layer2_size,self.layer1_size)
    self.actor_62 = nn.Linear(self.layer1_size,self.layer2_size)
    self.actor_63 = nn.Linear(self.layer2_size, self.actor_action_dim)
    
    self.actor_71 = nn.Linear(self.layer2_size,self.layer1_size)
    self.actor_72 = nn.Linear(self.layer1_size,self.layer2_size)
    self.actor_73 = nn.Linear(self.layer2_size, self.actor_action_dim)
    
    self.actor_81 = nn.Linear(self.layer2_size,self.layer1_size)
    self.actor_82 = nn.Linear(self.layer1_size,self.layer2_size)
    self.actor_83 = nn.Linear(self.layer2_size, self.actor_action_dim)
    
  def forward(self, stat):                         
    stat = F.relu(self.actor_1(stat))
    stat = F.relu(self.actor_2(stat))
    stat = F.relu(self.actor_3(stat))
 
    l1 = F.relu(self.actor_11(stat))
    l1 = F.relu(self.actor_12(l1))
    l1 = torch.sigmoid(self.actor_13(l1))
    
    l2 = F.relu(self.actor_21(stat))
    l2 = F.relu(self.actor_22(l2))
    l2 = torch.sigmoid(self.actor_23(l2))
     
    l3 = F.relu(self.actor_31(stat))
    l3 = F.relu(self.actor_32(l3))
    l3 = torch.sigmoid(self.actor_33(l3))
    
    l4 = F.relu(self.actor_41(stat))
    l4 = F.relu(self.actor_42(l4))
    l4 = torch.sigmoid(self.actor_43(l4))
    
    l5 = F.relu(self.actor_51(stat))
    l5 = F.relu(self.actor_52(l5))
    l5 = torch.sigmoid(self.actor_53(l5))
    
    l6 = F.relu(self.actor_61(stat))
    l6 = F.relu(self.actor_62(l6))
    l6 = torch.sigmoid(self.actor_63(l6))
     
    l7 = F.relu(self.actor_71(stat))
    l7 = F.relu(self.actor_72(l7))
    l7 = torch.sigmoid(self.actor_73(l7))
    
    l8 = F.relu(self.actor_81(stat))
    l8 = F.relu(self.actor_82(l8))
    l8 = torch.sigmoid(self.actor_83(l8))

    return l1,l2,l3,l4,l5,l6,l7,l8
# We build two neural networks for the two Critic models and two neural networks for the two Critic targets
class Critic_Network(nn.Module):
  
  def __init__(self, state_dim, critic_action_dim,layer1_size,layer2_size,min_actions,max_action):
    super(Critic_Network, self).__init__()
    
    self.critic_action_dim=critic_action_dim
    self.state_dim=state_dim
    self.layer1_size = layer1_size
    self.layer2_size = layer2_size   
# Forward_Propagation of the first Critic neural network
    self.critic_1 = nn.Linear(self.state_dim + self.critic_action_dim, self.layer1_size)
    self.critic_2 = nn.Linear(self.layer1_size,self.layer2_size)
    self.critic_3 = nn.Linear(self.layer2_size, 1)
    
# Forward_Propagation of the second Critic neural network
    self.critic_4 = nn.Linear(self.state_dim + self.critic_action_dim, self.layer1_size)
    self.critic_5 = nn.Linear(self.layer1_size,self.layer2_size)
    self.critic_6 = nn.Linear(self.layer2_size, 1)

  def forward(self, s, a):         
    SA = torch.cat([s, a], 1)       
    
# Forward-Propagation on the first Critic Neural Network
    q1_value = F.relu(self.critic_1(SA))
    q1_value = F.relu(self.critic_2(q1_value))
    q1_value = self.critic_3(q1_value)

# Forward-Propagation on the second Critic Neural Network
    q2_value = F.relu(self.critic_4(SA))
    q2_value = F.relu(self.critic_5(q2_value))
    q2_value = self.critic_6(q2_value)
    
    return q1_value,q2_value
   
  def Q1(self, s, a):
    SA = torch.cat([s, a], 1)
    q1_value = F.relu(self.critic_1(SA))
    q1_value = F.relu(self.critic_2(q1_value))
    q1_value = self.critic_3(q1_value)
    return q1_value



# Selecting the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Building the whole Training Process into a class
class Agent(object):
  
  def __init__(self, state_dim, actor_action_dim, critic_action_dim,layer1_size,layer2_size,min_actions,max_action):
    self.min_actions = 0.0
    self.max_action = 1.0      
    # # self.noise = noise
    self.layer1_size = layer1_size
    self.layer2_size = layer2_size
    
    self.actor = Actor_Network(state_dim, actor_action_dim,layer1_size,layer2_size,min_actions,max_action).to(device)
    self.actor_target = Actor_Network(state_dim, actor_action_dim,layer1_size,layer2_size,min_actions,max_action).to(device)
    self.actor_target.load_state_dict(self.actor.state_dict())
    self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
    
    self.critic = Critic_Network(state_dim, critic_action_dim,layer1_size,layer2_size,min_actions,max_action).to(device)
    self.critic_target = Critic_Network(state_dim, critic_action_dim,layer1_size,layer2_size,min_actions,max_action).to(device)
    self.critic_target.load_state_dict(self.critic.state_dict())
    self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
    

  def choose_action(self, observation):
      state = torch.tensor(observation.reshape(1,-1),dtype=torch.float).to(device)
        
      st = torch.cat(self.actor.forward(state)).to(device)
        
      action = []
      n = env.NumAcceNetwork
      for i in st:
          action.append(i.cpu().data.numpy().flatten())
      # print("action",action) 
      acn = [action[i:i + n] for i in range(0, len(action), n)]
      nr_actions =[]
      for i in range(len(acn)):
           totalAcc = sum(map(sum,acn[i]))
                 # print("totalAcc", totalAcc)
      for j in range(len(acn)):
           nr_actions.append(acn[j]/totalAcc)
      # nr_actions =[]
      # for i in range(len(acn)):
      #     for j in range(len(acn[i])):
      #         totalAcc = sum(acn[i][j])
      #         # print("totalAcc",totalAcc)
      #     nr_actions.append(acn[i]/totalAcc)
            
      # print("nr_actions",nr_actions)        
      return nr_actions    
           
  def learn(self, replay_buffer, iterations, batch_size=100, disc_factor=0.99,
            learning_rate=0.005, noise=0.1,noise_clip=0.1, policy_net_update=2):
    
    for itr in range(iterations):
      
# We sample a batch of transitions (s, s’, a, r) from the memory
      states, next_states, actions, rewards, dones = replay_buffer.SampleBuffer(batch_size)
      state = torch.Tensor(states).to(device)
      new_state = torch.Tensor(next_states).to(device)
      action = torch.Tensor(actions).to(device)
      reward = torch.Tensor(rewards).to(device)
      done = torch.Tensor(dones).to(device)
      
# From the new state s’, the Actor target plays the next action a’
      l1,l2,l3,l4,l5,l6,l7,l8 = self.actor_target(new_state)
      next_action = torch.cat([l1,l2,l3,l4,l5,l6,l7,l8],1)
      next_action = next_action + torch.clamp(torch.tensor(np.random.normal(scale=noise)), -noise_clip, noise_clip)
      next_action = torch.clamp(next_action, self.min_actions,self.max_action)
     
# The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
      Q1_value, Q2_value = self.critic_target(new_state, next_action)

# because the target Q is still in the computation graph, then it need to be dettached inorder to add with the reward       
# The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs      
      current_Q1_value, current_Q2_value= self.critic(state, action)
# We keep the minimum of these two Q-values: min(Qt1, Qt2)
      Q_critic_value = torch.min(Q1_value, Q2_value)
      
# We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the disc_factor factor
      Q_critic_value = reward + ((1 - done) * disc_factor * Q_critic_value).detach()
      self.critic_optimizer.zero_grad()         
# We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
      critic_loss = F.mse_loss(current_Q1_value, Q_critic_value) + F.mse_loss(current_Q2_value,Q_critic_value)

# We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
      # self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()
        
# Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
      if itr % policy_net_update == 0:
          return
      self.actor_optimizer.zero_grad()
      l1,l2,l3,l4,l5,l6,l7,l8 = self.actor(state)
      action = torch.cat([l1,l2,l3,l4,l5,l6,l7,l8],1)
      # actor_loss = self.critic.Q1(state,action).mean()
      actor_q1_loss = self.critic.Q1(state,action)
      actor_loss = -torch.mean(actor_q1_loss)
      # self.actor_optimizer.zero_grad()
      actor_loss.backward()
      self.actor_optimizer.step()
        
# Still once every two iterations, we update the weights of the Actor target by polyak averaging
      for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
          target_param.data.copy_(learning_rate * param.data + (1 - learning_rate) * target_param.data)
        
# Still once every two iterations, we update the weights of the Critic target by polyak averaging
      for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
          target_param.data.copy_(learning_rate * param.data + (1 - learning_rate) * target_param.data)
  
# Making a save method to save a trained model
  def save(self, filename, directory):
    torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
    torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
  
# Making a load method to load a pre-trained model
  def load(self, filename, directory):
    self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
    self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
    
    
# We make a function that evaluates the policy by calculating its average reward over 10 episodes
# ep_reward = []
# evl_latency = []
import timeit
def evaluate_policy(policy, eval_episodes=10):
    total_latency = 0.0
    total_reward = 0.0
    for i in range(eval_episodes):
        obs = env.get_observation()[2]
        done = False
        while not done:
           
            action = policy.choose_action(np.array(obs))
            
            obs, latency, reward, done = env.step(action)
            
            total_latency += latency
            total_reward += reward
  
        avg_reward = total_reward /(env.NumAcceNetwork*eval_episodes)
    avg_latency = total_latency /10
    
    return avg_latency, avg_reward


# We set the parameters
# Name of a environment (set it to any Continous environment you want)

seed = 0                # Random seed number
warmup_step =1000 #Number of iterations/timesteps before which the model randomly chooses an action, and after which it starts to use the policy network
eval_freq = 100      #5e3 # How often the evaluation step is performed (after how many timesteps)
max_timesteps = 10000 #5e5 # Total number of iterations/timesteps
save_models = True      #Boolean checker whether or not to save the pre-trained model
# save_models = False
expl_noise = 0.1        #Exploration noise - STD value of exploration Gaussian noise
batch_size = 10      #Size of the batch
disc_factor = 0.99      #disc_factor factor gamma, used in the calculation of the total disc_factored reward
learning_rate = 0.005
min_actions=0.0

max_action=1.0
noise = 0.1           #Target network update rate
layer1_size = 400
layer2_size = 300
# policy_noise = 0.2      #STD of Gaussian noise added to the actions for the exploration purposes
noise_clip = 0.5        #Maximum value of the Gaussian noise added to the actions (policy)
policy_net_update = 2         #Number of iterations to wait before the policy network (Actor model) is updated


# We create a file name for the two saved models: the Actor and Critic models

file_name = "%s_%s_%s" % ("TD3", "CEF", str(seed))
print ("---------------------------------------")
print ("---------------------------------------")
print ("Settings: %s" % (file_name))
print ("---------------------------------------")

# We create a folder inside which will be saved the trained models

if not os.path.exists("./results"):
  os.makedirs("./results")
if save_models and not os.path.exists("./pytorch_models"):
  os.makedirs("./pytorch_models")

env = Environment()

#set seeds and we get the necessary information on the states and actions in the chosen environment
state_dim = env.get_observation()[2].shape[0]
print("state_dim", state_dim)
Number_actions = env.NumAcceNetwork+env.NumCoreNetwork+env.NumCloudNetwork
print("actor_action_dim", Number_actions)
critic_action_dim = env.NumCoreNetwork*env.NumAcceNetwork*Number_actions
print("critic_action_dim", critic_action_dim)

# create the policy network (the Actor model)
policy = Agent(state_dim, Number_actions, critic_action_dim,layer1_size,layer2_size,min_actions,max_action)

# create the Experience Replay memory
replay_buffer = ReplayBuffer()

# define a list where all the evaluation results over 10 episodes are stored
evaluations = [evaluate_policy(policy)]

# create a new folder directory in which the final results (videos of the agent) will be populated
def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
work_dir = mkdir('exp', 'brs')
monitor_dir = mkdir(work_dir, 'monitor')

#initialize the variables
max_episode_steps = 10
total_timesteps = 0
timesteps_since_eval = 0
episode_latency = 0
episode_reward = 0
episode_timesteps = 0
episode_num = 0
done = True
intil_time = time.time()

# Training
start = timeit.default_timer()
# ep_latency = []
ep_reward = []
# eval_latency = []
# eval_reward = []

# We start the main loop over 500,000 timesteps
while total_timesteps < max_timesteps:
    if done: # If the episode is done
    
        if total_timesteps != 0:    # If we are not at the very beginning, we start the training process of the model
            print("Total Timesteps: {} Episode Num: {} latency: {} reward: {}".format(total_timesteps, episode_num, episode_latency,episode_reward))
            
            policy.learn(replay_buffer, episode_timesteps, batch_size, disc_factor, learning_rate, noise, noise_clip, policy_net_update)
            # ep_latency.append(episode_latency) 
            ep_reward.append(episode_reward)
# We evaluate the episode and we save the policy
        if timesteps_since_eval >= eval_freq:
            timesteps_since_eval %= eval_freq
            evaluations.append(evaluate_policy(policy))
            policy.save(file_name, directory="./pytorch_models")
            np.save("./results/%s" % (file_name), evaluations)
      
# When the training step is done, we reset the state of the environment
        ob = env.reset()
        obs1, obs2, obs = env.get_observation()
    
# Set the Done to False
        done = False
    
# Set rewards and episode timesteps to zero
        episode_latency = 0
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    
# Before some timesteps, we play random actions
    if total_timesteps < warmup_step:
        array, action = env.generate_actions()
        action = array
        # print("Action Random", action)
    else:
        start_dtime = timeit.default_timer()
        action = policy.choose_action(np.array(obs))
        # print("Action Agent", action)
        stop_dtime = timeit.default_timer()
        decision_time = stop_dtime - start_dtime
# The agent performs the action in the environment, 
# then reaches the next state and receives the reward
    start_step_time = timeit.default_timer()
    # print("actions", action)
    actionsave = copy.deepcopy(action)
    new_obs, latency, reward, done = env.step(action)
    stop_step_time = timeit.default_timer()
    Step_decision_time = stop_step_time - start_step_time

# We check if the episode is done
    done_bool = 0 if episode_timesteps + 1 == max_episode_steps else float(done)
    
# We increase the total reward
    episode_latency += latency
    # episode_latency /=10
    
    episode_reward += reward
    # episode_reward /=10
    action = actionsave
# We store the new transition into the Experience Replay memory (ReplayBuffer)
    xy = []
    for i in action:
        xy.append(np.concatenate(i).ravel())
    action = np.array(xy).flatten()
    # print("Acctions",action)
    replay_buffer.Transitions((obs, new_obs, action, reward, done_bool))

# We update the state, the episode timestep, the total timesteps, 
# and the timesteps since the evaluation of the policy
    obs = new_obs
    episode_timesteps += 1
    total_timesteps += 1
    timesteps_since_eval += 1
    # ep_reward.append(episode_reward)    
# print("Episode Reward = ", episode_reward)

# We add the last policy evaluation to our list of evaluations and we save our model
evaluations.append(evaluate_policy(policy))
if save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
np.save("./results/%s" % (file_name), evaluations)
stop = timeit.default_timer()
execution_time = stop - start

# stop_dtime = timeit.default_timer()
# decision_time = stop_dtime - start_dtime
evl_latency,evl_reward = evaluate_policy(policy)
# eval_latency.append(evl_latency)
# ep_reward.append(evl_reward)
from datetime import datetime
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

with open("result80k.txt", "a+") as file_object:
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
    file_object.write("Decision Time ")
    file_object.write(str(decision_time))
    file_object.write("\n")
    file_object.write("Evaluation latency = ")
    file_object.write(str(evl_latency))
    file_object.write("\n")
    file_object.write("Evaluation reward = ")
    file_object.write(str(evl_reward))
    file_object.write("\n")
    file_object.write("Program Executed in ")
    file_object.write(str(execution_time))
    file_object.write("\n")
    file_object.write("Step Decision Time ")
    file_object.write(str(Step_decision_time))
    file_object.write("\n")
    file_object.write("Max Reward = ")
    file_object.write(str(max(ep_reward)))
    file_object.write("\n")
    file_object.write("Index = ")
    file_object.write(str(ep_reward.index(max(ep_reward))))    
    file_object.write("\n")
    file_object.write("Cloud Num = ")
    file_object.write(str(env.NumCloudNetwork))
    file_object.write("\n")
    file_object.write("Core Num = ")
    file_object.write(str(env.NumCoreNetwork))    
    file_object.write("\n")
    file_object.write("AccessNetwork Num = ")
    file_object.write(str(env.NumAcceNetwork))
    file_object.write("\n")
    file_object.write("Cloud Capacity = ")
    file_object.write(str(env.miuCloudNetwork))
    file_object.write("\n")
    file_object.write("Core capacity = ")
    file_object.write(str(env.miuCoreNetwork))
    file_object.write("\n")
    file_object.write("Acces capacity = ")
    file_object.write(str(env.miuAcceNetwork))
    file_object.write("\n")    
    file_object.write("Arrival Traffic = ")
    file_object.write(str(env.lam))      
    file_object.write("\n")    
    file_object.write("Trafic Distribution in AccNetwork = ")
    file_object.write(str(env.AccesNetLamda))
    file_object.write("\n")
    file_object.write("Sum =: ")
    file_object.write(str(sum(map(sum, env.AccesNetLamda))))
    file_object.write("\n")
    file_object.write("Trafic Distribution in CoreNetwork = ")
    file_object.write(str(env.CoreNetLamda))
    file_object.write("\n")
    file_object.write("Sum =: ")
    file_object.write(str(sum(env.CoreNetLamda)))
    file_object.write("\n")
    file_object.write("Trafic Distribution in Cloud = ")
    file_object.write(str(env.CloudNetLamda))
    file_object.write("\n")
    file_object.write("Sum =: ")
    file_object.write(str(sum(env.CloudNetLamda)))
    file_object.write("\n")
    file_object.write("State Dimantions = ")
    file_object.write(str(state_dim))
    file_object.write("\n")    
    file_object.write("Number of Actions = ")
    file_object.write(str(Number_actions))      
    file_object.write("\n")    
    file_object.write("Number of Critic Actions =")
    file_object.write(str(critic_action_dim))
    file_object.write("\n")       
    
print ("---------------------------------------")

print ("Average Latency: %.20f" % (evl_latency))
print ("Average Reward: %.20f" % (evl_reward))
print("Decision Time "+str(decision_time))
print ("---------------------------------------")
print("  ")
print("Training done in "+str(execution_time))
print("  ")
print("Max", max(ep_reward), "index", ep_reward.index(max(ep_reward)))
print("Step Decision Time "+str(Step_decision_time))
print(" ")
print("Trafic Distribution in AccNetwork = : ", sum(map(sum, env.AccesNetLamda)))
print(" ")
print("Trafic Distribution in CoreNetwork = : ", sum(env.CoreNetLamda))
print(" ")
print("Trafic Distribution in Cloud = : ", sum(env.CloudNetLamda))
print()
print("Total trafic: ", sum(map(sum, env.AccesNetLamda))+sum(env.CoreNetLamda)+sum(env.CloudNetLamda))

pyplot.title("reward Values for Multi-Agent TD3", fontsize=12, fontweight='bold')
pyplot.xlabel("Number of Iterations")
pyplot.ylabel("Average reward")
x =[i+1 for i in range(max_timesteps)]
pyplot.plot(ep_reward)
pyplot.show()


#++++++++++++++++++++++===============================+++++++++++++++++++++

# class Actor_Network(nn.Module):
  
#   def __init__(self, state_dim, actor_action_dim,layer1_size,layer2_size,min_actions,max_action):
#     super(Actor_Network, self).__init__()
#     self.state_dim = state_dim
#     self.actor_action_dim = actor_action_dim
#     self.layer1_size = layer1_size
#     self.layer2_size = layer2_size
    
#     self.actor_1 = nn.Linear(self.state_dim, self.layer1_size)
#     self.actor_2 = nn.Linear(self.layer1_size,self.layer2_size)
#     self.actor_3 = nn.Linear(self.layer2_size,self.layer2_size)
    
#     self.actor_11 = nn.Linear(self.layer2_size,self.layer1_size)
#     self.actor_12 = nn.Linear(self.layer1_size,self.layer2_size)
#     self.actor_13 = nn.Linear(self.layer2_size, self.actor_action_dim)
    
#     self.actor_21 = nn.Linear(self.layer2_size,self.layer1_size)
#     self.actor_22 = nn.Linear(self.layer1_size,self.layer2_size)
#     self.actor_23 = nn.Linear(self.layer2_size, self.actor_action_dim)
    
#     self.actor_31 = nn.Linear(self.layer2_size,self.layer1_size)
#     self.actor_32 = nn.Linear(self.layer1_size,self.layer2_size)
#     self.actor_33 = nn.Linear(self.layer2_size, self.actor_action_dim)
    
#     self.actor_41 = nn.Linear(self.layer2_size,self.layer1_size)
#     self.actor_42 = nn.Linear(self.layer1_size,self.layer2_size)
#     self.actor_43 = nn.Linear(self.layer2_size, self.actor_action_dim)
    
#     self.actor_51 = nn.Linear(self.layer2_size,self.layer1_size)
#     self.actor_52 = nn.Linear(self.layer1_size,self.layer2_size)
#     self.actor_53 = nn.Linear(self.layer2_size, self.actor_action_dim)
    
#     self.actor_61 = nn.Linear(self.layer2_size,self.layer1_size)
#     self.actor_62 = nn.Linear(self.layer1_size,self.layer2_size)
#     self.actor_63 = nn.Linear(self.layer2_size, self.actor_action_dim)
    
#     self.actor_71 = nn.Linear(self.layer2_size,self.layer1_size)
#     self.actor_72 = nn.Linear(self.layer1_size,self.layer2_size)
#     self.actor_73 = nn.Linear(self.layer2_size, self.actor_action_dim)
    
#     self.actor_81 = nn.Linear(self.layer2_size,self.layer1_size)
#     self.actor_82 = nn.Linear(self.layer1_size,self.layer2_size)
#     self.actor_83 = nn.Linear(self.layer2_size, self.actor_action_dim)

    
    
#   def forward(self, stat):                         
#     stat = F.relu(self.actor_1(stat))
#     stat = F.relu(self.actor_2(stat))
#     stat = F.relu(self.actor_3(stat))
 
#     l1 = F.relu(self.actor_11(stat))
#     l1 = F.relu(self.actor_12(l1))
#     l1 = torch.sigmoid(self.actor_13(l1))
    
#     l2 = F.relu(self.actor_21(stat))
#     l2 = F.relu(self.actor_22(l2))
#     l2 = torch.sigmoid(self.actor_23(l2))
     
#     l3 = F.relu(self.actor_31(stat))
#     l3 = F.relu(self.actor_32(l3))
#     l3 = torch.sigmoid(self.actor_33(l3))
    
#     l4 = F.relu(self.actor_41(stat))
#     l4 = F.relu(self.actor_42(l4))
#     l4 = torch.sigmoid(self.actor_43(l4))
    
#     l5 = F.relu(self.actor_51(stat))
#     l5 = F.relu(self.actor_52(l5))
#     l5 = torch.sigmoid(self.actor_53(l5))
    
#     l6 = F.relu(self.actor_61(stat))
#     l6 = F.relu(self.actor_62(l6))
#     l6 = torch.sigmoid(self.actor_63(l6))
     
#     l7 = F.relu(self.actor_71(stat))
#     l7 = F.relu(self.actor_72(l7))
#     l7 = torch.sigmoid(self.actor_73(l7))
    
#     l8 = F.relu(self.actor_81(stat))
#     l8 = F.relu(self.actor_82(l8))
#     l8 = torch.sigmoid(self.actor_83(l8))

    
#     # return l1,l2,l3,l4,l5,l6
#     return l1,l2,l3,l4,l5,l6,l7,l8
# # We build two neural networks for the two Critic models and two neural networks for the two Critic targets
# class Critic_Network(nn.Module):
  
#   def __init__(self, state_dim, critic_action_dim,layer1_size,layer2_size,min_actions,max_action):
#     super(Critic_Network, self).__init__()
    
#     self.critic_action_dim=critic_action_dim
#     self.state_dim=state_dim
#     self.layer1_size = layer1_size
#     self.layer2_size = layer2_size   
# # Forward_Propagation of the first Critic neural network
#     self.critic_1 = nn.Linear(self.state_dim + self.critic_action_dim, self.layer1_size)
#     self.critic_2 = nn.Linear(self.layer1_size,self.layer2_size)
#     self.critic_3 = nn.Linear(self.layer2_size, 1)
    
# # Forward_Propagation of the second Critic neural network
#     self.critic_4 = nn.Linear(self.state_dim + self.critic_action_dim, self.layer1_size)
#     self.critic_5 = nn.Linear(self.layer1_size,self.layer2_size)
#     self.critic_6 = nn.Linear(self.layer2_size, 1)

#   def forward(self, s, a):         
#     SA = torch.cat([s, a], 1)       
    
# # Forward-Propagation on the first Critic Neural Network
#     q1_value = F.relu(self.critic_1(SA))
#     q1_value = F.relu(self.critic_2(q1_value))
#     q1_value = self.critic_3(q1_value)

# # Forward-Propagation on the second Critic Neural Network
#     q2_value = F.relu(self.critic_4(SA))
#     q2_value = F.relu(self.critic_5(q2_value))
#     q2_value = self.critic_6(q2_value)
    
#     return q1_value,q2_value
   
#   def Q1(self, s, a):
#     SA = torch.cat([s, a], 1)
#     q1_value = F.relu(self.critic_1(SA))
#     q1_value = F.relu(self.critic_2(q1_value))
#     q1_value = self.critic_3(q1_value)
#     return q1_value



# # Selecting the device (CPU or GPU)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Building the whole Training Process into a class
# class Agent(object):
  
#   def __init__(self, state_dim, actor_action_dim, critic_action_dim,layer1_size,layer2_size,min_actions,max_action):
#     self.min_actions = 0.0
#     self.max_action = 1.0      
#     # # self.noise = noise
#     self.layer1_size = layer1_size
#     self.layer2_size = layer2_size
    
#     self.actor = Actor_Network(state_dim, actor_action_dim,layer1_size,layer2_size,min_actions,max_action).to(device)
#     self.actor_target = Actor_Network(state_dim, actor_action_dim,layer1_size,layer2_size,min_actions,max_action).to(device)
#     self.actor_target.load_state_dict(self.actor.state_dict())
#     self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
    
#     self.critic = Critic_Network(state_dim, critic_action_dim,layer1_size,layer2_size,min_actions,max_action).to(device)
#     self.critic_target = Critic_Network(state_dim, critic_action_dim,layer1_size,layer2_size,min_actions,max_action).to(device)
#     self.critic_target.load_state_dict(self.critic.state_dict())
#     self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
    


#   # def choose_action(self, observation):
#   #       state = torch.Tensor(observation.reshape(1, -1)).to(device)
#   #       # print("state", state)
#   #       st = self.actor(state)
#   #       # print("xl", st)
#   #       lst = []
#   #       n = env.NumAcceNetwork
#   #       for i in st:
#   #           # print("I", i)
#   #           lst.append(i.cpu().data.numpy().flatten())
#   #           # print("LSt", lst)
#   #       acn = [lst[i:i + n] for i in range(0, len(lst), n)]
#   #       # ac = np.array(acn)
#   #       # if expl_noise != 0:
#   #       #      acn=noise(acn)
#   #       # print("acn", acn)
#   #       return acn
#   def choose_action(self, observation):
#       state = torch.tensor(observation.reshape(1,-1),dtype=torch.float).to(device)
        
#       st = torch.cat(self.actor.forward(state)).to(device)
        
#       action = []
#       n = env.NumAcceNetwork
#       for i in st:
#           action.append(i.cpu().data.numpy().flatten())
#       # print("action",action) 
#       acn = [action[i:i + n] for i in range(0, len(action), n)]
#       nr_actions =[]
#       for i in range(len(acn)):
#           totalAcc = sum(map(sum,acn[i]))
#                    # print("totalAcc", totalAcc)
#       for j in range(len(acn)):
#           nr_actions.append(acn[j]/totalAcc)
#       # nr_actions =[]
#       # for i in range(len(acn)):
#       #     for j in range(len(acn[i])):
#       #         totalAcc = sum(acn[i][j])
#       #           # print("totalAcc",totalAcc)
#       #     nr_actions.append(acn[i]/totalAcc)
            
#       # print("nr_actions",nr_actions)        
#       return nr_actions 
#         # state = torch.tensor(observation.reshape(1,-1),dtype=torch.float).to(device)
#         # # print("state",state)  
#         # muPr = torch.cat(self.actor(state))
#         # # print("muPr",muPr)
#         # mu_prime = muPr + torch.tensor(np.random.normal(scale=0.1),dtype=torch.float).to(device)
#         # # print("mu_Prime", mu_prime)
#         # mu_prime = torch.clamp(mu_prime, self.min_actions, self.max_action)
#         # # print("mu_PrimeClamp", mu_prime)
#         # n = env.NumAcceNetwork
#         # ls = []
#         # for i in mu_prime:
#         #     ls.append(i.cpu().detach().numpy().flatten())
#         #     # print("ls", ls)
#         # muPrime=[ls[i:i + n] for i in range(0, len(ls), n)]
#         # # print("muPrimeList", muPrime) 
#         # return muPrime                    
           
#   def learn(self, replay_buffer, iterations, batch_size=100, disc_factor=0.99,
#             learning_rate=0.005, noise=0.1,noise_clip=0.5, policy_net_update=2):
    
#     for itr in range(iterations):
      
# # We sample a batch of transitions (s, s’, a, r) from the memory
#       states, next_states, actions, rewards, dones = replay_buffer.SampleBuffer(batch_size)
#       state = torch.Tensor(states).to(device)
#       new_state = torch.Tensor(next_states).to(device)
#       action = torch.Tensor(actions).to(device)
#       reward = torch.Tensor(rewards).to(device)
#       done = torch.Tensor(dones).to(device)
      
# # From the new state s’, the Actor target plays the next action a’
#       l1,l2,l3,l4,l5,l6,l7,l8 = self.actor_target(new_state)
#       next_action = torch.cat([l1,l2,l3,l4,l5,l6,l7,l8],1)
#       next_action = next_action + torch.clamp(torch.tensor(np.random.normal(scale=noise)), -noise_clip, noise_clip)
#       next_action = torch.clamp(next_action, self.min_actions,self.max_action)
     
  
# # The two Critic targets take each the couple (s’, a’) as input and return two Q-values Qt1(s’,a’) and Qt2(s’,a’) as outputs
#       Q1_value, Q2_value = self.critic_target(new_state, next_action)

# # because the target Q is still in the computation graph, then it need to be dettached inorder to add with the reward       
# # The two Critic models take each the couple (s, a) as input and return two Q-values Q1(s,a) and Q2(s,a) as outputs      
#       # current_Q1_value, current_Q2_value= self.critic.forward(state, action)
#       current_Q1_value, current_Q2_value= self.critic(state, action)
# # We keep the minimum of these two Q-values: min(Qt1, Qt2)
#       Q_critic_value = torch.min(Q1_value, Q2_value)
      
# # We get the final target of the two Critic models, which is: Qt = r + γ * min(Qt1, Qt2), where γ is the disc_factor factor
#       Q_critic_value = reward + ((1 - done) * disc_factor * Q_critic_value).detach()
#       self.critic_optimizer.zero_grad()         
# # We compute the loss coming from the two Critic models: Critic Loss = MSE_Loss(Q1(s,a), Qt) + MSE_Loss(Q2(s,a), Qt)
#       critic_loss = F.mse_loss(current_Q1_value, Q_critic_value) + F.mse_loss(current_Q2_value,Q_critic_value)

# # We backpropagate this Critic loss and update the parameters of the two Critic models with a SGD optimizer
#       # self.critic_optimizer.zero_grad()
#       critic_loss.backward()
#       self.critic_optimizer.step()
        
# # Once every two iterations, we update our Actor model by performing gradient ascent on the output of the first Critic model
#       if itr % policy_net_update != 0:
#           return
#       self.actor_optimizer.zero_grad()
#       l1,l2,l3,l4,l5,l6,l7,l8 = self.actor(state)
#       action = torch.cat([l1,l2,l3,l4,l5,l6,l7,l8],1)
#       # actor_loss = self.critic.Q1(state,action).mean()
#       actor_q1_loss = self.critic.Q1(state,action)
#       actor_loss = -torch.mean(actor_q1_loss)
#       # self.actor_optimizer.zero_grad()
#       actor_loss.backward()
#       self.actor_optimizer.step()
        
# # Still once every two iterations, we update the weights of the Actor target by polyak averaging
#       for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
#           target_param.data.copy_(learning_rate * param.data + (1 - learning_rate) * target_param.data)
        
# # Still once every two iterations, we update the weights of the Critic target by polyak averaging
#       for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
#           target_param.data.copy_(learning_rate * param.data + (1 - learning_rate) * target_param.data)
  
# # Making a save method to save a trained model
#   def save(self, filename, directory):
#     torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
#     torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
  
# # Making a load method to load a pre-trained model
#   def load(self, filename, directory):
#     self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
#     self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
    
    
# # We make a function that evaluates the policy by calculating its average reward over 10 episodes
# eva_reward = []
# evl_latency = []
# import timeit
# def evaluate_policy(policy, eval_episodes=10):
#     total_latency = 0.0
#     total_reward = 0.0
#     for i in range(eval_episodes):
#         obs = env.get_observation()[2]
#         done = False
#         while not done:
           
#             action = policy.choose_action(np.array(obs))
            
#             obs, latency, reward, done = env.step(action)
            
#             total_latency += latency
#             total_reward += reward
#             evl_latency.append(total_latency)
#         avg_reward = total_reward /(env.NumAcceNetwork*eval_episodes)
#     # avg_latency = total_latency/10    
#     avg_latency = np.mean(evl_latency[-10:])
#     from datetime import datetime
#     now = datetime.now()
#     dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    
#     with open("result80k.txt", "a+") as file_object:
#         # Move read cursor to the start of file.
#         file_object.seek(0)
#         # If file is not empty then append '\n'
#         data = file_object.read(100)
#         if len(data) > 0 :
#             file_object.write("\n")
#         # Append text at the end of file
#         file_object.write("\n")
#         file_object.write("############")
#         file_object.write(dt_string)
#         file_object.write("#############")
#         file_object.write("\n")
        
#         file_object.write("Evaluation latency = ")
#         file_object.write(str(avg_latency))
#         file_object.write("\n")
#         file_object.write("Evaluation reward = ")
#         file_object.write(str(avg_reward))
        
#         file_object.write("\n")
#         file_object.write("State Dimantions = ")
#         file_object.write(str(state_dim))
#         file_object.write("\n")    
#         file_object.write("Number of Actions = ")
#         file_object.write(str(Number_actions))      
#         file_object.write("\n")    
#         file_object.write("Number of Critic Actions =")
#         file_object.write(str(critic_action_dim))
#         file_object.write("\n")       
#     return avg_latency, avg_reward   

# seed = 0

# file_name = "%s_%s_%s" % ("TD3", "CEF", str(seed))
# print ("---------------------------------------")
# print ("---------------------------------------")
# print ("Settings: %s" % (file_name))
# print ("---------------------------------------")

# eval_episodes = 10
# save_env_vid = False
# env = Environment()
# max_episode_steps = 10

# state_dim = env.get_observation()[2].shape[0]

# Number_actions = env.NumAcceNetwork+env.NumCoreNetwork+env.NumCloudNetwork

# critic_action_dim = env.NumCoreNetwork*env.NumAcceNetwork*Number_actions


# # create the policy network (the Actor model)
# policy = Agent(state_dim, Number_actions, critic_action_dim,layer1_size,layer2_size,min_actions,max_action)

# policy.load(file_name, './pytorch_models/')
# _ = evaluate_policy(policy, eval_episodes=eval_episodes)
