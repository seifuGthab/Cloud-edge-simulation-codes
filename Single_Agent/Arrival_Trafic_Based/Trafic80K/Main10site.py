#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 21:58:35 2022

@author: seifu
"""

import os
import numpy as np
from matplotlib import pyplot
from CEF_EnvironmentV1 import Environment
from CEF_TD3V22 import Agent
if __name__ == '__main__':
    env = Environment()
    agent = Agent(alpha=0.001,beta=0.001,state_dim = env.get_observation()[2].shape[0], 
                  tau=0.005, env=env,batch_size=100,layer1_size=400,layer2_size=300,
                  actor_action_dim = env.NumAcceNetwork + env.NumCoreNetwork + env.NumCloudNetwork,
                 critic_action_dim = (env.NumCoreNetwork*env.NumAcceNetwork)*
                 (env.NumAcceNetwork+env.NumCoreNetwork+env.NumCloudNetwork),
                 time_step =0, warmup=10)
    
    # def mkdir(base, name):
    #     path = os.path.join(base, name)
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     return path
    # work_dir = mkdir('exp', 'brs')
    # monitor_dir = mkdir(work_dir, 'monitor')
    
    state_dim = env.get_observation()[2].shape[0]
    print("state_dim", state_dim)
    actor_action_dim = env.NumAcceNetwork + env.NumCoreNetwork + env.NumCloudNetwork
    print("actor_action_dim", actor_action_dim)
    critic_action_dim = env.NumCoreNetwork*env.NumAcceNetwork*actor_action_dim
    print("critic_action_dim", critic_action_dim)
    # critic_action_dim = (env.NumAcceNetwork+env.NumCoreNetwork+env.NumCloudNetwork)*env.NumAcceNetwork
    # state_dim = env.get_observation()[2].shape[0]
    # print("state_dim ",state_dim)
    # actor_action_dim = env.NumAcceNetwork+env.NumCoreNetwork+env.NumCloudNetwork
    # print("Number_actions ",actor_action_dim)
    # critic_action_dim = (env.NumCoreNetwork+env.NumAcceNetwork+env.NumCloudNetwork)*env.NumAcceNetwork
    # print("critic_action_dim",critic_action_dim)

    n_games = 100
    filename = 'plots/' + 'CEF_EnvironmentV1_' + str(n_games) +'_games.png'
    # if not os.path.exists("./td3"):
    #   os.makedirs(".tmp/td3")
    # if agent.save_models and not os.path.exists(".tmp/td3"):
    #   os.makedirs(".tmp/td3")
    import timeit    
    start = timeit.default_timer()
    best_reward = 0.0
    score_history = []
    # agent.load_models()
    # print(env.action_space.shape[0])
    for i in range(n_games):
        observation = env.reset()
        # ob = env.reset()
        obs1, obs2, observation = env.get_observation()
        # print("observation", observation)   
        done = False
        epsoid_reward = 0
        while not done:
            # action = agent.choose_action(observation)
            if agent.warmup < agent.time_step:
                array, action = env.generate_actions()
                action = array
            # actiony = [action[y:y+int(critic_action_dim/env.NumCoreNetwork)] for y in range(0, len(action), int(critic_action_dim/env.NumCoreNetwork))] 
            # ino =0
            # for i in actiony:
            #     actiony[ino] = [i[y:y+actor_action_dim] for y in range(0, len(action), actor_action_dim)] 
            #     ino+=1
            # action = actiony
    #             print("Random action", action)
    # # print("     Random Action    ")
            else:
                action = agent.choose_action(np.array(observation))
    #     # print("TD3 action", policy)
    #     # print("    Neural Action   ", action)
    #     # if expl_noise != 0:
    #     #     action=noise(action)
    #             print("TD3 action with noise", action)            
            # action = agent.choose_action(np.array(observation))
            # print(action)
            agent.time_step+=1
            observation_, reward, done = env.step(action)
            xy = []
            for j in action:
                xy.append(np.concatenate(j).ravel())
            action = np.array(xy).flatten()
            
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            
            epsoid_reward += reward
            observation = observation_
        
        score_history.append(epsoid_reward)
        avg_reward =epsoid_reward /10
        # avg_score =np.mean(score_history)
        # agent.save_models()
        if avg_reward > best_reward:
            best_reward = avg_reward
            agent.save_models()
            
        print('Episode_Number ', i, 'Epsoid_Reward %.8f' % epsoid_reward, 'Average Reward %.8f' % avg_reward)
    
    stop = timeit.default_timer()
    execution_time = stop - start
    
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    with open("result.txt", "a+") as file_object:
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
        file_object.write("Average Reward = ")
        file_object.write(str(avg_reward))
        file_object.write("\n")
        file_object.write("Program Executed in ")
        file_object.write(str(execution_time))
        file_object.write("\n")
        file_object.write("Number of Cloud = ")
        file_object.write(str(env.NumCloudNetwork))
        file_object.write("\n")
        file_object.write("Number of Cores = ")
        file_object.write(str(env.NumCoreNetwork))    
        file_object.write("\n")
        file_object.write("Number of AccessNetwork = ")
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
        file_object.write("Traffic = ")
        file_object.write(str(env.lamda))
        file_object.write("\n")
        file_object.write("===Hyperparameters===")
        file_object.write("\n")
        
    print ("---------------------------------------")
    print ("Average Reward: %f" % (avg_reward))
    print ("---------------------------------------")
    print("Training done in "+str(execution_time))
    # print("Reward", Ev_Reward)
    print()

    pyplot.title("Reward Values for Single-Agent TD3", fontsize=12, fontweight='bold')
    pyplot.xlabel("Number of Iterations")
    pyplot.ylabel("Average Reward")
    pyplot.plot(score_history)
    pyplot.show()
        
        # x = [i+1 for i in range(n_games)]
        # plot_learning_curve(x, score_history, filename)
    # while total_timesteps < max_timesteps:
    #     if done: # If the episode is done
        
    #         if total_timesteps != 0:    # If we are not at the very beginning, we start the training process of the model
    #             print("Total Timesteps: {} Episode Num: {} Reward: {}".format(total_timesteps, episode_num, episode_reward))
    #             policy.learn(replay_buffer, episode_timesteps, batch_size, disc_factor, learning_rate, noise_clip, policy_noise, policy_net_update)
    #             ep_reward.append(episode_reward) 
    # # We evaluate the episode and we save the policy
    #         # evaluations = [evaluate_policy(policy)]
    #         if timesteps_since_eval >= eval_freq:
    #             timesteps_since_eval %= eval_freq
    #             evaluations.append(evaluate_policy(policy))
    #             policy.save(file_name, directory="./pytorch_models")
    #             np.save("./results/%s" % (file_name), evaluations)
          
    # # When the training step is done, we reset the state of the environment
    #         ob = env.reset()
    #         obs1, obs2, obs = env.get_observation()
        
    # # Set the Done to False
    #         done = False
        
    # # Set rewards and episode timesteps to zero
    #         episode_reward = 0
    #         episode_timesteps = 0
    #         episode_num += 1

        
    # # Before 2000 timesteps, we play random actions
    #     if total_timesteps < warmup_step:
    #         array, action = env.generate_actions()
    #         # print("action",action)
    #         action = array
    #         # print("action",action)
    #         # actiony = [action[y:y+int(critic_action_dim/env.NumCoreNetwork)] for y in range(0, len(action), int(critic_action_dim/env.NumCoreNetwork))] 
    #         # ino =0
    #         # for i in actiony:
    #         #     actiony[ino] = [i[y:y+actor_action_dim] for y in range(0, len(action), actor_action_dim)] 
    #         #     ino+=1
    #         # action = actiony
    #         # print("Random action", action)
    #     # print("     Random Action    ")
    #     else:
    #         action = policy.choose_action(np.array(obs))
    #         # print("Taction", action)
    #         # print("    Neural Action   ", action)
    #         # if expl_noise != 0:
    #         #     action=noise(action)
    #             # print("TD3 action with noise", action)
    # # If the explore_noise parameter is not 0, we add noise to the action and we clip it
    #         # actiony = [action[y:y+int(critic_action_dim/env.NumCoreNetwork)] for y in range(0, len(action), int(critic_action_dim/env.NumCoreNetwork))] 
    #         # ino =0
    #         # for i in actiony:
    #         #     actiony[ino] = [i[y:y+actor_action_dim] for y in range(0, len(action), actor_action_dim)] 
    #         #     ino+=1
    #         # action = actiony[0][0]
    #         # for i in action:
    #         # if expl_noise != 0:
    #         #     action=noise(action)
    #         #     print("AAction", action)
    # # The agent performs the action in the environment, 
    # # then reaches the next state and receives the reward
    #     new_obs, reward, done = env.step(action)
    #     # print("action", action)
    # # We check if the episode is done
    #     done_bool = 0 if episode_timesteps + 1 == max_episode_steps else float(done)
    #     # print("  step reward = ", reward)
      
    # # We increase the total reward
    #     episode_reward += reward
    # # We store the new transition into the Experience Replay memory (ReplayBuffer)
    #     xy = []
    #     for i in action:
    #         xy.append(np.concatenate(i).ravel())
    #     action = np.array(xy).flatten()
      
    #     replay_buffer.Transitions((obs, new_obs, action, reward, done_bool))

    # # We update the state, the episode timestep, the total timesteps, 
    # # and the timesteps since the evaluation of the policy
    #     obs = new_obs
    #     episode_timesteps += 1
    #     total_timesteps += 1
    #     timesteps_since_eval += 1
    # # ep_reward.append(episode_reward)    
    # # print("Episode Reward = ", episode_reward)

    # # We add the last policy evaluation to our list of evaluations and we save our model
    # evaluations.append(evaluate_policy(policy))
    # if save_models: policy.save("%s" % (file_name), directory="./pytorch_models")
    # np.save("./results/%s" % (file_name), evaluations)
    # stop = timeit.default_timer()
    # execution_time = stop - start
    # ev_reward = evaluate_policy(policy)

    