# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 12:25:46 2020
@author: mauro
"""
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import itertools 
import pickle

class PolicyIteration:
    def __init__(self):
        self.value = np.zeros((21,21))
        self.policy = np.zeros((21,21))
        self.gamma = 0.9
        self.epsilon = 0.1
    
    # Move cars
    def action(self, state):
        return self.policy[state[0], state[1]]           
    
    def get_reward(self, state, cars_requested, cars_returned, cars_moved):
        reward = 0
        reward -= np.abs(cars_moved)*2   # positive: from 1 to 2. negative: from 2 to 1
        cars_available = [min(state[0] - cars_moved + cars_returned[0], 20), min(state[1] + cars_moved + cars_returned[1], 20)]
        cars_requested_true = [min(cars_available[0], cars_requested[0]), min(cars_available[1], cars_requested[1])]
        reward += 10*cars_requested_true[0] + 10*cars_requested_true[1]
        new_state = [int(cars_available[0] - cars_requested_true[0]), int(cars_available[1] - cars_requested_true[1])]
        return reward, new_state  
    
    def plot_value_function(self):
        sns.heatmap(self.value)
        plt.title("Jack's Rental Problem: value function")
        plt.show()
    
    def plot_policy(self):
        sns.heatmap(self.policy)
        plt.title("Jack's Rental Problem: policy")
        plt.show()
     
        
class RentalCompany:
    def __init__(self, lambda_request, lambda_return):
        self.lambda_request = lambda_request
        self.lambda_return = lambda_return
        self.prob_request = np.array([(math.exp(-self.lambda_request)*self.lambda_request**i)/math.factorial(i) for i in range(0,21)])
        self.prob_return =  np.array([(math.exp(-self.lambda_return)*self.lambda_return**i)/math.factorial(i) for i in range(0,21)])
    
    def cars_returned(self):
        n_returns = random.choices(np.arange(0,21), weights=self.prob_return)  
        prob = self.prob_return[n_returns]
        return n_returns[0], prob
    
    def cars_requested(self):
        n_requests = random.choices(np.arange(0,21), weights=self.prob_request)   
        prob = self.prob_request[n_requests]
        return n_requests[0] , prob
    
if __name__ == '__main__':
    policy_iteration = PolicyIteration()
    rental_A = RentalCompany(3, 3)
    rental_B = RentalCompany(4, 2)
    while(1):
        # Policy evaluation
        print("Policy evaluation")
        policy_iteration.epsilon = policy_iteration.epsilon
        while(1):     
            delta = 0        
            for i in tqdm(range(0,21), position=0):
                for j in range(0,21):
                    state = [i, j]
                    value = policy_iteration.value[i, j]
                    cars_moved = policy_iteration.action(state)                     
                    cumulative_newvalue = 0
                    reqretAB = [p for p in itertools.product(list(range(0,10)), repeat=4)]    # Avoid 4 nested loops
                    for tupla in reqretAB:     # 0: reqA, 1: retA, 2: reqB, 3: retB
                        prob = rental_A.prob_request[tupla[0]] * rental_A.prob_return[tupla[1]] * \
                                        rental_B.prob_request[tupla[2]] * rental_B.prob_return[tupla[3]] 
                        reward, new_state = policy_iteration.get_reward(state, [tupla[0], tupla[2]], [tupla[1], tupla[3]], cars_moved)                    
                        cumulative_newvalue += (prob * (reward + policy_iteration.gamma * policy_iteration.value[new_state[0], new_state[1]]))
                    policy_iteration.value[i, j] = cumulative_newvalue
                    delta = max(delta, abs(value - policy_iteration.value[i, j]))
            # Stop when delta is lower than epsilon
            print(f"delta: {delta}, epsilon: {policy_iteration.epsilon}")
            if delta < policy_iteration.epsilon:
                policy_iteration.plot_policy()
                policy_iteration.plot_value_function()
                break
   
        
        # Policy improvement
        print("Policy improvement")
        policy_stable = True
        for i in tqdm(range(0,21), position=0):
            for j in range(0,21):
                state = [i, j]
                temp_action = policy_iteration.action(state)
                min_action_rng = -min(5,j)
                max_action_rng = min(5,i)
                action_values = dict()
                for action in range(min_action_rng, max_action_rng+1):
                    cars_moved = action 
                    list_values = []
                    reqretAB = [p for p in itertools.product(list(range(0,10)), repeat=4)]    # Avoid 4 nested loops
                    for tupla in reqretAB:     # 0: reqA, 1: retA, 2: reqB, 3: retB
                        prob = rental_A.prob_request[tupla[0]] * rental_A.prob_return[tupla[1]] * \
                                        rental_B.prob_request[tupla[2]] * rental_B.prob_return[tupla[3]] 
                        reward, new_state = policy_iteration.get_reward(state, [tupla[0], tupla[2]], [tupla[1], tupla[3]], cars_moved)                    
                        list_values.append(prob * (reward + policy_iteration.gamma * policy_iteration.value[new_state[0], new_state[1]]))
                    list_values = np.array(list_values)
                    action_values[action] = np.sum(list_values)#/194481
                best_value = np.amax(np.fromiter(action_values.values(), dtype=float))
                for key, value in action_values.items():
                    if value == best_value:
                        best_action = key
                if best_action!=temp_action:
                    policy_iteration.policy[i,j] = best_action
                    policy_stable = False
        if policy_stable:
            # Save policy and value function
            pickle.dump(policy_iteration.policy, open("policy.pkl", "wb"))
            pickle.dump(policy_iteration.value, open("value_function.pkl", "wb"))
            print("Done!")       
            break
    pickle.dump(policy_iteration.policy, open("policy.pkl", "wb"))
    pickle.dump(policy_iteration.value, open("value_function.pkl", "wb"))
    policy_iteration.plot_policy()
    policy_iteration.plot_value_function()
    
sns.heatmap(policy_iteration.policy)
plt.title("Jack's Rental Problem: policy")
plt.show()