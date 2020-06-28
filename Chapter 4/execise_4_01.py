# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 12:30:37 2020

@author: mauro

Exercise 4.1 from Introduction to Reinforcement Learning, Sutton and Barto
"""
import numpy as np
import random
import copy

class GridWorld:
    def __init__(self):
        self.value = np.zeros((4,4))

class State:
    def __init__(self):
        self.world = GridWorld()
        self.gamma = 0.9
        
    def get_value(self, coord):
        return self.world.value[coord[0],coord[1]] 

    def get_reward(self, coord):
        if (coord == [0,0] or coord == [3,3]):
            return 0
        else:
            return -1
        
    def move(self, coord, action):
        """
        Actions can be 0,1,2,3 : North, East, South, West
        """
        new_coord = copy.deepcopy(coord)
        if (coord != [0,0] and coord != [3,3]):
            if action == 0 and coord[1] - 1 >= 0:
                new_coord[1] = new_coord[1] - 1
            elif action == 1 and coord[0] + 1 <= 3:
                new_coord[0] = new_coord[0] + 1
            elif action == 2 and coord[1] + 1 <= 3:
                new_coord[1] = new_coord[1] + 1
            elif action == 3 and coord[0] -1 >= 0:
                new_coord[0] = new_coord[0] - 1
        return new_coord
    
    def update_value(self, old_coord):
        value = 0
        for a in range(0,4):
            r = self.get_reward(old_coord)
            new_coord = self.move(old_coord, a)
            value += r + self.gamma * self.get_value(new_coord)
        value = value / 4
        self.world.value[old_coord[0], old_coord[1]] = value 
        
    def get_q_value(self, coord, a):  
        r = self.get_reward(coord)
        next_coord = self.move(coord, a)
        return r + self.gamma * self.get_value(next_coord)
       

if __name__=='__main__':    
    agent = State()
    for epoch in range(1000):
        for x in range(0,4):
            for y in range(0,4):
                agent.update_value([x,y])
                
        print(f"Epoch {epoch}")
        print(agent.world.value)
        q_value = agent.get_q_value([3,1], 2)
        print(f"Q-value is equal to {q_value}")
        
        
        
        
        
    

            
            
 
        